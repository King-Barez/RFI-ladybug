/*
================================================================================
LadybugRFI sender — Architecture, threading model, and wire protocol (current)

Goal
- Grab frames from a FLIR Ladybug camera.
- Color-convert per-camera images (optionally downsampled by the Ladybug SDK).
- Crop a bottom-anchored square region from each camera, resize to OUT_ROWS x
  OUT_COLS, convert to BGR 3 BPP, and send over TCP to a receiver.

Threads
- Producer (capture/process, main thread):
  - ladybugGrabImage() -> ladybugConvertImage() -> crop/resize -> write
    into a mailbox buffer cell and publish its index atomically.
  - Only the latest fully written cell is made visible to the consumer.
  - First-frame diagnostics (sizes before/after convert and resize) are printed
    once per run and then suppressed.

- Consumer (sender thread):
  - Waits for a publication (condition_variable) and atomically grabs the latest
    published cell index (“latest-wins”).
  - Connects (or reconnects) to the receiver, sends a frame header (NetHeader),
    then N camera images back-to-back. If send fails, it closes and will
    reconnect on the next iteration.

Mailbox buffering (latest-wins)
- A small pool of buffers (typically 3) acts as a mailbox.
- Producer writes into a free cell and atomically publishes its index when done.
- Consumer reads the most recent published index and may skip older cells.
- Effect: low latency with graceful frame dropping under load. No hard
  backpressure; producer never blocks on a slow network.

Wire protocol (LDB0 v1)
- Transport: TCP (sender = client, receiver = server).
- Per-frame: 1 fixed-size header + numCams payloads (contiguous images).
- Header fields are big-endian (network order). Receiver validates magic "LDB0".
- Images are sent in BGR 3 BPP, each of size rows*cols*bpp bytes.

Image processing pipeline
- Color processing method is configurable via LBUG_COLOR_METHOD (e.g.,
  DOWNSAMPLE4, DOWNSAMPLE16, DOWNSAMPLE64, NEAREST_NEIGHBOR_FAST, EDGE_SENSING,
  HQLINEAR/HQLINEAR_GPU, RIGOROUS, MONO). Defaults to DOWNSAMPLE4.
- Some methods downsample inside ladybugConvertImage(). We map the method to an
  effective downsample factor (ds) to compute per-camera converted dimensions:
  - DOWNSAMPLE4 or MONO → ds=2; DOWNSAMPLE16 → ds=4; DOWNSAMPLE64 → ds=8; others → ds=1.
- Crop: bottom-anchored square of size min(srcRows, srcCols).
- Resize: nearest neighbor using precomputed x/y maps.
- Output: BGR, 3 bytes per pixel, OUT_ROWS x OUT_COLS for the first NUM_SEND_CAMS cams.

Networking behavior
- Persistent TCP connection; on send failure, the socket is closed and a reconnect
  is attempted on the next loop.
- TCP_NODELAY is enabled; SO_SNDBUF is increased for throughput.
- sendAll() handles partial sends and transient EINTR/EAGAIN; returns false on
  fatal errors so the thread can reconnect.
- Header is sent first, then each camera image. No per-image size fields.

Stats
- Producer: aggregates grab/convert/resize timings and prints about once per second.
- Consumer: aggregates send bandwidth and prints about once per second.

Shutdown
- The process runs until it is terminated externally (e.g., SIGINT/SIGTERM).
- On shutdown, the main thread wakes the sender (if needed), joins the thread,
  then stops and destroys the Ladybug context.

Env vars
- LBUG_TARGET_IP / LBUG_TARGET_PORT: receiver address (client connects to it).
- LBUG_COLOR_METHOD: color processing method (default DOWNSAMPLE4).
- LBUG_NUM_CAMS: number of cameras to send (1..6, default 3).

Edge cases considered
- Camera errors (grab/convert): the frame is skipped and the loop continues.
- Network errors: reconnect and continue; frames may be dropped by design.
- Resolution changes: mapping/scratch recompute on the first usable frame.

================================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstring>
#include <cerrno>
#include <algorithm>
#include <chrono>

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#include "ladybug.h"

// Fixed output geometry (square) for each camera image sent over the wire.
// Output pixels are BGR (3 bytes per pixel), no alpha.
static constexpr uint32_t OUT_ROWS = 640;
static constexpr uint32_t OUT_COLS = 640;
static constexpr uint32_t OUT_BPP  = 3;   // B, G, R

// Number of cameras to send per frame (1..6). Default is 3.
// This controls how many camera images are serialized after the header.
// Note: all six cameras are still converted by the SDK API (tmpDest[6]),
// but we only crop/resize/send the first NUM_SEND_CAMS to cut bandwidth/CPU.
static unsigned int NUM_SEND_CAMS = []{
    const char* v = getenv("LBUG_NUM_CAMS");
    int n = (v && *v) ? atoi(v) : 3;
    if (n < 1) n = 1;
    if (n > 6) n = 6;
    return (unsigned int)n;
}();

// Network target (receiver). Sender acts as TCP client connecting here.
static const char* getEnvOrDefault(const char* k, const char* defv) {
    const char* v = getenv(k);
    return v && *v ? v : defv;
}
static const char* TARGET_IP   = getEnvOrDefault("LBUG_TARGET_IP", "192.168.0.2");
static uint16_t    TARGET_PORT = []{
    const char* p = getenv("LBUG_TARGET_PORT");
    return (uint16_t)(p && *p ? atoi(p) : 5000);
}();

// Robust send() and TCP connect() helpers (see bottom of file for impls).
static bool sendAll(int sock, const void* buf, size_t len);
static int  connectTcp(const char* ip, uint16_t port);

// Frame header sent once per frame before the image payloads.
// All integers are big-endian (network byte order).
#pragma pack(push, 1)
struct NetHeader {
    char     magic[4];   // "LDB0"
    uint32_t version;    // protocol version (htonl)
    uint32_t frame;      // sender frame counter (htonl)
    uint32_t rows;       // output rows (htonl), equals OUT_ROWS
    uint32_t cols;       // output cols (htonl), equals OUT_COLS
    uint32_t bpp;        // bytes per pixel (htonl), here 3 (BGR)
    uint32_t numCams;    // number of images following (htonl)
};
#pragma pack(pop)

// One double-buffer "cell" contains per-camera output BGR buffers.
struct FrameCell {
    // cam[c] is a contiguous BGR buffer of size OUT_ROWS*OUT_COLS*OUT_BPP.
    std::vector<std::vector<unsigned char>> cam;
    int  frame = 0;     // frame index assigned by producer
    // ready flag removed in mailbox mode
};

// Resize/copy helper: from per-camera converted BGRU (4 BPP) to BGR (3 BPP).
// Applies a bottom-anchored square crop and nearest-neighbor resize using
// precomputed coordinate maps (ymap,xmap).
static void bgruBottomSquareToBgrResize(
    const unsigned char* src, uint32_t srcRows, uint32_t srcCols,
    unsigned char* dst, uint32_t outRows, uint32_t outCols,
    const std::vector<uint32_t>& ymap, const std::vector<uint32_t>& xmap)
{
    for (uint32_t y = 0; y < outRows; ++y) {
        const uint32_t sy = ymap[y];                       // source row index
        const size_t dstRowOff = (size_t)y * outCols * 3;  // 3 BPP
        const size_t srcRowOff = (size_t)sy * srcCols * 4; // 4 BPP (BGRU)
        for (uint32_t x = 0; x < outCols; ++x) {
            const uint32_t sx = xmap[x];                   // source col index
            const size_t sidx = srcRowOff + (size_t)sx * 4; // B,G,R,U
            const size_t didx = dstRowOff + (size_t)x * 3;  // B,G,R
            // Copy B, G, R (drop U/alpha)
            dst[didx + 0] = src[sidx + 0];
            dst[didx + 1] = src[sidx + 1];
            dst[didx + 2] = src[sidx + 2];
        }
    }
}

// Parse the color processing method from environment (LBUG_COLOR_METHOD).
// Defaults to LADYBUG_DOWNSAMPLE4 for speed. Accepts multiple aliases.
static LadybugColorProcessingMethod parseColorMethod(const char* s) {
    if (!s || !*s) return LADYBUG_DOWNSAMPLE4;
    auto eq = [](const char* a, const char* b){
        for (; *a && *b; ++a, ++b) if (tolower(*a) != tolower(*b)) return false;
        return *a == 0 && *b == 0;
    };
    if (eq(s,"DISABLE")) return LADYBUG_DISABLE;
    if (eq(s,"EDGE_SENSING") || eq(s,"EDGE")) return LADYBUG_EDGE_SENSING;
    if (eq(s,"NEAREST_NEIGHBOR_FAST") || eq(s,"NN_FAST")) return LADYBUG_NEAREST_NEIGHBOR_FAST;
    if (eq(s,"RIGOROUS")) return LADYBUG_RIGOROUS;
    if (eq(s,"DOWNSAMPLE4") || eq(s,"DS4")) return LADYBUG_DOWNSAMPLE4;
    if (eq(s,"DOWNSAMPLE16") || eq(s,"DS16")) return LADYBUG_DOWNSAMPLE16;
    if (eq(s,"DOWNSAMPLE64") || eq(s,"DS64")) return LADYBUG_DOWNSAMPLE64;
    if (eq(s,"MONO")) return LADYBUG_MONO;
    if (eq(s,"HQLINEAR")) return LADYBUG_HQLINEAR;
    if (eq(s,"HQLINEAR_GPU") || eq(s,"GPU")) return LADYBUG_HQLINEAR_GPU;
    if (eq(s,"DIRECTIONAL_FILTER") || eq(s,"DF")) return LADYBUG_DIRECTIONAL_FILTER;
    if (eq(s,"WEIGHTED_DIRECTIONAL_FILTER") || eq(s,"WDF")) return LADYBUG_WEIGHTED_DIRECTIONAL_FILTER;
    return LADYBUG_DOWNSAMPLE4;
}

// Map the chosen SDK color method to an effective downsample factor in width/height.
// For methods that do not downsample, the factor is 1.
static uint32_t downsampleFactorFor(LadybugColorProcessingMethod m) {
    switch (m) {
        case LADYBUG_DOWNSAMPLE64: return 8; // 1/8 width & height
        case LADYBUG_DOWNSAMPLE16: return 4; // 1/4 width & height
        case LADYBUG_DOWNSAMPLE4:  return 2; // 1/2 width & height
        case LADYBUG_MONO:         return 2; // documented as same size as DS4
        default:                   return 1; // no downsample
    }
}

int main()
{
    using Clock = std::chrono::steady_clock;
    auto msf = [](auto d){ return std::chrono::duration<double, std::milli>(d).count(); };

    // Create and start Ladybug context using color-separated JPEG stream.
    LadybugContext context;
    if (ladybugCreateContext(&context) != LADYBUG_OK) { printf("ladybugCreateContext failed\n"); return -1; }
    if (ladybugInitializeFromIndex(context, 0) != LADYBUG_OK) { printf("ladybugInitializeFromIndex failed\n"); ladybugDestroyContext(&context); return -1; }
    if (ladybugStart(context, LADYBUG_DATAFORMAT_COLOR_SEP_JPEG8) != LADYBUG_OK) { printf("ladybugStart failed\n"); ladybugDestroyContext(&context); return -1; }

    // Choose color processing method (performance/quality tradeoff).
    // If unsupported (e.g., GPU), fallback to DOWNSAMPLE4.
    const char* cmEnv = getenv("LBUG_COLOR_METHOD");
    LadybugColorProcessingMethod colorMethod = parseColorMethod(cmEnv);
    if (ladybugSetColorProcessingMethod(context, colorMethod) != LADYBUG_OK) {
        printf("ladybugSetColorProcessingMethod failed for '%s', falling back to DOWNSAMPLE4\n",
               cmEnv ? cmEnv : "(default)");
        colorMethod = LADYBUG_DOWNSAMPLE4;
        ladybugSetColorProcessingMethod(context, colorMethod);
    }
    const uint32_t ds = downsampleFactorFor(colorMethod);

    printf(
        "Streaming %ux%u BGR (3 BPP), first %u cams, to %s:%u; color=%d (ds=%ux)\n",
        OUT_ROWS, OUT_COLS, NUM_SEND_CAMS, TARGET_IP, TARGET_PORT, (int)colorMethod, ds);

    // Mailbox (latest-wins) buffers:
    // - published: latest completed buffer index (or -1 if none).
    // - consumerIdx: buffer currently being sent; producer avoids picking it.
    // - producerIdx: producer's next write target (only used by producer).
    // This design drops stale frames if producer is faster, keeping latency low.
    static constexpr int BUF_COUNT = 3;
    FrameCell cells[BUF_COUNT];
    for (int i = 0; i < BUF_COUNT; ++i) {
        cells[i].cam.assign(NUM_SEND_CAMS,
            std::vector<unsigned char>((size_t)OUT_ROWS * OUT_COLS * OUT_BPP));
    }
    std::atomic<bool> running{true};
    std::atomic<int>  published{-1};    // latest ready buffer index, or -1 if none
    std::atomic<int>  consumerIdx{-1};  // index currently being sent; producer avoids it
    int producerIdx = 0;                // producer’s current write target (only producer touches)
    std::mutex cv_m;                    // for fast wakeups (no data inside)
    std::condition_variable cv;

    // Per-camera scratch (BGRU) and resize maps
    std::vector<std::vector<unsigned char>> scratch(LADYBUG_NUM_CAMERAS);
    std::vector<uint32_t> ymap, xmap;
    bool mapsReady = false;
    uint32_t srcRows = 0, srcCols = 0;

    // First-frame logging and crop geometry
    uint32_t cropH = 0, cropW = 0;
    bool printedFrameInfo = false;

    // Producer-side stats: average grab/convert/resize time per second window.
    struct {
        uint64_t frames = 0;
        double grab_ms = 0, convert_ms = 0, resize_ms = 0;
        Clock::time_point t0 = Clock::now();
    } prod;

    // Consumer-side stats: average send time and bandwidth per second window.
    struct {
        uint64_t frames = 0;
        double send_ms = 0;
        uint64_t bytes = 0;
        Clock::time_point t0 = Clock::now();
    } cons;

    // Sender thread (consumer): waits for the latest published cell, connects if needed,
    // sends header + NUM_SEND_CAMS images, then releases the buffer for reuse.
    std::thread sender([&](){
        int sock = -1;
        while (running.load()) {
            int idx = -1;
            {
                std::unique_lock<std::mutex> lk(cv_m);
                cv.wait(lk, [&]{
                    return !running.load() || published.load(std::memory_order_acquire) >= 0;
                });
                if (!running.load()) break;
                // Atomically take the latest ready buffer and clear the mailbox.
                idx = published.exchange(-1, std::memory_order_acq_rel);
                if (idx < 0) continue; // spurious wake
                // Mark this buffer as in use so producer won’t pick it.
                consumerIdx.store(idx, std::memory_order_release);
            }

            // Ensure connection, build header, and send the chosen buffer
            if (sock < 0) {
                sock = connectTcp(TARGET_IP, TARGET_PORT);
                if (sock >= 0) {
                    int one = 1;
                    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
                    int snd = 4*1024*1024; // smoother bursts
                    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &snd, sizeof(snd));
                } else {
                    // brief backoff to avoid busy loop
                    usleep(200 * 1000);
                    continue;
                }
            }

            // Build the frame header with the current frame index and fixed dims.
            NetHeader h{};
            std::memcpy(h.magic, "LDB0", 4);
            h.version = htonl(1);
            h.frame   = htonl(cells[idx].frame);
            h.rows    = htonl(OUT_ROWS);
            h.cols    = htonl(OUT_COLS);
            h.bpp     = htonl(OUT_BPP);
            h.numCams = htonl(NUM_SEND_CAMS);

            const size_t payload = (size_t)OUT_ROWS * OUT_COLS * OUT_BPP;

            // Attempt to send header + images. On failure, close socket and retry later.
            auto ts0 = Clock::now();
            bool ok = sendAll(sock, &h, sizeof(h));
            for (unsigned int c = 0; ok && c < NUM_SEND_CAMS; ++c) {
                ok = sendAll(sock, cells[idx].cam[c].data(), payload);
            }
            auto ts1 = Clock::now();
            if (ok) {
                cons.frames++;
                cons.send_ms += msf(ts1 - ts0);
                cons.bytes += sizeof(h) + payload * NUM_SEND_CAMS;
                auto now = Clock::now();
                if (now - cons.t0 >= std::chrono::seconds(1)) {
                    double dt = msf(now - cons.t0) / 1000.0;
                    double fps = cons.frames / dt;
                    double avg_send = cons.frames ? cons.send_ms / cons.frames : 0.0;
                    double mbps = (cons.bytes / (1024.0*1024.0)) / dt;
                    printf("Sender: fps=%.1f avg_send=%.2f ms bw=%.1f MiB/s\n", fps, avg_send, mbps);
                    cons.frames = 0; cons.send_ms = 0; cons.bytes = 0; cons.t0 = now;
                }
            } else {
                close(sock); sock = -1;
            }

            // Done sending: release the buffer so producer can reuse it.
            consumerIdx.store(-1, std::memory_order_release);
        }
        if (sock >= 0) close(sock);
        printf("Sender: exiting\n");
    });

    // Producer: grab -> convert -> crop/resize -> publish latest
    LadybugImage image;
    int frame = 0;
    while (running.load()) {
        // 1) Grab raw image
        auto tg0 = Clock::now();
        if (ladybugGrabImage(context, &image) != LADYBUG_OK) continue;
        auto tg1 = Clock::now();

        // Print raw (pre-conversion) size only for the first frame.
        if (!printedFrameInfo) {
            printf("Frame0: before convert: raw=%ux%u, method=%d ds=%ux\n",
                   image.uiRows, image.uiCols, (int)colorMethod, ds);
        }

        // 2) One-time maps/scratch based on post-conversion dimensions
        if (!mapsReady) {
            // Dimensions after SDK color processing; downsample factor ds is derived
            // from the chosen color method (e.g., DS4=2, DS16=4, etc.).
            srcRows = image.uiRows / ds;
            srcCols = image.uiCols / ds;

            // Bottom-anchored square crop of the converted per-camera image.
            cropH = std::min(srcRows, srcCols);
            const uint32_t startRow = srcRows - cropH;
            cropW = cropH;

            // Precompute nearest-neighbor maps from output to source coordinates.
            ymap.resize(OUT_ROWS);
            xmap.resize(OUT_COLS);
            for (uint32_t y = 0; y < OUT_ROWS; ++y) {
                uint32_t sy = startRow + (uint64_t)y * cropH / OUT_ROWS;
                if (sy >= srcRows) sy = srcRows - 1;
                ymap[y] = sy;
            }
            for (uint32_t x = 0; x < OUT_COLS; ++x) {
                uint32_t sx = (uint64_t)x * cropW / OUT_COLS;
                if (sx >= srcCols) sx = srcCols - 1;
                xmap[x] = sx;
            }

            // Allocate per-camera converted BGRU buffers (4 BPP).
            const size_t srcBufSize = (size_t)srcRows * srcCols * 4;
            for (unsigned int c = 0; c < LADYBUG_NUM_CAMERAS; ++c) {
                scratch[c].assign(srcBufSize, 0u);
            }
            mapsReady = true;
        }

        // Choose a write buffer that is not currently sent by the consumer.
        // With triple buffers there’s always at least one free slot.
        int cand = (producerIdx + 1) % BUF_COUNT;
        const int cons = consumerIdx.load(std::memory_order_acquire);
        if (cand == cons) cand = (cand + 1) % BUF_COUNT;
        producerIdx = cand;

        // 3) Convert to per-camera BGRU (SDK applies the selected color method)
        unsigned char* tmpDest[LADYBUG_NUM_CAMERAS] = { nullptr };
        for (unsigned int c = 0; c < LADYBUG_NUM_CAMERAS; ++c) tmpDest[c] = scratch[c].data();

        auto tc0 = Clock::now();
        if (ladybugConvertImage(context, &image, tmpDest, LADYBUG_BGRU) != LADYBUG_OK) {
            continue; // skip frame on conversion error
        }
        auto tc1 = Clock::now();

        // Print converted (post-conversion) per-camera size only for the first frame.
        if (!printedFrameInfo) {
            printf("Frame0: after convert: per-cam BGRU=%ux%u (4 BPP)\n", srcRows, srcCols);
            printf("Frame0: before resize: crop=%ux%u (bottom-anchored)\n", cropH, cropW);
        }

        // 4) Crop (bottom square) + resize (nearest neighbor) into output buffers (BGR 3 BPP)
        auto tr0 = Clock::now();
        for (unsigned int c = 0; c < NUM_SEND_CAMS; ++c) {
            bgruBottomSquareToBgrResize(
                scratch[c].data(), srcRows, srcCols,
                cells[producerIdx].cam[c].data(), OUT_ROWS, OUT_COLS,
                ymap, xmap
            );
        }
        auto tr1 = Clock::now();

        // Print output size only for the first frame, then disable further prints.
        if (!printedFrameInfo) {
            printf("Frame0: after resize: out BGR=%ux%u (3 BPP)\n", OUT_ROWS, OUT_COLS);
            printedFrameInfo = true;
        }

        // 5) Publish the newest completed buffer and wake the sender.
        cells[producerIdx].frame = frame;
        published.store(producerIdx, std::memory_order_release);
        cv.notify_one();

        // 6) Stats accumulation (producer)
        prod.frames++;
        prod.grab_ms    += msf(tg1 - tg0);
        prod.convert_ms += msf(tc1 - tc0);
        prod.resize_ms  += msf(tr1 - tr0);
        auto now = Clock::now();
        if (now - prod.t0 >= std::chrono::seconds(1)) {
            double dt = msf(now - prod.t0) / 1000.0;
            double fps = prod.frames / dt;
            double avg_grab = prod.frames ? prod.grab_ms / prod.frames : 0.0;
            double avg_conv = prod.frames ? prod.convert_ms / prod.frames : 0.0;
            double avg_resz = prod.frames ? prod.resize_ms / prod.frames : 0.0;
            printf("Producer: fps=%.1f grab=%.2f ms conv=%.2f ms resize=%.2f ms\n",
                   fps, avg_grab, avg_conv, avg_resz);
            prod.frames = 0; prod.grab_ms = prod.convert_ms = prod.resize_ms = 0; prod.t0 = now;
        }

        frame++;
    }

    // Shutdown: stop camera, destroy context, stop sender thread cleanly.
    ladybugStop(context);
    ladybugDestroyContext(&context);
    running.store(false);
    cv.notify_all();
    if (sender.joinable())
        sender.join();
    printf("Done.\n");
    return 0;
}

// Robust sender that continues on EINTR/EAGAIN and returns false on fatal error.
// MSG_NOSIGNAL avoids SIGPIPE on Linux when the peer disconnects.
static bool sendAll(int sock, const void* buf, size_t len)
{
    const unsigned char* p = static_cast<const unsigned char*>(buf);
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(sock, p + sent, len - sent,
#ifdef MSG_NOSIGNAL
                         MSG_NOSIGNAL
#else
                         0
#endif
        );
        if (n > 0) { sent += (size_t)n; continue; }
        if (n < 0 && (errno == EINTR)) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { continue; }
        return false; // fatal error or connection closed
    }
    return true;
}

// Create a TCP socket, enable low-latency and larger buffers, and connect.
static int connectTcp(const char* ip, uint16_t port)
{
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return -1;
    int one = 1;
    int sndbuf = 4 * 1024 * 1024; // 4 MiB send buffer for higher throughput
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) { close(s); return -1; }
    if (connect(s, (sockaddr*)&addr, sizeof(addr)) < 0) { close(s); return -1; }
    return s;
}