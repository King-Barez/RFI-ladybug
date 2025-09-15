#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
// #include <time.h>          // unused
// #include <sys/stat.h>      // unused
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <algorithm> // for std::min/max

// Sockets (Linux)
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#include "ladybug.h"

// Forward declarations for networking helpers (ensure visible before use)
static bool sendAll(int sock, const void* buf, size_t len);
static int  connectTcp(const char* ip, uint16_t port);

// Config: network target
static const char* getEnvOrDefault(const char* k, const char* defv) {
    const char* v = getenv(k);
    return v && *v ? v : defv;
}
static const char* TARGET_IP   = getEnvOrDefault("LBUG_TARGET_IP", "192.168.0.2");
static uint16_t    TARGET_PORT = []{
    const char* p = getenv("LBUG_TARGET_PORT");
    return (uint16_t)(p && *p ? atoi(p) : 5000);
}();

// Output size (square)
static int getEnvIntOrDefault(const char* k, int defv) {
    const char* v = getenv(k);
    if (!v || !*v) return defv;
    return atoi(v);
}
static const int OUT_SIZE = getEnvIntOrDefault("LBUG_OUT_SIZE", 640);

// Helper non-bloccante
int kbhit(void)
{
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

#pragma pack(push, 1)
struct NetHeader {
    char     magic[4];   // "LDB0"
    uint32_t version;    // network byte order
    uint32_t frame;      // network byte order
    uint32_t rows;       // network byte order
    uint32_t cols;       // network byte order
    uint32_t bpp;        // network byte order
    uint32_t numCams;    // network byte order
};
#pragma pack(pop)

struct FrameCell
{
    // cam[c] is a contiguous BGRU buffer of size bufSize
    std::vector<std::vector<unsigned char>> cam;
    uint32_t rows = 0;
    uint32_t cols = 0;
    size_t   bufSize = 0;
    int      frame = 0;
    bool     allocated = false;
};

static void ensureAllocated(FrameCell& cell, uint32_t rows, uint32_t cols)
{
    const size_t need = static_cast<size_t>(rows) * cols * 4; // BGRU
    if (cell.allocated && cell.rows == rows && cell.cols == cols && cell.bufSize == need) return;

    cell.rows = rows;
    cell.cols = cols;
    cell.bufSize = need;
    cell.cam.assign(LADYBUG_NUM_CAMERAS, std::vector<unsigned char>(cell.bufSize));
    // Pre-imposta alpha a 0xFF per tutto il buffer
    for (auto& v : cell.cam) {
        std::fill(v.begin(), v.end(), 0xFF);
    }
    cell.allocated = true;
}

// Nearest-neighbor resize from a bottom square crop of a BGRU image
static void resizeBottomSquareCropBGRU(
    const unsigned char* src, uint32_t srcRows, uint32_t srcCols,
    unsigned char* dst, uint32_t outRows, uint32_t outCols)
{
    const uint32_t cropH = std::min(srcRows, srcCols);
    const uint32_t startRow = srcRows - cropH;
    const uint32_t cropW = cropH; // square

    // Precompute mappings (avoid division in inner loop)
    std::vector<uint32_t> ymap(outRows);
    std::vector<uint32_t> xmap(outCols);
    for (uint32_t y = 0; y < outRows; ++y) {
        uint32_t syOff = (uint32_t)((uint64_t)y * cropH / outRows);
        uint32_t sy = startRow + syOff;
        if (sy >= srcRows) sy = srcRows - 1;
        ymap[y] = sy;
    }
    for (uint32_t x = 0; x < outCols; ++x) {
        uint32_t sxOff = (uint32_t)((uint64_t)x * cropW / outCols);
        uint32_t sx = sxOff; // startCol=0
        if (sx >= srcCols) sx = srcCols - 1;
        xmap[x] = sx;
    }

    for (uint32_t y = 0; y < outRows; ++y) {
        const uint32_t sy = ymap[y];
        for (uint32_t x = 0; x < outCols; ++x) {
            const uint32_t sx = xmap[x];
            const size_t sidx = ((size_t)sy * srcCols + sx) * 4;
            const size_t didx = ((size_t)y * outCols + x) * 4;
            // safe 4-byte copy (unaligned)
            std::memcpy(dst + didx, src + sidx, 4);
        }
    }
}

int main(int /*argc*/, char** /*argv*/)
{
    LadybugError error;
    LadybugContext context;

    // Create context
    error = ladybugCreateContext(&context);
    if (error != LADYBUG_OK)
    {
        printf("Failed to create context: %s\n", ladybugErrorToString(error));
        return -1;
    }

    // Initialize camera 0
    printf("Initializing camera...\n");
    error = ladybugInitializeFromIndex(context, 0);
    if (error != LADYBUG_OK)
    {
        printf("Failed to initialize camera: %s\n", ladybugErrorToString(error));
        ladybugDestroyContext(&context);
        return -1;
    }

    // Start camera in COLOR_SEP_JPEG8
    printf("Starting camera...\n");
    error = ladybugStart(context, LADYBUG_DATAFORMAT_COLOR_SEP_JPEG8);
    if (error != LADYBUG_OK) {
        printf("Failed to start camera: %s\n", ladybugErrorToString(error));
        ladybugDestroyContext(&context);
        return -1;
    }

    // Debayering
    error = ladybugSetColorProcessingMethod(context, LADYBUG_HQLINEAR);
    if (error != LADYBUG_OK) {
        printf("Failed to set color processing: %s\n", ladybugErrorToString(error));
        ladybugStop(context);
        ladybugDestroyContext(&context);
        return -1;
    }

    printf("Double buffer + TCP sender to %s:%u; press 'q' to quit\n", TARGET_IP, TARGET_PORT);
    printf("Output: %dx%d, bottom square crop (H=W=min(rows,cols))\n", OUT_SIZE, OUT_SIZE);

    // Double-buffer
    FrameCell cells[2];
    std::mutex m;
    std::condition_variable cv;
    bool ready[2] = { false, false };
    int nextReadIdx = 0;
    std::atomic<bool> running{true};

    // Scratch buffers for full-size per-camera images (source of resize)
    std::vector<std::vector<unsigned char>> scratch(LADYBUG_NUM_CAMERAS);
    size_t scratchSize = 0;

    // Sender thread
    std::thread sender([&](){
        int sock = -1;
        // Optional: enlarge socket send buffer
        // int sndbuf = 4 * 1024 * 1024; setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

        while (running.load()) {
            int idx = -1;

            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&]{
                    return !running.load() || ready[nextReadIdx];
                });
                if (!running.load()) break;
                idx = nextReadIdx;
                nextReadIdx ^= 1; // alterna
                // IMPORTANT: do NOT clear ready[idx] here; keep it true while sending
            }

            // (re)connect if needed
            if (sock < 0) {
                sock = connectTcp(TARGET_IP, TARGET_PORT);
                if (sock < 0) {
                    fprintf(stderr, "Sender: connect to %s:%u failed, retrying...\n", TARGET_IP, TARGET_PORT);
                    std::this_thread::sleep_for(std::chrono::milliseconds(300));
                    // Release the buffer so producer can proceed (drop this frame)
                    {
                        std::lock_guard<std::mutex> lk(m);
                        ready[idx] = false;
                    }
                    cv.notify_all();
                    continue;
                }
                printf("Sender: connected to %s:%u\n", TARGET_IP, TARGET_PORT);
                // (Optionally set SO_SNDBUF here after creating 'sock')
            }

            const uint32_t rows = cells[idx].rows;
            const uint32_t cols = cells[idx].cols;
            const int      frameNo = cells[idx].frame;
            const size_t   bufSize = cells[idx].bufSize;

            NetHeader hdr{};
            std::memcpy(hdr.magic, "LDB0", 4);
            hdr.version = htonl(1);
            hdr.frame   = htonl(static_cast<uint32_t>(frameNo));
            hdr.rows    = htonl(rows);
            hdr.cols    = htonl(cols);
            hdr.bpp     = htonl(4);
            hdr.numCams = htonl(LADYBUG_NUM_CAMERAS);

            bool ok = true;
            if (!sendAll(sock, &hdr, sizeof(hdr))) {
                fprintf(stderr, "Sender: header send failed, reconnecting...\n");
                ok = false;
            } else {
                for (unsigned int c = 0; c < LADYBUG_NUM_CAMERAS; ++c) {
                    if (!sendAll(sock, cells[idx].cam[c].data(), bufSize)) {
                        fprintf(stderr, "Sender: payload send failed on cam %u, reconnecting...\n", c);
                        ok = false;
                        break;
                    }
                }
            }
            if (!ok) {
                if (sock >= 0) { close(sock); sock = -1; }
            }

            // Now release the buffer for producer (after send attempt)
            {
                std::lock_guard<std::mutex> lk(m);
                ready[idx] = false;
            }
            cv.notify_all();
        }

        if (sock >= 0) { close(sock); sock = -1; }
        printf("Sender: exiting\n");
    });

    int frame = 0;
    LadybugImage image;

    while (running.load())
    {
        // Grab frame
        error = ladybugGrabImage(context, &image);
        if (error != LADYBUG_OK) {
            printf("Failed to grab image: %s\n", ladybugErrorToString(error));
            continue;
        }

        const int idx = frame & 1;

        // Prepare scratch buffers and output cell (OUT_SIZE x OUT_SIZE)
        const uint32_t srcRows = image.uiRows;
        const uint32_t srcCols = image.uiCols;
        const size_t srcBufSize = (size_t)srcRows * srcCols * 4;

        if (scratchSize != srcBufSize) {
            for (auto& v : scratch) v.assign(srcBufSize, 0xFF);
            scratchSize = srcBufSize;
        }

        unsigned char* tmpDest[LADYBUG_NUM_CAMERAS] = { nullptr };
        for (unsigned int c = 0; c < LADYBUG_NUM_CAMERAS; ++c) {
            tmpDest[c] = scratch[c].data();
        }

        {
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&]{ return !ready[idx]; });
            ensureAllocated(cells[idx], OUT_SIZE, OUT_SIZE); // allocate output buffers
        }

        // Convert to per-camera full-res BGRU in scratch
        error = ladybugConvertImage(context, &image, tmpDest, LADYBUG_BGRU);
        if (error != LADYBUG_OK) {
            printf("Failed to convert image: %s\n", ladybugErrorToString(error));
            continue;
        }

        // Resize bottom band into the output cell buffers (square crop -> OUT_SIZE)
        for (unsigned int c = 0; c < LADYBUG_NUM_CAMERAS; ++c) {
            resizeBottomSquareCropBGRU(
                scratch[c].data(), srcRows, srcCols,
                cells[idx].cam[c].data(), OUT_SIZE, OUT_SIZE
            );
        }

        {
            std::lock_guard<std::mutex> lk(m);
            cells[idx].frame = frame;
            ready[idx] = true;
        }
        cv.notify_one();

        frame++;

        if (kbhit() && getchar() == 'q')
        {
            printf("'q' pressed, exiting.\n");
            running.store(false);
            cv.notify_all();
            break;
        }
    }

    // Cleanup
    printf("Stopping camera...\n");
    ladybugStop(context);
    ladybugDestroyContext(&context);

    running.store(false);
    cv.notify_all();
    if (sender.joinable()) sender.join();

    printf("Done.\n");
    return 0;
}

// Networking helpers
static bool sendAll(int sock, const void* buf, size_t len)
{
    const unsigned char* p = static_cast<const unsigned char*>(buf);
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(sock, p + sent, len - sent, 0);
        if (n <= 0) {
            if (errno == EINTR) continue;
            return false;
        }
        sent += (size_t)n;
    }
    return true;
}

static int connectTcp(const char* ip, uint16_t port)
{
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return -1;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) {
        close(s);
        return -1;
    }
    if (connect(s, (sockaddr*)&addr, sizeof(addr)) < 0) {
        close(s);
        return -1;
    }
    return s;
}