#!/usr/bin/env python3
import socket
import struct
import argparse
import os
import time

try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None
    np = None

MAGIC = b"LDB0"
HDR_SIZE = 4 + 4 * 6  # magic + 6 x uint32 big-endian

def recv_into_all(conn, mv):
    view = mv
    total = 0
    while total < len(mv):
        n = conn.recv_into(view, len(mv) - total)
        if n == 0:
            raise ConnectionError("connessione chiusa")
        total += n
        view = view[n:]

def save_png_bgr(path, rows, cols, raw):
    if Image is None or np is None:
        return
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols, 3)  # B,G,R
    rgb = arr[:, :, [2, 1, 0]].copy()
    # fastest PNG (no compression)
    Image.fromarray(rgb).save(path, compress_level=0, optimize=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--save-dir", default=None, help="dir per salvare (disabilitato se omesso)")
    ap.add_argument("--save-every", type=int, default=0, help="salva 1 frame ogni N (0=mai)")
    args = ap.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    s.bind((args.host, args.port))
    s.listen(1)
    print(f"Ascolto su {args.host}:{args.port}…")

    seq = 0
    while True:
        conn, addr = s.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("Connesso da", addr)
        bufs = []
        last_t = time.perf_counter()
        frames = 0
        payload_ms_sum = 0.0
        save_ms_sum = 0.0
        bytes_sum = 0
        prev_frame = None
        try:
            while True:
                t0 = time.perf_counter()
                hdr = conn.recv(HDR_SIZE, socket.MSG_WAITALL)
                if len(hdr) != HDR_SIZE or hdr[:4] != MAGIC:
                    raise ConnectionError("header/magic non valido")
                version, frame, rows, cols, bpp, numCams = struct.unpack("!6I", hdr[4:])
                payload_sz = rows * cols * bpp

                if not bufs or len(bufs) != numCams or len(bufs[0]) != payload_sz:
                    bufs = [bytearray(payload_sz) for _ in range(numCams)]

                tp0 = time.perf_counter()
                for i in range(numCams):
                    recv_into_all(conn, memoryview(bufs[i]))
                tp1 = time.perf_counter()
                payload_ms_sum += (tp1 - tp0) * 1000.0
                bytes_sum += payload_sz * numCams

                frames += 1
                now = time.perf_counter()
                if now - last_t >= 1.0:
                    dt = now - last_t
                    fps = frames / dt
                    avg_payload = (payload_ms_sum / frames) if frames else 0.0
                    avg_save = (save_ms_sum / frames) if (frames and save_ms_sum > 0) else 0.0
                    mb_s = (bytes_sum / (1024.0 * 1024.0)) / dt
                    print(f"RX: fps={fps:.1f} payload={avg_payload:.2f} ms save={avg_save:.2f} ms bw={mb_s:.1f} MiB/s (frame={frame} {rows}x{cols} bpp={bpp} cams={numCams})")
                    frames = 0
                    payload_ms_sum = 0.0
                    save_ms_sum = 0.0
                    bytes_sum = 0
                    last_t = now

                # Optional saving (sampled)
                if args.save_dir and args.save_every > 0 and (seq % args.save_every == 0):
                    ts0 = time.perf_counter()
                    for c, raw in enumerate(bufs):
                        out = os.path.join(args.save_dir, f"cam{c:02d}_f{seq:06d}.png")
                        save_png_bgr(out, rows, cols, raw)
                    ts1 = time.perf_counter()
                    save_ms_sum += (ts1 - ts0) * 1000.0
                seq += 1
        except (ConnectionError, BrokenPipeError) as e:
            print("Connessione chiusa:", e)
            conn.close()
            continue

if __name__ == "__main__":
    main()