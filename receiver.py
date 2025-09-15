#!/usr/bin/env python3
import socket
import struct
import argparse
import os
import time
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

MAGIC = b"LDB0"
HDR_SIZE = 4 + 4 * 6  # magic + 6 x uint32 big-endian

def recv_all(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connessione chiusa")
        buf.extend(chunk)
    return bytes(buf)

def save_png_bgru(path, rows, cols, raw):
    if Image is None:
        return
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols, 4)
    rgba = arr[:, :, [2, 1, 0, 3]].copy()
    im = Image.fromarray(rgba)
    im.save(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--save-dir", default="rx_previews", help="salva tutte le immagini ricevute")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((args.host, args.port))
    s.listen(1)
    print(f"Ascolto su {args.host}:{args.port}…")

    seq = 0  # progressive numbering across reconnects
    while True:
        conn, addr = s.accept()
        print("Connesso da", addr)
        last_t = time.time()
        frames = 0
        try:
            while True:
                hdr = recv_all(conn, HDR_SIZE)
                if hdr[:4] != MAGIC:
                    raise RuntimeError(f"Magic errato: {hdr[:4]}")
                version, frame, rows, cols, bpp, numCams = struct.unpack("!6I", hdr[4:])
                payload_sz = rows * cols * bpp
                cams = [recv_all(conn, payload_sz) for _ in range(numCams)]

                frames += 1
                now = time.time()
                if now - last_t >= 1.0:
                    fps = frames / (now - last_t)
                    print(f"RX frame={frame} {rows}x{cols} bpp={bpp} cams={numCams} ~{fps:.1f} fps")
                    frames = 0
                    last_t = now

                # save all images with progressive numbering
                if Image is not None:
                    for c, data in enumerate(cams):
                        out = os.path.join(args.save_dir, f"cam{c:02d}_f{seq:06d}.png")
                        save_png_bgru(out, rows, cols, data)
                seq += 1
        except (ConnectionError, BrokenPipeError) as e:
            print("Connessione chiusa:", e)
            conn.close()
            continue

if __name__ == "__main__":
    main()