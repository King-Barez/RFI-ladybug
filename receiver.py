#!/usr/bin/env python3
import socket
import struct
import argparse
import os
import time
from typing import List, Optional

# Optional deps
try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None
    np = None

# Optional YOLO (Ultralytics)
try:
    from ultralytics import YOLO  # noqa: F401
    _HAS_YOLO = True
except Exception:
    YOLO = None  # type: ignore
    _HAS_YOLO = False

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
    Image.fromarray(rgb).save(path, compress_level=0, optimize=False)

def save_png_bgr_array(path, bgr_array):
    # bgr_array: HxWx3 uint8 (BGR)
    if Image is None or np is None:
        return
    rgb = bgr_array[:, :, [2, 1, 0]]
    Image.fromarray(rgb).save(path, compress_level=0, optimize=False)

def load_yolo(model_path: str, device: str):
    """
    Load YOLO model (supports .pt or .engine). device: 'auto'|'cpu'|'cuda'
    Returns (model, resolved_device_str)
    """
    if not _HAS_YOLO:
        print("YOLO non disponibile: installa 'ultralytics' per abilitare l'inferenza.")
        return None, "cpu"

    # Lazy import torch for device probing
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    resolved = device
    if device == "auto":
        # If TensorRT engine is used, prefer cuda; else CUDA if available
        if model_path.endswith(".engine"):
            resolved = "cuda"
        else:
            resolved = "cuda" if has_cuda else "cpu"

    try:
        model = YOLO(model_path)
        # Note: 'device' is passed on predict; not needed at load time.
        print(f"YOLO caricato: {model_path} (device={resolved})")
        return model, resolved
    except Exception as e:
        print(f"Errore caricando YOLO ({model_path}): {e}")
        return None, "cpu"

def yolo_predict_batch(model, device: str, imgs: List, conf: float, iou: float, imgsz: int, person_class: int):
    """
    Run a single batch predict; returns list of Results (or None if disabled)
    """
    if model is None:
        return None
    try:
        # classes filters to 'person' (default COCO idx=0)
        # imgs can be numpy arrays (BGR); Ultralytics handles layout internally.
        results = model.predict(
            imgs,
            device=device,
            verbose=False,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=[person_class],
        )
        return results
    except Exception as e:
        print(f"Errore YOLO predict: {e}")
        return None

def main():
    ap = argparse.ArgumentParser()
    # Network and saving args (existing)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--save-dir", default=None, help="dir per salvare (disabilitato se omesso)")
    ap.add_argument("--save-every", type=int, default=0, help="salva 1 frame ogni N (0=mai)")

    # New: YOLO inference options
    ap.add_argument("--yolo-model", default="yolo11n.pt", help="peso YOLOv11 (.pt o .engine)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="dispositivo per YOLO")
    ap.add_argument("--imgsz", type=int, default=640, help="dimensione lato per YOLO (default 640)")
    ap.add_argument("--conf", type=float, default=0.25, help="conf threshold YOLO")
    ap.add_argument("--iou", type=float, default=0.45, help="IoU threshold NMS")
    ap.add_argument("--person-class", type=int, default=0, help="indice classe 'person' (default COCO=0)")
    ap.add_argument("--save-raw", action="store_true", help="salva le immagini raw BGR")
    ap.add_argument("--save-annotated", action="store_true", help="salva immagini annotate con bbox")

    args = ap.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Load YOLO once
    model, yolo_device = load_yolo(args.yolo_model, args.device)

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
        bufs: List[bytearray] = []
        last_t = time.perf_counter()
        frames = 0
        payload_ms_sum = 0.0
        infer_ms_sum = 0.0
        save_ms_sum = 0.0
        bytes_sum = 0
        try:
            while True:
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

                # Build numpy arrays view (no copy) for inference
                imgs_np: Optional[List] = None
                if np is not None:
                    try:
                        imgs_np = [np.frombuffer(bufs[i], dtype=np.uint8).reshape(rows, cols, 3) for i in range(numCams)]
                    except Exception as e:
                        imgs_np = None
                        print(f"Errore creando array numpy: {e}")

                # Inference (batched)
                tinf0 = time.perf_counter()
                results = None
                if imgs_np is not None and model is not None:
                    # If incoming is already 640x640, prefer use that size
                    imgsz = rows if rows == cols else args.imgsz
                    results = yolo_predict_batch(model, yolo_device, imgs_np, args.conf, args.iou, imgsz, args.person_class)
                tinf1 = time.perf_counter()
                if results is not None:
                    infer_ms_sum += (tinf1 - tinf0) * 1000.0

                # Stats print every ~1s
                frames += 1
                now = time.perf_counter()
                if now - last_t >= 1.0:
                    dt = now - last_t
                    fps = frames / dt
                    avg_payload = (payload_ms_sum / frames) if frames else 0.0
                    avg_infer = (infer_ms_sum / frames) if frames and infer_ms_sum > 0 else 0.0
                    avg_save = (save_ms_sum / frames) if frames and save_ms_sum > 0 else 0.0
                    mb_s = (bytes_sum / (1024.0 * 1024.0)) / dt
                    print(f"RX: fps={fps:.1f} payload={avg_payload:.2f} ms infer={avg_infer:.2f} ms save={avg_save:.2f} ms bw={mb_s:.1f} MiB/s (frame={frame} {rows}x{cols} bpp={bpp} cams={numCams})")
                    frames = 0
                    payload_ms_sum = 0.0
                    infer_ms_sum = 0.0
                    save_ms_sum = 0.0
                    bytes_sum = 0
                    last_t = now

                # Optional saving (sampled)
                do_save = args.save_dir and args.save_every > 0 and (seq % args.save_every == 0)
                if do_save:
                    ts0 = time.perf_counter()
                    # Save raw?
                    if args.save_raw:
                        for c, raw in enumerate(bufs):
                            out = os.path.join(args.save_dir, f"cam{c:02d}_f{seq:06d}.png")
                            save_png_bgr(out, rows, cols, raw)

                    # Save annotated (if we have results)
                    if args.save_annotated and results is not None:
                        for c, r in enumerate(results):
                            try:
                                ann_bgr = r.plot()  # numpy uint8 BGR
                                outa = os.path.join(args.save_dir, f"cam{c:02d}_f{seq:06d}_annot.png")
                                save_png_bgr_array(outa, ann_bgr)
                            except Exception as e:
                                print(f"Errore salvataggio annotazione cam {c}: {e}")

                    ts1 = time.perf_counter()
                    save_ms_sum += (ts1 - ts0) * 1000.0

                seq += 1
        except (ConnectionError, BrokenPipeError) as e:
            print("Connessione chiusa:", e)
            conn.close()
            continue

if __name__ == "__main__":
    main()