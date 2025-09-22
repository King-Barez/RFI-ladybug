#!/usr/bin/env python3
import argparse
import os
import signal
import socket
import struct
import sys
import time
from typing import List, Optional

import numpy as np

# Optional deps
try:
    from PIL import Image
except ImportError:
    Image = None

# Optional YOLO (Ultralytics)
try:
    from ultralytics import YOLO  # noqa: F401
    _HAS_YOLO = True
except Exception:
    YOLO = None  # type: ignore
    _HAS_YOLO = False

MAGIC = b"LDB0"
# magic + version + frame + rows + cols + bpp + numCams = 4 + 6*4 = 28
HDR_STRUCT = struct.Struct(">4s6I")
HDR_SIZE = HDR_STRUCT.size


def recv_into_all(conn: socket.socket, mv: memoryview):
    total = 0
    view = mv
    while total < len(mv):
        n = conn.recv_into(view, len(mv) - total)
        if n == 0:
            raise ConnectionError("socket closed")
        total += n
        view = view[n:]


def save_png_bgr(path, rows, cols, raw):
    if Image is None:
        return
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols, 3)  # B,G,R
    rgb = arr[:, :, [2, 1, 0]].copy()
    Image.fromarray(rgb).save(path, compress_level=0, optimize=False)


def save_png_bgr_array(path, bgr_array):
    # bgr_array: HxWx3 uint8 (BGR)
    if Image is None:
        return
    rgb = bgr_array[:, :, [2, 1, 0]]
    Image.fromarray(rgb).save(path, compress_level=0, optimize=False)


def load_yolo(model_path: str, device: str):
    """
    Load YOLO model (supports .pt or .engine). device: 'auto'|'cpu'|'cuda'
    Returns (model, resolved_device_str)
    """
    if not _HAS_YOLO:
        print("YOLO not available: install 'ultralytics' to enable inference.")
        return None, "cpu"

    # Lazy import torch for device probing
    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    resolved = device
    if device == "auto":
        if torch is not None and torch.cuda.is_available():
            resolved = "cuda"
        else:
            resolved = "cpu"

    try:
        model = YOLO(model_path)
        # Ultralytics handles device internally; for .engine it will use CUDA.
        # For .pt you can still force device at predict time.
        print(f"YOLO loaded: {model_path} (device={resolved})")
        return model, resolved
    except Exception as e:
        print(f"Error loading YOLO ({model_path}): {e}")
        return None, "cpu"


def yolo_predict_batch(model, device: str, imgs: List[np.ndarray], conf: float, iou: float, imgsz: int, person_class: int, allow_batch: bool, classes: Optional[List[int]]):
    """
    Prefer batch inference if the engine supports batch > 1, otherwise fallback to per-image.
    Returns (results_list, info) where info contains timing and mode.
    info = {'mode': 'batch'|'sequential', 'per_cam_ms': [...], 'total_ms': float}
    """
    if model is None:
        return None, None

    info = {'mode': 'sequential', 'per_cam_ms': [], 'total_ms': 0.0}

    def _predict_one(im):
        t0 = time.time()
        r = model.predict(
            source=im, imgsz=imgsz, conf=conf, iou=iou,
            device=None if device == "auto" else device,
            verbose=False, classes=classes
        )[0]
        t1 = time.time()
        info['per_cam_ms'].append((t1 - t0) * 1000.0)
        return r

    # Try batch first
    if allow_batch and len(imgs) > 1:
        try:
            t0 = time.time()
            res = model.predict(
                source=imgs, imgsz=imgsz, conf=conf, iou=iou,
                device=None if device == "auto" else device,
                verbose=False, classes=classes
            )
            t1 = time.time()
            total_ms = (t1 - t0) * 1000.0
            per_cam = total_ms / max(1, len(imgs))
            info['mode'] = 'batch'
            info['total_ms'] = total_ms
            info['per_cam_ms'] = [per_cam] * len(imgs)
            return list(res), info
        except Exception as e:
            print(f"YOLO batch predict disabled, falling back to sequential: {e}")

    # Fallback: sequential
    out = []
    for im in imgs:
        try:
            out.append(_predict_one(im))
        except Exception as e:
            print(f"YOLO predict error: {e}")
            out.append(None)
            info['per_cam_ms'].append(0.0)
    info['total_ms'] = float(sum(info['per_cam_ms']))
    return out, info


def main():
    ap = argparse.ArgumentParser()
    # Network and saving args
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--save-dir", default=None, help="output dir (disabled if omitted)")
    ap.add_argument("--save-every", type=int, default=0, help="save 1 frame every N (0=never)")

    # YOLO inference options
    ap.add_argument("--yolo-model", default=None, help="YOLOv11 weights (.pt or .engine), omit to disable")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="YOLO device")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO input size (default 640)")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    ap.add_argument("--person-class", type=int, default=0, help="COCO 'person' class index")
    ap.add_argument("--infer-mode", choices=["auto", "batch", "sequential"], default="auto",
                    help="Batch inference mode. 'auto' tries batch then falls back.")
    ap.add_argument("--all-classes", action="store_true",
                    help="If set, do not filter classes (default filters to 'person' only).")

    ap.add_argument("--save-raw", action="store_true", help="save raw BGR images")
    ap.add_argument("--save-annotated", action="store_true", help="save images with bounding boxes")

    args = ap.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Load YOLO once
    model = None
    yolo_device = "cpu"
    if args.yolo_model is not None:
        model, yolo_device = load_yolo(args.yolo_model, args.device)

    # Handle signals cleanly
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((args.host, args.port))
    s.listen(1)
    print(f"Listening on {args.host}:{args.port} …")

    # Aggregate stats over ~1s window
    win_t0 = time.time()
    agg_frames = 0
    agg_payload_ms = 0.0
    agg_unpack_ms = 0.0
    agg_infer_ms = 0.0
    agg_infer_cam_ms = 0.0
    agg_infer_cam_cnt = 0
    agg_save_ms = 0.0
    agg_bytes = 0
    agg_batch_uses = 0
    agg_seq_uses = 0

    while True:
        try:
            conn, addr = s.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Connected from {addr}")

            # per-connection loop
            while True:
                # Header
                hdr_buf = bytearray(HDR_SIZE)
                recv_into_all(conn, memoryview(hdr_buf))
                magic, version, frame_idx, rows, cols, bpp, numCams = HDR_STRUCT.unpack(hdr_buf)
                if magic != MAGIC:
                    raise ValueError(f"Bad magic: {magic!r}")
                if version != 1:
                    raise ValueError(f"Unsupported version: {version}")
                if bpp != 3:
                    raise ValueError(f"Unsupported BPP: {bpp}")
                img_bytes = rows * cols * bpp
                payload_bytes = img_bytes * numCams

                # Payload receive
                t_payload0 = time.time()
                payload = bytearray(payload_bytes)
                recv_into_all(conn, memoryview(payload))
                t_payload1 = time.time()

                # Build per-camera BGR numpy views (unpack)
                t_unpack0 = time.time()
                cam_imgs = []
                for ci in range(numCams):
                    off = ci * img_bytes
                    view = memoryview(payload)[off : off + img_bytes]
                    arr = np.frombuffer(view, dtype=np.uint8).reshape(rows, cols, 3)
                    cam_imgs.append(arr)
                t_unpack1 = time.time()

                # YOLO inference (batch preferred)
                t_infer0 = time.time()
                yolo_results = None
                yinfo = None
                if args.yolo_model is not None and model is not None:
                    allow_batch = args.infer_mode in ("auto", "batch")
                    classes = None if args.all_classes else [args.person_class]
                    yolo_results, yinfo = yolo_predict_batch(
                        model=model,
                        device=yolo_device,
                        imgs=cam_imgs,
                        conf=args.conf,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        person_class=args.person_class,
                        allow_batch=allow_batch,
                        classes=classes,
                    )
                t_infer1 = time.time()

                # Saving
                do_save = args.save_dir and args.save_every > 0 and (frame_idx % args.save_every == 0)
                t_save0 = time.time()
                if do_save:
                    for ci, img in enumerate(cam_imgs):
                        base = os.path.join(args.save_dir, f"cam{ci}_f{frame_idx:06d}")
                        if args.save_raw:
                            save_png_bgr_array(base + "_raw.png", img)
                        if args.save_annotated and yolo_results is not None:
                            res = yolo_results[ci] if ci < len(yolo_results) else None
                            if res is not None:
                                try:
                                    ann = res.plot()  # BGR uint8
                                    save_png_bgr_array(base + "_annot.png", ann)
                                except Exception as e:
                                    print(f"Save annotated error: {e}")
                t_save1 = time.time()

                # Aggregate window
                agg_frames += 1
                agg_payload_ms += (t_payload1 - t_payload0) * 1000.0
                agg_unpack_ms  += (t_unpack1  - t_unpack0)  * 1000.0
                if yinfo is not None:
                    agg_infer_ms += yinfo.get('total_ms', (t_infer1 - t_infer0) * 1000.0)
                    per_cam = yinfo.get('per_cam_ms', [])
                    agg_infer_cam_ms += float(sum(per_cam))
                    agg_infer_cam_cnt += len(per_cam)
                    if yinfo.get('mode') == 'batch':
                        agg_batch_uses += 1
                    else:
                        agg_seq_uses += 1
                else:
                    agg_infer_ms += (t_infer1 - t_infer0) * 1000.0
                agg_save_ms += (t_save1 - t_save0) * 1000.0
                agg_bytes += HDR_SIZE + payload_bytes

                # Periodic report (1s)
                now = time.time()
                if now - win_t0 >= 1.0:
                    dt = now - win_t0
                    fps = agg_frames / dt if dt > 0 else 0.0
                    bw_mib_s = (agg_bytes / dt) / (1024.0 * 1024.0)
                    avg_payload = (agg_payload_ms / max(1, agg_frames))
                    avg_unpack  = (agg_unpack_ms  / max(1, agg_frames))
                    avg_infer   = (agg_infer_ms   / max(1, agg_frames))
                    avg_save    = (agg_save_ms    / max(1, agg_frames))
                    avg_infer_cam = (agg_infer_cam_ms / max(1, agg_infer_cam_cnt)) if agg_infer_cam_cnt > 0 else 0.0
                    batch_ratio = (agg_batch_uses / max(1, (agg_batch_uses + agg_seq_uses))) * 100.0

                    print(
                        f"RX: fps={fps:.1f} bw={bw_mib_s:.1f} MiB/s | "
                        f"payload={avg_payload:.2f} ms unpack={avg_unpack:.2f} ms "
                        f"infer={avg_infer:.2f} ms (per-cam={avg_infer_cam:.2f} ms, batch%={batch_ratio:.0f}) "
                        f"save={avg_save:.2f} ms "
                        f"(frame={frame_idx} {rows}x{cols} bpp={bpp} cams={numCams})"
                    )

                    # reset window
                    win_t0 = now
                    agg_frames = 0
                    agg_payload_ms = agg_unpack_ms = agg_infer_ms = agg_save_ms = 0.0
                    agg_infer_cam_ms = 0.0
                    agg_infer_cam_cnt = 0
                    agg_bytes = 0
                    agg_batch_uses = 0
                    agg_seq_uses = 0

        except KeyboardInterrupt:
            print("Interrupted, exiting.")
            break
        except (ConnectionError, OSError) as e:
            print(f"Disconnected ({e})")
            # loop back to accept() for next client
            continue
        finally:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()