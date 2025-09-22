# LadybugRFI

C++ sender for FLIR Ladybug streaming BGR frames over TCP to a Python receiver. Receiver supports optional YOLOv11 inference (TensorRT or PyTorch) and annotated PNG saving.

Features
- Latest-wins mailbox on sender for low latency.
- Configurable Ladybug color method (default DOWNSAMPLE4).
- 640x640 BGR output per camera, 3 BPP, 1..6 cameras.
- Receiver can save raw and/or annotated images.
- YOLOv11 inference on Jetson (TensorRT) or CPU/GPU (.pt).
- Per-frame timing (payload/infer/save) and bandwidth stats.

Quick start

Sender (host with camera)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# Set receiver IP/port
export LBUG_TARGET_IP=<receiver_ip>
export LBUG_TARGET_PORT=5000
# Optional
export LBUG_COLOR_METHOD=DOWNSAMPLE4
export LBUG_NUM_CAMS=3
./build/ladybugRFI
```

Receiver (Jetson, Docker + TensorRT)
One-time export to TensorRT engine:
```bash
sudo docker run --rm -it \
  --runtime nvidia --ipc=host --network host \
  -v /home/implab/Documents/RFI-ladybug:/work -w /work \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  bash -lc 'yolo export model=yolo11n.pt format=engine device=0 imgsz=640 half=True dynamic=False && mv -f yolo11n.engine models/'
```

Run the server with TensorRT:
```bash
sudo docker run --rm -it --init --name rfi-receiver \
  --runtime nvidia --ipc=host --network host \
  -v /home/implab/Documents/RFI-ladybug:/work -w /work \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  python3 receiver.py --host 0.0.0.0 --port 5000 \
    --yolo-model models/yolo11n.engine --device auto \
    --conf 0.15 --save-dir rx --save-every 10 --save-annotated
```

Receiver (fallback, PyTorch .pt)
```bash
sudo docker run --rm -it \
  --runtime nvidia --ipc=host --network host \
  -v /home/implab/Documents/RFI-ladybug:/work -w /work \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  bash -lc 'pip install -q --no-cache-dir pillow numpy && \
            python3 receiver.py --host 0.0.0.0 --port 5000 \
              --yolo-model yolo11n.pt --device auto \
              --conf 0.15 --save-dir rx --save-every 10 --save-annotated'
```

Notes
- The TensorRT engine above is exported with batch=1. The receiver runs YOLO per-camera to avoid batch-size errors.
- Annotated PNGs are saved in rx/ as camXX_fNNNNNN_annot.png. Add --save-raw to also save unannotated images.
- Ensure the sender points to your receiver IP (USB-C, Wi‑Fi, or Ethernet).

Configuration (env vars)
- LBUG_TARGET_IP: receiver IP (default 192.168.0.2)
- LBUG_TARGET_PORT: receiver port (default 5000)
- LBUG_COLOR_METHOD: DISABLE, EDGE_SENSING, NEAREST_NEIGHBOR_FAST, RIGOROUS, DOWNSAMPLE4/16/64, MONO, HQLINEAR, HQLINEAR_GPU, DIRECTIONAL_FILTER, WEIGHTED_DIRECTIONAL_FILTER
- LBUG_NUM_CAMS: 1..6 (default 3)

Runtime logs
- The receiver prints a 1 s report with averages:
  - fps, bandwidth (MiB/s)
  - payload ms (network read), unpack ms (bytes→arrays)
  - infer ms (total), per-cam ms average, batch% (share of batch mode over sequential)
  - save ms
- By default only the “person” class (COCO id 0) is inferred and drawn. Use --all-classes to disable filtering.