# Wire protocol (LDB0 v1)

Transport: TCP client (sender) -> TCP server (receiver).

Per frame:
1) Header (packed, big-endian)
- magic: 4 bytes ASCII "LDB0"
- version: uint32 (currently 1)
- frame: uint32 (sender frame counter)
- rows: uint32 (output rows, e.g., 640)
- cols: uint32 (output cols, e.g., 640)
- bpp: uint32 (bytes per pixel, 3 for BGR)
- numCams: uint32 (1..6)

2) Payload
- numCams consecutive images
- Each image is rows*cols*bpp bytes
- Pixel format: BGR (3 bytes per pixel)

Notes
- No per-image size fields; receiver derives payload size from the header.
- The sender may drop frames on reconnect or under load (latest-wins policy).

## Architecture

Components
- Producer (capture/process): grabs Ladybug frames, converts per-camera using the selected color method, crops a bottom-anchored square, resizes to OUT_ROWS x OUT_COLS, writes to a mailbox cell, then publishes that cell index atomically.
- Mailbox buffer (triple-buffer typical): a small pool of cells; only the latest fully-written cell is visible to the consumer.
- Consumer (sender): waits for a publication (condvar), grabs the latest index, sends header + images, and repeats. If the network stalls, older cells can be skipped (low-latency, “latest-wins”).

Threading and invariants
- Producer never overwrites a cell being read; indices are exchanged atomically.
- Consumer always prefers the most recently published index.
- No strict alternation and no hard backpressure; throughput is gated by compute and network.

Image processing
- Color method is selectable via LBUG_COLOR_METHOD (default DOWNSAMPLE4). Methods with built-in downsample in ladybugConvertImage() adjust the source size:
  - DOWNSAMPLE4 or MONO → ds=2
  - DOWNSAMPLE16 → ds=4
  - DOWNSAMPLE64 → ds=8
  - Others (EDGE_SENSING, NEAREST_NEIGHBOR_FAST, HQLINEAR, RIGOROUS, etc.) → ds=1
- For each of the first NUM_SEND_CAMS cameras:
  - Convert to BGRU in SDK scratch (downsample applied per method).
  - Crop bottom square: size = min(srcRows, srcCols), bottom-anchored.
  - Resize to OUT_ROWS x OUT_COLS (nearest-neighbor maps).
  - Strip alpha to produce BGR (3 BPP).

Configuration (env vars)
- LBUG_TARGET_IP: receiver IP (default 192.168.0.2)
- LBUG_TARGET_PORT: receiver port (default 5000)
- LBUG_COLOR_METHOD: DISABLE, EDGE_SENSING, NEAREST_NEIGHBOR_FAST, RIGOROUS,
  DOWNSAMPLE4, DOWNSAMPLE16, DOWNSAMPLE64, MONO, HQLINEAR, HQLINEAR_GPU,
  DIRECTIONAL_FILTER, WEIGHTED_DIRECTIONAL_FILTER (aliases accepted)
- LBUG_NUM_CAMS: 1..6 (default 3)

## Build (sender)

Release build:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
```

## Run

Receiver (Jetson, Docker + TensorRT)
```bash
# 1) Export to TensorRT once (saves to models/yolo11n.engine)
sudo docker run --rm -it \
  --runtime nvidia --ipc=host --network host \
  -v /home/implab/Documents/RFI-ladybug:/work -w /work \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  bash -lc 'yolo export model=yolo11n.pt format=engine device=0 imgsz=640 half=True dynamic=False && \
            mkdir -p models && mv -f yolo11n.engine models/'

# 2) Start the receiver server
sudo docker run --rm -it \
  --runtime nvidia --ipc=host --network host \
  -v /home/implab/Documents/RFI-ladybug:/work -w /work \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  bash -lc 'pip install -q --no-cache-dir pillow numpy && \
            python3 receiver.py --host 0.0.0.0 --port 5000 \
              --yolo-model models/yolo11n.engine --device auto \
              --conf 0.15 --save-dir rx --save-every 1 --save-annotated'
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
              --conf 0.15 --save-dir rx --save-every 1 --save-annotated'
```

Sender (example)
```bash
LBUG_TARGET_IP=<receiver_ip> LBUG_TARGET_PORT=5000 \
LBUG_COLOR_METHOD=DOWNSAMPLE4 LBUG_NUM_CAMS=3 \
./build/ladybugRFI
```

Notes
- The provided TensorRT engine is exported with batch=1 (static). The receiver runs YOLO per-camera to be compatible with TRT engines that do not support multi-image batches.
- Annotated images are written under rx/ as camXX_fNNNNNN_annot.png. Use --save-raw to also write the raw BGR images.
- Start the receiver before the sender and ensure the sender uses the correct receiver IP (USB-C, Wi‑Fi, or Ethernet).