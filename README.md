# LadybugRFI

C++ sender that captures from FLIR Ladybug, processes frames with a selectable Ladybug SDK color method (default: DOWNSAMPLE4), and streams them over TCP to a Python receiver. It uses a low-latency mailbox (latest-wins) buffer so the sender always transmits the freshest frame.

Features
- Mailbox triple-buffer (latest-wins) for low latency; frames may be dropped if the consumer lags.
- Selectable color processing via LBUG_COLOR_METHOD; built-in downsample mapping for DOWNSAMPLE4/16/64 and MONO.
- Bottom square crop then resize to 640x640; payload format: BGR, 3 bytes/pixel.
- Python receiver that can save sampled frames and survives reconnects.
- First-frame-only size diagnostics.
- NEW: YOLOv11 inference in the receiver (persons only), compatible with NVIDIA Jetson (see guide below).
- NEW: Per-frame inference timing added to receiver stats (prints “infer=XX.XX ms”).

## Configurazione (env vars)
- LBUG_TARGET_IP: receiver IP (default 192.168.0.2)
- LBUG_TARGET_PORT: receiver port (default 5000)
- LBUG_COLOR_METHOD: DISABLE, EDGE_SENSING, NEAREST_NEIGHBOR_FAST, RIGOROUS, DOWNSAMPLE4, DOWNSAMPLE16, DOWNSAMPLE64, MONO, HQLINEAR, HQLINEAR_GPU, DIRECTIONAL_FILTER, WEIGHTED_DIRECTIONAL_FILTER
- LBUG_NUM_CAMS: numero di camere inviate (1..6, default 3)

## Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
```

## Esecuzione
Receiver:
```bash
python receiver.py --host 0.0.0.0 --port 5000 --save-dir rx_previews --save-every 10
```

Sender:
```bash
LBUG_TARGET_IP=127.0.0.1 LBUG_TARGET_PORT=5000 LBUG_COLOR_METHOD=DOWNSAMPLE4 LBUG_NUM_CAMS=3 ./build/ladybugRFI
```