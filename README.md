# LadybugRFI

C++ sender that captures from FLIR Ladybug, processes frames, and streams them over TCP to a Python receiver. It uses a lock-stepped double buffer to avoid stalls and crops/rescales images to square output.

Features
- Double-buffer producer/sender with strict alternation.
- Bottom square crop (height = min(rows, cols)) then resize to OUT_SIZE x OUT_SIZE (default 640).
- Wire format: simple header + 6 per-camera images, BGRU (BGRA) 4 bytes/pixel.
- Python receiver that saves all frames with progressive numbering and survives reconnects.

Requirements
- Linux
- FLIR Ladybug SDK installed (headers in /usr/include/ladybug, libs in /lib/ladybug)
- CMake ≥ 3.10, GCC
- Optional (Python receiver): Conda or Python 3.11 with numpy and pillow

Repository layout
- main.cpp — C++ sender
- CMakeLists.txt — build config
- receiver.py — Python receiver that listens and saves frames
- environment.yml — Conda env for Python tools (optional)
- docs/PROTOCOL.md — wire protocol
- docs/ARCHITECTURE.md — system design
- CONTRIBUTING.md — build/debug guidelines

Quickstart (local loopback)
1) Receiver (Terminal 1, in Conda env)
```bash
cd /usr/src/ladybug/src/ladybugRFI
conda env create -f environment.yml  # one-time
conda activate ladybugrfi
python receiver.py --host 127.0.0.1 --port 5000 --save-dir rx_previews
```

2) Sender (Terminal 2)
```bash
cd /usr/src/ladybug/src/ladybugRFI
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
LBUG_TARGET_IP=127.0.0.1 LBUG_TARGET_PORT=5000 LBUG_OUT_SIZE=640 ./build/ladybugRFI
```

Configuration (env vars)
- LBUG_TARGET_IP: receiver IP (default 192.168.0.2)
- LBUG_TARGET_PORT: receiver port (default 5000)
- LBUG_OUT_SIZE: output side in pixels, square (default 640)

Troubleshooting
- Include error on ladybug.h in VS Code: add includePath /usr/include/ladybug or point IntelliSense to build/compile_commands.json.
- Link errors for ladybug libs: ensure libs exist in /lib/ladybug and RPATH is set by CMake (see generated link line).
- Receiver shows 2992x4096: after crop+resize it should show 640x640; rebuild if not.
- Disconnections: receiver auto-accepts; sender reconnects and resumes.

License
Add a LICENSE file if you plan to share publicly (e.g., MIT).