# Wire protocol (LDB0 v1)

Transport: TCP client (sender) -> TCP server (receiver).

Message per frame:
1) Header (packed, big-endian for integers)
- magic: 4 bytes ASCII "LDB0"
- version: uint32 (currently 1)
- frame: uint32 (sender frame counter)
- rows: uint32 (output rows, e.g., 640)
- cols: uint32 (output cols, e.g., 640)
- bpp: uint32 (bytes per pixel, 4 for BGRU/BGRA)
- numCams: uint32 (typically 6)

2) Payload
- numCams consecutive images
- Each image is rows*cols*bpp bytes
- Pixel format: BGRU (BGRA), 4 bytes per pixel, byte-order B,G,R,A

Notes
- No per-image size fields beyond the header; receiver computes payload size as rows*cols*bpp.
- Drop policy: sender may drop a frame on reconnect; receiver keeps a progressive counter for saved files.

# Architecture

Components
- Capture (producer): grabs Ladybug frames, converts to per-camera BGRU into scratch, crops bottom square and resizes to OUT_SIZE x OUT_SIZE, writes into cell[idx].
- Double buffer: two FrameCell instances (idx = frame & 1) with ready flags and a condition_variable.
- Sender (consumer): waits for the next ready cell in alternating order, sends header + 6 images, then marks the cell free.

Threading and invariants
- Producer sets ready[idx]=true only after writing the full cell.
- Sender reads only the index that matches nextReadIdx; after send attempt, sets ready[idx]=false.
- Producer blocks if the target cell is still ready (backpressure).
- Sender reconnects TCP on failure and may drop one frame to free the cell.

Image processing
- For each camera:
  - Convert to full-res BGRU.
  - Crop bottom square band: size = min(srcRows, srcCols), anchored to bottom.
  - Resize (nearest neighbor) to OUT_SIZE x OUT_SIZE.
- OUT_SIZE configurable via LBUG_OUT_SIZE (default 640).

Failure modes
- Camera grab/convert error: logged and the loop continues.
- Network failure: sender reconnects and resumes.
- Resolution change: scratch and cells reallocate to match output size.

# Contributing / Dev notes

Build (Release)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
```

Run locally
```bash
conda activate ladybugrfi
python receiver.py --host 127.0.0.1 --port 5000 --save-dir rx_previews
LBUG_TARGET_IP=127.0.0.1 LBUG_TARGET_PORT=5000 ./build/ladybugRFI
```

C++ style
- C++17, -Wall -Wextra -Wpedantic in Debug builds.
- Keep networking helpers and resizer pure functions.
- Avoid clearing a cell’s ready flag until after the send attempt completes to preserve the producer/consumer contract.

Where things live
- Ladybug SDK headers: /usr/include/ladybug
- Ladybug SDK libs: /lib/ladybug
- IntelliSense: use build/compile_commands.json or add includePath to /usr/include/ladybug.