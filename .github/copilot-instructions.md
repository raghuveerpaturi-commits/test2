# Copilot Instructions for Face Detection Project

## Project Overview
Real-time face detection and recognition system optimized for **NVIDIA Jetson Nano**, supporting RTSP camera streams with optional face recognition for security/access control scenarios. This is an embedded system project with strict resource constraints - all optimizations target 2-4GB RAM and limited GPU.

## Architecture & Key Components

### Core Detection Pipeline
1. **VideoStream class** ([face_detection.py](face_detection.py#L22-L73)): Threaded video capture for RTSP streams, uses queue-based buffering to reduce latency
2. **FaceDetector class** ([face_detection.py](face_detection.py#L265-L338)): Dual detection methods
   - Haar Cascade (default): Fast, lower resource usage, good for embedded systems
   - DNN (optional): More accurate, requires model download, supports CUDA acceleration
3. **FaceRecognizer class** ([face_detection.py](face_detection.py#L113-L263)): Multi-image voting system for identity verification
4. **MotionDetector class** ([face_detection.py](face_detection.py#L270-L355)): Optional background subtraction using MOG2 to filter static false positives (chairs, posters)

### Face Recognition Strategy
- **Multiple images per person** (3-4 recommended): Each person can have multiple reference images (`name_1.jpg`, `name_2.jpg`, etc.)
- **Voting algorithm** ([face_detection.py](face_detection.py#L219-L256)): Aggregates matches across all reference images using weighted average of face distances. Lower average distance + more matching images = higher confidence
- **EXIF orientation handling** ([face_detection.py](face_detection.py#L76-L109)): `load_image_with_orientation()` auto-rotates phone/camera images using PIL - critical because phone photos often have incorrect orientation without EXIF parsing

### Motion Detection Strategy (Optional)
- **MOG2 background subtraction** ([face_detection.py](face_detection.py#L273-L281)): Learns background over ~500 frames to detect moving objects
- **Purpose**: Eliminates false positives from static objects (photos on walls, printed faces on posters, chairs that look like faces)
- **Trade-off**: Requires ~10-20 seconds of stationary camera at startup to learn background. Won't detect faces that aren't moving
- **Use case**: Security scenarios where you only care about people entering the frame, not static objects

## Critical Workflows

### Running on Target Hardware (Jetson Nano)
```bash
# Development → Deployment workflow:
# 1. Edit in dev container workspace
# 2. git add . && git commit -m "changes" && git push
# 3. On Jetson: cd ~/test2 && git pull

# Performance-optimized run (common for Jetson):
python3 face_detection.py --scale 0.5 --skip-frames 1 --width 640 --height 480
```

### Model Download (DNN Method)
DNN requires separate model files not in repo (size). Run `python3 download_models.py` first to download:
- `deploy.prototxt` (config)
- `res10_300x300_ssd_iter_140000.caffemodel` (weights)

### Debugging Face Recognition
Use `diagnose_images.py` to test why specific images fail encoding:
- Tests 3 detection methods (CNN, HOG, Haar Cascade) in order
- Tries multiple encoding strategies (large/small model, jitter variations)
- Reports face size, location, and specific failure reasons
- **Critical for Jetson**: Auto-resizes large images to 800px to avoid GPU OOM errors
- **Run this FIRST** if face recognition isn't loading certain images

### Threat Capture Feature
Auto-saves screenshots when unknown faces detected ([face_detection.py](face_detection.py#L485-L505)):
- Triggered when `threat_count >= --threat-threshold` (default: 1)
- Respects `--capture-cooldown` (default: 5s) to avoid duplicate captures
- Saves to `threat_captures/` with timestamp: `threat_20260125_143022_2threats.jpg`
- Filename includes threat count for quick severity assessment

## Project-Specific Patterns

### Performance Optimization Pattern
Every detection parameter is tunable for Jetson's limited resources:
```python
# Frame processing strategy: skip frames + scale down (face_detection.py#L421-L425)
if args.skip_frames == 0 or frame_count % (args.skip_frames + 1) == 1:
    if args.scale != 1.0:
        process_frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
```
Coordinates are scaled back to original frame size for drawing. This pattern is critical for real-time performance on embedded hardware.

### RTSP Stream Handling
RTSP streams use **threaded capture** ([face_detection.py](face_detection.py#L22-L73)) to prevent frame buffering lag. Regular VideoCapture is used for USB cameras. Check for `rtsp://` prefix to decide:
```python
if source.startswith('rtsp'):
    cap = VideoStream(source, buffer_size=1).start()
```

### Threat Detection Colors
Visual convention ([face_detection.py](face_detection.py#L469-L471)): Green box = known person, Red box = unknown/threat. Thickness also differs (2 vs 3) for quick visual identification.

## Dependencies & Environment

### Optional Dependencies
- `face-recognition` + `dlib`: Required only for `--recognize` flag, not default detection
- `dlib` compilation on Jetson takes significant time (~30 min)
- Basic detection works with just `opencv-python` and `numpy`

### CUDA Acceleration
DNN detection auto-enables CUDA if available ([face_detection.py](face_detection.py#L293-L296)):
```python
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

## Common Pitfalls

1. **Don't use `setup_known_faces.py` output directly**: It's a helper that prints instructions, not automated setup
2. **Image naming matters**: `person_name_1.jpg` format is parsed by regex `r'_\d+$'` to group images by person
3. **RTSP requires threading**: Regular VideoCapture will lag on network streams
4. **Frame skipping caches results**: When `--skip-frames 2`, last detection is reused for skipped frames to maintain consistent UI
5. **Motion detection needs warmup**: MOG2 requires ~10-20s of stationary camera to learn background before filtering works well
6. **Diagnose first, debug later**: Always run `diagnose_images.py` before debugging why recognition isn't working - saves 90% of troubleshooting time

## Testing & Validation

No formal test suite. Manual testing approach:
1. Test detection: `python3 face_detection.py` (should show live feed)
2. Test recognition: Add images to `known_faces/`, run `python3 face_detection.py --recognize` (check console for load statistics)
3. Diagnose failed images: `python3 diagnose_images.py` (shows why each image works/fails)
