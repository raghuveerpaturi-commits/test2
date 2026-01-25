# Face Detection for Jetson Nano

Real-time face detection system optimized for NVIDIA Jetson Nano with support for RTSP camera streams and face recognition capabilities.

## Features

- **Dual Detection Methods:**
  - Haar Cascade (faster, lower resource usage)
  - DNN-based detection (more accurate, requires model download)
- **RTSP Camera Support** with threaded video capture for reduced latency
- **Face Recognition** to identify known vs unknown faces
- **Performance Optimization:**
  - Adjustable frame skipping
  - Configurable processing scale
  - CUDA acceleration support
- **Real-time FPS display**
- **Threat Detection** with visual alerts for unknown faces

## Hardware Requirements

- NVIDIA Jetson Nano (2GB or 4GB)
- USB/CSI Camera or RTSP network camera
- Display (for viewing output)

## Installation on Jetson Nano

### 1. Clone the Repository

```bash
cd ~
git clone https://github.com/raghuveerpaturi-commits/test2.git
cd test2
```

### 2. Install Dependencies

```bash
# Install OpenCV and NumPy
pip3 install -r requirements.txt

# Optional: Install face_recognition for identity recognition
# Note: This requires compiling dlib which may take time on Jetson
pip3 install face-recognition
```

### 3. Download DNN Models (Optional)

If you want to use the more accurate DNN method:

```bash
python3 download_models.py
```

## Usage

### Basic Usage (Haar Cascade - Fastest)

```bash
# Use default USB camera
python3 face_detection.py

# Specify camera index
python3 face_detection.py --source 0
```

### RTSP Camera Stream

```bash
python3 face_detection.py --source "rtsp://username:password@camera_ip:port/stream"
```

### DNN Method (More Accurate)

```bash
python3 face_detection.py --method dnn --conf 0.5
```

### Face Recognition Mode

```bash
# First, add known face images to known_faces/ directory
# Name format: person_name.jpg

python3 face_detection.py --recognize --known-faces known_faces
```

### Performance Optimization Options

```bash
# Process at half resolution (2x faster)
python3 face_detection.py --scale 0.5

# Process every other frame
python3 face_detection.py --skip-frames 1

# Combine optimizations for max performance
python3 face_detection.py --scale 0.5 --skip-frames 2 --width 640 --height 480
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | `haar` | Detection method: `haar` or `dnn` |
| `--source` | `0` | Video source: camera index or RTSP URL |
| `--conf` | `0.5` | Confidence threshold for DNN (0-1) |
| `--width` | `640` | Camera resolution width |
| `--height` | `480` | Camera resolution height |
| `--skip-frames` | `0` | Process every Nth frame (0=all frames) |
| `--scale` | `1.0` | Processing scale factor (0.5=half size) |
| `--recognize` | `False` | Enable face recognition |
| `--known-faces` | `known_faces` | Directory with known face images |
| `--tolerance` | `0.6` | Face recognition strictness (lower=stricter) |

## Face Recognition Setup

### Adding Known Faces (3-4 Images Per Person Recommended)

For best accuracy, add **multiple images** per person with varying conditions:

```bash
# Run setup helper
python3 setup_known_faces.py

# Example structure - Multiple images per person
known_faces/
  ├── john_doe_1.jpg         # Front facing
  ├── john_doe_2.jpg         # Slight angle
  ├── john_doe_3.jpg         # With glasses
  ├── jane_smith_1.jpg       # Smiling
  ├── jane_smith_2.jpg       # Neutral expression
  ├── jane_smith_3.jpg       # Different lighting
  ├── raghuveer_paturi_1.jpg # Normal
  ├── raghuveer_paturi_2.jpg # Side angle
  ├── raghuveer_paturi_3.jpg # With accessories
  └── raghuveer_paturi_4.jpg # Different time/lighting

# Run with recognition
python3 face_detection.py --recognize
```

**Naming Convention:**
- Format: `firstname_lastname_NUMBER.jpg`
- Numbers can be `_1`, `_2`, `_3`, `_4`, etc.
- All images with the same base name are treated as the same person
- The system uses **voting** across all images for better accuracy

**Image Tips:**
- ✅ 3-4 images per person (optimal)
- ✅ Vary angles slightly
- ✅ Include with/without glasses, hats
- ✅ Different lighting conditions
- ✅ Clear, well-lit faces (300x300 pixels minimum)
- ❌ Avoid blurry images
- ❌ One face per image only

**Detection Colors:**
- 🟢 Green = Known/Authorized person
- 🔴 Red = Unknown/Threat detected

## Performance Tips for Jetson Nano

1. **Use Haar Cascade for real-time:** Fastest method, good accuracy
2. **Reduce resolution:** `--width 640 --height 480` or lower
3. **Enable frame skipping:** `--skip-frames 1` or `2`
4. **Scale down processing:** `--scale 0.5` for 2x performance boost
5. **Close other applications** to free up resources
6. **Use CUDA:** DNN method automatically uses CUDA if available

## Troubleshooting

### Camera not detected
```bash
# List available cameras
ls /dev/video*

# Test camera
gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink
```

### Low FPS
- Reduce resolution: `--width 320 --height 240`
- Skip frames: `--skip-frames 2`
- Use Haar method instead of DNN
- Scale processing: `--scale 0.5`

### Face recognition errors
```bash
# Install dlib dependencies
sudo apt-get install build-essential cmake
pip3 install dlib face-recognition
```

## Project Structure

```
test2/
├── face_detection.py       # Main detection script
├── download_models.py      # DNN model downloader
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── known_faces/           # Directory for known face images (auto-created)
├── deploy.prototxt        # DNN config (after download_models.py)
└── res10_300x300_ssd_iter_140000.caffemodel  # DNN weights (after download)
```

## Development Workflow

### Making Changes

1. Edit code in this workspace
2. Commit and push changes
3. Pull on Jetson to update

```bash
# On development machine (this workspace)
git add .
git commit -m "Your changes"
git push

# On Jetson Nano
cd ~/test2
git pull
```

## Examples

### Example 1: USB Camera with Basic Detection
```bash
python3 face_detection.py --source 0
```

### Example 2: RTSP Camera with DNN
```bash
python3 face_detection.py --method dnn --source "rtsp://admin:pass@192.168.1.100:554/stream"
```

### Example 3: Optimized for Performance
```bash
python3 face_detection.py --scale 0.5 --skip-frames 1 --width 640 --height 480
```

### Example 4: Face Recognition with Alerts
```bash
python3 face_detection.py --recognize --known-faces ./authorized_persons
```

## License

MIT License - Feel free to modify and distribute

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Press 'q' to quit the application when running**