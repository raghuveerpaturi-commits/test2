#!/usr/bin/env python3
"""
Face Detection for Jetson Nano
Supports both Haar Cascade (fast) and DNN-based detection (more accurate)
Optimized for RTSP camera streaming with threading
"""

import cv2
import argparse
import time
import threading
import queue
import os
import numpy as np
from datetime import datetime
from PIL import Image, ImageOps
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not installed. Install with: pip3 install face-recognition")

class VideoStream:
    """Threaded video capture for reduced latency"""
    def __init__(self, src, buffer_size=1):
        self.stream = cv2.VideoCapture(src)
        
        # RTSP optimizations
        if isinstance(src, str) and src.startswith('rtsp'):
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
            self.stream.set(cv2.CAP_PROP_FPS, 30)
        
        self.stopped = False
        self.q = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        
    def start(self):
        """Start the thread to read frames"""
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self
    
    def update(self):
        """Continuously read frames and keep only the latest"""
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stop()
                return
            
            # Keep only the latest frame
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)
    
    def read(self):
        """Return the latest frame"""
        try:
            return True, self.q.get(timeout=1.0)
        except queue.Empty:
            return False, None
    
    def stop(self):
        """Stop the thread"""
        self.stopped = True
    
    def release(self):
        """Release the video stream"""
        self.stopped = True
        self.stream.release()
    
    def isOpened(self):
        """Check if stream is opened"""
        return self.stream.isOpened()


def load_image_with_orientation(image_path, max_size=1600):
    """Load image and apply EXIF orientation for proper rotation
    
    This handles images taken from phones/cameras at different angles.
    PIL automatically rotates the image based on EXIF orientation data.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) to resize large images
    """
    # Open image with PIL
    pil_image = Image.open(image_path)
    
    # Apply EXIF orientation (auto-rotates based on camera orientation)
    pil_image = ImageOps.exif_transpose(pil_image)
    
    # Resize if too large (helps with memory and processing)
    width, height = pil_image.size
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        # Use Image.LANCZOS for compatibility with older Pillow versions
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
    
    # Convert to RGB if needed (handles RGBA, grayscale, etc.)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert PIL image to numpy array for face_recognition
    return np.array(pil_image), pil_image.size


class FaceRecognizer:
    """Face recognition to identify known vs unknown faces"""
    def __init__(self, known_faces_dir='known_faces', tolerance=0.6):
        self.known_face_encodings = []
        self.known_face_names = []
        self.tolerance = tolerance
        
        if not FACE_RECOGNITION_AVAILABLE:
            print("Face recognition disabled - library not available")
            return
        
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            print(f"Created directory: {known_faces_dir}")
            print(f"Add known face images (jpg/png) to this directory")
            print(f"Name format: person_name_1.jpg, person_name_2.jpg, etc.")
            print(f"Add 3-4 images per person for better accuracy")
            return
        
        # Load known faces - support multiple images per person
        print(f"Loading known faces from {known_faces_dir}...")
        person_image_count = {}
        
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(known_faces_dir, filename)
                
                # Extract person name (remove _1, _2, etc. suffixes)
                base_name = os.path.splitext(filename)[0]
                # Remove trailing numbers like _1, _2, _3
                import re
                name = re.sub(r'_\d+$', '', base_name).replace('_', ' ').title()
                
                try:
                    # Use custom loader with EXIF orientation handling
                    image, img_size = load_image_with_orientation(path)
                    
                    # Try multiple detection models for better success rate
                    encodings = face_recognition.face_encodings(image, model='large')
                    
                    # If large model fails, try small model
                    if not encodings:
                        encodings = face_recognition.face_encodings(image, model='small')
                    
                    # If still no face, try with different number of jitters
                    if not encodings:
                        encodings = face_recognition.face_encodings(image, num_jitters=10, model='large')
                    
                    if encodings:
                        # If multiple faces found, use the largest one (most prominent)
                        if len(encodings) > 1:
                            print(f"  ⚠ Multiple faces ({len(encodings)}) found in {filename}, using largest")
                        
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                        
                        # Track count per person
                        person_image_count[name] = person_image_count.get(name, 0) + 1
                        print(f"  ✓ Loaded: {name} (image #{person_image_count[name]}) - {img_size[0]}x{img_size[1]}")
                    else:
                        print(f"  ✗ No face found in: {filename} (size: {img_size[0]}x{img_size[1]})")
                        print(f"      → Try: Better lighting, frontal face, clear image")
                except Exception as e:
                    print(f"  ✗ Error loading {filename}: {e}")
        
        print(f"\nTotal encodings loaded: {len(self.known_face_encodings)}")
        print(f"Unique persons: {len(set(self.known_face_names))}")
        for person, count in sorted(person_image_count.items()):
            print(f"  - {person}: {count} image(s)")
    
    def recognize_faces(self, frame, face_locations):
        """Recognize faces in the frame
        Args:
            frame: BGR image
            face_locations: List of (x, y, w, h) tuples
        Returns:
            List of (name, is_known) tuples for each face
        """
        if not FACE_RECOGNITION_AVAILABLE or not self.known_face_encodings:
            return [("Unknown", False)] * len(face_locations)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert (x, y, w, h) to (top, right, bottom, left) format
        face_locations_rgb = []
        for (x, y, w, h) in face_locations:
            face_locations_rgb.append((y, x+w, y+h, x))
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations_rgb)
        
        results = []
        for face_encoding in face_encodings:
            # Compare with all known face encodings
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding, 
                tolerance=self.tolerance
            )
            name = "THREAT"
            is_known = False
            
            if True in matches:
                # Calculate distances to all encodings
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                
                # Find all matching indices
                matching_indices = [i for i, match in enumerate(matches) if match]
                
                if matching_indices:
                    # Get the best match among all matches
                    best_match_index = min(matching_indices, 
                                         key=lambda i: face_distances[i])
                    
                    # Use voting: count which person appears most in matches
                    person_votes = {}
                    for idx in matching_indices:
                        person = self.known_face_names[idx]
                        distance = face_distances[idx]
                        if person not in person_votes:
                            person_votes[person] = []
                        person_votes[person].append(distance)
                    
                    # Find person with most votes and best average distance
                    best_person = None
                    best_score = float('inf')
                    
                    for person, distances in person_votes.items():
                        # Average distance for this person across all their images
                        avg_distance = sum(distances) / len(distances)
                        # Weight by number of matches (more matches = more confidence)
                        score = avg_distance / (1 + len(distances) * 0.1)
                        
                        if score < best_score:
                            best_score = score
                            best_person = person
                    
                    if best_person:
                        name = best_person
                        is_known = True
            
            results.append((name, is_known))
        
        return results


class MotionDetector:
    """Detect motion in video frames to filter out static objects"""
    def __init__(self, threshold=25, min_area=500, history=500):
        """
        Initialize motion detector
        Args:
            threshold: Sensitivity threshold (lower=more sensitive, 15-50 recommended)
            min_area: Minimum contour area to consider as motion (pixels)
            history: Number of frames for background learning (higher=slower adaptation)
        """
        self.threshold = threshold
        self.min_area = min_area
        # Use MOG2 background subtractor (works well for stationary cameras)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=threshold,
            detectShadows=False  # Disable shadow detection for better performance
        )
        self.motion_mask = None
        
    def detect_motion(self, frame):
        """Detect motion regions in frame
        Returns:
            motion_mask: Binary mask where white = motion detected
            motion_detected: Boolean if any significant motion found
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up the mask
        # Remove noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours to identify motion regions
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create clean motion mask
        motion_mask = np.zeros_like(fg_mask)
        motion_detected = False
        
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                # Draw filled contour on motion mask
                cv2.drawContours(motion_mask, [contour], -1, 255, -1)
                motion_detected = True
        
        # Dilate motion regions to include nearby areas (catches full face)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        motion_mask = cv2.dilate(motion_mask, kernel_dilate, iterations=2)
        
        self.motion_mask = motion_mask
        return motion_mask, motion_detected
    
    def is_in_motion_region(self, x, y, w, h, margin=20):
        """Check if a bounding box overlaps with motion regions
        Args:
            x, y, w, h: Bounding box coordinates
            margin: Extra margin around box to check (pixels)
        Returns:
            True if box is in or near a motion region
        """
        if self.motion_mask is None:
            return True  # If no motion mask, allow all detections
        
        # Check expanded region around the face
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(self.motion_mask.shape[1], x + w + margin)
        y2 = min(self.motion_mask.shape[0], y + h + margin)
        
        # Extract region of interest from motion mask
        roi = self.motion_mask[y1:y2, x1:x2]
        
        # If any pixels in ROI are white (motion), return True
        motion_pixels = cv2.countNonZero(roi)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        # Require at least 5% of the region to have motion
        return (motion_pixels / total_pixels) > 0.05 if total_pixels > 0 else False


class FaceDetector:
    def __init__(self, method='haar', min_face_size=20):
        """
        Initialize face detector
        Args:
            method: 'haar' for Haar Cascade or 'dnn' for DNN-based detection
            min_face_size: Minimum face size to detect (smaller = catch distant faces)
        """
        self.method = method
        self.min_face_size = min_face_size
        
        if method == 'haar':
            # Load Haar Cascade classifier (faster, good for Jetson Nano)
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        elif method == 'dnn':
            # Load DNN model (more accurate)
            self.model_file = "res10_300x300_ssd_iter_140000.caffemodel"
            self.config_file = "deploy.prototxt"
            try:
                self.net = cv2.dnn.readNetFromCaffe(self.config_file, self.model_file)
                # Enable CUDA if available on Jetson Nano
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    print("Using CUDA acceleration")
            except:
                print("DNN model files not found. Please download them first.")
                print("Run: python3 download_models.py")
                exit(1)
    
    def detect_faces_haar(self, frame):
        """Detect faces using Haar Cascade
        
        Optimized for detecting faces at various distances:
        - scaleFactor=1.05: Finer steps for better small face detection
        - minNeighbors=4: Stricter to reduce false positives (chairs, tables)
        - minSize: Configurable minimum to catch distant faces
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,  # Increased from 3 to reduce false positives
            minSize=(self.min_face_size, self.min_face_size)
        )
        return faces
    
    def detect_faces_dnn(self, frame, conf_threshold=0.5):
        """Detect faces using DNN"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x, y, x2, y2) = box.astype("int")
                faces.append((x, y, x2-x, y2-y, confidence))
        
        return faces
    
    def filter_faces(self, faces, frame_width, frame_height):
        """Filter detected faces to reduce false positives
        
        Filters based on:
        - Aspect ratio (faces should be roughly square)
        - Minimum size (too small = noise)
        - Maximum size (too large = likely not a face)
        """
        filtered = []
        
        for face in faces:
            if self.method == 'haar':
                x, y, w, h = face
                conf = 1.0
            else:
                x, y, w, h, conf = face
            
            # Check aspect ratio (faces are roughly square: 0.7 to 1.5)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.7 or aspect_ratio > 1.5:
                continue  # Skip - chairs/tables often have wrong aspect ratio
            
            # Check minimum face size (at least 30x30 pixels after scaling)
            if w < 30 or h < 30:
                continue  # Too small - likely noise
            
            # Check maximum face size (not more than 80% of frame)
            max_size = min(frame_width, frame_height) * 0.8
            if w > max_size or h > max_size:
                continue  # Too large - likely false positive
            
            # Check face position (should not be at extreme edges)
            if x < 5 or y < 5 or x + w > frame_width - 5 or y + h > frame_height - 5:
                continue  # At edge - likely partial detection
            
            filtered.append(face)
        
        return filtered
    
    def detect(self, frame, conf_threshold=0.5):
        """Detect faces using the selected method with filtering"""
        frame_height, frame_width = frame.shape[:2]
        
        if self.method == 'haar':
            faces = self.detect_faces_haar(frame)
        else:
            faces = self.detect_faces_dnn(frame, conf_threshold)
        
        # Apply filtering to reduce false positives
        filtered_faces = self.filter_faces(faces, frame_width, frame_height)
        
        return filtered_faces


def main():
    parser = argparse.ArgumentParser(description='Face Detection on Jetson Nano')
    parser.add_argument('--method', type=str, default='haar', 
                        choices=['haar', 'dnn'],
                        help='Detection method: haar (faster) or dnn (more accurate)')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source: camera index (0,1) or video file path')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for DNN method (0-1)')
    parser.add_argument('--width', type=int, default=640,
                        help='Camera width')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera height')
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='Process every Nth frame (0=process all, 2=every other frame)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for processing (0.5 = half size, faster)')
    parser.add_argument('--known-faces', type=str, default='known_faces',
                        help='Directory containing known face images')
    parser.add_argument('--tolerance', type=float, default=0.6,
                        help='Face recognition tolerance (lower=stricter, 0.6=default)')
    parser.add_argument('--recognize', action='store_true',
                        help='Enable face recognition (identify known vs unknown)')
    parser.add_argument('--min-face-size', type=int, default=20,
                        help='Minimum face size in pixels (lower=detect smaller/distant faces)')
    parser.add_argument('--motion-detection', action='store_true',
                        help='Enable motion detection to filter static objects (chairs, benches)')
    parser.add_argument('--motion-threshold', type=int, default=25,
                        help='Motion sensitivity (15-50, lower=more sensitive)')
    parser.add_argument('--motion-min-area', type=int, default=500,
                        help='Minimum motion area in pixels (reduce false positives)')
    parser.add_argument('--capture-threats', action='store_true',
                        help='Auto-capture images when threats detected (RECOMMENDED for debugging false positives)')
    parser.add_argument('--threat-threshold', type=int, default=1,
                        help='Number of threats to trigger capture (default: 1)')
    parser.add_argument('--capture-dir', type=str, default='threat_captures',
                        help='Directory to save captured threat images')
    parser.add_argument('--capture-cooldown', type=int, default=5,
                        help='Seconds between captures to avoid duplicates')
    parser.add_argument('--min-detections', type=int, default=1,
                        help='Minimum consecutive detections before considering valid face (reduces false positives)')
    
    args = parser.parse_args()
    
    # Initialize face detector
    detector = FaceDetector(method=args.method, min_face_size=args.min_face_size)
    
    # Initialize motion detector if enabled
    motion_detector = None
    if args.motion_detection:
        motion_detector = MotionDetector(
            threshold=args.motion_threshold,
            min_area=args.motion_min_area
        )
        print(f"Motion detection enabled (threshold={args.motion_threshold}, min_area={args.motion_min_area})")
        print("Static objects (chairs, benches) will be filtered out")
    
    # Setup threat capture if enabled
    last_capture_time = 0
    capture_count = 0
    if args.capture_threats:
        if not os.path.exists(args.capture_dir):
            os.makedirs(args.capture_dir)
            print(f"Created threat capture directory: {args.capture_dir}/")
        print(f"Auto-capture enabled: Will save images when {args.threat_threshold}+ threat(s) detected")
        print(f"Captures saved to: {os.path.abspath(args.capture_dir)}/")
        print(f"Cooldown: {args.capture_cooldown}s between captures")
    
    # Initialize face recognizer if enabled
    recognizer = None
    if args.recognize:
        recognizer = FaceRecognizer(args.known_faces, args.tolerance)
    
    # Open video source with threading for RTSP
    if args.source.isdigit():
        source = int(args.source)
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        source = args.source
        # Use threaded capture for RTSP streams
        if source.startswith('rtsp'):
            print("Using threaded capture for RTSP stream...")
            cap = VideoStream(source, buffer_size=1).start()
            time.sleep(2.0)  # Allow stream to initialize
        else:
            cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video source {args.source}")
        return
    
    print(f"Starting face detection using {args.method} method...")
    print(f"Processing scale: {args.scale}x")
    if args.skip_frames > 0:
        print(f"Processing every {args.skip_frames + 1} frames")
    print("Press 'q' to quit")
    
    # FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    frame_count = 0
    faces = []  # Cache faces for skipped frames
    
    # Face detection stability tracking (reduce false positives)
    face_detection_history = {}  # key: (x, y, w, h), value: count
    detection_threshold = args.min_detections
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot read frame")
            break
        
        frame_count += 1
        
        # Process frames based on skip_frames setting
        if args.skip_frames == 0 or frame_count % (args.skip_frames + 1) == 1:
            # Scale down for faster processing
            if args.scale != 1.0:
                process_frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
            else:
                process_frame = frame
            
            # Detect motion first (if enabled)
            motion_mask = None
            if motion_detector:
                motion_mask, has_motion = motion_detector.detect_motion(process_frame)
                if not has_motion:
                    # No motion detected, skip face detection
                    faces = []
                    detected_faces = []
                else:
                    # Motion detected, proceed with face detection
                    detected_faces = detector.detect(process_frame, args.conf)
            else:
                # No motion detection, always run face detection
                detected_faces = detector.detect(process_frame, args.conf)
            
            # Scale face coordinates back to original frame size
            if args.scale != 1.0:
                faces = []
                for face in detected_faces:
                    if args.method == 'haar':
                        x, y, w, h = face
                        scaled_face = (int(x/args.scale), int(y/args.scale), 
                                     int(w/args.scale), int(h/args.scale))
                    else:
                        x, y, w, h, conf = face
                        scaled_face = (int(x/args.scale), int(y/args.scale), 
                                     int(w/args.scale), int(h/args.scale), conf)
                    
                    # Filter by motion if enabled
                    if motion_detector:
                        x_check, y_check, w_check, h_check = scaled_face[:4]
                        # Scale motion check back to process_frame size
                        x_motion = int(x_check * args.scale)
                        y_motion = int(y_check * args.scale)
                        w_motion = int(w_check * args.scale)
                        h_motion = int(h_check * args.scale)
                        if motion_detector.is_in_motion_region(x_motion, y_motion, w_motion, h_motion):
                            faces.append(scaled_face)
                    else:
                        faces.append(scaled_face)
            else:
                # No scaling, filter by motion if enabled
                if motion_detector:
                    faces = []
                    for face in detected_faces:
                        x, y, w, h = face[:4]
                        if motion_detector.is_in_motion_region(x, y, w, h):
                            faces.append(face)
                else:
                    faces = detected_faces
            
            # Apply detection stability filter (reduce transient false positives)
            if detection_threshold > 1:
                stable_faces = []
                for face in faces:
                    x, y, w, h = face[:4]
                    # Create a region key (rounded to reduce noise)
                    region_key = (round(x / 20) * 20, round(y / 20) * 20, round(w / 20) * 20, round(h / 20) * 20)
                    
                    # Increment detection count
                    face_detection_history[region_key] = face_detection_history.get(region_key, 0) + 1
                    
                    # Only include if detected enough times
                    if face_detection_history[region_key] >= detection_threshold:
                        stable_faces.append(face)
                
                # Clean old entries (keep only last 100 regions)
                if len(face_detection_history) > 100:
                    # Keep only the most recent detections
                    sorted_keys = sorted(face_detection_history.items(), key=lambda x: x[1], reverse=True)
                    face_detection_history = dict(sorted_keys[:100])
                
                faces = stable_faces
            
            # Recognize faces if enabled
            if recognizer and len(faces) > 0:
                # Extract just locations for recognition
                face_locations = [(x, y, w, h) for x, y, w, h, *_ in 
                                [face if len(face) == 5 else (*face, 0) for face in faces]]
                face_identities = recognizer.recognize_faces(frame, face_locations)
            else:
                face_identities = [("Unknown", False)] * len(faces)
            
            # Check for threats and capture if enabled
            if args.capture_threats and len(faces) > 0:
                threat_count = sum(1 for _, is_known in face_identities if not is_known)
                current_time = time.time()
                
                # Capture if threshold met and cooldown expired
                if (threat_count >= args.threat_threshold and 
                    current_time - last_capture_time >= args.capture_cooldown):
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"threat_{timestamp}_{threat_count}threats.jpg"
                    filepath = os.path.join(args.capture_dir, filename)
                    
                    # Save full-resolution frame with bounding boxes drawn
                    capture_frame = frame.copy()
                    
                    # Draw detections on capture for easy review
                    for idx_cap, face_cap in enumerate(faces):
                        x_cap, y_cap, w_cap, h_cap = face_cap[:4]
                        name_cap, is_known_cap = face_identities[idx_cap] if idx_cap < len(face_identities) else ("Unknown", False)
                        color_cap = (0, 255, 0) if is_known_cap else (0, 0, 255)
                        cv2.rectangle(capture_frame, (x_cap, y_cap), (x_cap+w_cap, y_cap+h_cap), color_cap, 3)
                        cv2.putText(capture_frame, name_cap, (x_cap, y_cap-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_cap, 2)
                    
                    cv2.imwrite(filepath, capture_frame)
                    
                    capture_count += 1
                    last_capture_time = current_time
                    print(f"⚠ THREAT CAPTURED [{capture_count}]: {filename} ({threat_count} threat(s))")
                    print(f"   → Saved with bounding boxes: {os.path.abspath(filepath)}")
        
        # Draw rectangles around faces
        for idx, face in enumerate(faces):
            if args.method == 'haar':
                x, y, w, h = face
                conf = None
            else:
                x, y, w, h, conf = face
            
            # Get identity if recognition enabled
            name, is_known = face_identities[idx] if idx < len(face_identities) else ("Unknown", False)
            
            # Color: Green for known, Red for unknown/threat
            color = (0, 255, 0) if is_known else (0, 0, 255)
            thickness = 2 if is_known else 3
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            # Label
            if args.recognize:
                label = name
                if not is_known:
                    label = f"⚠ {name}"  # Warning symbol for threats
            else:
                label = f'Face: {conf:.2f}' if conf else 'Face'
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y-label_size[1]-10), (x+label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Calculate and display FPS
        fps_counter += 1
        if time.time() - fps_start_time > 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        # Display info
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show motion detection status
        if motion_detector:
            motion_text = "Motion: ON"
            cv2.putText(frame, motion_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # Show capture status if enabled
        if args.capture_threats:
            # Count current threats
            threat_count = sum(1 for _, is_known in face_identities if not is_known)
            capture_text = f"Captures: {capture_count} | Threats: {threat_count}"
            color = (0, 0, 255) if threat_count >= args.threat_threshold else (0, 255, 0)
            cv2.putText(frame, capture_text, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Optional: overlay motion mask (semi-transparent blue)
            if motion_detector.motion_mask is not None:
                # Resize motion mask to match display frame
                if args.scale != 1.0:
                    motion_overlay = cv2.resize(motion_detector.motion_mask, 
                                              (frame.shape[1], frame.shape[0]))
                else:
                    motion_overlay = motion_detector.motion_mask
                # Create blue overlay where motion is detected
                motion_viz = np.zeros_like(frame)
                motion_viz[:, :, 0] = motion_overlay  # Blue channel
                # Blend with original frame (20% opacity)
                frame = cv2.addWeighted(frame, 1.0, motion_viz, 0.2, 0)
        
        # Display frame
        cv2.imshow('Face Detection - Jetson Nano', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    if isinstance(cap, VideoStream):
        cap.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Face detection stopped")


if __name__ == "__main__":
    main()
