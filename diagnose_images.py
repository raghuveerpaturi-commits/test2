#!/usr/bin/env python3
"""
Diagnostic tool to check known face images and identify issues
Helps debug why certain images aren't being loaded for face recognition
"""

import os
import sys
from PIL import Image, ImageOps
import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Error: face_recognition not installed")
    print("Install with: pip3 install face-recognition")
    sys.exit(1)

import cv2

KNOWN_FACES_DIR = "known_faces"

def load_image_with_orientation(image_path, max_size=800):
    """Load image with proper EXIF orientation and resize for Jetson Nano
    
    Args:
        image_path: Path to image
        max_size: Maximum dimension (smaller for Jetson to avoid OOM)
    """
    pil_image = Image.open(image_path)
    pil_image = ImageOps.exif_transpose(pil_image)
    
    # Resize if too large (critical for Jetson Nano memory limits)
    width, height = pil_image.size
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        # Use Image.LANCZOS for compatibility with older Pillow versions
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
        print(f"  Resized from {width}x{height} to {new_size[0]}x{new_size[1]} (GPU memory)")
    
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    return np.array(pil_image), pil_image.size

def diagnose_image(filepath):
    """Diagnose a single image and report detailed information"""
    filename = os.path.basename(filepath)
    print(f"\n{'='*70}")
    print(f"Analyzing: {filename}")
    print('='*70)
    
    try:
        # Load with orientation
        image, (width, height) = load_image_with_orientation(filepath)
        print(f"✓ Image loaded successfully")
        print(f"  Size: {width}x{height} pixels")
        print(f"  Aspect ratio: {width/height:.2f}")
        
        # Check if too small
        if width < 100 or height < 100:
            print(f"  ⚠ WARNING: Image is very small (minimum recommended: 300x300)")
        
        # Try to detect faces with different models
        print(f"\nFace Detection Attempts:")
        
        # Attempt 1: HOG model first (faster, less memory, good for Jetson)
        print(f"  1. HOG model (CPU, faster, less memory)...")
        face_locations_hog = face_recognition.face_locations(image, model='hog')
        if face_locations_hog:
            print(f"     ✓ Found {len(face_locations_hog)} face(s)")
            for i, (top, right, bottom, left) in enumerate(face_locations_hog, 1):
                face_w = right - left
                face_h = bottom - top
                print(f"       Face {i}: {face_w}x{face_h} pixels at ({left},{top})")
        else:
            print(f"     ✗ No faces detected")
        
        # Attempt 2: CNN model (GPU, more accurate but memory intensive)
        print(f"  2. CNN model (GPU, more accurate but memory intensive)...")
        try:
            face_locations = face_recognition.face_locations(image, model='cnn')
            if face_locations:
                print(f"     ✓ Found {len(face_locations)} face(s)")
            else:
                print(f"     ✗ No faces detected")
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"     ✗ GPU out of memory (image too large for Jetson)")
                face_locations = []
            else:
                raise
        
        # Attempt 3: OpenCV Haar Cascade
        print(f"  3. OpenCV Haar Cascade...")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces_haar = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
        )
        if len(faces_haar) > 0:
            print(f"     ✓ Found {len(faces_haar)} face(s)")
            for i, (x, y, w, h) in enumerate(faces_haar, 1):
                print(f"       Face {i}: {w}x{h} pixels at ({x},{y})")
        else:
            print(f"     ✗ No faces detected")
        
        # Try encoding with different parameters
        if face_locations or face_locations_hog:
            print(f"\nFace Encoding Attempts:")
            
            # Use detected locations (prefer HOG for Jetson stability)
            locations_to_use = face_locations_hog if face_locations_hog else face_locations
            
            # Attempt with large model
            print(f"  1. Large model...")
            encodings_large = face_recognition.face_encodings(
                image, known_face_locations=locations_to_use, model='large'
            )
            print(f"     {'✓' if encodings_large else '✗'} {len(encodings_large)} encoding(s)")
            
            # Attempt with small model
            print(f"  2. Small model...")
            encodings_small = face_recognition.face_encodings(
                image, known_face_locations=locations_to_use, model='small'
            )
            print(f"     {'✓' if encodings_small else '✗'} {len(encodings_small)} encoding(s)")
            
            # Attempt with more jitters
            print(f"  3. Large model with 10 jitters (slower but more robust)...")
            encodings_jitter = face_recognition.face_encodings(
                image, known_face_locations=locations_to_use, 
                num_jitters=10, model='large'
            )
            print(f"     {'✓' if encodings_jitter else '✗'} {len(encodings_jitter)} encoding(s)")
            
            if encodings_large or encodings_small or encodings_jitter:
                print(f"\n✓ SUCCESS: This image can be used for face recognition!")
                return True
            else:
                print(f"\n✗ FAILED: Face detected but encoding failed")
                return False
        else:
            print(f"\n✗ FAILED: No faces detected with any method")
            print(f"\nSuggestions:")
            print(f"  • Ensure face is clearly visible and well-lit")
            print(f"  • Face should be looking towards camera (frontal view works best)")
            print(f"  • Avoid extreme angles, shadows, or obstructions")
            print(f"  • Try a different photo of the same person")
            print(f"  • Minimum face size should be about 80x80 pixels")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

def main():
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Error: Directory '{KNOWN_FACES_DIR}' not found")
        sys.exit(1)
    
    files = [f for f in os.listdir(KNOWN_FACES_DIR) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not files:
        print(f"No images found in {KNOWN_FACES_DIR}/")
        sys.exit(1)
    
    print(f"\n{'#'*70}")
    print(f"# FACE IMAGE DIAGNOSTIC TOOL")
    print(f"# Analyzing {len(files)} image(s) in {KNOWN_FACES_DIR}/")
    print(f"{'#'*70}")
    
    results = {}
    for filename in sorted(files):
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        success = diagnose_image(filepath)
        results[filename] = success
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    
    print(f"\nTotal images: {len(results)}")
    print(f"✓ Usable: {successful}")
    print(f"✗ Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed images:")
        for filename, success in results.items():
            if not success:
                print(f"  • {filename}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
