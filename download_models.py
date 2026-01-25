#!/usr/bin/env python3
"""
Download DNN models for face detection
These are required for --method dnn option
"""

import os
import urllib.request
import sys

# Model URLs
DEPLOY_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

def download_file(url, filename):
    """Download file with progress bar"""
    print(f"Downloading {filename}...")
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r{filename}: {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"\n✗ Error downloading {filename}: {e}")
        return False

def main():
    print("Downloading DNN face detection models...")
    print("These files are needed for --method dnn option\n")
    
    # Check if files already exist
    if os.path.exists("deploy.prototxt") and os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"):
        print("Model files already exist!")
        overwrite = input("Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("Skipping download")
            return
    
    # Download files
    success = True
    success &= download_file(DEPLOY_PROTOTXT_URL, "deploy.prototxt")
    success &= download_file(CAFFEMODEL_URL, "res10_300x300_ssd_iter_140000.caffemodel")
    
    if success:
        print("\n✓ All models downloaded successfully!")
        print("You can now use: python3 face_detection.py --method dnn")
    else:
        print("\n✗ Some downloads failed. Please check your internet connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()
