#!/usr/bin/env python3
"""
Review and analyze captured threat images
Helps identify false positives (chairs, tables) vs real threats
"""

import os
import sys
from datetime import datetime
from collections import defaultdict

CAPTURE_DIR = "threat_captures"

def analyze_captures():
    """Analyze all captured threat images"""
    
    if not os.path.exists(CAPTURE_DIR):
        print(f"❌ No captures directory found: {CAPTURE_DIR}/")
        print(f"Run face detection with --capture-threats flag first")
        return
    
    files = sorted([f for f in os.listdir(CAPTURE_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not files:
        print(f"📁 No captured images found in {CAPTURE_DIR}/")
        print(f"Run: python3 face_detection.py --recognize --capture-threats")
        return
    
    print(f"\n{'='*70}")
    print(f"THREAT CAPTURE ANALYSIS")
    print(f"{'='*70}\n")
    print(f"📊 Total captures: {len(files)}")
    print(f"📁 Location: {os.path.abspath(CAPTURE_DIR)}/\n")
    
    # Analyze by date
    captures_by_date = defaultdict(list)
    threat_counts = defaultdict(int)
    
    for filename in files:
        filepath = os.path.join(CAPTURE_DIR, filename)
        file_size = os.path.getsize(filepath) / 1024  # KB
        
        # Parse filename: threat_YYYYMMDD_HHMMSS_Nthreats.jpg
        parts = filename.split('_')
        if len(parts) >= 4:
            date_str = parts[1]
            time_str = parts[2]
            threat_str = parts[3].split('.')[0]  # Remove extension
            
            # Format for display
            try:
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                date_display = date_obj.strftime("%Y-%m-%d")
            except:
                date_display = date_str
            
            try:
                time_obj = datetime.strptime(time_str, "%H%M%S")
                time_display = time_obj.strftime("%H:%M:%S")
            except:
                time_display = time_str
            
            # Extract threat count
            threat_count = 1
            if 'threat' in threat_str:
                try:
                    threat_count = int(threat_str.replace('threats', '').replace('threat', ''))
                except:
                    pass
            
            captures_by_date[date_display].append({
                'filename': filename,
                'filepath': filepath,
                'time': time_display,
                'threats': threat_count,
                'size_kb': file_size
            })
            
            threat_counts[threat_count] += 1
    
    # Display by date
    for date, captures in sorted(captures_by_date.items(), reverse=True):
        print(f"📅 {date} ({len(captures)} captures)")
        print(f"   {'Time':<10} {'Threats':<8} {'Size':<10} {'Filename'}")
        print(f"   {'-'*60}")
        
        for capture in sorted(captures, key=lambda x: x['time'], reverse=True):
            threat_icon = "🔴" if capture['threats'] > 1 else "🟡"
            print(f"   {capture['time']:<10} {threat_icon} {capture['threats']:<6} "
                  f"{capture['size_kb']:>6.1f} KB  {capture['filename']}")
        print()
    
    # Summary statistics
    print(f"{'='*70}")
    print(f"STATISTICS")
    print(f"{'='*70}\n")
    
    total_threats = sum(c['threats'] for captures in captures_by_date.values() for c in captures)
    avg_threats = total_threats / len(files) if files else 0
    
    print(f"Total threat detections: {total_threats}")
    print(f"Average threats per capture: {avg_threats:.1f}")
    print(f"\nThreat count distribution:")
    for count in sorted(threat_counts.keys()):
        print(f"  {count} threat(s): {threat_counts[count]} captures")
    
    # Recommendations
    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    print(f"🔍 Review captured images to identify patterns:")
    print(f"   1. View images: cd {CAPTURE_DIR}/ && ls -lh")
    print(f"   2. Look for common false positives (chairs, tables, posters)")
    print(f"   3. Check if faces are correctly bounded in red boxes")
    print(f"\n💡 To reduce false positives:")
    print(f"   • Use --motion-detection flag (filters static objects)")
    print(f"   • Increase --min-detections 2 or 3 (requires stable detection)")
    print(f"   • Adjust --motion-threshold (lower = more sensitive)")
    print(f"   • Add false positive images to known_faces/ as 'background'")
    print(f"\n🎯 To improve recognition accuracy:")
    print(f"   • Add 3-4 images per person to known_faces/")
    print(f"   • Run: python3 diagnose_images.py to test images")
    print(f"   • Adjust --tolerance (default: 0.6, lower = stricter)")
    
    print(f"\n{'='*70}\n")

def main():
    analyze_captures()

if __name__ == "__main__":
    main()
