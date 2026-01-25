#!/usr/bin/env python3
"""
Helper script to set up known faces directory with multiple images per person
"""

import os
import sys

KNOWN_FACES_DIR = "known_faces"

def create_structure():
    """Create example directory structure with instructions"""
    
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"✓ Created directory: {KNOWN_FACES_DIR}/")
    else:
        print(f"✓ Directory exists: {KNOWN_FACES_DIR}/")
    
    print("\n" + "="*60)
    print("KNOWN FACES SETUP GUIDE")
    print("="*60)
    
    print(f"\n📁 Add images to: {os.path.abspath(KNOWN_FACES_DIR)}/")
    
    print("\n📝 Naming Convention:")
    print("   For multiple images per person, use:")
    print("   • person_name_1.jpg")
    print("   • person_name_2.jpg")
    print("   • person_name_3.jpg")
    print("   • person_name_4.jpg")
    
    print("\n💡 Examples:")
    print("   known_faces/")
    print("   ├── john_doe_1.jpg      ← John facing front")
    print("   ├── john_doe_2.jpg      ← John slight angle")
    print("   ├── john_doe_3.jpg      ← John with glasses")
    print("   ├── jane_smith_1.jpg    ← Jane smiling")
    print("   ├── jane_smith_2.jpg    ← Jane neutral")
    print("   ├── jane_smith_3.jpg    ← Jane different lighting")
    print("   └── raghuveer_paturi_1.jpg")
    
    print("\n✅ Best Practices:")
    print("   • Use 3-4 images per person for accuracy")
    print("   • Vary angles slightly (front, slight left/right)")
    print("   • Include with/without accessories (glasses, hat)")
    print("   • Different lighting conditions")
    print("   • Clear, well-lit faces")
    print("   • One face per image")
    print("   • Images should be .jpg, .jpeg, or .png")
    
    print("\n📸 Image Quality Tips:")
    print("   • Resolution: At least 300x300 pixels")
    print("   • Face should be clearly visible")
    print("   • Good lighting (avoid shadows)")
    print("   • Face looking towards camera")
    print("   • Avoid blurry images")
    
    print("\n🚀 Next Steps:")
    print(f"   1. Add your images to {KNOWN_FACES_DIR}/ directory")
    print("   2. Name them correctly (person_name_1.jpg, etc.)")
    print("   3. Run: python3 face_detection.py --recognize")
    print("   4. The system will load all images and show statistics")
    
    print("\n" + "="*60)
    
    # Check current contents
    files = [f for f in os.listdir(KNOWN_FACES_DIR) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if files:
        print(f"\n📊 Current contents ({len(files)} images):")
        
        # Group by person
        from collections import defaultdict
        import re
        
        person_files = defaultdict(list)
        for f in sorted(files):
            base_name = os.path.splitext(f)[0]
            # Remove trailing numbers
            person = re.sub(r'_\d+$', '', base_name).replace('_', ' ').title()
            person_files[person].append(f)
        
        for person, images in sorted(person_files.items()):
            print(f"\n   {person} ({len(images)} images):")
            for img in images:
                size = os.path.getsize(os.path.join(KNOWN_FACES_DIR, img))
                size_kb = size / 1024
                print(f"      • {img} ({size_kb:.1f} KB)")
    else:
        print(f"\n⚠️  No images found in {KNOWN_FACES_DIR}/")
        print("   Add images and run this script again to verify")
    
    print("\n" + "="*60)

def main():
    create_structure()

if __name__ == "__main__":
    main()
