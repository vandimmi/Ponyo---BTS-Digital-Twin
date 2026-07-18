import os
import zipfile
import csv
import sys
import tempfile
import shutil
from PIL import Image

def verify_submission(zip_path="submission.zip", data_dir="data"):
    if not os.path.exists(zip_path):
        print(f"[ERROR] Submission file {zip_path} not found!")
        sys.exit(1)

    print(f"Unzipping {zip_path} for verification...")
    temp_dir = tempfile.mkdtemp()
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        scenes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        all_passed = True
        
        for scene in scenes:
            print(f"\nVerifying scene: {scene}...")
            scene_data_dir = os.path.join(data_dir, scene)
            test_csv = os.path.join(scene_data_dir, "test", "test_poses.csv")
            
            if not os.path.exists(test_csv):
                print(f"  [ERROR] {test_csv} not found in dataset! Skipping.")
                all_passed = False
                continue
                
            expected_poses = []
            with open(test_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    expected_poses.append({
                        'width': int(row['width']),
                        'height': int(row['height'])
                    })
            
            scene_zip_folder = os.path.join(temp_dir, f"scene_{scene}")
            if not os.path.exists(scene_zip_folder):
                print(f"  [FAIL] Expected folder 'scene_{scene}' not found in zip!")
                all_passed = False
                continue
                
            extracted_images = sorted([f for f in os.listdir(scene_zip_folder) if f.endswith('.png')])
            
            if len(extracted_images) != len(expected_poses):
                print(f"  [FAIL] Image count mismatch! Expected {len(expected_poses)}, found {len(extracted_images)} in zip.")
                all_passed = False
                continue
                
            scene_passed = True
            for i, expected in enumerate(expected_poses):
                expected_filename = f"{i:04d}.png"
                img_path = os.path.join(scene_zip_folder, expected_filename)
                
                if not os.path.exists(img_path):
                    print(f"  [FAIL] Missing image: {expected_filename}")
                    scene_passed = False
                    break
                    
                with Image.open(img_path) as img:
                    w, h = img.size
                    if w != expected['width'] or h != expected['height']:
                        print(f"  [FAIL] Dimension mismatch on {expected_filename}. Expected {expected['width']}x{expected['height']}, got {w}x{h}")
                        scene_passed = False
                        break
                        
            if scene_passed:
                print(f"  [PASS] {scene}: {len(expected_poses)} images correctly formatted and sized.")
            else:
                all_passed = False
                
        if all_passed:
            print("\n[SUCCESS] ALL SCENES PASSED VERIFICATION! Ready to submit.")
            sys.exit(0)
        else:
            print("\n[ERROR] One or more scenes failed verification. DO NOT SUBMIT.")
            sys.exit(1)
            
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    verify_submission()
