import os
import glob
import subprocess
import csv
import zipfile
from PIL import Image

def verify_scene(scene_dir, renders_dir):
    """
    Verifies that the renders for a scene match the test_poses.csv requirements.
    """
    test_csv = os.path.join(scene_dir, "test", "test_poses.csv")
    if not os.path.exists(test_csv):
        print(f"  [FAIL] test_poses.csv not found in {scene_dir}")
        return False

    expected_poses = []
    with open(test_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected_poses.append({
                'image_name': row['image_name'],
                'width': int(row['width']),
                'height': int(row['height'])
            })

    rendered_files = sorted(glob.glob(os.path.join(renders_dir, "*.png")))
    if len(rendered_files) != len(expected_poses):
        print(f"  [FAIL] Image count mismatch! Expected {len(expected_poses)}, found {len(rendered_files)}")
        return False

    # Check dimensions
    for i, expected in enumerate(expected_poses):
        render_path = rendered_files[i]
        # In a real scenario, you'd check that the filenames match or correspond in order
        with Image.open(render_path) as img:
            w, h = img.size
            if w != expected['width'] or h != expected['height']:
                print(f"  [FAIL] Dimension mismatch on {os.path.basename(render_path)}. Expected {expected['width']}x{expected['height']}, got {w}x{h}")
                return False

    print(f"  [PASS] Verified {len(expected_poses)} images for {os.path.basename(scene_dir)}")
    return True

def main():
    data_dir = "data"
    output_root = "output"
    submission_zip = "submission.zip"
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found.")
        return

    scenes = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found {len(scenes)} scenes to process: {[os.path.basename(s) for s in scenes]}")

    all_passed = True
    valid_scenes = []

    for scene in scenes:
        scene_name = os.path.basename(scene)
        print(f"\n================ Processing {scene_name} ================")
        
        scene_out_dir = os.path.join(output_root, scene_name)
        renders_dir = os.path.join(scene_out_dir, "renders")
        
        # 1. Run Training
        print(f"-> Training {scene_name}...")
        train_cmd = [
            "python", "gaussian-splatting/train.py",
            "-s", scene,
            "-m", scene_out_dir,
            "--iterations", "1000"
        ]
        subprocess.run(train_cmd)

        # 2. Render test poses
        print(f"-> Rendering test poses for {scene_name}...")
        render_cmd = [
            "python", "gaussian-splatting/render_test_poses.py",
            "-m", scene_out_dir,
            "--test_poses_csv", os.path.join(scene, "test", "test_poses.csv"),
            "--output_dir", renders_dir
        ]
        subprocess.run(render_cmd)

        # For the sake of local testing, we'll simulate a failure by checking the non-existent renders
        # On Kaggle, this will actually check the real renders
        print(f"-> Verifying {scene_name}...")
        passed = verify_scene(scene, renders_dir)
        if passed:
            valid_scenes.append((scene_name, renders_dir))
        else:
            all_passed = False

    if not all_passed:
        print("\n[WARNING] Not all scenes passed verification. Please fix the failed scenes before submission.")
        # We don't zip if verification fails to enforce the rule, but on Kaggle you might want to zip whatever passed
    else:
        print(f"\n================ Zipping Submission ================")
        with zipfile.ZipFile(submission_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for scene_name, renders_dir in valid_scenes:
                images = sorted(glob.glob(os.path.join(renders_dir, "*.png")))
                for i, img_path in enumerate(images):
                    arcname = f"scene_{scene_name}/{i:04d}.png"
                    zipf.write(img_path, arcname)
                    print(f"  Added {arcname}")
        print(f"\n[SUCCESS] submission.zip created with {len(valid_scenes)} scenes.")

if __name__ == "__main__":
    main()
