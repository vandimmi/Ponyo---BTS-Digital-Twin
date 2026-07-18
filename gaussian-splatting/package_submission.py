import os
import zipfile
import shutil

def package_submission():
    renders_dir = "renders"
    output_zip = "submission.zip"
    
    if not os.path.exists(renders_dir):
        print(f"Error: {renders_dir} not found. Render the test poses first.")
        return

    print("Creating submission.zip...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # We need to package it as scene_xxx/000x.png
        # Let's say the scene is HCM0421
        scene_name = "scene_HCM0421"
        for i, f in enumerate(sorted(os.listdir(renders_dir))):
            if f.endswith('.png') or f.endswith('.jpg'):
                # Rename to 0000.png, 0001.png etc.
                arcname = f"{scene_name}/{i:04d}.png"
                file_path = os.path.join(renders_dir, f)
                zipf.write(file_path, arcname)
                print(f"Added {file_path} -> {arcname}")
                
    print(f"Submission packaged into {output_zip}")

if __name__ == "__main__":
    package_submission()
