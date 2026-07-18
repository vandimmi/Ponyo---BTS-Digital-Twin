import os
import sys
import csv

def main():
    sys.path.append("gaussian-splatting")
    from scene.dataset_readers import readColmapSceneInfo

    scene_path = r"data\HCM0421"
    print(f"--- Testing readColmapSceneInfo on {scene_path} ---")
    
    try:
        scene_info = readColmapSceneInfo(scene_path, images=None, depths="", eval=False, train_test_exp=False)
        print(f"Successfully loaded scene!")
        print(f"Number of train cameras: {len(scene_info.train_cameras)}")
        print(f"Number of test cameras: {len(scene_info.test_cameras)}")
        if scene_info.point_cloud:
            print(f"Number of 3D points: {scene_info.point_cloud.points.shape[0]}")
        else:
            print("No point cloud loaded.")
    except Exception as e:
        print(f"Failed to load scene using readColmapSceneInfo: {e}")
        import traceback
        traceback.print_exc()

    # Also parse test_poses.csv manually to prove we can read it
    test_poses_path = os.path.join(scene_path, "test", "test_poses.csv")
    print(f"\n--- Parsing first few lines of {test_poses_path} ---")
    try:
        with open(test_poses_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"Header: {header}")
            for i, row in enumerate(reader):
                if i >= 3:
                    break
                print(f"Row {i+1}: {row}")
    except Exception as e:
        print(f"Error reading test_poses.csv: {e}")

if __name__ == '__main__':
    main()
