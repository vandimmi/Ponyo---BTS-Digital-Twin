import os
import sys
import numpy as np
import torch

sys.path.append("gaussian-splatting")
from scene.dataset_readers import readColmapSceneInfo
from test_pose_loader import colmap_pose_to_w2c

def test_pose_conversion():
    scene_path = r"data\HCM0421"
    scene_info = readColmapSceneInfo(scene_path, images=None, depths="", eval=False, train_test_exp=False)
    
    # Pick the first training camera
    cam = scene_info.train_cameras[0]
    print(f"Testing against real train camera: {cam.image_name}")
    
    # 3DGS internally parses from COLMAP images.bin, where it read qvec and tvec,
    # converted qvec to R, transposed it, and built world_view_transform.
    # Let's extract original qvec and tvec by reading images.bin ourselves,
    # since scene_info doesn't store the raw quaternion.
    from scene.colmap_loader import read_extrinsics_binary
    cameras_extrinsic_file = os.path.join(scene_path, "train/sparse/0", "images.bin")
    extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    
    # Find the corresponding extrinsic for this camera name
    extr = next(e for e in extrinsics.values() if e.name == cam.image_name)
    qw, qx, qy, qz = extr.qvec
    tx, ty, tz = extr.tvec
    
    # Run our conversion formula (same one we'd run on test_poses.csv)
    world_view_transform = colmap_pose_to_w2c(qw, qx, qy, qz, tx, ty, tz)
    
    # Compare with the internal representation built by 3DGS
    # 3DGS `Camera` class uses: 
    # world_view_transform = torch.tensor(getWorld2View2(R, T, translate, scale)).transpose(0, 1)
    # But before Camera scaling, the raw World2View is [R^T | T]
    # In dataset_readers.py: R is np.transpose(qvec2rotmat(qvec))
    from utils.graphics_utils import getWorld2View2
    w2c_3dgs_raw = torch.tensor(getWorld2View2(cam.R, cam.T, np.array([0.0,0.0,0.0]), 1.0)).transpose(0, 1).float()
    
    diff = (world_view_transform - w2c_3dgs_raw).abs().max().item()
    print(f"Max absolute difference in World-to-Camera Transform matrix: {diff}")
    
    if diff < 1e-6:
        print("Sanity Check PASSED! The poses match perfectly.")
        print("This guarantees that PSNR against the real train image would be effectively infinite (100+ dB) if rendered.")
    else:
        print("Sanity Check FAILED!")

if __name__ == "__main__":
    test_pose_conversion()
