import os
import torch
import csv
from scene import Scene
from gaussian_renderer import render
from argparse import ArgumentParser
from utils.system_utils import mkdir_p
from torchvision.utils import save_image
from test_pose_loader import TestCamera, focal2fov

def main():
    parser = ArgumentParser(description="Render test poses from test_poses.csv")
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("--test_poses_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="renders")
    args = parser.parse_args()

    # Load checkpoint
    # (Assuming we have GaussianModel instantiated; simplified for demonstration)
    from scene.gaussian_model import GaussianModel
    gaussians = GaussianModel(sh_degree=3)
    
    # Normally we load the scene/checkpoint here
    print(f"Loading checkpoint from {args.model_path}")
    # gaussians.load_ply(...)
    
    mkdir_p(args.output_dir)
    
    # Read CSV
    with open(args.test_poses_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height = row
            
            # Convert COLMAP qvec/tvec to R, T
            # R should be World-to-Camera, transposed for 3DGS
            qw, qx, qy, qz = map(float, (qw, qx, qy, qz))
            tx, ty, tz = map(float, (tx, ty, tz))
            
            R = np.array([
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
            ])
            Rt = np.transpose(R)
            T = np.array([tx, ty, tz])
            
            FovY = focal2fov(float(fy), float(height))
            FovX = focal2fov(float(fx), float(width))
            
            cam = TestCamera(uid=0, R=Rt, T=T, FovX=FovX, FovY=FovY, 
                             image_name=image_name, width=int(width), height=int(height))
            
            # rendering call
            print(f"Rendering {image_name}")
            # render_pkg = render(cam, gaussians, pipeline_args, background)
            # image = render_pkg["render"]
            # save_image(image, os.path.join(args.output_dir, image_name))

if __name__ == "__main__":
    main()
