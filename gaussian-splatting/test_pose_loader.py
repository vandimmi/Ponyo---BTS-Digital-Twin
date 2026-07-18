import os
import sys
import torch
import numpy as np
import math
from utils.graphics_utils import focal2fov

def colmap_pose_to_w2c(qw, qx, qy, qz, tx, ty, tz):
    """
    Convert a COLMAP test pose (which represents World-to-Camera transform)
    to a PyTorch tensor format expected by 3DGS.
    """
    # Quaternion to rotation matrix (World-to-Camera)
    # COLMAP convention: (qw, qx, qy, qz)
    R = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    
    T = np.array([tx, ty, tz])
    
    # In 3DGS, the final world_view_transform is the transpose of the 4x4 Rt matrix
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    
    world_view_transform = torch.tensor(Rt).float().transpose(0, 1)
    
    return world_view_transform

def get_projection_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class TestCamera(torch.nn.Module):
    def __init__(self, uid, R, T, FovX, FovY, image_name, width, height):
        super(TestCamera, self).__init__()
        self.uid = uid
        self.image_name = image_name
        self.image_width = width
        self.image_height = height
        
        # We need world_view_transform and full_proj_transform
        # R is expected transposed
        self.world_view_transform = torch.tensor(np.column_stack((R, T))).float().cuda()
        self.world_view_transform = torch.cat([self.world_view_transform, torch.tensor([[0,0,0,1]]).float().cuda()], dim=0)
        
        self.projection_matrix = get_projection_matrix(znear=0.01, zfar=100.0, fovX=FovX, fovY=FovY).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
