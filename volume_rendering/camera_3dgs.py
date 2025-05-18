#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from tools.gs_utils.graphics_utils import (
    getWorld2View2, getWorld2View2_torch,
    getProjectionMatrix, getProjectionMatrixShift
)

#-------------------------------------------------------------------------------#

class Camera(nn.Module):
    def __init__(
            self, R, T, FoVx, FoVy, img_res, intrinsics = None,
            trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
        ):
        super(Camera, self).__init__()

        # self.R = R.squeeze(0).cpu().numpy()
        # self.T = T.squeeze(0).cpu().numpy()
        self.R = R.squeeze(0)
        self.T = T.squeeze(0)
        self.FoVx = FoVx
        self.FoVy = FoVy

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = img_res[1]
        self.image_height = img_res[0]

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, trans, scale)).transpose(0, 1).cuda()
        self.world_view_transform = getWorld2View2_torch(self.R, self.T).transpose(0, 1).cuda()

        # in case non-zero principle point offset
        if intrinsics is None:
            self.projection_matrix = getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            ).transpose(0,1).cuda()
        else:
            self.intrinsics = intrinsics.squeeze(0)
            focal_x = self.intrinsics[0, 0]
            focal_y = self.intrinsics[1, 1]
            cx = self.intrinsics[0, 2]
            cy = self.intrinsics[1, 2]
            self.projection_matrix = getProjectionMatrixShift(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy,
                focal_x=focal_x, focal_y=focal_y, cx=cx, cy=cy,
                width=self.image_width, height=self.image_height
            ).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

#-------------------------------------------------------------------------------#

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]