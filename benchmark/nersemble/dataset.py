import math
import torch
import torch.nn as nn

from typing import List, Literal

from train.dataset import FaceDataset

from nersemble_benchmark.data.benchmark_data import MonoFlameAvatarDataManager
from nersemble_benchmark.data.benchmark_data import FlameTracking

class NersembleBenchmarkDataset(FaceDataset):
    def __init__(
            self,
            root_path:  str,
            participant_id: int,
            serial: str,
            sequence_list: List,
            mode: Literal['train', 'test'] = 'train'
        ):

        self.optimize_tracking      = False

        width = 512
        height = 512

        self.serial = serial
        self.mode = mode
        
        mono_flame_data_manager = MonoFlameAvatarDataManager(root_path, participant_id)
        self.data_manager: MonoFlameAvatarDataManager = mono_flame_data_manager

        camera_calibration = mono_flame_data_manager.load_camera_calibration()
        intrinsics = camera_calibration.intrinsics[serial]
        world_2_cam_pose = camera_calibration.world_2_cam[serial]

        fov_x = 2 * math.atan(width / (2 * intrinsics[0, 0]))
        fov_y = 2 * math.atan(height / (2 * intrinsics[1, 1]))

        self.fov_x = fov_x
        self.fov_y = fov_y
        self.intrinsics = intrinsics
        self.world_2_cam_pose = world_2_cam_pose

        self.index_list = []
        self.tracking_data = {}

        for sequence_name in sequence_list:
            flame_tracking = mono_flame_data_manager.load_flame_tracking(sequence_name)
            self.tracking_data[sequence_name] = flame_tracking
            T = flame_tracking.frames.shape[0]
            for t in range(T):
                self.index_list.append((sequence_name, t))

    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, idx):
        
        seq_name, t = self.index_list[idx]

        tracking: FlameTracking = self.tracking_data[seq_name]

        world_2_cam_pose = torch.Tensor(self.world_2_cam_pose.tolist())

        cam_R = world_2_cam_pose[:3, :3]
        cam_t = world_2_cam_pose[:3, 3]

        cam_R_new = torch.linalg.inv(cam_R)
        cam_t_new = cam_t

        world_2_cam_pose_new = torch.eye(4)
        world_2_cam_pose_new[:3, :3] = cam_R_new
        world_2_cam_pose_new[:3, 3] = cam_t_new

        sample = {
            "idx": torch.LongTensor([idx]),
            # camera
            "intrinsics": torch.Tensor(self.intrinsics.tolist()),
            "cam_pose": world_2_cam_pose_new,
            "fovx": self.fov_x,
            "fovy": self.fov_y,
            # flame
            "shape": torch.from_numpy(tracking.shape[0]),
            "expression": torch.from_numpy(tracking.expression[t]),
            "rotation": torch.from_numpy(tracking.rotation[t]),
            "rotation_matrix": torch.from_numpy(tracking.rotation_matrices[t]),
            "translation": torch.from_numpy(tracking.translation[t]),
            "jaw": torch.from_numpy(tracking.jaw[t]),
            "scale": torch.from_numpy(tracking.scale[0]),
            "neck": torch.from_numpy(tracking.neck[t]),
            "eyes": torch.from_numpy(tracking.eyes[t]),
        }

        if self.mode == 'train':
            image = self.data_manager.load_image(seq_name, self.serial, t)
            alpha = self.data_manager.load_alpha_map(seq_name, self.serial, t)

            image = torch.from_numpy(image).permute(2, 0, 1).float()
            alpha = torch.from_numpy(alpha).permute(2, 0, 1).float()

            ground_truth = {
                'rgb': image,
                'object_mask': alpha
            }
        else:
            ground_truth = {}

        return idx, sample, ground_truth

    
                





        