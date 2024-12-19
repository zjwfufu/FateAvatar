import glob
import os
import time
import torch
import torchvision
import tqdm
import imageio
import numpy as np

from rich.console import Console

from tools.util         import EasyDict, load_to_gpu

from train.deserialize  import (
    deserialize_checkpoints_fateavatar,
    deserialize_checkpoints_flashavatar,
    deserialize_checkpoints_gaussianavatars,
    deserialize_checkpoints_monogaussianavatar,
    deserialize_checkpoints_splattingavatar
)

from typing import Type
from model import ModelClass

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------------------- #

class Loader(object):
    def __init__(
            self,
            name:              str,
            cfg:               EasyDict,
            model:             Type[ModelClass],
            device:            torch.device,
            workspace:         str='workspace',
            use_checkpoint:    str='scratch',
            max_keep_ckpt:     int=2
        ):
        
        self.name           = name
        self.cfg            = cfg
        self.model          = model
        self.device         = device
        self.workspace      = workspace
        self.use_checkpoint = use_checkpoint
        self.max_keep_ckpt  = max_keep_ckpt

        self.model_name     = model.__class__.__name__

        self.console    = Console()
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        # variable init
        self.epoch          = 0
        self.global_step    = 0
        self.local_step     = 0

        self.deserialize_callbacks_hooks = {
            'FateAvatar':           deserialize_checkpoints_fateavatar,
            'FlashAvatar':          deserialize_checkpoints_flashavatar,
            'GaussianAvatars':      deserialize_checkpoints_gaussianavatars,
            'MonoGaussianAvatar':   deserialize_checkpoints_monogaussianavatar,
            'SplattingAvatar':      deserialize_checkpoints_splattingavatar  
        }

        self.register_workspace()

        self.load_checkpoint_process()

    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        # load model
        self.deserialize_callbacks_hooks[self.model_name](self, checkpoint_dict)

        try:
            self.train_expression = checkpoint_dict['train_expression']['weight']
            self.train_flame_pose = checkpoint_dict['train_flame_pose']['weight']
            self.train_cam_pose = checkpoint_dict['train_cam_pose']['weight']
        except Exception as e:
            print("[INFO] Fail to load optimized train set tracking", e)

        try:
            self.test_expression = checkpoint_dict['test_expression']['weight']
            self.test_flame_pose = checkpoint_dict['test_flame_pose']['weight']
            self.test_cam_pose = checkpoint_dict['test_cam_pose']['weight']
        except Exception as e:
            print("[INFO] Fail to load optimized test set tracking:", e)

        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

    def log(self, *args, **kwargs):
        self.console.print(*args, **kwargs)
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()  # write immediately to file

    def log_file_only(self, *args, **kwargs):
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()  # write immediately to file

    def load_checkpoint_process(self):
        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def register_workspace(self):
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(self.workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

# ------------------------------------------------------------------------------- #

class Reenactor(Loader):
    def __init__(
            self,
            name:              str,
            cfg:               EasyDict,
            model:             Type[ModelClass],
            device:            torch.device,
            workspace:         str='workspace',
            use_checkpoint:    str='scratch',
            max_keep_ckpt:     int=2
        ):
        super().__init__(name, cfg, model, device, workspace, use_checkpoint, max_keep_ckpt)

        self.register_media_save()

    def register_media_save(self):
        
        self.media_save_path = {
            "eval_render": {
                "folder": os.path.join(self.workspace, "reenact", "eval_render")
            },
            "video": {
                "folder": os.path.join(self.workspace, "reenact", "video")
            }
        }
        
        for _, value in self.media_save_path.items():
            folder_path = value["folder"]
            os.makedirs(folder_path, exist_ok=True)

    def reenacting(self, dst_name, dst_loader, delta_exp):
        self.log(f'++> Reenact to {dst_name} ...')

        self.model.eval()

        total_len = len(dst_loader)

        pbar = tqdm.tqdm(total=total_len, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            join_video_frames       = []
            render_video_frames     = []
            save_path_join_videos   = os.path.join(self.media_save_path["video"]["folder"], 'video', f'epoch_{self.epoch}_join_render_{dst_name}.mp4')
            save_path_videos        = os.path.join(self.media_save_path["video"]["folder"], 'video', f'epoch_{self.epoch}_render_{dst_name}.mp4')
            os.makedirs(os.path.dirname(save_path_videos), exist_ok=True)

            for idx, (_, input_data, ground_truth) in enumerate(dst_loader):

                frame_idx = input_data['idx'].item()

                input_data['expression'] += delta_exp

                load_to_gpu(input_data, ground_truth, self.device)

                self.local_step += 1

                output_data     = self.model(input_data)
                render_image    = output_data['rgb_image']
                gt_image        = ground_truth['rgb']

                # ------------------------ save video frame ------------------------ #
                render_np   = render_image[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                gt_np       = gt_image[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                video_frame = np.concatenate((render_np, gt_np), axis=1)

                render_video_frames.append((render_np * 255.).astype('uint8'))
                join_video_frames.append((video_frame * 255.).astype('uint8'))

                # ------------------------ save pre frame render result ------------------------ #
                if self.local_step % 100 == 0:
                    save_render_path = os.path.join(self.media_save_path["eval_render"]["folder"], dst_name, f"{frame_idx:04d}.png")
                    os.makedirs(os.path.dirname(save_render_path), exist_ok=True)
                    render_plot = render_image[0].detach().cpu()
                    torchvision.utils.save_image(render_plot, save_render_path, normalize=True, value_range=(0, 1))

                pbar.update(1)

        all_render_np = np.stack(render_video_frames, axis=0)
        all_join_up   = np.stack(join_video_frames, axis=0)
        imageio.mimwrite(save_path_videos, all_render_np, fps=25)
        imageio.mimwrite(save_path_join_videos, all_join_up, fps=25)

        pbar.close()
        self.log_file_only(pbar)

        self.log(f"++> Reenact to {dst_name} Finished.")