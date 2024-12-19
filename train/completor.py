import  copy
import  json
import  math
import  os
import  re

import  cv2
import  imageio
import  numpy as np
import  torch
import  torchvision
import  tqdm

from    torch import device

from train.trainer      import Trainer

from tools.util import (
        EasyDict,
        load_to_gpu,
    )
from tools.eg3d_utils.camera_eg3d       import LookAtPoseSampler

from typing         import Type
from model          import ModelClass
from train.loss     import LossClass
from train.dataset  import DatasetClass

# ------------------------------------------------------------------------------- #

class CompletionTrainer(Trainer):
    def __init__(
        self,
        name:           str,
        cfg:            EasyDict,
        model:          Type[ModelClass],
        device:         device,
        train_dataset:  Type[DatasetClass],
        test_dataset:   Type[DatasetClass],
        criterions:     Type[LossClass],
        metrics:        list = ...,
        workspace:      str = 'workspace',
        use_checkpoint: str = 'scratch',
        max_keep_ckpt:  int = 2
    ):
        
        super().__init__(name, cfg, model, device, train_dataset, test_dataset, criterions, metrics, workspace, use_checkpoint, max_keep_ckpt)

        # override
        self.register_media_save()

    def register_media_save(self):
        
        self.media_save_path = {
            "train_snapshot_pseudo": {
                "folder": os.path.join(self.workspace, "completion", "train_snapshot_pseudo"),
                "interval": 1000
            },
            "train_snapshot_regular": {
                "folder": os.path.join(self.workspace, "completion", "train_snapshot_regular"),
                "interval": 1000
            },
            "train_metric": {
                "folder": os.path.join(self.workspace, "completion", "train_metrics")
            },
            "eval_metric": {
                "folder": os.path.join(self.workspace, "completion", "eval_metric")
            },
            "eval_snapshot": {
                "folder": os.path.join(self.workspace, "completion", "eval_snapshot"),
                "interval": 50
            },
            "eval_render": {
                "folder": os.path.join(self.workspace, "completion", "eval_render")
            },
            "video": {
                "folder": os.path.join(self.workspace, "completion", "video")
            },
            "inverse_transform":  {
                "folder": os.path.join(self.workspace, "augmentation", "paste_back")
            },
        }

        for _, value in self.media_save_path.items():
            folder_path = value["folder"]
            os.makedirs(folder_path, exist_ok=True)
    
    def augmentation(self, finetune_epoch = 5):

        from train.dataset  import   load_rgb, load_mask
        from tools.util     import   get_bg_color

        def natural_sort_key(s):
            return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

        self.log("++> Run Augmentation Framework")

        self.model.train()

        pseudo_dir          = self.media_save_path["inverse_transform"]["folder"]

        if not os.path.exists(pseudo_dir):
            raise FileNotFoundError(f"The path '{pseudo_dir}' does not exist, run 'generate_pseudo.py' first.")

        pseudo_image_dir    = os.path.join(pseudo_dir, 'image')
        pseudo_mask_dir     = os.path.join(pseudo_dir, 'mask_modnet')

        pseudo_image_list = sorted(os.listdir(pseudo_image_dir), key=natural_sort_key)
        pseudo_mask_list  = sorted(os.listdir(pseudo_mask_dir),  key=natural_sort_key)

        pseudo_name_list  = [filename.split('.')[0] for filename in pseudo_image_list]

        with open(os.path.join(pseudo_dir, 'trajectory.json'), 'r') as f:
            traj = json.load(f)

        # start_index = len(pseudo_name_list) // 3
        # end_index = len(pseudo_name_list) * 2 // 3

        start_index = len(pseudo_name_list) // 4
        end_index = len(pseudo_name_list) * 3 // 4

        # start_index = len(pseudo_name_list) // 6
        # end_index = len(pseudo_name_list) * 5 // 6

        # start_index = len(pseudo_name_list) // 8
        # end_index = len(pseudo_name_list) * 7 // 8

        pseudo_name_list    = pseudo_name_list[start_index:end_index]
        pseudo_image_list   = pseudo_image_list[start_index:end_index]
        pseudo_mask_list    = pseudo_mask_list[start_index:end_index]
        num_pseudo          = len(pseudo_name_list)

        input_data_pseudo      = {}
        ground_truth_pseudo    = {}

        input_data_pseudo['expression']     = self.model.canonical_expression
        input_data_pseudo['flame_pose']     = self.model.flame.canonical_pose
        input_data_pseudo['fovx']           = [self.cfg.camera_fovx]
        input_data_pseudo['fovy']           = [self.cfg.camera_fovy]

        input_data_pseudo['cam_pose']       = torch.cat((self.cfg.camera_rotation, self.cfg.camera_translation[..., None]), dim=1)
        input_data_pseudo['cam_pose'][0, 3] = 0;   input_data_pseudo['cam_pose'][1, 3] = 0    # erase translation in x, y
        input_data_pseudo['cam_pose']       = input_data_pseudo['cam_pose'][None, ...]

        torsor_mask = load_mask(
            os.path.join(pseudo_dir, 'torsor_boundary.png'),
            self.train_dataset.img_res
        )

        max_epoch = finetune_epoch
        self.local_step = 0

        if self.model_name == 'FateAvatar':
            self.model.add_default_points(self.optimizers_group['gs'])

        for epoch in range(0, max_epoch):

            pbar = tqdm.tqdm(total=len(self.train_loader) * 2, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

            for idx, (_, input_data, ground_truth) in enumerate(self.train_loader):

                self.local_step += 1
                self.global_step += 1

                # ---------- train pseudo data ---------- #
                pseudo_idx = np.random.randint(0, num_pseudo)
                pseudo_epoch = idx // num_pseudo

                pseudo_name = pseudo_name_list[pseudo_idx]
                pseudo_cam_pose = torch.tensor(traj[pseudo_name]).to(self.device)

                bg_color = get_bg_color('random')
                self.model.bg_color = bg_color.to(self.device)

                # input_data_pseudo['cam_pose'][:, :3, :3] = pseudo_cam_pose[:3, :3]
                input_data_pseudo['cam_pose'][:, :3, :4] = pseudo_cam_pose[:3, :4]

                # # overwrite expression
                input_data_pseudo["expression"] = copy.deepcopy(input_data["expression"])

                rgb = load_rgb(
                    os.path.join(pseudo_image_dir, pseudo_image_list[pseudo_idx]),
                    self.train_dataset.img_res,
                    bg_color = self.cfg.bg_color    # dummy here, GAN output does not contain alpha channel
                )

                mask = load_mask(
                    os.path.join(pseudo_mask_dir, pseudo_mask_list[pseudo_idx]),
                    self.train_dataset.img_res,
                )

                # kernel = np.ones((3, 3), np.uint8)
                # dilate_mask = cv2.dilate(mask, kernel, iterations=3)

                rgb = rgb * mask[None, ...] + (1 - mask[None, ...]) * bg_color.numpy()[:, None, None]

                ground_truth_pseudo['rgb'] = torch.from_numpy(rgb[None, ...]).float()
                # ground_truth_pseudo['torsor_mask'] = torch.from_numpy(torsor_mask[None, ...]).float()

                load_to_gpu(input_data_pseudo, ground_truth_pseudo, self.device)

                if self.model_name == 'GaussianAvatars' or self.model_name == 'SplattingAvatar':
                    rgb_weight = self.criterions.rgb_weight
                    self.criterions.rgb_weight = 0.0

                output_dict_pseudo         = self.train_step(input_data_pseudo, ground_truth_pseudo)
                loss_output_dict_pseudo    = output_dict_pseudo['loss_output']
                render_image_pseudo        = output_dict_pseudo['render_image']
                gt_image_pseudo            = output_dict_pseudo['gt_image']

                if self.model_name == 'GaussianAvatars' or self.model_name == 'SplattingAvatar':
                    self.criterions.rgb_weight = rgb_weight

                if pseudo_epoch % 10 == 0 and idx % num_pseudo == 0:
                    save_path = os.path.join(self.media_save_path["train_snapshot_pseudo"]["folder"], f'training_step_{(epoch + 1) * pseudo_epoch}.png')
                    self.save_full_snap_shot(input_data_pseudo, ground_truth_pseudo, save_path)

                pbar.update(1)

                self.local_step += 1
                self.global_step += 1

                # ---------- train regular data ---------- #
                load_to_gpu(input_data, ground_truth, self.device)

                self.local_step += 1
                self.global_step += 1

                if self.cfg.optimize_tracking:
                    input_data["expression"]            = self.train_expression(input_data["idx"]).squeeze(1)
                    input_data["flame_pose"]            = self.train_flame_pose(input_data["idx"]).squeeze(1)
                    input_data["cam_pose"][:, :3, 3]    = self.train_cam_pose(input_data["idx"]).squeeze(1)

                bg_color = get_bg_color('white')
                self.model.bg_color = bg_color.to(self.device)

                output_dict = self.train_step(input_data, ground_truth)
                loss_output_dict    =   output_dict['loss_output']
                render_image        =   output_dict['render_image']
                gt_image            =   output_dict['gt_image']

                if (self.local_step % 1000 == 0):
                    save_path = os.path.join(self.media_save_path["train_snapshot_regular"]["folder"], f'training_step_{self.local_step}.png')
                    self.save_full_snap_shot(input_data, ground_truth, save_path)

                for each_loss_name, each_loss_monitor in self.loss_monitor.items():
                    each_loss_monitor.update(loss_output_dict[each_loss_name].item())

                for metric in self.metrics:
                    metric.update(render_image, gt_image)

                pbar.set_description(f"loss={loss_output_dict['loss'].item():.4f} ({self.loss_meter.avg:.4f})")
                pbar.update(1)

            pbar.close()
            self.log_file_only(pbar)

            self.log(f"==> Metrics @ Epoch {epoch}")
            for metric in self.metrics:
                self.log(metric.report(), style="red")
                metric.clear()
            self.log(f"==> Loss @ Epoch {epoch}")
            for each_loss_name, each_loss_monitor in self.loss_monitor.items():
                self.log(each_loss_monitor.report(), style="red")
                each_loss_monitor.clear()



    def render_dynamic_novel_view(self, name=None, mode=None):

        self.log(f"++> Render dynamic novel view...")

        self.model.eval()

        pbar = tqdm.tqdm(total=len(self.test_loader), bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            
            render_video_frames = []

            save_path_videos = os.path.join(self.media_save_path["video"]["folder"], f'epoch_{self.epoch}_{name if name else "dynamic_novel_view"}.mp4')
            os.makedirs(os.path.dirname(save_path_videos), exist_ok=True)

            rounds = 4

            for idx, (_, input_data, ground_truth) in enumerate(self.test_loader):

                frame_idx = input_data['idx'].item()
 
                cam2world_pose  = LookAtPoseSampler.sample(
                    math.pi / 2 + 2 * math.pi * idx / (len(self.test_loader)) * rounds,
                    math.pi / 2,
                    torch.tensor([0, 0, 0], device=self.device),
                    radius  = 2.75,   # not used in here
                    device  = self.device
                )

                world2cam = torch.linalg.inv(cam2world_pose)
                
                # R = cam2world_pose[:, :3, :3]
                # R[:, 1] = R[:, 1] * -1
                # R[:, 2] = R[:, 2] * -1
                # rot_vec = matrix_to_axis_angle(R)

                # overwrite flame root rotation
                # rot neck jaw left-eye right-eye
                input_data["flame_pose"][0, 3:6] = 0
                # input_data["flame_pose"][0, :3] = rot_vec



                # set translation to zero
                input_data["cam_pose"][:, :2, 3] = input_data["cam_pose"][:, :2, 3] * 0

                # set camera pose
                input_data["cam_pose"][:, :3, :3] = world2cam[:, :3, :3]

                # set camera pose
                # input_data["cam_pose"][:, 0, 0] = 1; input_data["cam_pose"][:, 0, 1] = 0;  input_data["cam_pose"][:, 0, 2] = 0
                # input_data["cam_pose"][:, 1, 0] = 0; input_data["cam_pose"][:, 1, 1] = -1; input_data["cam_pose"][:, 1, 2] = 0
                # input_data["cam_pose"][:, 2, 0] = 0; input_data["cam_pose"][:, 2, 1] = 0;  input_data["cam_pose"][:, 2, 2] = -1

                load_to_gpu(input_data, ground_truth, self.device)

                self.local_step += 1

                output_dict = self.model(input_data)
                render_image = output_dict['rgb_image']

                # save video frame
                render_np = render_image[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                render_video_frames.append((render_np * 255.).astype('uint8'))

                # save pre frame
                if self.local_step % 100 == 0:
                    save_render_path = os.path.join(
                        self.media_save_path["eval_render"]["folder"],
                        f'{name if name else "dynamic_novel_view"}',
                        f'{frame_idx:04d}.png'
                    )
                    os.makedirs(os.path.dirname(save_render_path), exist_ok=True)
                    render_plot = render_image[0].detach().cpu()
                    torchvision.utils.save_image(render_plot, save_render_path, normalize=True, value_range=(0, 1))

                pbar.update(1)

            all_render_np = np.stack(render_video_frames, axis=0)
            imageio.mimwrite(save_path_videos, all_render_np, fps=25)

            pbar.close()
            self.log_file_only(pbar)
     

    def render_dynamic_fixed_view(self, name=None, mode=None):

        self.log(f"++> Render fixed novel view...")

        self.model.eval()

        pbar = tqdm.tqdm(total=len(self.test_loader), bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            
            render_videos = [[] for _ in range(6)]

            camera_positions = [
                1 / 6,
                2 / 6,
                3 / 6,
                4 / 6,
                5 / 6,
                1.0
            ]

            save_path_videos = os.path.join(self.media_save_path["video"]["folder"], f'epoch_{self.epoch}_{name if name else "fixed_views"}')
            os.makedirs(save_path_videos, exist_ok=True)

            for idx, (_, input_data, ground_truth) in enumerate(self.test_loader):
                frame_idx = input_data['idx'].item()

                for view_idx, (ratio) in enumerate(camera_positions):

                    cam2world_pose  = LookAtPoseSampler.sample(
                        math.pi / 2 + 2 * math.pi * ratio,
                        math.pi / 2,
                        torch.tensor([0, 0, 0], device=self.device),
                        radius  = 2.75,   # not used in here
                        device  = self.device
                    )

                    world2cam = torch.linalg.inv(cam2world_pose)

                    input_data["flame_pose"][0, 3:6] = 0  # Zero out flame pose
                    input_data["cam_pose"][:, :3, :3] = world2cam[:, :3, :3]
                    input_data["cam_pose"][:, :2, 3] = input_data["cam_pose"][:, :2, 3] * 0  # Set translation to zero

                    load_to_gpu(input_data, ground_truth, self.device)

                    self.local_step += 1

                    output_dict = self.model(input_data)
                    render_image = output_dict['rgb_image']

                    render_np = render_image[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                    render_videos[view_idx].append((render_np * 255.).astype('uint8'))

                    if frame_idx % 100 == 0:
                        save_render_path = os.path.join(self.media_save_path["eval_render"]["folder"], 
                                                        f'view_{view_idx}', f'{name if name else "dynamic_fixed_view"}', f'{frame_idx:04d}.png')
                        os.makedirs(os.path.dirname(save_render_path), exist_ok=True)
                        render_plot = render_image[0].detach().cpu()
                        torchvision.utils.save_image(render_plot, save_render_path, normalize=True, value_range=(0, 1))

                pbar.update(1)

            for view_idx, frames in enumerate(render_videos):
                save_path_video = os.path.join(save_path_videos, f'view_{view_idx}.mp4')
                all_render_np = np.stack(frames, axis=0)
                imageio.mimwrite(save_path_video, all_render_np, fps=25)

            pbar.close()
            self.log_file_only(pbar)






                


                






        

        







            


    
