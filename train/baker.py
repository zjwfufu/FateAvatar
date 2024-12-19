import  os
import  cv2
import  glob
import  imageio
import  torch
import  torchvision
import  torchvision._utils
import  tqdm
import  json
import  numpy as np

from    PIL import Image
from    torch           import device
from    torchvision     import transforms

from    train.trainer   import Trainer, BaseTrainer
from    train.dataset   import FaceDataset
from    train.loss      import BaseLoss

from    model.uv_decoder   import UVDecoder
from    model.fateavatar   import FateAvatar

from    tools.gs_utils.sh_utils import C0
from    tools.util import EasyDict
from    tools.util import (load_to_gpu, save_image_grid)

from    torchvision.transforms.functional import vflip


# ------------------------------------------------------------------------------- #

class UVBaker(Trainer):
    def __init__(
        self,
        name:           str,
        cfg:            EasyDict,
        model:          UVDecoder,
        avatar:         FateAvatar,
        device:         device,
        train_dataset:  FaceDataset,
        test_dataset:   FaceDataset,
        criterions:     BaseLoss,
        metrics:        list = ...,
        workspace:      str = 'workspace',
        use_checkpoint: str = 'scratch',
        use_full_head_resume: bool = False,
        max_keep_ckpt:  int = 2
    ):
        self.use_full_head_resume = use_full_head_resume
        
        super().__init__(name, cfg, model, device, train_dataset, test_dataset, criterions, metrics, workspace, use_checkpoint, max_keep_ckpt)

        self.avatar_model   = avatar

        self.register_media_save()

    def register_media_save(self):

        baking_dir = "baking_full_head" if self.use_full_head_resume else "baking" 
        
        self.media_save_path = {
            "dump_texture": {
                "folder": os.path.join(self.workspace, baking_dir, "dump_texture"),
            },
            "visualize_texture": {
                "folder": os.path.join(self.workspace, baking_dir, "visualize_texture"),
            },
            "train_snapshot": {
                "folder": os.path.join(self.workspace, baking_dir, "train_snapshot"),
                "interval": 1000,
                "interval_fullhead": 50
            },
            "eval_snapshot": {
                "folder": os.path.join(self.workspace, baking_dir, "eval_snapshot"),
                "interval": 50
            },
            "checkpoints_baked": {
                "folder": os.path.join(self.workspace, baking_dir, "checkpoints_baked"),
            },
            "eval_render": {
                "folder": os.path.join(self.workspace, baking_dir, "eval_render")
            },
            "train_metric": {
                "folder": os.path.join(self.workspace, baking_dir, "train_metric")
            },
            "eval_metric": {
                "folder": os.path.join(self.workspace, baking_dir, "eval_metric")
            },
            "video": {
                "folder": os.path.join(self.workspace, baking_dir, "video")
            },
            "inverse_transform":  {
                "folder": os.path.join(self.workspace, "augmentation", "paste_back")
            },
        }

    def register_optimizer_group(self):

        nn_l = [
            {'params': list(self.model.decoder.parameters()), 'lr': 0.001}
        ]
        
        nn_optimizer = torch.optim.Adam(nn_l)

        self.optimizers_group.update({'nn': nn_optimizer})

        return BaseTrainer.register_optimizer_group(self)
    
    def bake(self, max_epochs):
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch  = epoch

            if self.use_full_head_resume:
                self.bake_full_head_epoch()
            else:
                self.bake_epoch()

            self.save_checkpoint()

    def bake_epoch(self):
        self.log(f"==> Start Training Epoch {self.epoch} ...")

        self.model.train()

        pbar = tqdm.tqdm(total=len(self.train_loader), bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        metrics_per_frame = {}
        for metric in self.metrics:
            metrics_per_frame[metric.name] = {}

        for idx, (_, input_data, ground_truth) in enumerate(self.train_loader):

            frame_idx = input_data['idx'].item()

            load_to_gpu(input_data, ground_truth, self.device)

            self.local_step += 1
            self.global_step += 1

            #------------------------ override tracking result if we optimize tracking ------------------------#
            if self.cfg.optimize_tracking:
                input_data["expression"]            = self.train_expression(input_data["idx"]).squeeze(1)
                input_data["flame_pose"]            = self.train_flame_pose(input_data["idx"]).squeeze(1)
                input_data["cam_pose"][:, :3, 3]    = self.train_cam_pose(input_data["idx"]).squeeze(1)

            #------------------------ execute training ------------------------#
            output_dict         = self.bake_step(input_data, ground_truth)
            loss_output_dict    = output_dict['loss_output']
            render_image        = output_dict['render_image']
            gt_image            = output_dict['gt_image']

            #------------------------ save snapshot ------------------------#
            if (self.global_step % self.media_save_path["train_snapshot"]["interval"]) == 0 or self.global_step == 1:
                save_path = os.path.join(self.media_save_path["train_snapshot"]["folder"], f'train_step_{self.global_step:06d}.png')
                self.save_full_snap_shot(input_data, ground_truth, save_path)
                self.texture_visualize(output_dict['texture_dict'], pre_fix=f'{self.global_step:06d}')

            #------------------------ gather loss ------------------------#
            for each_loss_name, each_loss_monitor in self.loss_monitor.items():
                each_loss_monitor.update(loss_output_dict[each_loss_name].item())

            #------------------------ gather metrics ------------------------#
            with torch.no_grad():
                for metric in self.metrics:
                    metric.update(render_image, gt_image)
                    # log current to dict for logging
                    metrics_per_frame[metric.name][frame_idx] = metric.V_cur

            pbar.set_description(f"loss={loss_output_dict['loss'].item():.4f} ({self.loss_meter.avg:.4f})")
            pbar.update(1)

        self.texture_dump(output_dict['texture_dict'], pre_fix=f'{self.epoch:03d}')

        self.export_avatar_model()

        pbar.close()
        self.log_file_only(pbar)

        #------------------------ gather metrics of one epoch ------------------------#
        self.save_one_epoch_metrics(metrics_per_frame, self.metrics, self.media_save_path["train_metric"]["folder"])

        self.log(f"==> Metrics @ Epoch {self.epoch}.")
        for metric in self.metrics:
            self.log(metric.report(), style="red")
            metric.clear()
        self.log(f"==> Loss @ Epoch {self.epoch}.")
        for each_loss_name, each_loss_monitor in self.loss_monitor.items():
            self.log(each_loss_monitor.report(), style="red")
            each_loss_monitor.clear()

    def bake_full_head_epoch(self):

        import re
        import copy
        from train.dataset  import   load_rgb, load_mask
        from tools.util     import   get_bg_color

        def natural_sort_key(s):
            return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

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

        start_index = len(pseudo_name_list) // 4
        end_index = len(pseudo_name_list) * 3 // 4

        pseudo_name_list    = pseudo_name_list[start_index:end_index]
        pseudo_image_list   = pseudo_image_list[start_index:end_index]
        pseudo_mask_list    = pseudo_mask_list[start_index:end_index]
        num_pseudo          = len(pseudo_name_list)

        input_data_pseudo      = {}
        ground_truth_pseudo    = {}

        input_data_pseudo['expression']     = self.model.avatar_model.canonical_expression
        input_data_pseudo['flame_pose']     = self.model.avatar_model.flame.canonical_pose
        input_data_pseudo['fovx']           = [self.cfg.camera_fovx]
        input_data_pseudo['fovy']           = [self.cfg.camera_fovy]

        input_data_pseudo['cam_pose']       = torch.cat((self.cfg.camera_rotation, self.cfg.camera_translation[..., None]), dim=1)
        input_data_pseudo['cam_pose'][0, 3] = 0;   input_data_pseudo['cam_pose'][1, 3] = 0    # erase translation in x, y
        input_data_pseudo['cam_pose']       = input_data_pseudo['cam_pose'][None, ...]

        pbar = tqdm.tqdm(total=len(self.train_loader) * 2, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for idx, (_, input_data, ground_truth) in enumerate(self.train_loader):

            self.local_step += 1
            self.global_step += 1

            # ---------------------------------------- #
            # ---------- train pseudo data ----------- #
            # ---------------------------------------- #
            pseudo_idx      = np.random.randint(0, num_pseudo)
            pseudo_epoch    = idx // num_pseudo

            pseudo_name     = pseudo_name_list[pseudo_idx]
            pseudo_cam_pose = torch.tensor(traj[pseudo_name]).to(self.device)

            bg_color        = get_bg_color('random')
            self.model.bg_color = bg_color.to(self.device)

            # input_data_pseudo['cam_pose'][:, :3, :3] = pseudo_cam_pose[:3, :3]
            input_data_pseudo['cam_pose'][:, :3, :4] = pseudo_cam_pose[:3, :4]

            # # overwrite expression
            input_data_pseudo["expression"] = copy.deepcopy(input_data["expression"])

            rgb = load_rgb(
                os.path.join(pseudo_image_dir, pseudo_image_list[pseudo_idx]),
                self.train_dataset.img_res,
                bg_color = self.cfg.bg_color
            )
            mask = load_mask(
                os.path.join(pseudo_mask_dir, pseudo_mask_list[pseudo_idx]),
                self.train_dataset.img_res,
            )

            rgb = rgb * mask[None, ...] + (1 - mask[None, ...]) * bg_color.numpy()[:, None, None]
            ground_truth_pseudo['rgb'] = torch.from_numpy(rgb[None, ...]).float()

            load_to_gpu(input_data_pseudo, ground_truth_pseudo, self.device)

            # ---------------------- execute training --------------------- #
            output_dict_pseudo         = self.bake_step(input_data_pseudo, ground_truth_pseudo)
            loss_output_dict_pseudo    = output_dict_pseudo['loss_output']
            render_image_pseudo        = output_dict_pseudo['render_image']
            gt_image_pseudo            = output_dict_pseudo['gt_image']

            # ---------------------- save snapshot ----------------------- #
            if pseudo_epoch % self.media_save_path["train_snapshot"]["interval_fullhead"] == 0 and idx % num_pseudo == 0:
                save_path = os.path.join(self.media_save_path["train_snapshot"]["folder"], f'training_full_head_step_{(self.epoch + 1) * pseudo_epoch}.png')
                self.save_full_snap_shot(input_data_pseudo, ground_truth_pseudo, save_path)

            # ----------------------- gather loss ------------------------ #
            for each_loss_name, each_loss_monitor in self.loss_monitor.items():
                each_loss_monitor.update(loss_output_dict_pseudo[each_loss_name].item())

            # ----------------------- gather metric ------------------------ #
            for metric in self.metrics:
                metric.update(render_image_pseudo, gt_image_pseudo)

            pbar.update(1)

            self.local_step += 1
            self.global_step += 1

            # ---------------------------------------- #
            # ---------- train regular data ---------- #
            # ---------------------------------------- #
            load_to_gpu(input_data, ground_truth, self.device)

            self.local_step += 1
            self.global_step += 1

            if self.cfg.optimize_tracking:
                input_data["expression"]            = self.train_expression(input_data["idx"]).squeeze(1)
                input_data["flame_pose"]            = self.train_flame_pose(input_data["idx"]).squeeze(1)
                input_data["cam_pose"][:, :3, 3]    = self.train_cam_pose(input_data["idx"]).squeeze(1)

            bg_color = get_bg_color('white')
            self.model.bg_color = bg_color.to(self.device)

            # ------------------------ execute training ------------------------ #
            rgb_weight = self.criterions.Params.rgb_weight
            self.criterions.Params.rgb_weight = 0.0
            output_dict = self.bake_step(input_data, ground_truth)
            loss_output_dict    =   output_dict['loss_output']
            render_image        =   output_dict['render_image']
            gt_image            =   output_dict['gt_image']
            self.criterions.Params.rgb_weight = rgb_weight

            # ------------------------ save snapshot ------------------------ #
            if (self.local_step % self.media_save_path["train_snapshot"]["interval"] == 0):
                save_path = os.path.join(self.media_save_path["train_snapshot"]["folder"], f'training_step_{self.local_step}.png')
                self.save_full_snap_shot(input_data, ground_truth, save_path)
                self.texture_visualize(output_dict['texture_dict'], pre_fix=f'{self.global_step:06d}')

            # ----------------------- gather loss ------------------------ #
            for each_loss_name, each_loss_monitor in self.loss_monitor.items():
                each_loss_monitor.update(loss_output_dict[each_loss_name].item())

            # ----------------------- gather metric ------------------------ #
            for metric in self.metrics:
                metric.update(render_image, gt_image)

            pbar.set_description(f"loss={loss_output_dict['loss'].item():.4f} ({self.loss_meter.avg:.4f})")
            pbar.update(1)

        self.texture_dump(output_dict['texture_dict'], pre_fix=f'{self.epoch:03d}')

        self.export_avatar_model()

        pbar.close()
        self.log_file_only(pbar)

        self.log(f"==> Metrics @ Epoch {self.epoch}")
        for metric in self.metrics:
            self.log(metric.report(), style="red")
            metric.clear()
        self.log(f"==> Loss @ Epoch {self.epoch}")
        for each_loss_name, each_loss_monitor in self.loss_monitor.items():
            self.log(each_loss_monitor.report(), style="red")
            each_loss_monitor.clear()

    def bake_step(self, input_data, ground_truth):

        output_data  = self.model(input_data)
        render_image = output_data['rgb_image']
        gt_image     = ground_truth['rgb']

        loss_output  = self.criterions(output_data, ground_truth)

        loss = loss_output['loss']

        bs                  = output_data['bs']
        viewspace_points    = output_data['viewspace_points']
        visibility_filter   = output_data['visibility_filter']
        radii               = output_data['radii']

        texture_dict        = output_data['texture_dict']

        #------------------------ zero grad ------------------------#
        for name, optimizer in self.optimizers_group.items():
            optimizer.zero_grad(set_to_none=True)

        loss.backward()

        #------------------------ optimize ------------------------#
        for name, optimizer in self.optimizers_group.items():
            optimizer.step()

        return {'loss_output':  loss_output,
                'render_image': render_image,
                'gt_image':     gt_image,
                'texture_dict': texture_dict}
    
    def evalulate_step(self, input_data, ground_truth):
        
        output_data  = self.model(input_data)
        render_image = output_data['rgb_image']
        gt_image     = ground_truth['rgb']
        loss_output = self.criterions(output_data, ground_truth)

        return {'loss_output': loss_output,
                'render_image': render_image,
                'gt_image': gt_image}
    
    def texture_dump(self, texture_dict: dict, pre_fix: str=None):

        save_path = os.path.join(
            self.media_save_path["dump_texture"]["folder"],
            f'{pre_fix}_tex_dump.pth' if pre_fix else f'tex_dump.pth'
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(texture_dict, save_path)

    def texture_visualize(self, texture_dict: dict, pre_fix: str=None):

        for name, tex in texture_dict.items():

            if name == 'color':
                act_tex = self.model._color_activation(tex)
                _min = -1.78
                _max = 1.78

            elif name == 'opacity':
                act_tex = torch.sigmoid(tex)
                _min = 0
                _max = 1

            elif name == 'offset':
                act_tex = self.model._offset_activation(tex)
                _min = -1
                _max = 1

            else:
                act_tex = tex
                _min = act_tex.min().item()
                _max = act_tex.max().item()
            
            save_path = os.path.join(self.media_save_path["visualize_texture"]["folder"], f'{pre_fix}_{name}.png' if pre_fix else f'{name}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image_grid(vflip(act_tex).detach().cpu().numpy(), save_path, drange=[_min, _max], grid_size=(1, 1))
    
    def export_avatar_model(self, remove_old=True, name=None):
        
        baked_avatar = self.model._export_avatar_model()

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
        }

        state['model'] = baked_avatar.state_dict()

        state['train_expression'] = self.train_expression.state_dict()
        state['train_flame_pose'] = self.train_flame_pose.state_dict()
        state['train_cam_pose']   = self.train_cam_pose.state_dict()

        state['test_expression']  = self.test_expression.state_dict()
        state['test_flame_pose']  = self.test_flame_pose.state_dict()
        state['test_cam_pose']    = self.test_cam_pose.state_dict()

        file_base = os.path.join(self.media_save_path["checkpoints_baked"]["folder"], name)
        os.makedirs(os.path.dirname(file_base), exist_ok=True)

        if remove_old:
            self.ckpt_stats.append(file_base)

            if len(self.ckpt_stats) > self.max_keep_ckpt:
                old_file = self.ckpt_stats.pop(0)
                if os.path.exists(f'{old_file}.pth'):
                    os.remove(f'{old_file}.pth')
        
        torch.save(state, f"{file_base}.pth")

    def save_checkpoint(self, name=None, remove_old=True, save_path=None):
        pass

    def load_checkpoint(self, checkpoint=None):
        pass

# ------------------------------------------------------------------------------- #

class UVEditor(Trainer):
    def __init__(
        self,
        name:           str,
        cfg:            EasyDict,
        model:          UVDecoder,
        avatar:         FateAvatar,
        device:         device,
        train_dataset:  FaceDataset,
        test_dataset:   FaceDataset,
        criterions:     BaseLoss,
        metrics:        list = ...,
        workspace:      str = 'workspace',
        use_checkpoint: str = 'scratch',
        use_full_head_resume: bool = False,
        max_keep_ckpt:  int = 2
    ):
        self.use_full_head_resume = use_full_head_resume
        
        super().__init__(name, cfg, model, device, train_dataset, test_dataset, criterions, metrics, workspace, use_checkpoint, max_keep_ckpt)

        self.avatar_model   = avatar

        self.register_media_save()

    def register_optimizer_group(self):
        pass

    def load_checkpoint_process(self):
        pass

    def register_media_save(self):

        baking_dir = "baking_full_head" if self.use_full_head_resume else "baking" 

        self.media_save_path = {
            "dump_texture": {
                "folder": os.path.join(self.workspace, baking_dir, "dump_texture"),
            },
            "visualize_texture": {
                "folder": os.path.join(self.workspace, baking_dir, "edit", "visualize_edited_texture"),
            },
            "checkpoints_edited": {
                "folder": os.path.join(self.workspace, baking_dir, "edit", "checkpoints_edited"),
            },
            "eval_render": {
                "folder": os.path.join(self.workspace, baking_dir, "edit", "eval_render"),
                "interval": 50
            },
            "video": {
                "folder": os.path.join(self.workspace, baking_dir, "edit", "video")
            },
            "edit_assets": {
                "sticker_assets": "./edit_assets/sticker_assets",
                "style_transfer": "./edit_assets/style_transfer",
                "uv_mask"       :  "./edit_assets/uv_mask"
            }
        }

    def texture_load(self, path):
        texture_dict = torch.load(path)
        self.model.texture_dict_cache = texture_dict
        return texture_dict
    
    def run_animation(self, texture_dict = None, save_name = 'edit_animation'):
        self.log(f"++> Run animation ...")

        self.model.eval()

        pbar = tqdm.tqdm(total=len(self.test_loader), bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            render_video_frames = []
            save_path_videos = os.path.join(self.media_save_path["video"]["folder"], f'{save_name}.mp4')
            os.makedirs(os.path.dirname(save_path_videos), exist_ok=True)

            for idx, (_, input_data, ground_truth) in enumerate(self.test_loader):

                frame_idx = input_data['idx'].item()

                load_to_gpu(input_data, ground_truth, self.device)

                self.local_step += 1

                # ------------------------ eval step ------------------------ #
                output_data  = self.model.render_from_texture_dict(input_data, texture_dict)
                render_image = output_data['rgb_image']

                # ------------------------ save video frame ------------------------ #
                render_np   = render_image[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                render_video_frames.append((render_np * 255.).astype('uint8'))

                # ------------------------ save pre frame render result ------------------------ #
                if frame_idx % self.media_save_path["eval_render"]["interval"] == 0:
                    save_render_path = os.path.join(self.media_save_path["eval_render"]["folder"], f'{save_name}', f'{frame_idx:04d}.png')
                    os.makedirs(os.path.dirname(save_render_path), exist_ok=True)
                    render_plot = render_image[0].detach().cpu()
                    torchvision.utils.save_image(render_plot, save_render_path, normalize=True, value_range=(0, 1))

                pbar.update(1)

        all_render_np = np.stack(render_video_frames, axis=0)
        imageio.mimwrite(save_path_videos, all_render_np, fps=25)

        pbar.close()
        self.log_file_only(pbar)

        self.log(f"++> Animated Done.")

    def export_avatar_model(self, texture_dict, name):
        
        baked_avatar = self.model._export_avatar_model(texture_dict)

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
        }

        state['model'] = baked_avatar.state_dict()

        state['train_expression'] = self.train_expression.state_dict()
        state['train_flame_pose'] = self.train_flame_pose.state_dict()
        state['train_cam_pose']   = self.train_cam_pose.state_dict()

        state['test_expression']  = self.test_expression.state_dict()
        state['test_flame_pose']  = self.test_flame_pose.state_dict()
        state['test_cam_pose']    = self.test_cam_pose.state_dict()

        file_base = os.path.join(self.media_save_path["checkpoints_edited"]["folder"], name)
        os.makedirs(os.path.dirname(file_base), exist_ok=True)
        
        torch.save(state, f"{file_base}.pth")

    def sticker_editing(self, sticker_name: str):

        texture_dict_list = sorted(glob.glob(f'{self.media_save_path["dump_texture"]["folder"]}/*.pth'))
        texture_dict_path = texture_dict_list[-1]

        texture_dict = self.texture_load(texture_dict_path)

        # active color for editing
        texture_dict['color'] = UVDecoder._color_activation(texture_dict['color'])

        self.apply_sticker_editing(
            texture_dict,
            content_path  = os.path.join(self.media_save_path["edit_assets"]["sticker_assets"], f"{sticker_name}_content.png"),
            mask_path     = os.path.join(self.media_save_path["edit_assets"]["sticker_assets"], f"{sticker_name}_mask.png"),
            save_tex_path = os.path.join(self.media_save_path["visualize_texture"]["folder"], f"{sticker_name}.png")
        )

        self.run_animation(
            texture_dict = texture_dict,
            save_name    = sticker_name
        )

        self.export_avatar_model(
            texture_dict = texture_dict,
            name    = sticker_name
        )

    @staticmethod
    def apply_sticker_editing(
            texture_dict:   dict,
            content_path:   str,
            mask_path:      str,
            save_tex_path:  str
        ):
        
        device = texture_dict['color'].device
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        content         = Image.open(content_path).convert("RGB")
        mask            = Image.open(mask_path).convert("L")

        content_tensor  = transform(content)[None, ...]
        mask_tensor     = transform(mask)[None, ...]

        content_tensor = (2 * content_tensor - 1) * (0.5 / C0)  # rescale to [-1.78, 1.78]

        tatoo_dict = {
            'content': content_tensor.to(device),
            'mask':    mask_tensor.to(device)
        }

        content = tatoo_dict['content']
        mask    = tatoo_dict['mask']

        texture_dict['color'] = mask * content + (1 - mask) * texture_dict['color']

        os.makedirs(os.path.dirname(save_tex_path), exist_ok=True)
        save_image_grid(vflip(texture_dict['color']).detach().cpu().numpy(), save_tex_path, drange=[-1.78, 1.78], grid_size=(1, 1))

    def style_transfer(self, transfer_mdoel: str):

        texture_dict_list = sorted(glob.glob(f'{self.media_save_path["dump_texture"]["folder"]}/*.pth'))
        texture_dict_path = texture_dict_list[-1]

        texture_dict = self.texture_load(texture_dict_path)

        # active color for editing
        texture_dict['color'] = UVDecoder._color_activation(texture_dict['color'])

        self.apply_style_transfer(
            texture_dict,
            model_path      = os.path.join(self.media_save_path["edit_assets"]["style_transfer"], f"{transfer_mdoel}.t7"),
            save_tex_path   = os.path.join(self.media_save_path["edit_assets"]["style_transfer"], f"{transfer_mdoel}.png")
        )

        self.run_animation(
            texture_dict = texture_dict,
            save_name    = transfer_mdoel
        )

        self.export_avatar_model(
            texture_dict = texture_dict,
            name    = transfer_mdoel
        )

    @staticmethod
    def apply_style_transfer(
            texture_dict:   dict,
            model_path:     str,
            save_tex_path:  str
        ):
        
        device = texture_dict['color'].device

        color_tex = texture_dict['color'] * (C0 / 0.5)
        color_tex = (color_tex + 1) / 2
        
        color_np = color_tex.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        color_np = color_np * 255
        color_np = color_np.astype(np.uint8)

        net = cv2.dnn.readNetFromTorch(model_path)

        (h, w) = color_np.shape[:2]
        blob = cv2.dnn.blobFromImage(color_np, 1.0, (w, h),
                                    (103.939, 116.779, 123.680), swapRB=False, crop=False)

        net.setInput(blob)
        output = net.forward()

        output = output.reshape((3, output.shape[2], output.shape[3]))
        output = output.transpose(1, 2, 0)
        output += np.array([103.939, 116.779, 123.680])
        output = np.clip(output, 0, 255).astype("uint8")

        output_tensor = torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0).float() / 255
        output_tensor = ((output_tensor - 0.5) * 2) * (0.5 / C0)

        texture_dict['color'] = output_tensor.to(device)

        os.makedirs(os.path.dirname(save_tex_path), exist_ok=True)
        save_image_grid(vflip(texture_dict['color']).detach().cpu().numpy(), save_tex_path, drange=[-1.78, 1.78], grid_size=(1, 1))


# ------------------------------------------------------------------------------- #




                


                






        

        







            


    
