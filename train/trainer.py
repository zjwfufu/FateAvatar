import  copy
import  glob
import  json
import  math
import  os
import  re
import  time

import  cv2
import  imageio
import  matplotlib.pyplot as plt
import  numpy as np
import  torch
import  torchvision
import  tqdm

from    torch import device
from    torchvision.transforms.functional import vflip
from    pytorch3d.io import save_obj
from    pytorch3d.transforms import matrix_to_axis_angle

from train.base         import BaseTrainer
from train.optim        import (
        register_optimizer_group_flashavatar,
        register_optimizer_gruop_gaussianavatars,
        register_optimizer_group_monogaussianavatar,
        register_optimizer_group_fateavatar,
        register_optimizer_group_splattingavatar,
    )
from train.iteration    import (
        iteration_step_flashavatar,
        iteration_step_gaussianavatars,
        iteration_step_monogaussianavatar,
        iteration_step_fateavatar,
        iteration_step_splattingavatar,
    )
from train.deserialize  import (
        deserialize_checkpoints_flashavatar,
        deserialize_checkpoints_gaussianavatars,
        deserialize_checkpoints_monogaussianavatar,
        deserialize_checkpoints_fateavatar,
        deserialize_checkpoints_splattingavatar,
    )

from tools.util import (
        EasyDict,
        colorize_weights_map,
        load_to_gpu,
        measure_fps,
    )
from tools.gs_utils.general_utils       import get_expon_lr_func
from tools.eg3d_utils.camera_eg3d       import LookAtPoseSampler

from typing         import Type
from model          import ModelClass
from train.loss     import LossClass
from train.dataset  import DatasetClass

# ------------------------------------------------------------------------------- #

class Trainer(BaseTrainer):
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
        
        self.model_name     = model.__class__.__name__

        self.optimizer_callbacks_hooks = {
            'FateAvatar':           register_optimizer_group_fateavatar,
            'FlashAvatar':          register_optimizer_group_flashavatar,
            'GaussianAvatars':      register_optimizer_gruop_gaussianavatars,
            'MonoGaussianAvatar':   register_optimizer_group_monogaussianavatar,
            'SplattingAvatar':      register_optimizer_group_splattingavatar
        }

        self.iteration_callbacks_hooks = {
            'FateAvatar':           iteration_step_fateavatar,
            'FlashAvatar':          iteration_step_flashavatar,
            'GaussianAvatars':      iteration_step_gaussianavatars,
            'MonoGaussianAvatar':   iteration_step_monogaussianavatar,
            'SplattingAvatar':      iteration_step_splattingavatar  
        }

        self.deserialize_callbacks_hooks = {
            'FateAvatar':           deserialize_checkpoints_fateavatar,
            'FlashAvatar':          deserialize_checkpoints_flashavatar,
            'GaussianAvatars':      deserialize_checkpoints_gaussianavatars,
            'MonoGaussianAvatar':   deserialize_checkpoints_monogaussianavatar,
            'SplattingAvatar':      deserialize_checkpoints_splattingavatar  
        }

        super().__init__(name, cfg, model, device, train_dataset, test_dataset, criterions, metrics, workspace, use_checkpoint, max_keep_ckpt)

        self.register_media_save()

    def register_media_save(self):
        
        self.media_save_path = {
            "train_snapshot": {
                "folder": os.path.join(self.workspace, "monocular", "train_snapshot"),
                "interval": 1000
            },
            "eval_snapshot": {
                "folder": os.path.join(self.workspace, "monocular", "eval_snapshot"),
                "interval": 50
            },
            "train_metric": {
                "folder": os.path.join(self.workspace, "monocular", "train_metric")
            },
            "eval_metric": {
                "folder": os.path.join(self.workspace, "monocular", "eval_metric")
            },
            "eval_render": {
                "folder": os.path.join(self.workspace, "monocular", "eval_render")
            },
            "video": {
                "folder": os.path.join(self.workspace, "monocular", "video")
            }
        }

        for _, value in self.media_save_path.items():
            folder_path = value["folder"]
            os.makedirs(folder_path, exist_ok=True)

    def register_optimizer_group(self):

        self.optimizers_group   = self.optimizer_callbacks_hooks[self.model_name](self.model, self.cfg)

        if self.model_name == 'GaussianAvatars':
            self.xyz_scheduler_args = get_expon_lr_func(
                                                lr_init         = self.cfg.training.position_lr_init,
                                                lr_final        = self.cfg.training.position_lr_final,
                                                lr_delay_mult   = self.cfg.training.position_lr_delay_mult,
                                                max_steps       = self.cfg.training.position_lr_max_steps
                                            )

        return super().register_optimizer_group()

    def train_epoch(self):
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
            output_dict         = self.train_step(input_data, ground_truth)
            loss_output_dict    = output_dict['loss_output']
            render_image        = output_dict['render_image']
            gt_image            = output_dict['gt_image']

            #------------------------ save snapshot ------------------------#
            if (self.global_step % self.media_save_path["train_snapshot"]["interval"]) == 0 or self.global_step == 1:
                save_path = os.path.join(self.media_save_path["train_snapshot"]["folder"], f'train_step_{self.global_step:06d}.png')
                self.save_full_snap_shot(input_data, ground_truth, save_path)

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

        # 'MonoGaussianAvatar' retain upsampling and prune via 'epoch'
        if self.model_name == 'MonoGaussianAvatar':

            # pruning
            if self.epoch != 0 and self.epoch % self.cfg.training.upsample_freq == 0:
                self.model.pc.prune(self.model.visible_points)
                self.optimizers_group['nn'] = torch.optim.Adam([
                    {'params': list(self.model.parameters())}
                ],  lr=self.cfg.training.lr)

            # upsampling
            if self.epoch % self.cfg.training.upsample_freq == 0:
                old_radius  = self.model.radius
                old_points  = self.model.pc.points.data.shape[0]
                self.model._upsample_points(self.epoch)
                if self.epoch >= 100:
                    self.log("old radius: {}, new radius: {}, sample radius: {}".format(old_radius, self.model.radius, 0.004))
                else:
                    self.log("old radius: {}, new radius: {}, sample radius: {}".format(old_radius, self.model.radius, old_radius))
                self.log("old points: {}, new points: {}".format(old_points, self.model.pc.points.data.shape[0]))
                self.optimizers_group['nn'] = torch.optim.Adam([
                        {'params': list(self.model.parameters())}
                    ],  lr=self.cfg.training.lr)
            # re-init visible point
            self.model.visible_points = torch.zeros(self.model.pc.points.shape[0]).bool().to(self.device)  
            
        elif self.model_name in ['FateAvatar', 'SplattingAvatar', 'GaussianAvatars']:
            self.log("cur points: {}".format(self.model.num_points))

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

    def train_step(self, input_data, ground_truth):

        kwargs = {}

        if self.model_name == 'GaussianAvatars':
            kwargs['xyz_scheduler_args'] = self.xyz_scheduler_args
        
        output_dict = self.iteration_callbacks_hooks[self.model_name](
            input_data,
            ground_truth,
            self.model,
            self.criterions,
            self.optimizers_group,
            self.cfg,
            self.global_step,
            self.epoch,
            self.log,
            **kwargs,
        )

        return output_dict
    

    def evaluate_epoch(self, mode, optim_epoch=None, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        for metric in self.metrics:
            metric.clear()

        # in imavatar dataset, we shall optimize FLAME coefficients before evaluation.
        if self.cfg.optimize_tracking:
            # usually, we will optimize testset flame after training is done
            # which 'mode' is set to 'train'
            if mode == 'train':
                self.optimize_tracking(optim_epoch=optim_epoch)
            # when 'mode' is set to 'eval', we already load finetuned flame
            # so we dont need optimize tracking again.
            elif mode =='eval':
                self.log("++> Tracking in evaluation has been optimized.")

            elif mode == 'train_full_head':
                self.optimize_tracking(optim_epoch=50)

        self.model.eval()

        pbar = tqdm.tqdm(total=len(self.test_loader), bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        metrics_per_frame = {}
        for metric in self.metrics:
            metrics_per_frame[metric.name] = {}

        with torch.no_grad():
            self.local_step = 0

            join_video_frames       = []
            render_video_frames     = []
            save_path_join_videos   = os.path.join(self.media_save_path["video"]["folder"], f'epoch_{self.epoch}_join_render{"_" + name if name else ""}.mp4')
            save_path_videos        = os.path.join(self.media_save_path["video"]["folder"], f'epoch_{self.epoch}_render{"_" + name if name else ""}.mp4')
            os.makedirs(os.path.dirname(save_path_videos), exist_ok=True)

            for idx, (_, input_data, ground_truth) in enumerate(self.test_loader):

                frame_idx = input_data['idx'].item()

                load_to_gpu(input_data, ground_truth, self.device)

                self.local_step += 1

                # ------------------------ override tracking ------------------------ #
                if self.cfg.optimize_tracking:
                    input_data["expression"]            = self.test_expression(input_data["idx"]).squeeze(1)
                    input_data["flame_pose"]            = self.test_flame_pose(input_data["idx"]).squeeze(1)
                    input_data["cam_pose"][:, :3, 3]    = self.test_cam_pose(input_data["idx"]).squeeze(1)

                # ------------------------ eval step ------------------------ #
                output_dict         = self.evalulate_step(input_data, ground_truth)
                loss_output_dict    = output_dict['loss_output']
                render_image        = output_dict['render_image']
                gt_image            = output_dict['gt_image']

                # ------------------------ save video frame ------------------------ #
                render_np   = render_image[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                gt_np       = gt_image[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                video_frame = np.concatenate((render_np, gt_np), axis=1)

                render_video_frames.append((render_np * 255.).astype('uint8'))
                join_video_frames.append((video_frame * 255.).astype('uint8'))

                # ------------------------ gather loss ------------------------ #
                for each_loss_name, each_loss_monitor in self.loss_monitor.items():
                    each_loss_monitor.update(loss_output_dict[each_loss_name].item())

                # ------------------------ gather metrices ------------------------ #
                for metric in self.metrics:
                    metric.update(render_image, gt_image)
                    # log current to dict for logging
                    metrics_per_frame[metric.name][frame_idx] = metric.V_cur

                # ------------------------ save snapshot ------------------------ #
                if (idx % self.media_save_path["eval_snapshot"]["interval"]) == 0 or self.global_step == 1:
                    save_path = os.path.join(self.media_save_path["eval_snapshot"]["folder"], f"eval_step_{frame_idx:06d}{'_' + name if name else ''}.png")
                    self.save_full_snap_shot(input_data, ground_truth, save_path)

                # ------------------------ save pre frame render result ------------------------ #
                save_render_path = os.path.join(self.media_save_path["eval_render"]["folder"], f"evaluation_render{'_' + name if name else ''}", f"{frame_idx:04d}.png")
                os.makedirs(os.path.dirname(save_render_path), exist_ok=True)
                render_plot = render_image[0].detach().cpu()
                torchvision.utils.save_image(render_plot, save_render_path, normalize=True, value_range=(0, 1))

                pbar.set_description(f"loss={loss_output_dict['loss'].item():.4f} ({self.loss_meter.avg:.4f})")
                pbar.update(1)

        all_render_np = np.stack(render_video_frames, axis=0)
        all_join_up   = np.stack(join_video_frames, axis=0)
        imageio.mimwrite(save_path_videos, all_render_np, fps=25)
        imageio.mimwrite(save_path_join_videos, all_join_up, fps=25)

        pbar.close()
        self.log_file_only(pbar)

        #------------------------ gather metrics of one epoch ------------------------#
        save_metrics_folder = os.path.join(self.media_save_path["eval_metric"]["folder"], f"evaluation_metrics_per_epoch{'_' + name if name else ''}")
        self.save_one_epoch_metrics(metrics_per_frame, self.metrics, save_metrics_folder)

        self.log(f"==> Metrics @ Epoch {self.epoch}")
        for metric in self.metrics:
            self.log(metric.report(), style="blue")
            metric.clear()
        self.log(f"==> Loss @ Epoch {self.epoch}.")
        for each_loss_name, each_loss_monitor in self.loss_monitor.items():
            self.log(each_loss_monitor.report(), style="blue")
            each_loss_monitor.clear()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")


    def evalulate_step(self, input_data, ground_truth):
        
        output_data  = self.model(input_data)
        render_image = output_data['rgb_image']
        gt_image     = ground_truth['rgb']
        loss_output = self.criterions(output_data, ground_truth)

        return {'loss_output': loss_output,
                'render_image': render_image,
                'gt_image': gt_image}


    def save_checkpoint(self, name=None, remove_old=True, save_path=None):

        if save_path is None:
            save_path = self.ckpt_path

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
        }

        state['model'] = self.model.state_dict()

        state['train_expression'] = self.train_expression.state_dict()
        state['train_flame_pose'] = self.train_flame_pose.state_dict()
        state['train_cam_pose']   = self.train_cam_pose.state_dict()

        state['test_expression']  = self.test_expression.state_dict()
        state['test_flame_pose']  = self.test_flame_pose.state_dict()
        state['test_cam_pose']    = self.test_cam_pose.state_dict()

        file_base = os.path.join(save_path, name)

        if remove_old:
            self.ckpt_stats.append(file_base)

            if len(self.ckpt_stats) > self.max_keep_ckpt:
                old_file = self.ckpt_stats.pop(0)
                if os.path.exists(f'{old_file}.pth'):
                    os.remove(f'{old_file}.pth')
        
        torch.save(state, f"{file_base}.pth")


    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
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
            self.train_expression.load_state_dict(checkpoint_dict['train_expression'])
            self.train_flame_pose.load_state_dict(checkpoint_dict['train_flame_pose'])
            self.train_cam_pose.load_state_dict(checkpoint_dict['train_cam_pose'])
            self.log("[INFO] Load optimized train set tracking")
        except Exception as e:
            self.log("[INFO] Fail to load optimized train set tracking", e)

        try:
            self.test_expression.load_state_dict(checkpoint_dict['test_expression'])
            self.test_flame_pose.load_state_dict(checkpoint_dict['test_flame_pose'])
            self.test_cam_pose.load_state_dict(checkpoint_dict['test_cam_pose'])
            self.log("[INFO] Load optimized test set tracking")
        except Exception as e:
            self.log("[INFO] Fail to load optimized test set tracking:", e)

        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")


    def save_full_snap_shot(self, input_data, ground_truth, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        output_data_vis = self.model.visualization(input_data)

        if self.model_name == 'FateAvatar':

            gt_image        = ground_truth['rgb']
            pred_image      = output_data_vis['rgb_image']
            cano_image      = output_data_vis['cano_image']
            point_image     = output_data_vis['point_image']
            grad_image      = output_data_vis['grad_image']

            # visualization output from render directly
            pred_plot       = pred_image.detach().cpu()
            gt_plot         = gt_image.detach().cpu()
            cano_plot       = cano_image.detach().cpu()
            point_plot      = point_image.detach().cpu()
            grad_plot       = grad_image.detach().cpu()

            # render mesh via nvidiffrast
            dr_verts = output_data_vis['verts']
            dr_faces = output_data_vis['faces']
            dr_cam   = output_data_vis['camera']

            face_colors = self.mesh_colors[:3].to(dr_verts)[None, None, :].repeat(1, dr_faces.shape[0], 1)

            dr_output = self.mesh_render.render_from_camera(dr_verts,
                                                            dr_faces,
                                                            dr_cam,
                                                            face_colors = face_colors)
            
            mesh_path = save_path.replace('.png', '.obj')
            save_obj(mesh_path, dr_verts[0], dr_faces)
            
            rgba_mesh   = dr_output['rgba']
            mesh_plot    = rgba_mesh[..., :3].permute(0, 3, 1, 2).to('cpu')

            # L1 heat map
            err = (gt_plot - pred_plot).abs().max(dim=1)[0].clip(0, 1)
            err_plot = colorize_weights_map(err, min_val=0, max_val=1)

            grad_plot = colorize_weights_map(grad_plot[:, 0], min_val=0, max_val=1)

            # images = torch.cat([pred_plot, gt_plot, mesh_plot, err_plot, cano_plot, point_plot], dim=0)
            images = torch.cat([pred_plot, gt_plot, mesh_plot, err_plot, cano_plot, grad_plot], dim=0)
            grid   = torchvision.utils.make_grid(images, nrow=3, normalize=False, value_range=(0, 1))

        else:

            gt_image        = ground_truth['rgb']
            pred_image      = output_data_vis['rgb_image']

            # visualization output from render directly
            pred_plot       = pred_image.detach().cpu()
            gt_plot         = gt_image.detach().cpu()

            # L1 heat map
            err = (gt_plot - pred_plot).abs().max(dim=1)[0].clip(0, 1)
            err_plot = colorize_weights_map(err, min_val=0, max_val=1)

            images = torch.cat([pred_plot, gt_plot, err_plot], dim=0)
            grid   = torchvision.utils.make_grid(images, nrow=3, normalize=True, value_range=(0, 1))

        torchvision.utils.save_image(grid, save_path)


    def fps_performance_test(self):
        self.log(f"++> Run FPS test ...")
        self.model.eval()

        total_time = 0.0
        total_frames = 0

        fps_list = []

        with torch.no_grad():
            for idx, (_, input_data, ground_truth) in enumerate(self.test_loader):
                load_to_gpu(input_data, ground_truth, self.device)

                t_start = time.time()
                fps  = self.fps_unit(input_data)
                t_end = time.time()

                total_time += (t_end - t_start)
                total_frames += 1

                fps_list.append(fps)

        print_interval = 10
        for i in range(0, len(fps_list), print_interval):
            self.log(f"FPS at frame {i}: {fps_list[i]:.2f}")

        average_fps = total_frames / total_time if total_time > 0 else float('inf')
        self.log(f"{self.name} Average FPS: {average_fps:.2f}")


    @measure_fps
    def fps_unit(self, input_data):
        return self.model(input_data)


    def save_one_epoch_metrics(self, per_frame_dict: dict, metrics_list: list, save_folder):
        os.makedirs(save_folder, exist_ok=True)

        plt.figure(figsize=(16, 9))
        sorted_per_frame_dict = {}

        for metric in metrics_list:
            metric_dict = per_frame_dict[metric.name]
            sorted_metrics = {key: metric_dict[key] for key in sorted(metric_dict.keys())}

            plt.plot(sorted_metrics.keys(), sorted_metrics.values())
            plt.xlabel('Frame ID')
            plt.ylabel(f'Metric: {metric.name}')
            plt.title('Metric per Frame during Training')

            plt.savefig(os.path.join(save_folder, f'{self.epoch:03d}_{metric.name}_plot.png'), dpi=300)

            plt.clf()

            plt.close()

            sorted_per_frame_dict[metric.name] = sorted_metrics

        with open(os.path.join(save_folder, f'{self.epoch:03d}.json'), 'w') as f:
            json.dump(sorted_per_frame_dict, f)

        del sorted_per_frame_dict, per_frame_dict






                


                






        

        







            


    
