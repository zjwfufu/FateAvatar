import os
import tqdm
import time

import torch

from tools.util     import load_to_gpu, print_tree, EasyDict
from rich.console   import Console
from train.dataset  import FaceDataset
from train.loss     import BaseLoss
from train.metrics  import LossMeter
from mesh_rendering import NVDiffRenderer

class BaseTrainer(object):
    def __init__(
        self,
        name:           str,
        cfg:            EasyDict,
        model:          torch.nn.Module,
        device:         torch.device,
        train_dataset:  FaceDataset,
        test_dataset:   FaceDataset,
        criterions:     BaseLoss,
        metrics:        list = [],
        workspace:      str = 'workspace',
        use_checkpoint: str = 'scratch',
        max_keep_ckpt:  int = 2
    ):
        #---------- Assign input arguments to instance variables ----------#
        self.name           = name
        self.cfg            = cfg
        self.model          = model
        self.device         = device
        self.train_dataset  = train_dataset
        self.test_dataset   = test_dataset
        self.criterions     = criterions
        self.metrics        = metrics
        self.workspace      = workspace
        self.use_checkpoint = use_checkpoint
        self.max_keep_ckpt  = max_keep_ckpt

        self.console        = Console()
        self.time_stamp     = time.strftime("%Y-%m-%d_%H-%M-%S")

        self.epoch          = 0
        self.global_step    = 0
        self.local_step     = 0
        self.ckpt_stats     = []

        self.mesh_render    = NVDiffRenderer(use_opengl=False)
        self.mesh_colors    = torch.tensor([1, 1, 1, 0.5])

        #------- Create data loaders for training and testing datasets ----------#
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size      = 1,
            shuffle         = True,
            collate_fn      = train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None,
            num_workers     = 4,
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size      = 1,
            shuffle         = False,
            collate_fn      = test_dataset.collate_fn if hasattr(test_dataset, 'collate_fn') else None,
            num_workers     = 4
        )
        
        #------- Prepare workspace for saving checkpoints, log, etc. ----------#
        self.register_workspace()

        #------- Set up loss monitor ----------#
        self.register_loss_monitor()

        #------- Prepare tracking optimization (optional) ----------#
        self.register_tracking_optimization()

        #------- Load checkpoints ----------#
        self.load_checkpoint_process()

        #------- Register several optimizers ----------#
        self.optimizers_group = {}
        self.register_optimizer_group()
        
        #------- Log config yaml to file ----------#
        print_tree(self.log_file_only, self.cfg)

        
    def register_loss_monitor(self):
        self.loss_meter = LossMeter(name='loss')
        self.loss_monitor = {'loss': self.loss_meter}

        loss_weight_dict = self.cfg.loss.weight

        for loss_name, loss_weight in loss_weight_dict.items():
            if loss_weight > 0:
                self.loss_monitor.update({loss_name: LossMeter(name=loss_name)})

    def register_workspace(self):
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(self.workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {self.workspace}')

    def register_tracking_optimization(self):
        if self.cfg.optimize_tracking:
            num_train_frames        = len(self.train_dataset)

            self.train_expression   = torch.nn.Embedding(num_train_frames, self.cfg.n_exp,
                                                        _weight    = self.train_dataset.data["expressions"],
                                                        sparse     = True).to(self.device)
            self.train_flame_pose   = torch.nn.Embedding(num_train_frames, 15,
                                                        _weight    = self.train_dataset.data["flame_pose"],
                                                        sparse     = True).to(self.device)
            self.train_cam_pose     = torch.nn.Embedding(num_train_frames, 3,
                                                        _weight    = self.train_dataset.data["world_mats"][:, :3, 3],
                                                        sparse     = True).to(self.device)
            
            num_test_frames         = len(self.test_dataset)

            self.test_expression    = torch.nn.Embedding(num_test_frames, self.cfg.n_exp,
                                                        _weight    = self.test_dataset.data["expressions"],
                                                        sparse     = True).to(self.device)
            self.test_flame_pose    = torch.nn.Embedding(num_test_frames, 15,
                                                        _weight    = self.test_dataset.data["flame_pose"],
                                                        sparse     = True).to(self.device)
            self.test_cam_pose      = torch.nn.Embedding(num_test_frames, 3,
                                                        _weight    = self.test_dataset.data["world_mats"][:, :3, 3],
                                                        sparse     = True).to(self.device)
            
            l_train_tracking        = [
                {'params': list(self.train_expression.parameters()), 'lr': self.cfg.training.tracking_lr, 'name': "train_expression"},
                {'params': list(self.train_flame_pose.parameters()), 'lr': self.cfg.training.tracking_lr, 'name': "train_flame_pose"},
                {'params': list(self.train_cam_pose.parameters()),  'lr': self.cfg.training.tracking_lr, 'name': "train_cam_pose"},
            ]
            self.optimizer_train_tracking   = torch.optim.SparseAdam(l_train_tracking, lr=self.cfg.training.tracking_lr)

            l_test_tracking         = [
                {'params': list(self.test_expression.parameters()), 'lr': self.cfg.training.tracking_lr, 'name': "test_expression"},
                {'params': list(self.test_flame_pose.parameters()), 'lr': self.cfg.training.tracking_lr, 'name': "test_flame_pose"},
                {'params': list(self.test_cam_pose.parameters()), 'lr': self.cfg.training.tracking_lr, 'name': "test_cam_pose"},     
            ]
            self.optimizer_test_tracking   = torch.optim.SparseAdam(l_test_tracking, lr=self.cfg.training.tracking_lr)

    def register_optimizer_group(self):
        
        if self.cfg.optimize_tracking:
            self.optimizers_group.update({'train_tracking': self.optimizer_train_tracking})

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

    def save_checkpoint(self, name=None, remove_old=True, save_path=None):
        raise NotImplementedError

    def load_checkpoint(self, checkpoint=None):
        raise NotImplementedError
    
    def train(self, max_epochs):
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch  = epoch

            self.train_epoch()
            self.save_checkpoint()

    def train_epoch(self):
        raise NotImplementedError
    
    def train_step(self, input_data, ground_truth):
        raise NotImplementedError

    def evaluate(self, mode, optim_epoch=None):
        self.evaluate_epoch(mode, optim_epoch=optim_epoch)
        self.save_checkpoint()

    def evaluate_epoch(self, mode, optim_epoch=None):
        raise NotImplementedError
    
    def evalulate_step(self, input_data, ground_truth):
        raise NotImplementedError
    
    def optimize_tracking(self, optim_epoch=None):
        """
        optimize test set FLAME coefficients
        """

        if optim_epoch is None:
            optim_epoch = 50

        self.model.train()

        if self.cfg.optimize_tracking:
            self.log("==> Optimizing tracking for evaluation...")
            self.log(f"==> Finetune {self.epoch} epochs...")
            pbar = tqdm.tqdm(total=len(self.test_loader), bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

            for idx, (_, input_data, ground_truth) in enumerate(self.test_loader):

                load_to_gpu(input_data, ground_truth, self.device)

                for _ in range(optim_epoch):
                    # overwrite test flame coefficients
                    input_data["expression"]            = self.test_expression(input_data["idx"]).squeeze(1)
                    input_data["flame_pose"]            = self.test_flame_pose(input_data["idx"]).squeeze(1)
                    input_data["cam_pose"][:, :3, 3]    = self.test_cam_pose(input_data["idx"]).squeeze(1)

                    output_data     = self.model(input_data)
                    loss_output     = self.criterions(output_data, ground_truth)
                    loss            = loss_output["loss"]
                    self.optimizer_test_tracking.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer_test_tracking.step()

                    pbar.set_description(f"loss={loss.item():.4f} frame={idx}")

                pbar.update(1)

            pbar.close()
            self.log_file_only(pbar)

    def log(self, *args, **kwargs):
        self.console.print(*args, **kwargs)
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()

    def log_file_only(self, *args, **kwargs):
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()