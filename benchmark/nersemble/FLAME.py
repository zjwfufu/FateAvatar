"""
FLAME Layer: Implementation of the 3D Statistical Face model in PyTorch

It is designed in a way to directly plug in as a decoder layer in a
Deep learning framework for training and testing

It can also be used for 2D or 3D optimisation applications

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.

For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""
# Modified from smplx code [https://github.com/vchoutas/smplx] for FLAME

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from elias.util.io import download_file
from smplx.lbs import lbs, batch_rodrigues, vertices2landmarks
from smplx.utils import Struct, to_tensor, to_np, rot_mat_to_euler

from flame_model.assets import FLAME_VT_PATH, FLAME_FT_PATH

from dreifus.matrix import Pose


@dataclass
class FlameConfig:
    flame_folder: str = f"./weights"
    shape_params: int = 100
    expression_params: int = 50
    pose_params: int = 6
    use_face_contour: bool = True  # If true apply the landmark loss on also on the face contour
    use_3D_translation: bool = True  # If true apply the landmark loss on also on the face contour
    optimize_eyeballpose: bool = True  # If true optimize for the eyeball pose
    optimize_neckpose: bool = True  # If true optimize for the neck pose
    num_worker: int = 4
    batch_size: int = 8
    ring_margin: float = 0.5
    ring_loss_weight: float = 1.0
    flame_version: str = 'flame2023.pkl'
    static_landmark_embedding: str = 'flame_static_embedding.pkl'
    dynamic_landmark_embedding: str = 'flame_dynamic_embedding.npy'


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs a mesh and 3D facial landmarks
    """

    def __init__(self, config: FlameConfig = FlameConfig()):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")

        flame_model_path = f"{config.flame_folder}/{config.flame_version}"
        if not Path(flame_model_path).exists():
            # Download FLAME
            raise FileNotFoundError(f"FLAME model not found! Please download {config.flame_version} "
                                    f"from https://flame.is.tue.mpg.de/download.php and put it into {config.flame_folder}")

        static_landmark_embedding_path = f"{config.flame_folder}/flame_static_embedding.pkl"
        dynamic_landmark_embedding_path = f"{config.flame_folder}/flame_dynamic_embedding.npy"
        if not Path(static_landmark_embedding_path).exists():
            download_file("https://raw.githubusercontent.com/soubhiksanyal/RingNet/refs/heads/master/flame_model/flame_static_embedding.pkl",
                          static_landmark_embedding_path)
            download_file("https://github.com/soubhiksanyal/RingNet/raw/refs/heads/master/flame_model/flame_dynamic_embedding.npy",
                          dynamic_landmark_embedding_path)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            with open(flame_model_path, 'rb') as f:
                self.flame_model = Struct(**pickle.load(f, encoding='latin1'))

        self.NECK_IDX = 1
        self.batch_size = config.batch_size
        self.dtype = torch.float32
        self.use_face_contour = config.use_face_contour

        self.faces = self.flame_model.f
        self.vt = np.load(FLAME_VT_PATH)
        self.ft = np.load(FLAME_FT_PATH)
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        # Fixing remaining Shape betas
        # There are total 300 shape parameters to control FLAME; But one can use the first few parameters to express
        # the shape. For example 100 shape parameters are used for RingNet project
        default_shape = torch.zeros([self.batch_size, 300 - config.shape_params],
                                    dtype=self.dtype, requires_grad=False)
        self.register_parameter('shape_betas', nn.Parameter(default_shape,
                                                            requires_grad=False))

        # Fixing remaining expression betas
        # There are total 100 shape expression parameters to control FLAME; But one can use the first few parameters to express
        # the expression. For example 50 expression parameters are used for RingNet project
        default_exp = torch.zeros([self.batch_size, 100 - config.expression_params],
                                  dtype=self.dtype, requires_grad=False)
        self.register_parameter('expression_betas', nn.Parameter(default_exp,
                                                                 requires_grad=False))

        # Eyeball and neck rotation
        default_eyball_pose = torch.zeros([self.batch_size, 6],
                                          dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                         requires_grad=False))

        default_neck_pose = torch.zeros([self.batch_size, 3],
                                        dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                          requires_grad=False))

        # Fixing 3D translation since we use translation in the image plane

        self.use_3D_translation = config.use_3D_translation

        default_transl = torch.zeros([self.batch_size, 3],
                                     dtype=self.dtype, requires_grad=False)
        self.register_parameter(
            'transl',
            nn.Parameter(default_transl, requires_grad=False))

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(self.flame_model.v_template),
                                       dtype=self.dtype))

        # The shape components
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(to_np(
            self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=self.dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))

        # Static and Dynamic Landmark embeddings for FLAME

        with open(static_landmark_embedding_path, 'rb') as f:
            static_embeddings = Struct(**pickle.load(f, encoding='latin1'))

        lmk_faces_idx = (static_embeddings.lmk_face_idx).astype(np.int64)
        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = static_embeddings.lmk_b_coords
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_bary_coords, dtype=self.dtype))

        if self.use_face_contour:
            conture_embeddings = np.load(dynamic_landmark_embedding_path,
                                         allow_pickle=True, encoding='latin1')
            conture_embeddings = conture_embeddings[()]
            dynamic_lmk_faces_idx = np.array(conture_embeddings['lmk_face_idx']).astype(np.int64)
            dynamic_lmk_faces_idx = torch.tensor(
                dynamic_lmk_faces_idx,
                dtype=torch.long)
            self.register_buffer('dynamic_lmk_faces_idx',
                                 dynamic_lmk_faces_idx)

            dynamic_lmk_bary_coords = conture_embeddings['lmk_b_coords']
            dynamic_lmk_bary_coords = torch.tensor(
                np.array(dynamic_lmk_bary_coords), dtype=self.dtype)
            self.register_buffer('dynamic_lmk_bary_coords',
                                 dynamic_lmk_bary_coords)

            neck_kin_chain = []
            curr_idx = torch.tensor(self.NECK_IDX, dtype=torch.long)
            while curr_idx != -1:
                neck_kin_chain.append(curr_idx)
                curr_idx = self.parents[curr_idx]
            self.register_buffer('neck_kin_chain',
                                 torch.stack(neck_kin_chain))

        self._separate_transformation = True

    def _find_dynamic_lmk_idx_and_bcoords(self, vertices, pose, dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
            Source: Modified for batches from https://github.com/vchoutas/smplx
        """

        batch_size = vertices.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_vecs = aa_pose.view(-1, 3)

        assert rot_vecs.dtype == dtype

        rot_mats = batch_rodrigues(
            rot_vecs).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=vertices.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)
        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)

        return dyn_lmk_faces_idx, dyn_lmk_b_coords
    
    def apply_model_to_world_transformation(self, points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        points_world = points
        if self._separate_transformation:
            B = points.shape[0]
            V = points.shape[1]
            model_transformations = torch.stack([torch.from_numpy(
                Pose.from_euler(rotation[0].detach().cpu().numpy(), translation[0].detach().cpu().numpy(), 'XYZ'))]
            ).to(scale.device)
            model_transformations[:, :3, :3] *= scale[0]
            points_world = torch.cat([points_world, torch.ones((B, V, 1)).to(points_world.device)], dim=-1)
            points_world = torch.bmm(points_world, model_transformations.permute(0, 2, 1))
            points_world = points_world[..., :3]

        return points_world

    def forward(
            self,
            shape_params=None,
            expression_params=None,
            pose_params=None,
            neck_pose=None,
            eye_pose=None,
            transl=None,
            rotation=None,
            scale=None,
            model_view_transform: bool = True,
        ):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters. Typically, the first 3 will be global rotation, and the last 3 the jaw pose
            return:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        B = shape_params.shape[0]
        betas = torch.cat(
            [shape_params, self.shape_betas.repeat(B, 1), expression_params, self.expression_betas.repeat(B, 1)],
            dim=1)
        neck_pose = (neck_pose if neck_pose is not None else self.neck_pose)
        eye_pose = (eye_pose if eye_pose is not None else self.eye_pose)
        transl = (transl if transl is not None else self.transl)
        full_pose = torch.cat([pose_params[:, :3], neck_pose, pose_params[:, 3:], eye_pose], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(self.batch_size, 1, 1)

        assert betas.dtype == self.dtype

        vertices, _ = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).repeat(
            self.batch_size, 1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            self.batch_size, 1, 1)
        if self.use_face_contour:
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain, dtype=self.dtype)

            lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx.repeat(B, 1)], 1)
            lmk_bary_coords = torch.cat(
                [dyn_lmk_bary_coords, lmk_bary_coords.repeat(B, 1, 1)], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)
        
        if model_view_transform:
            vertices = self.apply_model_to_world_transformation(
                vertices,
                rotation    = rotation,
                translation = transl,
                scale       = scale
            )

        return vertices, landmarks, None    # None for compatibility


    def forward_with_delta_blendshape(
            self,
            shape_params=None,
            expression_params=None,
            pose_params=None,
            neck_pose=None,
            eye_pose=None,
            transl=None,
            rotation=None,
            scale=None,
            delta_shapedirs:    nn.Parameter = None,
            delta_posedirs:     nn.Parameter = None,
            delta_vertex:       nn.Parameter = None,
            model_view_transform:       bool = True,
        ):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters. Typically, the first 3 will be global rotation, and the last 3 the jaw pose
            return:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        B = shape_params.shape[0]
        betas = torch.cat(
            [shape_params, self.shape_betas.repeat(B, 1), expression_params, self.expression_betas.repeat(B, 1)],
            dim=1)
        neck_pose = (neck_pose if neck_pose is not None else self.neck_pose)
        eye_pose = (eye_pose if eye_pose is not None else self.eye_pose)
        transl = (transl if transl is not None else self.transl)
        full_pose = torch.cat([pose_params[:, :3], neck_pose, pose_params[:, 3:], eye_pose], dim=1)

        assert betas.dtype == self.dtype

        if delta_vertex is None:
            verts = self.v_template.unsqueeze(0).expand(B, -1, -1)
        else:
            verts = self.v_template.unsqueeze(0).expand(B, -1, -1) + delta_vertex.unsqueeze(0).expand(B, -1, -1)

        if delta_shapedirs is None:
            shapedirs = self.shapedirs
        else:
            shapedirs = self.shapedirs + delta_shapedirs

        if delta_posedirs is None:
            posedirs = self.posedirs
        else:
            posedirs = self.posedirs + delta_posedirs

        vertices, _ = lbs(betas, full_pose, verts,
                          shapedirs, posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).repeat(
            self.batch_size, 1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            self.batch_size, 1, 1)
        if self.use_face_contour:
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain, dtype=self.dtype)

            lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx.repeat(B, 1)], 1)
            lmk_bary_coords = torch.cat(
                [dyn_lmk_bary_coords, lmk_bary_coords.repeat(B, 1, 1)], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        if model_view_transform:
            vertices = self.apply_model_to_world_transformation(
                vertices,
                rotation    = rotation,
                translation = transl,
                scale       = scale
            )

        return vertices, landmarks, None    # None for compatibility

