# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import numpy as np

from .lbs import *
import pickle


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)
    
def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """
    def __init__(
        self,
        flame_model_path:   str,
        lmk_embedding_path: str,
        n_shape:            int,
        n_exp:              int,
        shape_params:       torch.Tensor,
        canonical_expression,
        canonical_pose:     int,
        device:             torch.device,
        factor:             int = 1
    ):
        super(FLAME, self).__init__()
        print("Setting up [FLAME]")
        print(f"Loading model from: {flame_model_path}")
        self.device = device
        self.dtype = torch.float32

        with open(flame_model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)
        
        # begin: for plotting landmarks, to make sure that camera is correct. This is so hard to debug otherwise...
        lmk_embeddings = np.load(lmk_embedding_path, allow_pickle=True, encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]

        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1

        self.register_buffer('parents', parents)
        self.register_buffer('lmk_faces_idx', torch.from_numpy(lmk_embeddings['static_lmk_faces_idx']).long())
        self.register_buffer('dynamic_lmk_faces_idx', lmk_embeddings['dynamic_lmk_faces_idx'].long())
        self.register_buffer('dynamic_lmk_bary_coords', lmk_embeddings['dynamic_lmk_bary_coords'].to(self.dtype))
        self.register_buffer('full_lmk_faces_idx', torch.from_numpy(lmk_embeddings['full_lmk_faces_idx']).long())
        self.register_buffer(
            'lmk_bary_coords',
            torch.from_numpy(lmk_embeddings['static_lmk_bary_coords']).to(self.dtype)
        )
        self.register_buffer(
            'full_lmk_bary_coords',
            torch.from_numpy(lmk_embeddings['full_lmk_bary_coords']).to(self.dtype)
        )

        neck_kin_chain = []; NECK_IDX=1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]

        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))
        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template) * factor, dtype=self.dtype))
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:,:,:n_shape], shapedirs[:,:,300:300+n_exp]], 2)
        self.register_buffer('shapedirs', shapedirs * factor)
        canonical_expression =canonical_expression[..., :n_exp]
        self.v_template = self.v_template + torch.einsum('bl,mkl->bmk', [shape_params.cpu(), self.shapedirs[:, :, :n_shape]]).squeeze(0)

        self.canonical_pose = torch.zeros(1, 15).float().to(self.device)
        self.canonical_pose[:, 6] = canonical_pose
        self.canonical_exp = canonical_expression.float().to(self.device)

        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs) * factor, dtype=self.dtype))
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long(); parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter(
            'eye_pose',
            nn.Parameter(default_eyball_pose,requires_grad=False)
        )
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter(
            'neck_pose',
            nn.Parameter(default_neck_pose,requires_grad=False)
        )

        self.n_shape = n_shape
        self.n_exp = n_exp

    def forward(self, expression_params: torch.Tensor, full_pose: torch.Tensor):
        """
        FLAME mesh morphing

        Args:
            expression_params: torch.Size([bs, n_exp])
            full_pose:         torch.Size([bs, 15])

        Return:
            vertices:          torch.Size([bs, 5023, 3])
            pose_feature:      torch.Size([bs, 36]), flattening of the rotation metrices excluding the root
            transformations:   torch.Size([bs, 5, 4, 4]), rigid transformations for five nodes
        """
        batch_size = expression_params.shape[0]
        betas = torch.cat([torch.zeros(batch_size, self.n_shape).to(expression_params.device), expression_params[:, :self.n_exp]], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, pose_feature, transformations = lbs(
            betas, full_pose, template_vertices,
            self.shapedirs, self.posedirs,
            self.J_regressor, self.parents,
            self.lbs_weights, dtype=self.dtype
        )
        
        return vertices, pose_feature, transformations
    
    def forward_with_delta_blendshape(
            self,
            expression_params:  torch.Tensor,
            full_pose:          torch.Tensor,
            delta_shapedirs:    nn.Parameter = None,
            delta_posedirs:     nn.Parameter = None,
            delta_vertex:       nn.Parameter = None
        ):
        """
            FLAME mesh morphing with personalized blendshape and offset

            Args:
                expression_params: torch.Size([bs, n_exp])
                full_pose:         torch.Size([bs, 15])
                delta_shapedirs:   torch.Size([5023, 3, 400])
                delta_posedirs:    torch.Size([36, 5023 * 3])
                delta_vertex:      torch.Size([5023, 3])

            Return:
                vertices:          torch.Size([bs, 5023, 3])
                pose_feature:      torch.Size([bs, 36]), flattening of the rotation metrices excluding the root
                transformations:   torch.Size([bs, 5, 4, 4]), rigid transformations for five nodes
        """
        batch_size = expression_params.shape[0]
        betas = torch.cat([torch.zeros(batch_size, self.n_shape).to(expression_params.device), expression_params[:, :self.n_exp]], dim=1)
        
        if delta_vertex is None:
            verts = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            verts = self.v_template.unsqueeze(0).expand(batch_size, -1, -1) + delta_vertex.unsqueeze(0).expand(batch_size, -1, -1)

        if delta_shapedirs is None:
            shapedirs = self.shapedirs
        else:
            shapedirs = self.shapedirs + delta_shapedirs

        if delta_posedirs is None:
            posedirs = self.posedirs
        else:
            posedirs = self.posedirs + delta_posedirs

        vertices, pose_feature, transformations = lbs(
            betas, full_pose, verts,
            shapedirs, posedirs,
            self.J_regressor, self.parents,
            self.lbs_weights, dtype=self.dtype
        )

        return vertices, pose_feature, transformations
    

    def forward_pts(self, pnts_c, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32, mask=None):
        """
            FLAME morphing for individual points, used in PointAvatar[1] and MonoGaussianAvatar[2]

            [1] PointAvatar, Zheng et al. https://arxiv.org/abs/2212.08377
            [2] MonoGaussianAvatar, Chen et al. https://dl.acm.org/doi/abs/10.1145/3641519.3657499
        """
        assert len(pnts_c.shape) == 2
        if mask is not None:
            pnts_c = pnts_c[mask]
            betas = betas[mask]
            transformations = transformations[mask]
            pose_feature = pose_feature[mask]
        num_points = pnts_c.shape[0]
        if shapedirs.shape[-1] > self.n_exp:
            canonical_exp = torch.cat([self.canonical_exp, torch.zeros(1, shapedirs.shape[-1] - self.n_exp).cuda()], dim=1)
        else:
            canonical_exp = self.canonical_exp
        pnts_c_original = inverse_pts(
            pnts_c, canonical_exp.expand(num_points, -1),
            self.canonical_transformations.expand(num_points, -1, -1, -1),
            self.canonical_pose_feature.expand(num_points, -1),
            shapedirs, posedirs, lbs_weights, dtype=dtype
        )
        pnts_p = forward_pts(
            pnts_c_original, betas,
            transformations,
            pose_feature,
            shapedirs, posedirs, lbs_weights, dtype=dtype
        )
        return pnts_p

