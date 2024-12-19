import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from flame.FLAME            import FLAME

from pytorch3d.io   import load_obj
from pytorch3d.ops  import knn_points

from volume_rendering.mesh_sampling import (
    uniform_sampling_barycoords,
    reweight_uvcoords_by_barycoords,
    reweight_verts_by_barycoords
)

from volume_rendering.camera_3dgs       import Camera
from volume_rendering.gaussian_model    import GaussianModel
from volume_rendering.render_3dgs       import render

from tools.util import get_bg_color
from tools.gs_utils.general_utils       import inverse_sigmoid

import warnings
warnings.filterwarnings("ignore", message="No mtl file provided")
warnings.filterwarnings("ignore", message="Mtl file does not exist")

#-------------------------------------------------------------------------------#

def load_binary_pickle(filepath):
    with open(filepath, 'rb') as f:
        if sys.version_info >= (3, 0):
            data = pickle.load(f, encoding='latin1')
        else:
            data = pickle.load(f)
    return data

def a_in_b_torch(a, b):
    ainb = torch.isin(a, b)
    return ainb

#-------------------------------------------------------------------------------#

class FlashAvatar(nn.Module):
    def __init__(
            self,
            shape_params,
            img_res,
            canonical_expression,
            canonical_pose,
            background_color,
            cfg_model,
            device
        ):
        super().__init__()
        """
        official code:
            https://github.com/USTC3DV/FlashAvatar-code

        bibtex:
            @inproceedings{xiang2024flashavatar,
                author    = {Jun Xiang and Xuan Gao and Yudong Guo and Juyong Zhang},
                title     = {FlashAvatar: High-fidelity Head Avatar with Efficient Gaussian Embedding},
                booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
                year      = {2024},
            }
        """

        self.uv_resolution = cfg_model.tex_size
        self.max_sh_degree = 3

        self.device = device

        self.bg_color = get_bg_color(background_color).to(self.device)

        self.img_res = img_res

        self.cfg_model = cfg_model

        self.pts_freq = 8

        self.shape_params           = shape_params
        self.canonical_expression   = canonical_expression
        self.canonical_pose         = canonical_pose

        self._register_flame(
            flame_path='./weights/generic_model.pkl',
            flame_mask_path='./weights/FLAME_masks.pkl',
            landmark_embedding_path='./weights/landmark_embedding.npy'
        )
        
        self._register_deformer()

        self._register_template_mesh(template_path='./weights/head_template_mouth_close.obj')

        mean_scaling, max_scaling, scale_init = self.get_init_scale_by_knn(self.verts_sampling)

        self.register_buffer('mean_scaling', mean_scaling)
        self.register_buffer('max_scaling', max_scaling)
        self.register_buffer('scale_init', scale_init)

        self._register_init_gaussian()


    def _register_flame(self, flame_path, flame_mask_path,
                        landmark_embedding_path):

        self.flame = FLAME(
            flame_path,
            landmark_embedding_path,
            n_shape              = self.cfg_model.n_shape,
            n_exp                = self.cfg_model.n_exp,
            shape_params         = self.shape_params,
            canonical_expression = self.canonical_expression,
            canonical_pose       = self.canonical_pose,
            device               = self.device
        ).to(self.device)
        
        canonical_verts, canonical_pose_feature, canonical_transformations = self.flame(
            expression_params = self.flame.canonical_exp,
            full_pose         = self.flame.canonical_pose
        )
        
        # make sure call of FLAME is successful
        self.canonical_verts                    = canonical_verts
        self.flame.canonical_verts              = canonical_verts.squeeze(0)
        self.flame.canonical_pose_feature       = canonical_pose_feature
        self.flame.canonical_transformations    = canonical_transformations

        # flame mask
        flame_mask_dic          = load_binary_pickle(flame_mask_path)
        boundary_id             = flame_mask_dic['boundary']
        full_id                 = np.array(range(5023)).astype(int)
        neckhead_id_list        = list(set(full_id) - set(boundary_id))
        self.neckhead_id_list   = neckhead_id_list
        self.neckhead_id_tensor = torch.tensor(self.neckhead_id_list, dtype=torch.int64).to(self.device)

    def _register_deformer(self):
        self.pts_embedder = Embedder(self.pts_freq)

        self.deformNet = MLP(
            input_dim = self.pts_embedder.dim_embeded + self.cfg_model.n_exp + 3 + 6,
            output_dim = 10,
            hidden_dim = 256,
            hidden_layers = 6
        )

    def _register_template_mesh(self, template_path):

        #----------------   load head mesh & process UV ----------------#
        verts, faces, aux = load_obj(template_path)

        uvcoords    = aux.verts_uvs
        uvfaces     = faces.textures_idx
        faces       = faces.verts_idx

        face_index, bary_coords = uniform_sampling_barycoords(
            num_points    = self.uv_resolution * self.uv_resolution,
            tex_coord     = uvcoords,
            uv_faces      = uvfaces,
            strict        = False
        )

        uvcoords_sample = reweight_uvcoords_by_barycoords(
            uvcoords    = uvcoords,
            uvfaces     = uvfaces,
            face_index  = face_index,
            bary_coords = bary_coords
        )
        
        uvcoords_sample = uvcoords_sample[...,:2]

        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('faces', faces)
        self.register_buffer('template_verts', verts)
        self.register_buffer('face_index', face_index)
        self.register_buffer('bary_coords', bary_coords)
        self.register_buffer('uvcoords_sample', uvcoords_sample)

        #----------------   sample points in template mesh  ----------------#
        verts_sampling = reweight_verts_by_barycoords(
            verts         = verts.unsqueeze(0),
            faces         = faces,
            face_index    = face_index,
            bary_coords   = bary_coords
        )
        
        cano_init_embeded = self.pts_embedder(verts_sampling)

        self.register_buffer('verts_sampling', verts_sampling)
        self.register_buffer('cano_init_embeded', cano_init_embeded)

    def _register_init_gaussian(self):
        
        self.num_points = self.verts_sampling.shape[1]  # verts_sampling: [1, N, 3]

        #--------- Gaussian attribute initialization ---------#
        init_rgb = inverse_sigmoid(torch.Tensor([0.5, 0.5, 0.5]))[None, ...].float()
        init_rgb = init_rgb.repeat_interleave(self.num_points, dim=0).to(self.device)
        features = torch.zeros((init_rgb.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = init_rgb
        features[:, 3:, 1:] = 0.0
        scales      = self.scale_init[...,None].repeat(self.num_points, 3).to(self.device)
        rots        = torch.zeros((self.num_points, 4), device=self.device)
        rots[:, 0]  = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((self.num_points, 1), dtype=torch.float, device=self.device))

        self._features_dc   = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling       = nn.Parameter(scales.requires_grad_(True))
        self._rotation      = nn.Parameter(rots.requires_grad_(True))
        self._opacity       = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D        = torch.zeros((self.num_points), device=self.device)
        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=self.device)
        self.denom              = torch.zeros((self.num_points, 1), device=self.device)

    def forward(self, input):
        cam_pose = input["cam_pose"].clone()
        fovx = input["fovx"][0]
        fovy = input["fovy"][0]
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]

        camera = Camera(R=R, T=T, FoVx=fovx, FoVy=fovy, img_res=self.img_res)

        flame_pose = input["flame_pose"]
        expression = input["expression"]
        bs = flame_pose.shape[0]    # 1, essentially

        exp_code    = expression
        jaw_pose    = flame_pose[:, 6:9]
        eyes_pose   = flame_pose[:, 9:]

        cond = torch.cat([exp_code, jaw_pose, eyes_pose], dim=1)
        cond = cond.unsqueeze(1).repeat(1, self.num_points, 1)
        embeded_cond = torch.cat([self.cano_init_embeded, cond], dim=2)

        deforms = self.deformNet(embeded_cond)
        deforms = torch.tanh(deforms)

        pos_delta = deforms[..., :3]
        rot_delta = deforms[..., 3:7]
        rot_delta_r = torch.exp(rot_delta[..., 0]).unsqueeze(-1)
        rot_delta_v = rot_delta[..., 1:]
        rot_delta = torch.cat([rot_delta_r, rot_delta_v], dim=-1)
        scale_delta = deforms[..., 7:]
        scale_delta = torch.exp(scale_delta)

        verts, _, _ = self.flame.forward(expression_params      = expression,
                                        full_pose               = flame_pose)
        
        pos_val = reweight_verts_by_barycoords(
            verts         = verts,
            faces         = self.faces,
            face_index    = self.face_index,
            bary_coords   = self.bary_coords
        )
        
        gaussian = GaussianModel(sh_degree = 0)

        render_image = []

        # dummy loop
        for bs_ in range(bs):
            gaussian._features_dc   = self._features_dc
            gaussian._features_rest = self._features_rest
            gaussian._opacity       = self._opacity
            gaussian._scaling       = self._scaling * scale_delta[bs_]
            gaussian._rotation      = self.quatProduct_batch(self._rotation, rot_delta[bs_])
            gaussian._xyz           = pos_val[bs_] + pos_delta[bs_]

            render_out = render(
                camera,
                gaussian,
                self.bg_color.to(self.device),
                device=self.device,
                override_color=None
            )
            
            render_image_ = render_out['render']
            render_image.append(render_image_)

        render_image = torch.stack(render_image)

        output = {
            'rgb_image': render_image,
        }

        return output
    
    @torch.no_grad()
    def visualization(self, input):
        output = self.forward(input)
        return output

    @torch.no_grad()
    def inference(self, expression, flame_pose, camera):
        bs = flame_pose.shape[0]    # 1, essentially

        exp_code    = expression
        jaw_pose    = flame_pose[:, 6:9]
        eyes_pose   = flame_pose[:, 9:]

        cond = torch.cat([exp_code, jaw_pose, eyes_pose], dim=1)
        cond = cond.unsqueeze(1).repeat(1, self.num_points, 1)
        embeded_cond = torch.cat([self.cano_init_embeded, cond], dim=2)

        deforms = self.deformNet(embeded_cond)
        deforms = torch.tanh(deforms)

        pos_delta = deforms[..., :3]
        rot_delta = deforms[..., 3:7]
        rot_delta_r = torch.exp(rot_delta[..., 0]).unsqueeze(-1)
        rot_delta_v = rot_delta[..., 1:]
        rot_delta = torch.cat([rot_delta_r, rot_delta_v], dim=-1)
        scale_delta = deforms[..., 7:]
        scale_delta = torch.exp(scale_delta)

        verts, _, _ = self.flame.forward(expression_params      = expression,
                                        full_pose               = flame_pose)
        
        pos_val = reweight_verts_by_barycoords(
            verts         = verts,
            faces         = self.faces,
            face_index    = self.face_index,
            bary_coords   = self.bary_coords
        )
        
        gaussian = GaussianModel(sh_degree = 0)

        render_image = []

        # dummy loop
        for bs_ in range(bs):
            gaussian._features_dc   = self._features_dc
            gaussian._features_rest = self._features_rest
            gaussian._opacity       = self._opacity
            gaussian._scaling       = self._scaling * scale_delta[bs_]
            gaussian._rotation      = self.quatProduct_batch(self._rotation, rot_delta[bs_])
            gaussian._xyz           = pos_val[bs_] + pos_delta[bs_]

            render_out = render(
                camera,
                gaussian,
                self.bg_color.to(self.device),
                device=self.device,
                override_color=None
            )
            
            render_image_ = render_out['render']
            render_image.append(render_image_)

        render_image = torch.stack(render_image)

        return render_image[0]

    @staticmethod
    def get_init_scale_by_knn(points:torch.Tensor):
        """KNN in PyTorch3D"""
        knn_points_     = points.float().cuda()
        knn             = knn_points(knn_points_, knn_points_, K=6)
        dists           = torch.sqrt(knn.dists[..., 1])
        mean_scaling    = dists.mean()
        max_scaling     = 10 * mean_scaling
        scale_init      = torch.log(mean_scaling).cpu()

        del knn_points_

        return mean_scaling, max_scaling, scale_init
    
    @staticmethod
    def quatProduct_batch(q1, q2):
        r1 = q1[:,0] # [B]
        r2 = q2[:,0]
        v1 = torch.stack((q1[:,1], q1[:,2], q1[:,3]), dim=-1) #[B,3]
        v2 = torch.stack((q2[:,1], q2[:,2], q2[:,3]), dim=-1)

        r = r1 * r2 - torch.sum(v1*v2, dim=1) # [B]
        v = r1.unsqueeze(1) * v2 + r2.unsqueeze(1) * v1 + torch.cross(v1, v2) #[B,3]
        q = torch.stack((r, v[:,0], v[:,1], v[:,2]), dim=1)

        return q

#-------------------------------------------------------------------------------#
                    # utils functions in FlashAvatar # 
#-------------------------------------------------------------------------------#

class Embedder(nn.Module):
    def __init__(
            self,
            N_freqs,
            input_dims = 3,
            include_input = True
        ):
        super().__init__()
        self.log_sampling = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq = N_freqs - 1
        self.N_freqs = N_freqs
        self.include_input = include_input
        self.input_dims = input_dims
        embed_fns = []
        if self.include_input:
            embed_fns.append(lambda x: x)

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
        self.embed_fns = embed_fns
        self.dim_embeded = self.input_dims*len(self.embed_fns)

    def forward(self, inputs):
        output = torch.cat([fn(inputs) for fn in self.embed_fns], 2)
        return output
    
#-------------------------------------------------------------------------------#

class MLP(nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            hidden_dim=256, 
            hidden_layers=8
        ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers-1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        # input: B,V,d
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size*N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs):
            h = self.fcs[i](h)
            h = F.relu(h)
        output = self.output_linear(h)
        output = output.reshape(batch_size, N_v, -1)

        return output
