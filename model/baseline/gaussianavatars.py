import numpy as np
import torch
import torch.nn as nn

from flame.FLAME import FLAME

from pytorch3d.transforms import (matrix_to_quaternion, quaternion_multiply)

from tools.util import get_bg_color
from tools.gs_utils.general_utils import inverse_sigmoid, build_rotation

from volume_rendering.camera_3dgs       import Camera
from volume_rendering.mesh_compute      import compute_face_orientation
from volume_rendering.gaussian_model    import GaussianModel
from volume_rendering.render_3dgs       import render

#-------------------------------------------------------------------------------#

class GaussianAvatars(nn.Module):
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
            https://github.com/ShenhanQian/GaussianAvatars

        bibtex:
            @inproceedings{qian2024gaussianavatars,
                title       = {Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians},
                author      = {Qian, Shenhan and Kirschstein, Tobias and Schoneveld, Liam and Davoli, Davide and Giebenhain, Simon and Nie{\ss}ner, Matthias},
                booktitle   = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
                pages       = {20299--20309},
                year        = {2024}
            }
        """
        
        self.max_sh_degree      = cfg_model.sh_degree
        self.active_sh_degree   = 0
        self.percent_dense      = 0.01

        self.device = device

        self.bg_color   = get_bg_color(background_color).to(self.device)

        self.img_res    = img_res

        self.cfg_model  = cfg_model

        self.shape_params   = shape_params
        self.canonical_expression   = canonical_expression
        self.canonical_pose = canonical_pose

        self._register_flame(flame_path='./weights/generic_model.pkl',
                             landmark_embedding_path='./weights/landmark_embedding.npy')
        
        # binding mechanism in GaussianAvatars
        binding    = torch.arange(len(self.flame.faces_tensor)).to(self.device)
        binding_counter    = torch.ones(len(self.flame.faces_tensor), dtype=torch.int32).to(self.device)

        self.register_buffer('binding', binding)
        self.register_buffer('binding_counter', binding_counter)

        self._register_init_gaussian()

    def _register_flame(self, flame_path, landmark_embedding_path):

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

    def _register_init_gaussian(self):

        self.num_points = self.binding.shape[0]
        fused_point_cloud   = torch.zeros((self.num_points, 3)).float().to(self.device)
        fused_color =   torch.tensor(np.random.random((self.num_points, 3)) / 255.0).float().to(self.device)

        features    = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0]  = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation: ", self.num_points)

        scales  = torch.log(torch.ones((self.num_points, 3), device=self.device))
        rots    = torch.zeros((self.num_points, 4), device=self.device)
        rots[:, 0]  = 1

        opacities   = inverse_sigmoid(0.1 * torch.ones((self.num_points, 1), dtype=torch.float, device=self.device))

        self._xyz           = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc   = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling       = nn.Parameter(scales.requires_grad_(True))
        self._rotation      = nn.Parameter(rots.requires_grad_(True))
        self._opacity       = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D        = torch.zeros((self.num_points)).to(self.device)
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

        verts, _, _ = self.flame.forward(expression_params      = expression,
                                        full_pose               = flame_pose)
        
        faces = self.flame.faces_tensor
        triangles = verts[:, faces]

        face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        face_orien_mat, face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)

        # keep the 'face_scaling' for proper split densification
        self.face_scaling   = face_scaling

        face_orien_quat     = matrix_to_quaternion(face_orien_mat)
        flame_orien_quat    = torch.nn.functional.normalize(face_orien_quat[self.binding])

        gaussian = GaussianModel(sh_degree = self.active_sh_degree)

        render_image = []
        viewspace_points = []
        visibility_filter = []
        radii = []

        # dummy loop
        for bs_ in range(bs):
            gaussian._features_dc   = self._features_dc
            gaussian._features_rest = self._features_rest
            gaussian._opacity      = self._opacity
            gaussian._scaling      = self._scaling + torch.log(face_scaling[self.binding])
            gaussian._rotation     = quaternion_multiply(flame_orien_quat, self._rotation)
            gaussian._xyz          = torch.bmm(face_orien_mat[self.binding], self._xyz[..., None]).squeeze(-1) * face_scaling[self.binding] + face_center[self.binding]

            render_out = render(
                camera,
                gaussian,
                self.bg_color.to(self.device),
                device=self.device,
                override_color=None
            )
            
            render_image_ = render_out['render']
            viewspace_points_ = render_out['viewspace_points']
            visibility_filter_ = render_out['visibility_filter']
            radii_ = render_out['radii']

            render_image.append(render_image_)
            viewspace_points.append(viewspace_points_)
            visibility_filter.append(visibility_filter_)
            radii.append(radii_)

        
        render_image = torch.stack(render_image)

        output = {
            'rgb_image': render_image,
            'scale': torch.exp(self._scaling),
            'xyz':  self._xyz,
            # ----- gaussian maintainer ----- #
            'viewspace_points': viewspace_points,   # List
            'visibility_filter': visibility_filter, # List
            'radii': radii, # List
            'bs': bs,
        }

        return output
    
    @torch.no_grad()
    def visualization(self, input):
        output = self.forward(input)
        return output
    
    @torch.no_grad()
    def inference(self, expression, flame_pose, camera):

        bs = flame_pose.shape[0]    # 1, essentially

        verts, _, _ = self.flame.forward(expression_params      = expression,
                                        full_pose               = flame_pose)
        
        faces = self.flame.faces_tensor
        triangles = verts[:, faces]

        face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        face_orien_mat, face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)

        # keep the 'face_scaling' for proper split densification
        self.face_scaling   = face_scaling

        face_orien_quat     = matrix_to_quaternion(face_orien_mat)
        flame_orien_quat    = torch.nn.functional.normalize(face_orien_quat[self.binding])

        gaussian = GaussianModel(sh_degree = self.active_sh_degree)

        render_image = []
        viewspace_points = []
        visibility_filter = []
        radii = []

        # dummy loop
        for bs_ in range(bs):
            gaussian._features_dc   = self._features_dc
            gaussian._features_rest = self._features_rest
            gaussian._opacity      = self._opacity
            gaussian._scaling      = self._scaling + torch.log(face_scaling[self.binding])
            gaussian._rotation     = quaternion_multiply(flame_orien_quat, self._rotation)
            gaussian._xyz          = torch.bmm(face_orien_mat[self.binding], self._xyz[..., None]).squeeze(-1) * face_scaling[self.binding] + face_center[self.binding]

            render_out = render(
                camera,
                gaussian,
                self.bg_color.to(self.device),
                device=self.device,
                override_color=None
            )
            
            render_image_ = render_out['render']
            viewspace_points_ = render_out['viewspace_points']
            visibility_filter_ = render_out['visibility_filter']
            radii_ = render_out['radii']

            render_image.append(render_image_)
            viewspace_points.append(viewspace_points_)
            visibility_filter.append(visibility_filter_)
            radii.append(radii_)

        
        render_image = torch.stack(render_image)

        return render_image[0]
        
    def _add_densification_stats(self, viewspace_point_tensor, update_filter):

        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def _densify_and_prune(self, gs_optimizer: torch.optim.Adam, 
                           max_grad, min_opacity, extent, max_screen_size):

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self._clone_densify(gs_optimizer, grads, max_grad, extent)
        self._split_densify(gs_optimizer, grads, max_grad, extent)

        prune_mask  = (torch.sigmoid(self._opacity) < min_opacity).squeeze()
        if max_screen_size:
            big_point_vs = self.max_radii2D > max_screen_size
            big_point_ws = torch.exp(self._scaling).max(dim=1).values > 0.1 * extent
            prune_mask   = torch.logical_or(torch.logical_or(prune_mask, big_point_vs), big_point_ws)

        self._prune(gs_optimizer, prune_mask)

        torch.cuda.empty_cache()

    def _clone_densify(self, gs_optimizer: torch.optim.Adam,
                        grads, max_grad, extent):

        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= max_grad, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(torch.exp(self._scaling), dim=1).values <= self.percent_dense * extent
        )

        new_xyz             = self._xyz[selected_pts_mask]
        new_scaling         = self._scaling[selected_pts_mask]
        new_rotation        = self._rotation[selected_pts_mask]
        new_features_dc     = self._features_dc[selected_pts_mask]
        new_features_rest   = self._features_rest[selected_pts_mask]
        new_opacity         = self._opacity[selected_pts_mask]

        new_binding         = self.binding[selected_pts_mask]
        self.binding        = torch.cat([self.binding, new_binding])
        self.binding_counter.scatter_add_(0, new_binding, torch.ones_like(new_binding, dtype=torch.int32, device=self.device))

        gs_clone_dict = {
            '_xyz':              new_xyz,
            '_scaling':          new_scaling,
            '_rotation':         new_rotation,
            '_features_dc':      new_features_dc,
            '_features_rest':    new_features_rest,
            '_opacity':          new_opacity,
        }

        optimizable_tensors = {}
        for group in gs_optimizer.param_groups:
            # assert len(group['params']) == 1
            if len(group['params']) != 1:
                continue

            extension_tensor = gs_clone_dict[group['name']]
            if extension_tensor is None:
                continue

            dd = 1 if group['name'] == 'xyz_comp' else 0
            stored_state = gs_optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(extension_tensor)), dim=dd)
                stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor)), dim=dd)

                del gs_optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=dd).requires_grad_(True))
                gs_optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=dd).requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]

        self._densification_postfix(optimizable_tensors)

    def _split_densify(self, gs_optimizer: torch.optim.Adam,
                        grads, max_grad, extent, N=2):
        n_init_points = self.num_points
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= max_grad, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(torch.exp(self._scaling), dim=1).values > self.percent_dense * extent)

        stds = torch.exp(self._scaling)[selected_pts_mask].repeat(N, 1)
        means =torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean = means, std = stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = torch.log((torch.exp(self._scaling[selected_pts_mask])).repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_binding = self.binding[selected_pts_mask].repeat(N)

        self.binding = torch.cat((self.binding, new_binding))
        self.binding_counter.scatter_add_(0, new_binding, torch.ones_like(new_binding, dtype=torch.int32, device=self.device))

        gs_split_dict = {
            '_xyz':              new_xyz,
            '_scaling':          new_scaling,
            '_rotation':         new_rotation,
            '_features_dc':      new_features_dc,
            '_features_rest':    new_features_rest,
            '_opacity':          new_opacity,
        }

        optimizable_tensors = {}
        for group in gs_optimizer.param_groups:
            # assert len(group['params']) == 1
            if len(group['params']) != 1:
                continue

            extension_tensor = gs_split_dict[group['name']]
            if extension_tensor is None:
                continue

            dd = 1 if group['name'] == 'xyz_comp' else 0
            stored_state = gs_optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(extension_tensor)), dim=dd)
                stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor)), dim=dd)

                del gs_optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=dd).requires_grad_(True))
                gs_optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=dd).requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]

        self._densification_postfix(optimizable_tensors)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self._prune(gs_optimizer, prune_filter)

    def _prune(self, gs_optimizer: torch.optim.Adam, mask):

        binding_to_prune = self.binding[mask]
        counter_prune = torch.zeros_like(self.binding_counter)
        counter_prune.scatter_add_(0, binding_to_prune, torch.ones_like(binding_to_prune, dtype=torch.int32, device="cuda"))
        mask_redundant = (self.binding_counter - counter_prune) > 0
        mask[mask.clone()] = mask_redundant[binding_to_prune]

        valid_points_mask   = ~mask        

        optimizable_tensors = {}
        for group in gs_optimizer.param_groups:
            # rule out parameters that are not properties of gaussians
            if len(group["params"]) != 1 or group["params"][0].shape[0] != valid_points_mask.shape[0]:
                continue

            stored_state = gs_optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_points_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_points_mask]

                del gs_optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][valid_points_mask].requires_grad_(True)))
                gs_optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][valid_points_mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz           = optimizable_tensors["_xyz"]
        self._features_dc   = optimizable_tensors["_features_dc"]
        self._features_rest = optimizable_tensors["_features_rest"]
        self._opacity       = optimizable_tensors["_opacity"]
        self._scaling       = optimizable_tensors["_scaling"]
        self._rotation      = optimizable_tensors["_rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom              = self.denom[valid_points_mask]
        self.max_radii2D        = self.max_radii2D[valid_points_mask]

        self.binding_counter.scatter_add_(0, self.binding[mask], -torch.ones_like(self.binding[mask], dtype=torch.int32, device="cuda"))
        self.binding = self.binding[valid_points_mask]

    def _densification_postfix(self, optimizable_tensors):
        self._xyz           = optimizable_tensors["_xyz"]
        self._features_dc   = optimizable_tensors["_features_dc"]
        self._features_rest = optimizable_tensors["_features_rest"]
        self._opacity       = optimizable_tensors["_opacity"]
        self._scaling       = optimizable_tensors["_scaling"]
        self._rotation      = optimizable_tensors["_rotation"]

        self.num_points     = self._xyz.shape[0]

        # stats
        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=self.device)
        self.denom = torch.zeros((self.num_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((self.num_points), device=self.device)

    def _reset_opacity(self, gs_optimizer):
        opacities_new = inverse_sigmoid(torch.min(torch.sigmoid(self._opacity), torch.ones_like(self._opacity)*0.01))
        tensor = opacities_new
        name = '_opacity'

        optimizable_tensors = {}
        for group in gs_optimizer.param_groups:
            if group['name'] == name:
                stored_state = gs_optimizer.state.get(group['params'][0], None)
                stored_state['exp_avg'] = torch.zeros_like(tensor)
                stored_state['exp_avg_sq'] = torch.zeros_like(tensor)

                del gs_optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(tensor.requires_grad_(True))
                gs_optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group['name']] = group['params'][0]

        self._opacity = optimizable_tensors['_opacity']

    def _update_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1