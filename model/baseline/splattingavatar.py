import igl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flame.FLAME import FLAME

from simple_knn._C      import distCUDA2
from simple_phongsurf   import PhongSurfacePy3d

from pytorch3d.transforms import quaternion_multiply
import pytorch3d.structures.meshes as py3d_meshes

from volume_rendering.camera_3dgs       import Camera
from volume_rendering.gaussian_model    import GaussianModel
from volume_rendering.render_3dgs       import render

from tools.gs_utils.general_utils import inverse_sigmoid, build_rotation
from tools.gs_utils.sh_utils import RGB2SH

from tools.util import get_bg_color

#-------------------------------------------------------------------------------#

class SplattingAvatar(nn.Module):
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
        """
        official code:
            https://github.com/initialneil/SplattingAvatar

        bibtex:
            @inproceedings{shao2024splattingavatar,
                title       = {{SplattingAvatar: Realistic Real-Time Human Avatars with Mesh-Embedded Gaussian Splatting}},
                author      = {Shao, Zhijing and Wang, Zhaolong and Li, Zhuang and Wang, Duotun and Lin, Xiangru and Zhang, Yu and Fan, Mingming and Wang, Zeyu},
                booktitle   = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
                year        = {2024}
            }
        """
        super().__init__()

        self.max_sh_degree = 0
        self.percent_dense = 0.01

        self.device = device

        self.bg_color = get_bg_color(background_color).to(self.device)

        self.img_res = img_res

        self.cfg_model = cfg_model

        self.shape_params           = shape_params
        self.canonical_expression   = canonical_expression
        self.canonical_pose         = canonical_pose

        self._register_flame(flame_path='./weights/generic_model.pkl',
                             landmark_embedding_path='./weights/landmark_embedding.npy')
        
        self.mesh_py3d = py3d_meshes.Meshes(self.flame.v_template[None, ...].float(), # torch.Size([1, 5023, 3])
                                            self.flame.faces_tensor[None, ...].int()) # torch.Size([1, 9976, 3])
        
        flame_mesh = self.mesh_py3d.update_padded(self.canonical_verts)
        
        cano_mesh = {}
        cano_mesh['mesh_verts'] = flame_mesh.verts_packed()
        cano_mesh['mesh_norms'] = flame_mesh.verts_normals_packed()
        cano_mesh['mesh_faces'] = flame_mesh.faces_padded().squeeze(0)

        self._register_canonical(cano_mesh = cano_mesh)

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

    def _register_canonical(self, cano_mesh):
        cano_verts = cano_mesh['mesh_verts'].float().to(self.device)
        cano_norms = cano_mesh['mesh_norms'].float().to(self.device)
        cano_faces = cano_mesh['mesh_faces'].long().to(self.device)

        self.cano_verts = cano_verts
        self.cano_norms = cano_norms
        self.cano_faces = cano_faces

        # quaternion from cano to pose
        self.quat_helper = PerVertQuaternion(cano_verts, cano_faces).to(self.device)

        # phong surface for triangle walk
        self.phongsurf = PhongSurfacePy3d(cano_verts, cano_faces, cano_norms,
                                          outer_loop=2, inner_loop=50, method='uvd').to(self.device)

        self.mesh_verts = self.cano_verts
        self.mesh_norms = self.cano_norms
        
        # sample on mesh
        num_samples = self.cfg_model.num_init_samples # 10000
        sample_fidxs, sample_bary = sample_bary_on_triangles(cano_faces.shape[0], num_samples)
        sample_fidxs = sample_fidxs.to(self.device)
        sample_bary = sample_bary.to(self.device)

        self.register_buffer('sample_fidxs', sample_fidxs)
        self.register_buffer('sample_bary', sample_bary)

        sample_verts = retrieve_verts_barycentric(cano_verts, cano_faces, self.sample_fidxs, self.sample_bary)
        sample_norms = retrieve_verts_barycentric(cano_norms, cano_faces, self.sample_fidxs, self.sample_bary)
        sample_norms = F.normalize(sample_norms, dim=-1)

        self.sample_verts = sample_verts
        self.sample_norms = sample_norms


    def _register_init_gaussian(self):

        points = self.sample_verts.detach().cpu().numpy()
        normals = self.sample_norms.detach().cpu().numpy()
        colors = torch.full_like(self.sample_verts, 0.5).float().cpu()

        self.num_points = points.shape[0]

        fused_point_cloud = torch.tensor(np.asarray(points)).float().to(self.device) # torch.Size([10000, 3])
        fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().to(self.device)) # torch.Size([10000, 3])
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device) # torch.Size([10000, 3, 1])
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(points)).float().to(self.device)), 0.0000001) # torch.Size([10000])
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        # in SplattingAvatar, Gaussians are optimized through uvd represention.
        self._uvd = nn.Parameter(torch.zeros_like(fused_point_cloud))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous())
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D        = torch.zeros((self.num_points), device=self.device)
        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=self.device)
        self.denom              = torch.zeros((self.num_points, 1), device=self.device)

        self.active_sh_degree = self.max_sh_degree


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
        
        cur_flame_mesh = self.mesh_py3d.update_padded(verts)
        cur_mesh = {}
        cur_mesh['mesh_verts'] = cur_flame_mesh.verts_packed()
        cur_mesh['mesh_norms'] = cur_flame_mesh.verts_normals_packed()
        cur_mesh['mesh_faces'] = cur_flame_mesh.faces_padded().squeeze(0)
        
        self.mesh_verts = cur_mesh['mesh_verts'].float().to(self.device)    # torch.Size([5023, 3])
        self.mesh_norms = cur_mesh['mesh_norms'].float().to(self.device)    # torch.Size([5023, 3])

        self.per_vert_quat = self.quat_helper(self.mesh_verts)  # torch.Size([5023, 4])
        self.tri_quats = self.per_vert_quat[self.cano_faces]    # torch.Size([9976, 3, 4])

        self._face_scaling = self.quat_helper.calc_face_area_change(self.mesh_verts)

        gaussian = GaussianModel(sh_degree = 0)

        render_image = []
        viewspace_points = []
        visibility_filter = []
        radii = []

        base_xyz = retrieve_verts_barycentric(self.mesh_verts, self.cano_faces,
                                              self.sample_fidxs, self.sample_bary)
        
        base_normal = F.normalize(retrieve_verts_barycentric(self.mesh_norms, self.cano_faces, 
                                                        self.sample_fidxs, self.sample_bary), 
                                                        dim=-1)
        
        base_quat = torch.einsum('bij,bi->bj', self.tri_quats[self.sample_fidxs], self.sample_bary)

        scaling_alter = self._face_scaling[self.sample_fidxs]
        
        # dummy loop
        for bs_ in range(bs):
            gaussian._features_dc   = self._features_dc
            gaussian._features_rest = self._features_rest
            gaussian._opacity       = self._opacity
            gaussian._scaling       = self._scaling * scaling_alter
            gaussian._rotation      = quaternion_multiply(base_quat, self._rotation)
            gaussian._xyz           = base_xyz + base_normal * self._uvd[..., -1:]

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
        
        cur_flame_mesh = self.mesh_py3d.update_padded(verts)
        cur_mesh = {}
        cur_mesh['mesh_verts'] = cur_flame_mesh.verts_packed()
        cur_mesh['mesh_norms'] = cur_flame_mesh.verts_normals_packed()
        cur_mesh['mesh_faces'] = cur_flame_mesh.faces_padded().squeeze(0)
        
        self.mesh_verts = cur_mesh['mesh_verts'].float().to(self.device)    # torch.Size([5023, 3])
        self.mesh_norms = cur_mesh['mesh_norms'].float().to(self.device)    # torch.Size([5023, 3])

        self.per_vert_quat = self.quat_helper(self.mesh_verts)  # torch.Size([5023, 4])
        self.tri_quats = self.per_vert_quat[self.cano_faces]    # torch.Size([9976, 3, 4])

        self._face_scaling = self.quat_helper.calc_face_area_change(self.mesh_verts)

        gaussian = GaussianModel(sh_degree = 0)

        render_image = []
        viewspace_points = []
        visibility_filter = []
        radii = []

        base_xyz = retrieve_verts_barycentric(self.mesh_verts, self.cano_faces,
                                              self.sample_fidxs, self.sample_bary)
        
        base_normal = F.normalize(retrieve_verts_barycentric(self.mesh_norms, self.cano_faces, 
                                                        self.sample_fidxs, self.sample_bary), 
                                                        dim=-1)
        
        base_quat = torch.einsum('bij,bi->bj', self.tri_quats[self.sample_fidxs], self.sample_bary)

        scaling_alter = self._face_scaling[self.sample_fidxs]
        
        # dummy loop
        for bs_ in range(bs):
            gaussian._features_dc   = self._features_dc
            gaussian._features_rest = self._features_rest
            gaussian._opacity       = self._opacity
            gaussian._scaling       = self._scaling * scaling_alter
            gaussian._rotation      = quaternion_multiply(base_quat, self._rotation)
            gaussian._xyz           = base_xyz + base_normal * self._uvd[..., -1:]

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
            # ----- gaussian maintainer ----- #
            'viewspace_points': viewspace_points,   # List
            'visibility_filter': visibility_filter, # List
            'radii': radii, # List
            'bs': bs,
        }

        return output
    
    def _add_densification_stats(self, viewspace_point_tensor, update_filter):

        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def _densify_and_prune(self, gs_optimizer: torch.optim.Adam, 
                           max_grad, min_opacity, extent, max_screen_size):

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self._clone_densify(gs_optimizer, grads, max_grad, extent)
        self._split_densify(gs_optimizer, grads, max_grad, extent)

        opacity = torch.sigmoid(self._opacity)

        prune_mask = (opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = torch.exp(self._scaling).max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self._prune(gs_optimizer, prune_mask)


    def _clone_densify(self, gs_optimizer, grads, max_grad, extent):

        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= max_grad, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(torch.exp(self._scaling), dim=1).values <= self.percent_dense * extent
        )

        new_uvd             = self._uvd[selected_pts_mask]
        new_scaling         = self._scaling[selected_pts_mask]
        new_rotation        = self._rotation[selected_pts_mask]
        new_features_dc     = self._features_dc[selected_pts_mask]
        new_features_rest   = self._features_rest[selected_pts_mask]
        new_opacity         = self._opacity[selected_pts_mask]

        new_sample_fidxs    = self.sample_fidxs[selected_pts_mask]
        new_sample_bary     = self.sample_bary[selected_pts_mask]

        densify_out = {
            'new_uvd':              new_uvd,
            'new_scaling':          new_scaling,
            'new_rotation':         new_rotation,
            'new_features_dc':      new_features_dc,
            'new_features_rest':    new_features_rest,
            'new_opacity':          new_opacity,
            'new_sample_fidxs':     new_sample_fidxs,
            'new_sample_bary':      new_sample_bary
        }

        gs_clone_dict = {
            '_uvd':              new_uvd,
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

        self._densification_postfix(optimizable_tensors, densify_out)


    def _split_densify(self, gs_optimizer, grads, max_grad, extent, N=2):
    
        n_init_points = self.num_points
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= max_grad, True, False)

        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(torch.exp(self._scaling), dim=1).values > self.percent_dense * extent)

        stds    = torch.exp(self._scaling)[selected_pts_mask].repeat(N, 1)
        means   = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots    = build_rotation(torch.nn.functional.normalize(self._rotation[selected_pts_mask])).repeat(N, 1, 1)

        base_xyz = retrieve_verts_barycentric(self.mesh_verts, self.cano_faces, 
                                          self.sample_fidxs, self.sample_bary)

        base_normal_cano = torch.nn.functional.normalize(retrieve_verts_barycentric(self.cano_norms, self.cano_faces, 
                                                        self.sample_fidxs, self.sample_bary),
                                                        dim=-1)

        xyz_cano   = base_xyz + base_normal_cano * self._uvd[..., -1:]
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz_cano[selected_pts_mask].repeat(N, 1)

        new_scaling = torch.log(
            torch.exp(self._scaling)[selected_pts_mask].repeat(N, 1) / (0.8 * N)
            )
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)

        # fit uvd to new_xyz
        # find new embedding point
        fidx    = self.sample_fidxs[selected_pts_mask].repeat(N)
        uv      = self.sample_bary[selected_pts_mask, :2].repeat(N, 1)
        d       = self._uvd[selected_pts_mask, -1:].repeat(N, 1)

        fidx, uv = self.phongsurf.update_corres_spt(new_xyz.detach(), None, fidx, uv)

        bary = torch.concat([uv, 1.0 - uv[:, 0:1] - uv[:, 1:2]], dim=-1)
        new_uvd = torch.concat([torch.zeros_like(uv), d], dim=-1)
    
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        split_out = {
            'new_uvd':              new_uvd,
            'new_scaling':          new_scaling,
            'new_rotation':         new_rotation,
            'new_features_dc':      new_features_dc,
            'new_features_rest':    new_features_rest,
            'new_opacity':          new_opacity,
            'new_sample_fidxs':     fidx,
            'new_sample_bary':      bary
        }

        gs_split_dict = {
            '_uvd':              new_uvd,
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

        self._densification_postfix(optimizable_tensors, split_out)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device='cuda', dtype=bool)))
        self._prune(gs_optimizer, prune_filter)


    def _densification_postfix(self, optimizable_tensors, densify_out):
        self._uvd = optimizable_tensors.get('_uvd', self._uvd)

        if '_scaling' in optimizable_tensors:
            self._scaling = optimizable_tensors['_scaling']
        else:
            self._scaling = torch.cat([self._scaling, densify_out['new_scaling']], dim=0)

        if '_rotation' in optimizable_tensors:
            self._rotation = optimizable_tensors['_rotation']
        else:
            self._rotation = torch.cat([self._rotation, densify_out['new_rotation']], dim=0)

        self._opacity = optimizable_tensors.get('_opacity', self._opacity)
        self._features_dc = optimizable_tensors.get('_features_dc', self._features_dc)
        self._features_rest = optimizable_tensors.get('_features_rest', self._features_rest)

        # mesh embedding
        self.sample_fidxs = torch.cat([self.sample_fidxs, densify_out['new_sample_fidxs']], dim=0)
        self.sample_bary = torch.cat([self.sample_bary, densify_out['new_sample_bary']], dim=0)

        self.num_points = self.sample_bary.shape[0]

        # stats
        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=self.device)
        self.denom = torch.zeros((self.num_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((self.num_points), device=self.device)


    def _prune(self, gs_optimizer, mask):

        valid_points_mask = ~mask

        optimizable_tensors = {}
        for group in gs_optimizer.param_groups:
            if len(group['params']) != 1:
                continue

            stored_state = gs_optimizer.state.get(group['params'][0], None)

            if group['name'] != 'xyz_comp':
                if stored_state is not None:
                    stored_state['exp_avg'] = stored_state['exp_avg'][valid_points_mask]
                    stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][valid_points_mask]

                    del gs_optimizer.state[group['params'][0]]
                    group['params'][0] = nn.Parameter((group['params'][0][valid_points_mask].requires_grad_(True)))
                    gs_optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group['name']] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][valid_points_mask].requires_grad_(True))
                    optimizable_tensors[group['name']] = group['params'][0]
            else:
                if stored_state is not None:
                    stored_state['exp_avg'] = stored_state['exp_avg'][:, valid_points_mask]
                    stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][:, valid_points_mask]

                    del gs_optimizer.state[group['params'][0]]
                    group['params'][0] = nn.Parameter((group['params'][0][:, valid_points_mask].requires_grad_(True)))
                    gs_optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group['name']] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][:, valid_points_mask].requires_grad_(True))
                    optimizable_tensors[group['name']] = group['params'][0]

        self._uvd = optimizable_tensors['_uvd']

        if '_scaling' in optimizable_tensors:
            self._scaling = optimizable_tensors['_scaling']
        else:
            self._scaling = self._scaling[valid_points_mask]

        if '_rotation' in optimizable_tensors:
            self._rotation = optimizable_tensors['_rotation']
        else:
            self._rotation = self._rotation[valid_points_mask]

        self._opacity = optimizable_tensors.get('_opacity', self._opacity)
        self._features_dc = optimizable_tensors.get('_features_dc', self._features_dc)
        self._features_rest = optimizable_tensors.get('_features_rest', self._features_rest)

        self.sample_fidxs = self.sample_fidxs[valid_points_mask]
        self.sample_bary = self.sample_bary[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def _walking_on_triangles(self, gs_optimizer):

        fidx = self.sample_fidxs.detach().cpu().numpy().astype(np.int32)
        uv = self.sample_bary[..., :2].detach().cpu().numpy().astype(np.double)
        delta = self._uvd[..., :2].detach().cpu().numpy().astype(np.double)
        fidx, uv = self.phongsurf.triwalk.updateSurfacePoints(fidx, uv, delta)

        self.sample_fidxs = torch.tensor(fidx).long().to(self.device)
        self.sample_bary[..., :2] = torch.tensor(uv).float().to(self.device)
        self.sample_bary[..., 2] = 1.0 - self.sample_bary[..., 0] - self.sample_bary[..., 1]

        for group in gs_optimizer.param_groups:
            if group["name"] == "_uvd":
                assert len(group["params"]) == 1
                stored_state = gs_optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"][..., :2] = torch.zeros_like(stored_state["exp_avg"][..., :2])
                    stored_state["exp_avg_sq"][..., :2] = torch.zeros_like(stored_state["exp_avg_sq"][..., :2])

                    del gs_optimizer.state[group['params'][0]]
                    _uvd = group["params"][0]
                    group['params'][0] = nn.Parameter(torch.cat((torch.zeros_like(_uvd[..., :2]), 
                                                                 _uvd[..., 2:]), dim=-1).requires_grad_(True))
                    gs_optimizer.state[group['params'][0]] = stored_state
                else:
                    _uvd = group["params"][0]
                    group['params'][0] = nn.Parameter(torch.cat((torch.zeros_like(_uvd[..., :2]), 
                                                                 _uvd[..., 2:]), dim=-1).requires_grad_(True))
                    
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





#-------------------------------------------------------------------------#
                # utils functions in SplattingAvatar #
#-------------------------------------------------------------------------#

def sample_bary_on_triangles(num_faces, num_samples):
    sample_bary = torch.zeros(num_samples, 3)
    sample_bary[:, 0] = torch.rand(num_samples)
    sample_bary[:, 1] = torch.rand(num_samples) * (1.0 - sample_bary[:, 0])
    sample_bary[:, 2] = 1.0 - sample_bary[:, 0] - sample_bary[:, 1]
    sample_fidxs = torch.randint(0, num_faces, size=(num_samples,))

    # shuffle bary
    indices = torch.argsort(torch.rand_like(sample_bary), dim=-1)
    sample_bary = torch.gather(sample_bary, dim=-1, index=indices)

    return sample_fidxs, sample_bary

#-------------------------------------------------------------------------------#

def retrieve_verts_barycentric(vertices, faces, fidxs, barys):
    triangle_verts = vertices[faces].float() # torch.Size([9976, 3, 3])
    # fidxs.shape = torch.Size([10000])
    # barys.shape = torch.Size([10000, 3])
    if len(triangle_verts.shape) == 3:
        # triangle_verts[fidxs].shape = torch.Size([10000, 3, 3])
        sample_verts = torch.einsum('nij,ni->nj', triangle_verts[fidxs], barys)
    elif len(triangle_verts.shape) == 4:
        sample_verts = torch.einsum('bnij,ni->bnj', triangle_verts[:, fidxs, ...], barys)
    else:
        raise NotImplementedError
    
    return sample_verts

#-------------------------------------------------------------------------------#

def tbn(triangles):
    a, b, c = triangles.unbind(-2)
    n = F.normalize(torch.cross(b - a, c - a), dim=-1)
    d = b - a

    X = F.normalize(torch.cross(d, n), dim=-1)
    Y = F.normalize(torch.cross(d, X), dim=-1)
    Z = F.normalize(d, dim=-1)

    return torch.stack([X, Y, Z], dim=3)

#-------------------------------------------------------------------------------#

def triangle2projection(triangles):
    R = tbn(triangles)
    T = triangles.unbind(-2)[0]
    I = torch.repeat_interleave(torch.eye(4, device=triangles.device)[None, None, ...], R.shape[1], 1)

    I[:, :, 0:3, 0:3] = R
    I[:, :, 0:3, 3] = T

    return I

#-------------------------------------------------------------------------------#

def calc_face_areas(mesh_verts, mesh_faces):
    vertices_faces = mesh_verts[mesh_faces]

    faces_normals = torch.cross(
        vertices_faces[:, 2] - vertices_faces[:, 1],
        vertices_faces[:, 0] - vertices_faces[:, 1],
        dim=1,
    )

    face_areas = faces_normals.norm(dim=-1, keepdim=True) / 2.0
    return face_areas

#-------------------------------------------------------------------------------#

def calc_per_face_Rt(cano_triangles, deform_triangles):
    # c2w for triangles
    cano_Rt = triangle2projection(cano_triangles)[0]
    deform_Rt = triangle2projection(deform_triangles)[0]

    # for X_c in cano
    # X_d = deform_Rt @ cano_Rt.inv() @ X_c
    return torch.einsum('bij,bjk->bik', deform_Rt, torch.inverse(cano_Rt))

#-------------------------------------------------------------------------------#

def rotation_matrix_to_quaternion(R):
    # tr = torch.eye(3)[None,...].repeat(R.shape[0], 1, 1).to(R)
    # w = torch.pow(1 + (tr * R).sum(-1).sum(-1), 0.5)/2
    # x = (R[:, 2, 1] - R[:, 1, 2])/4/w
    # y = (R[:, 0, 2] - R[:, 2, 0])/4/w
    # z = (R[:, 1, 0] - R[:, 0, 1])/4/w
    # quat = torch.stack([w, x, y, z], dim=-1)
    # return quat
    from pytorch3d import transforms as tfs
    return tfs.matrix_to_quaternion(R)

#-------------------------------------------------------------------------------#

class PerVertQuaternion(nn.Module):
    def __init__(self, cano_verts, cano_faces, use_numpy=False):
        super().__init__()
        self.use_numpy = use_numpy
        self.prepare_cano_per_vert(cano_verts, cano_faces)

    def prepare_cano_per_vert(self, cano_verts, cano_faces):
        self.register_buffer('cano_verts', cano_verts)
        self.register_buffer('cano_faces', cano_faces)
        self.register_buffer('cano_triangles', cano_verts[cano_faces].unsqueeze(dim=0))

        if self.use_numpy:
            # https://github.com/libigl/libigl/blob/3cf08b7f681ed0e170d16d7a2efea61c3084be78/include/igl/per_vertex_normals.cpp#L61C8-L61C8
            A = igl.doublearea(cano_verts.detach().cpu().numpy(), cano_faces.detach().cpu().numpy()) # 计算每个三角形的面积
            W = torch.from_numpy(A)[:, None].float()

            per_vert_w_sum = torch.zeros(cano_verts.shape[0])
            per_vert_w_sum = per_vert_w_sum.scatter_(0, cano_faces.view(-1), W.repeat([1, 3]).view(-1), reduce='add')
            # W.repeat([1, 3]).view(-1): 表示每个顶点的权重值
            # cano_faces.view(-1): 表述每个权重值应该分到哪个顶点

            self.register_buffer('W', W)
            self.register_buffer('per_vert_w_sum', per_vert_w_sum[:, None])
        else:
            face_areas = calc_face_areas(cano_verts, cano_faces) # 计算每个三角形的面积
            self.register_buffer('face_areas', face_areas.clone())

    def calc_per_vert_quaternion(self, mesh_verts):
        cano_verts = self.cano_verts
        cano_faces = self.cano_faces
        per_face_quat = self.calc_per_face_quaternion(mesh_verts)

        if self.use_numpy:
            # face quat weighted to vert
            per_vert_quat = torch.zeros(cano_verts.shape[0], 4).to(per_face_quat.device)
            per_vert_quat = per_vert_quat.scatter_(0, 
                                                cano_faces[:, :, None].repeat([1, 1, 4]).view(-1, 4), 
                                                (self.W * per_face_quat)[:, None, :].repeat([1, 3, 1]).view(-1, 4), reduce='add')
            per_vert_quat = per_vert_quat / self.per_vert_w_sum

            # normalize
            per_vert_quat = F.normalize(per_vert_quat, eps=1e-6, dim=-1)

        else:
            faces_packed = cano_faces

            verts_quats = torch.zeros(cano_verts.shape[0], 4).to(per_face_quat.device)

            # NOTE: this is already applying the area weighting as the magnitude
            # of the cross product is 2 x area of the triangle.
            verts_quats = verts_quats.index_add(
                0, faces_packed[:, 0], self.face_areas * per_face_quat
            )
            verts_quats = verts_quats.index_add(
                0, faces_packed[:, 1], self.face_areas * per_face_quat
            )
            verts_quats = verts_quats.index_add(
                0, faces_packed[:, 2], self.face_areas * per_face_quat
            )

            per_vert_quat = F.normalize(verts_quats, eps=1e-6, dim=1)

        return per_vert_quat
    
    def calc_per_face_Rt(self, mesh_verts):
        cano_verts = self.cano_verts
        cano_faces = self.cano_faces
        cano_triangles = cano_verts[cano_faces].unsqueeze(dim=0)
        deform_triangles = mesh_verts[cano_faces].unsqueeze(dim=0)
        per_face_Rt = calc_per_face_Rt(cano_triangles, deform_triangles)
        return per_face_Rt
    
    def calc_per_face_quaternion(self, mesh_verts):
        per_face_Rt = self.calc_per_face_Rt(mesh_verts)
        per_face_quat = rotation_matrix_to_quaternion(per_face_Rt[:, :3, :3])
        return per_face_quat
    
    def forward(self, mesh_verts):
        return self.calc_per_vert_quaternion(mesh_verts)
    
    def calc_face_area_change(self, mesh_verts, damping=1e-4):
        areas = calc_face_areas(mesh_verts, self.cano_faces)
        change_ratio = (areas + damping) / (self.face_areas + damping)
        return change_ratio
        

