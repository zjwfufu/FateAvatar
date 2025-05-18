import copy
import torch
import torch.nn as nn

from pytorch3d.io   import load_obj
from pytorch3d.ops  import knn_points
from pytorch3d.transforms import (
    quaternion_to_axis_angle,
    matrix_to_quaternion,
    quaternion_multiply
)

from volume_rendering.camera_3dgs       import Camera
from volume_rendering.gaussian_model    import GaussianModel
from volume_rendering.render_3dgs       import render
from volume_rendering.mesh_sampling     import (
    uniform_sampling_barycoords,
    reweight_uvcoords_by_barycoords,
    reweight_verts_by_barycoords
)
from volume_rendering.mesh_compute      import (
    compute_face_orientation,
    compute_face_normals
)

from tools.util import get_bg_color
from tools.gs_utils.general_utils   import inverse_sigmoid
from tools.gs_utils.sh_utils        import RGB2SH

import warnings
warnings.filterwarnings("ignore", message="No mtl file provided")
warnings.filterwarnings("ignore", message="Mtl file does not exist")

from benchmark.nersemble.FLAME import FLAME, FlameConfig

#-------------------------------------------------------------------------------#

class FateAvatar(nn.Module):
    def __init__(
            self,
            img_res,
            background_color,
            cfg_model,
            device
        ):
        super().__init__()

        self.uv_resolution = cfg_model.tex_size
        self.shell_len     = cfg_model.normal_offset
        self.rodriguez_rotation = True
        self.max_sh_degree = 0

        self.device = device

        self.bg_color = get_bg_color(background_color).to(self.device)

        self.img_res = img_res

        self.cfg_model = cfg_model

        self._register_flame()

        self._register_template_mesh(template_path='./weights/head_template_mouth_close.obj')

        mean_scaling, max_scaling, scale_init = self.get_init_scale_by_knn(self.verts_sampling)

        self.register_buffer('mean_scaling', mean_scaling)
        self.register_buffer('max_scaling', max_scaling)
        self.register_buffer('scale_init', scale_init)

        # self._register_texture_map(tex_size=self.uv_resolution)
        self._register_init_gaussian()

        _, face_scaling_canonical    = compute_face_orientation(self.canonical_verts.squeeze(0), self.faces, return_scale=True)
        self.register_buffer('face_scaling_canonical', face_scaling_canonical)

        self.delta_shapedirs    = torch.zeros_like(self.flame.shapedirs).to(self.device)
        self.delta_shapedirs    = nn.Parameter(self.delta_shapedirs.requires_grad_(True))

        self.delta_posedirs     = torch.zeros_like(self.flame.posedirs).to(self.device)
        self.delta_posedirs     = nn.Parameter(self.delta_posedirs.requires_grad_(True))

        self.delta_vertex       = torch.zeros_like(self.flame.v_template).to(self.device)
        self.delta_vertex       = nn.Parameter(self.delta_vertex.requires_grad_(True))

    def _register_flame(self):

        flame_config = FlameConfig(
            shape_params=300,
            expression_params=100,
            batch_size=1,
        )

        self.flame = FLAME(flame_config).to(self.device)
        
        canonical_verts = self.flame.v_template

        # make sure call of FLAME is successful
        self.canonical_verts                    = canonical_verts[None, ...]
        self.flame.canonical_verts              = canonical_verts
    
    def _register_template_mesh(self, template_path):

        #----------------   load head mesh & process UV ----------------#
        verts, faces, aux = load_obj(template_path)

        uvcoords    = aux.verts_uvs
        uvfaces     = faces.textures_idx
        faces       = faces.verts_idx

        face_index, bary_coords = uniform_sampling_barycoords(
            num_points    = self.uv_resolution * self.uv_resolution,
            tex_coord     = uvcoords,
            uv_faces      = uvfaces
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
    
        self.register_buffer('verts_sampling', verts_sampling)

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
        offset = torch.zeros((self.num_points, 1), device=self.device)

        self._offset        = nn.Parameter(offset.requires_grad_(True))
        self._features_dc   = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling       = nn.Parameter(scales.requires_grad_(True))
        self._rotation      = nn.Parameter(rots.requires_grad_(True))
        self._opacity       = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D        = torch.zeros((self.num_points), device=self.device)
        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=self.device)
        self.denom              = torch.zeros((self.num_points, 1), device=self.device)

        # to visualize densification points
        self.sample_flag        = torch.zeros((self.num_points), device=self.device)

    def forward(self, input):

        cam_pose = input["cam_pose"].clone()
        fovx = input["fovx"][0]
        fovy = input["fovy"][0]
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]

        intrinsics = input["intrinsics"]

        camera = Camera(
            R=R, T=T, FoVx=fovx, FoVy=fovy,
            img_res=self.img_res, intrinsics=intrinsics
        )

        shape_param = input["shape"]
        experssion_param = input["expression"]
        pose_param = torch.cat([
            torch.zeros_like(input["rotation"]),
            input["jaw"]], dim=-1
        )
        neck_pose_param = input["neck"]
        eye_pose_param = input["eyes"]
        transl_param = input["translation"]

        rotation = input["rotation"]
        scale = input["scale"]
        
        bs = pose_param.shape[0]    # 1, essentially

        #-------------------------------    prepare splats position     -------------------------------#

        verts, _, _ = self.flame.forward_with_delta_blendshape(
            shape_params        = shape_param,
            expression_params   = experssion_param,
            pose_params         = pose_param,
            neck_pose           = neck_pose_param,
            eye_pose            = eye_pose_param,
            transl              = transl_param,
            rotation            = rotation,
            scale               = scale,
            delta_shapedirs     = self.delta_shapedirs if self.cfg_model.delta_blendshape else None,
            delta_posedirs      = self.delta_posedirs if self.cfg_model.delta_blendshape else None,
            delta_vertex        = self.delta_vertex if self.cfg_model.delta_vertex else None
        )

        verts_orig, _, _ = self.flame.forward(
            shape_params        = shape_param,
            expression_params   = experssion_param,
            pose_params         = pose_param,
            neck_pose           = neck_pose_param,
            eye_pose            = eye_pose_param,
            transl              = transl_param,
            rotation            = rotation,
            scale               = scale,
        )
        
        face_orien_mat, face_scaling    = compute_face_orientation(verts, self.faces, return_scale=True)
        face_normals                    = compute_face_normals(verts, self.faces)

        scaling_ratio       = face_scaling / self.face_scaling_canonical
        flame_scaling_ratio = scaling_ratio[:, self.face_index]

        flame_orien_mat     = face_orien_mat[:, self.face_index]
        flame_orien_quat    = matrix_to_quaternion(flame_orien_mat)
        flame_normals       = face_normals[:, self.face_index]

        pos_val = reweight_verts_by_barycoords(
            verts         = verts,
            faces         = self.faces,
            face_index    = self.face_index,
            bary_coords   = self.bary_coords
        )

        #-------------------------------    render gaussian     -------------------------------#

        gaussian = GaussianModel(sh_degree = 0)

        render_image = []
        viewspace_points = []
        visibility_filter = []
        radii = []

        # dummy loop
        for bs_ in range(bs):
            gaussian._features_dc   = self._features_dc
            gaussian._features_rest = self._features_rest
            gaussian._opacity       = self._opacity
            gaussian._scaling       = (self._scaling + torch.log(flame_scaling_ratio[bs_])) if self.cfg_model.resize_scale else self._scaling
            gaussian._rotation      = quaternion_multiply(flame_orien_quat[bs_], self._rotation)
            gaussian._xyz           = pos_val[bs_] + flame_normals[bs_] * self.shell_len * torch.tanh(self._offset)

            render_out = render(
                camera,
                gaussian,
                self.bg_color.to(self.device),
                device=self.device,
                override_color=None
            )
            
            render_image_       = render_out['render']
            viewspace_points_   = render_out['viewspace_points']
            visibility_filter_  = render_out['visibility_filter']
            radii_              = render_out['radii']

            render_image.append(render_image_)
            viewspace_points.append(viewspace_points_)
            visibility_filter.append(visibility_filter_)
            radii.append(radii_)
        
        render_image = torch.stack(render_image)

        output = {
            'rgb_image': render_image,
            'scale': torch.exp(self._scaling),
            'raw_rot': quaternion_to_axis_angle(self._rotation),
            # ----- gaussian maintainer ----- #
            'viewspace_points': viewspace_points,   # List
            'visibility_filter': visibility_filter, # List
            'radii': radii, # List
            'bs': bs,
            # ----- mesh loss ----- #
            'verts': verts,
            'verts_orig': verts_orig,
            'faces': self.faces,
        }

        return output
    
    def inference(self, expression, flame_pose, camera):

        bs = flame_pose.shape[0]    # 1, essentially

        #-------------------------------    prepare splats position     -------------------------------#

        verts, _, _ = self.flame.forward_with_delta_blendshape(
            expression_params   = expression,
            full_pose           = flame_pose,
            delta_shapedirs     = self.delta_shapedirs if self.cfg_model.delta_blendshape else None,
            delta_posedirs      = self.delta_posedirs if self.cfg_model.delta_blendshape else None,
            delta_vertex        = self.delta_vertex if self.cfg_model.delta_vertex else None
        )

        face_orien_mat, face_scaling    = compute_face_orientation(verts, self.faces, return_scale=True)
        face_normals                    = compute_face_normals(verts, self.faces)

        scaling_ratio       = face_scaling / self.face_scaling_canonical
        flame_scaling_ratio = scaling_ratio[:, self.face_index]

        flame_orien_mat     = face_orien_mat[:, self.face_index]
        flame_orien_quat    = matrix_to_quaternion(flame_orien_mat)
        flame_normals       = face_normals[:, self.face_index]

        pos_val = reweight_verts_by_barycoords(
            verts         = verts,
            faces         = self.faces,
            face_index    = self.face_index,
            bary_coords   = self.bary_coords
        )

        #-------------------------------    render gaussian     -------------------------------#

        gaussian = GaussianModel(sh_degree = 0)

        render_image = []
        viewspace_points = []
        visibility_filter = []
        radii = []

        # dummy loop
        for bs_ in range(bs):
            gaussian._features_dc   = self._features_dc
            gaussian._features_rest = self._features_rest
            gaussian._opacity       = self._opacity
            gaussian._scaling       = (self._scaling + torch.log(flame_scaling_ratio[bs_])) if self.cfg_model.resize_scale else self._scaling
            gaussian._rotation      = quaternion_multiply(flame_orien_quat[bs_], self._rotation)
            gaussian._xyz           = pos_val[bs_] + flame_normals[bs_] * self.shell_len * torch.tanh(self._offset)

            render_out = render(
                camera,
                gaussian,
                self.bg_color.to(self.device),
                device=self.device,
                override_color=None
            )
            
            render_image_       = render_out['render']
            viewspace_points_   = render_out['viewspace_points']
            visibility_filter_  = render_out['visibility_filter']
            radii_              = render_out['radii']

            render_image.append(render_image_)
            viewspace_points.append(viewspace_points_)
            visibility_filter.append(visibility_filter_)
            radii.append(radii_)
        
        render_image = torch.stack(render_image)

        return render_image[0]
    
    @torch.no_grad()
    def visualization(self, input):

        cam_pose = input["cam_pose"].clone()
        fovx = input["fovx"][0]
        fovy = input["fovy"][0]
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]

        R_I = torch.tensor(
            [[[ 1.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0, -1.0]]]
        )

        R_cano = R_I.to(self.device)
        T_cano = copy.deepcopy(T);  T_cano[:, 0] = 0; T_cano[:, 1] = 0

        intrinsics = input["intrinsics"]

        camera = Camera(
            R=R, T=T, FoVx=fovx, FoVy=fovy,
            img_res=self.img_res, intrinsics=intrinsics
        )

        camera_cano = Camera(
            R=R_cano, T=T_cano, FoVx=fovx, FoVy=fovy,
            img_res=self.img_res, intrinsics=intrinsics
        )

        shape_param = input["shape"]
        experssion_param = input["expression"]
        pose_param = torch.cat([
            torch.zeros_like(input["rotation"]),
            input["jaw"]], dim=-1
        )
        neck_pose_param = input["neck"]
        eye_pose_param = input["eyes"]
        transl_param = input["translation"]

        rotation = input["rotation"]
        scale = input["scale"]
        
        bs = pose_param.shape[0]    # 1, essentially

        #-------------------------------    prepare splats position     -------------------------------#

        verts, _, _ = self.flame.forward_with_delta_blendshape(
            shape_params        = shape_param,
            expression_params   = experssion_param,
            pose_params         = pose_param,
            neck_pose           = neck_pose_param,
            eye_pose            = eye_pose_param,
            transl              = transl_param,
            rotation            = rotation,
            scale               = scale,
            delta_shapedirs     = self.delta_shapedirs if self.cfg_model.delta_blendshape else None,
            delta_posedirs      = self.delta_posedirs if self.cfg_model.delta_blendshape else None,
            delta_vertex        = self.delta_vertex if self.cfg_model.delta_vertex else None
        )

        verts_orig, _, _ = self.flame.forward(
            shape_params        = shape_param,
            expression_params   = experssion_param,
            pose_params         = pose_param,
            neck_pose           = neck_pose_param,
            eye_pose            = eye_pose_param,
            transl              = transl_param,
            rotation            = rotation,
            scale               = scale,
        )
        
        face_orien_mat, _               = compute_face_orientation(verts, self.faces, return_scale=False)
        face_normals                    = compute_face_normals(verts, self.faces)

        _, face_scaling                 = compute_face_orientation(verts_orig, self.faces, return_scale=True)
        scaling_ratio       = face_scaling / self.face_scaling_canonical
        flame_scaling_ratio = scaling_ratio[:, self.face_index]

        flame_orien_mat     = face_orien_mat[:, self.face_index]
        flame_orien_quat    = matrix_to_quaternion(flame_orien_mat)
        flame_normals       = face_normals[:, self.face_index]

        pos_val = reweight_verts_by_barycoords(
            verts         = verts,
            faces         = self.faces,
            face_index    = self.face_index,
            bary_coords   = self.bary_coords
        )
        
        #-----------------------------------------------------#
        
        verts_cano, _, _ = self.flame.forward_with_delta_blendshape(
            shape_params         = shape_param,
            expression_params    = torch.zeros_like(experssion_param),
            pose_params          = torch.zeros_like(pose_param),
            neck_pose            = neck_pose_param,
            eye_pose             = eye_pose_param,
            transl               = transl_param,
            rotation             = rotation,
            scale                = scale,
            delta_shapedirs      = self.delta_shapedirs if self.cfg_model.delta_blendshape else None,
            delta_posedirs       = self.delta_posedirs if self.cfg_model.delta_blendshape else None,
            delta_vertex         = self.delta_vertex if self.cfg_model.delta_vertex else None,
            model_view_transform = False,
        )
        
        verts_cano_orig, _, _ = self.flame.forward(
            shape_params         = shape_param,
            expression_params    = torch.zeros_like(experssion_param),
            pose_params          = torch.zeros_like(pose_param),
            neck_pose            = neck_pose_param,
            eye_pose             = eye_pose_param,
            transl               = transl_param,
            rotation             = rotation,
            scale                = scale,
            model_view_transform = False,
            
        )

        face_orien_mat_cano, face_scaling_cano    = compute_face_orientation(verts_cano, self.faces, return_scale=True)
        face_normals_cano                         = compute_face_normals(verts_cano, self.faces)

        _, face_scaling_cano                    = compute_face_orientation(verts_cano_orig, self.faces, return_scale=True)
        scaling_ratio_cano                      = face_scaling_cano / self.face_scaling_canonical
        flame_scaling_ratio_cano                = scaling_ratio_cano[:, self.face_index]

        flame_orien_mat_cano        = face_orien_mat_cano[:, self.face_index]
        flame_orien_quat_cano       = matrix_to_quaternion(flame_orien_mat_cano)
        flame_normals_cano          = face_normals_cano[:, self.face_index]

        pos_val_cano = reweight_verts_by_barycoords(
            verts         = verts_cano,
            faces         = self.faces,
            face_index    = self.face_index,
            bary_coords   = self.bary_coords
        )
        
        #-------------------------------    render gaussian     -------------------------------#

        gaussian        = GaussianModel(sh_degree = 0)
        cano_gaussian   = GaussianModel(sh_degree = 0)
        point_gaussian  = GaussianModel(sh_degree = 0)
        grad_gaussian   = GaussianModel(sh_degree = 0)

        render_image = []
        render_image_point = []
        render_image_cano = []
        render_image_grad = []

        # dummy loop
        for bs_ in range(bs):
            gaussian._features_dc   = self._features_dc
            gaussian._features_rest = self._features_rest
            gaussian._opacity      = self._opacity
            gaussian._scaling      = (self._scaling + torch.log(flame_scaling_ratio[bs_])) if self.cfg_model.resize_scale else self._scaling
            gaussian._rotation     = quaternion_multiply(flame_orien_quat[bs_], self._rotation)
            gaussian._xyz          = pos_val[bs_] + flame_normals[bs_] * self.shell_len * torch.tanh(self._offset)

            render_out = render(
                camera,
                gaussian,
                self.bg_color.to(self.device),
                device=self.device,
                override_color=None
            )
            
            # render avatar in canoical space
            cano_gaussian._features_dc      = self._features_dc
            cano_gaussian._features_rest    = self._features_rest
            cano_gaussian._opacity          = self._opacity
            cano_gaussian._scaling          = (self._scaling + torch.log(flame_scaling_ratio_cano[bs_])) if self.cfg_model.resize_scale else self._scaling
            cano_gaussian._rotation         = quaternion_multiply(flame_orien_quat[bs_], self._rotation)
            cano_gaussian._xyz              = pos_val_cano[bs_] + flame_normals_cano[bs_] * self.shell_len * torch.tanh(self._offset)


            render_out_cano = render(
                camera_cano,
                cano_gaussian,
                self.bg_color.to(self.device),
                device=self.device,
                override_color=None
            )
            
            # render grad
            gray_value   = torch.tensor([0.99, 0, 0], dtype=torch.float32).to(self.device)

            grad_accum = self.xyz_gradient_accum / self.denom
            grad_accum[grad_accum.isnan()] = 0.0

            threshold = torch.quantile(grad_accum, 0.95)

            # print(threshold)

            normalized_grad = torch.zeros_like(self.xyz_gradient_accum)

            mask = grad_accum <= threshold
            normalized_grad[mask] = grad_accum[mask] / torch.max(grad_accum[mask])

            normalized_grad[~mask] = 1.0

            grad_vis = normalized_grad * gray_value
            grad_sh  = RGB2SH(grad_vis).unsqueeze(1)

            grad_gaussian._features_dc      = grad_sh
            grad_gaussian._features_rest    = self._features_rest
            grad_gaussian._opacity          = self._opacity + 5
            grad_gaussian._scaling          = (self._scaling + torch.log(flame_scaling_ratio_cano[bs_])) if self.cfg_model.resize_scale else self._scaling
            grad_gaussian._rotation         = quaternion_multiply(flame_orien_quat[bs_], self._rotation)
            grad_gaussian._xyz              = pos_val_cano[bs_] + flame_normals_cano[bs_] * self.shell_len * torch.tanh(self._offset)

            render_out_grad = render(
                camera_cano,
                grad_gaussian,
                torch.tensor([0, 0, 0], dtype=torch.float32).to(self.device),
                device=self.device,
                override_color=None
            )
            
            # render sampled splat position
            point_gaussian._features_dc             = torch.zeros_like(gaussian._features_dc.data); point_gaussian._features_dc[:, 0, 2]    = (180 / 255 - 0.5) / 0.282    # C0 = 0.28209479177387814
            point_gaussian._features_rest           = self._features_rest
            point_gaussian._opacity                 = torch.ones_like(self._opacity.data) + 5
            point_gaussian._scaling                 = - torch.ones_like(self._scaling.data) * 8
            point_gaussian._rotation                = quaternion_multiply(flame_orien_quat[bs_], self._rotation)
            point_gaussian._xyz                     = pos_val_cano[bs_] + flame_normals_cano[bs_] * self.shell_len * torch.tanh(self._offset)

            sampled_mask    = torch.where(self.sample_flag == 1)

            point_gaussian._features_dc             = point_gaussian._features_dc[sampled_mask]
            # point_gaussian._features_rest           = point_gaussian._features_rest[sampled_mask]
            point_gaussian._opacity                 = point_gaussian._opacity[sampled_mask]
            point_gaussian._scaling                 = point_gaussian._scaling[sampled_mask]
            point_gaussian._rotation                = point_gaussian._rotation[sampled_mask]
            point_gaussian._xyz                     = point_gaussian._xyz[sampled_mask]

            render_out_point = render(
                camera_cano,
                point_gaussian,
                self.bg_color.to(self.device),
                device=self.device,
                override_color=None
            )

            render_image_       = render_out['render']
            render_image_cano_  = render_out_cano['render']
            render_image_point_ = render_out_point['render']
            render_image_grad_  = render_out_grad['render']

            render_image.append(render_image_)
            render_image_cano.append(render_image_cano_)
            render_image_point.append(render_image_point_)
            render_image_grad.append(render_image_grad_)

        
        render_image        = torch.stack(render_image)
        render_image_cano   = torch.stack(render_image_cano)
        render_image_point  = torch.stack(render_image_point)
        render_image_grad   = torch.stack(render_image_grad)

        output = {
            'rgb_image': render_image,
            'cano_image': render_image_cano,
            'point_image': render_image_point,
            'grad_image': render_image_grad,
            'scale': torch.exp(self._scaling),
            'raw_rot': quaternion_to_axis_angle(self._rotation),
            # ----- nvidiffrast ----- #
            'verts': verts,
            'faces': self.faces,
            # 'camera': camera
            'camera': camera
        }

        return output

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
    
    def _uv_densify(
            self,
            gs_optimizer: torch.optim.Adam,
            increase_num = 1000
        ):

        xyz_gradient_accum = self.xyz_gradient_accum.squeeze(1)
        sampled_indices = xyz_gradient_accum.multinomial(increase_num, replacement=True)

        new_face_index  = self.face_index[sampled_indices]
        uvw = torch.rand((new_face_index.shape[0], 3), device=self.device)
        new_bary_coords = uvw / uvw.sum(dim=-1, keepdim=True)

        new_opacity     = self._opacity[sampled_indices]
        new_offset      = self._offset[sampled_indices]
        new_features_dc = self._features_dc[sampled_indices]
        new_rotation    = self._rotation[sampled_indices]
        new_scaling     = torch.log(torch.exp(self._scaling[sampled_indices]) * 0.75)

        new_attribute = {
            "opacity": new_opacity,
            "offset": new_offset,
            "color": new_features_dc,
            "rotation": new_rotation,
            "scaling": new_scaling
        }

        optimizable_tensors = {}
        for group in gs_optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = new_attribute[group["name"]]
            stored_state = gs_optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del gs_optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                gs_optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        self._opacity       = optimizable_tensors["opacity"]
        self._offset        = optimizable_tensors["offset"]
        self._features_dc   = optimizable_tensors["color"]
        self._rotation      = optimizable_tensors["rotation"]
        self._scaling       = optimizable_tensors["scaling"]

        self.face_index     = torch.cat([self.face_index, new_face_index], dim=0)
        self.bary_coords    = torch.cat([self.bary_coords, new_bary_coords], dim=0)

        new_sample_flag     = torch.ones((increase_num), device=self.device)
        self.sample_flag    = torch.cat([self.sample_flag, new_sample_flag], dim=0)

        self.num_points     = self.bary_coords.shape[0]

        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=self.device)
        self.denom = torch.zeros((self.num_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((self.num_points), device=self.device)

    def _prune_low_opacity_points(
            self,
            gs_optimizer: torch.optim.Adam,
            min_opacity=0.05
        ):
        
        prune_mask = (torch.sigmoid(self._opacity) < min_opacity).squeeze()
        valid_points_mask = ~prune_mask

        optimizable_tensors = {}
        for group in gs_optimizer.param_groups:
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

        self._opacity       = optimizable_tensors["opacity"]
        self._offset        = optimizable_tensors["offset"]
        self._features_dc   = optimizable_tensors["color"]
        self._rotation      = optimizable_tensors["rotation"]
        self._scaling       = optimizable_tensors["scaling"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom              = self.denom[valid_points_mask]
        self.max_radii2D        = self.max_radii2D[valid_points_mask]
        self.sample_flag        = self.sample_flag[valid_points_mask]

        self.face_index     = self.face_index[valid_points_mask]
        self.bary_coords    = self.bary_coords[valid_points_mask]

        self.num_points         = self.bary_coords.shape[0]

    def _reset_opacity(self, gs_optimizer: torch.optim.Adam):
        cur_opacity = torch.sigmoid(self._opacity)
        opacities_new = inverse_sigmoid(torch.min(cur_opacity, torch.ones_like(cur_opacity)*0.01))

        optimizable_tensors = {}
        for group in gs_optimizer.param_groups:
            if group["name"] == "opacity":
                stored_state = gs_optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(opacities_new)
                stored_state["exp_avg_sq"] = torch.zeros_like(opacities_new)

                del gs_optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(opacities_new.requires_grad_(True))
                gs_optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]

        self._opacity = optimizable_tensors["opacity"]

    def _add_densification_stats(self, viewspace_point_tensor, update_filter):

        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_default_points(self, gs_optimizer: torch.optim.Adam):
        """
        Add a set of default Gaussians, as points on the back of the head are pruned during monocular training.
        """

        default_number = self.verts_sampling.shape[1]

        init_rgb = inverse_sigmoid(torch.Tensor([0.5, 0.5, 0.5]))[None, ...].float()
        init_rgb = init_rgb.repeat_interleave(default_number, dim=0).to(self.device)
        features = torch.zeros((init_rgb.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = init_rgb
        features[:, 3:, 1:] = 0.0
        scales      = self.scale_init[...,None].repeat(default_number, 3).to(self.device)
        rots        = torch.zeros((default_number, 4), device=self.device)
        rots[:, 0]  = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((default_number, 1), dtype=torch.float, device=self.device))
        offset = torch.zeros((default_number, 1), device=self.device)

        new_offset        = nn.Parameter(offset.requires_grad_(True))
        new_features_dc   = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling       = nn.Parameter(scales.requires_grad_(True))
        new_rotation      = nn.Parameter(rots.requires_grad_(True))
        new_opacity       = nn.Parameter(opacities.requires_grad_(True))

        new_face_index, new_bary_coords = uniform_sampling_barycoords(
            num_points    = self.uv_resolution * self.uv_resolution,
            tex_coord     = self.uvcoords,
            uv_faces      = self.uvfaces
        )

        new_attribute = {
            "opacity": new_opacity,
            "offset": new_offset,
            "color": new_features_dc,
            "rotation": new_rotation,
            "scaling": new_scaling
        }

        optimizable_tensors = {}
        for group in gs_optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = new_attribute[group["name"]]
            stored_state = gs_optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del gs_optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                gs_optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        self._opacity       = optimizable_tensors["opacity"]
        self._offset        = optimizable_tensors["offset"]
        self._features_dc   = optimizable_tensors["color"]
        self._rotation      = optimizable_tensors["rotation"]
        self._scaling       = optimizable_tensors["scaling"]

        self.face_index     = torch.cat([self.face_index, new_face_index], dim=0)
        self.bary_coords    = torch.cat([self.bary_coords, new_bary_coords], dim=0)

        new_sample_flag     = torch.ones((default_number), device=self.device)
        self.sample_flag    = torch.cat([self.sample_flag, new_sample_flag], dim=0)

        self.num_points     = self.bary_coords.shape[0]

        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=self.device)
        self.denom = torch.zeros((self.num_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((self.num_points), device=self.device)