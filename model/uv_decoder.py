import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unet.arch import UNet, UNetDecoder, FeatureMap

from pytorch3d.ops import knn_points
from pytorch3d.io import load_obj
from pytorch3d.transforms import (
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    matrix_to_quaternion,
    quaternion_multiply,
)

from model.fateavatar   import FateAvatar

from volume_rendering.render_3dgs       import render
from volume_rendering.gaussian_model    import GaussianModel
from volume_rendering.camera_3dgs       import Camera
from volume_rendering.mesh_sampling     import (
    uniform_sampling_barycoords,
    reweight_uvcoords_by_barycoords,
    reweight_verts_by_barycoords
)
from volume_rendering.mesh_compute      import (
    compute_face_orientation,
    compute_face_normals
)

from tools.gs_utils.sh_utils import C0

# ------------------------------------------------------------------------------- #

class UVSampling(nn.Module):
    def __init__(self):
        super().__init__()
        """
        A class implementing Gaussian attribute sampling from attribute maps.
        """

    def _register_template_mesh(self, template_path):

        #----------------   load head mesh & process UV ----------------#
        verts, faces, aux = load_obj(template_path)

        uvcoords    = aux.verts_uvs
        uvfaces     = faces.textures_idx
        faces       = faces.verts_idx

        face_index, bary_coords = uniform_sampling_barycoords(
            num_points    = 256 * 256,
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

    def _gather_attribute(self, neural_texture, texture_layers_dict, uv_coords):
        texture_dict = {}
        act_texture_dict = {}
        value_dict = {}

        for name, layer in texture_layers_dict:
            texture_dict[name] = layer(neural_texture)

            if name == 'color':
                act_texture_dict[name] = self._color_activation(texture_dict[name])
            elif name == 'scaling':
                act_texture_dict[name] = self._scaling_activation(texture_dict[name])
            elif name == 'offset':
                act_texture_dict[name] = self._offset_activation(texture_dict[name])
            elif name == 'rotation':
                act_texture_dict[name] = self._rotation_activation(texture_dict[name])
            else:
                act_texture_dict[name] = texture_dict[name]

            # texture look up
            value_dict[name] = self._texture_look_up(act_texture_dict[name], uv_coords=uv_coords)

        return texture_dict, act_texture_dict, value_dict
    
    def _gather_attribute_from_texture_dict(self, texture_dict, uv_coords):

        value_dict = {}

        for name, texture in texture_dict.items():

            if name == 'color':
                # color will be activated outside
                pass

            elif name == 'scaling':
                texture = self._scaling_activation(texture)
            elif name == 'offset':
                texture = self._offset_activation(texture)
            elif name == 'rotation':
                texture = self._rotation_activation(texture)
            else:
                texture = texture_dict[name]

            # texture look up
            value_dict[name] = self._texture_look_up(texture, uv_coords=uv_coords)

        return value_dict

    @staticmethod
    def _color_activation(tensor):
        """
        Force the NN output serves as color to be between [-1.78, 1.78].
        """
        return torch.tanh(tensor) * (0.5 / C0)
    
    def _scaling_activation(self, tensor):
        """
        Force the NN output serves as scaling within proper range [-4, -6].
        After exp() in GaussianModel,  it will fall into [0.002, 0.018].
        softplus() is used to constrain huge splat.

        https://github.com/tobias-kirschstein/gghead/tree/master

        """
        return self._max_scaling - torch.nn.functional.softplus(- (tensor + self._mean_scaling) + self._max_scaling)
    
    @staticmethod
    def _offset_activation(tensor):
        """
        Force the NN output serves as offset within [-1, 1].
        """
        return torch.tanh(tensor)

    def _rotation_activation(self, tensor):
        """
        Force the NN output serves as rotation within proper range and no ambiguity.
        """
        tensor = torch.tanh(tensor) * (2 * torch.pi)
        if tensor.shape[-1] != 3:
            tensor = tensor.permute(0, 3, 2, 1)
            tensor = axis_angle_to_quaternion(tensor)
            tensor = torch.cat([tensor[..., [3]], tensor[..., :3]], dim=-1)  # xyzr -> rxyz
            tensor = tensor.permute(0, 3, 2, 1)
        else:
            tensor = axis_angle_to_quaternion(tensor)
            tensor = torch.cat([tensor[..., [3]], tensor[..., :3]], dim=-1)  # xyzr -> rxyz

        tensor = tensor.contiguous()

        return tensor
    
    def slice_function(self, x, start, end):
            return x[:, start:end, :, :]

    def _texture_look_up(self, texture, uv_coords = None):
        """U
        se sampling to get texture value

        Args:
            texture:    [B, C, H, W] texture color
            texc:       [B, H, W, 2] texture uv
        Returns:
            values:     [B, H, W, C] texture sampling
        """
        if uv_coords is None:
            uv_coords    = self.uvcoords_sample
        shift_uv = (2 * uv_coords - 1).unsqueeze(0)
        shift_uv = shift_uv.repeat(texture.shape[0], 1, 1, 1)

        out = F.grid_sample(
            texture,
            shift_uv,
            mode            = "bilinear",
            padding_mode    = "border",
            align_corners   = True,
        )
        # map back to [B, H, W, C]
        return out.permute(0, 2, 3, 1).squeeze(1)

# ------------------------------------------------------------------------------- #

class UVDecoder(UVSampling):
    def __init__(self,
                 avatar_model:  FateAvatar,
                 decode_type:   str = 'UNet',
                 bake_type:     list = ['color', 'opacity'],
            ):
        super().__init__()

        self.avatar_model   = avatar_model

        self._register_template_mesh(template_path='./weights/head_template_mouth_close.obj')
        self._parsing_avatar_model(avatar_model)
        self._build_bakenet(decode_type)

        self.texture_dict_cache = None
        self.bake_attribute     = bake_type

    def _build_bakenet(self, decode_type: str):

        self.texture_channels = {
            'color':    3,
            'opacity':  1,
            'scaling':  3,
            'rotation': 3,
            'offset':   1,
        }

        color_end       = self.texture_channels['color']
        opacity_end     = color_end + self.texture_channels['opacity']
        scaling_end     = opacity_end + self.texture_channels['scaling']
        rotation_end    = scaling_end + self.texture_channels['rotation']
        offset_end      = rotation_end + self.texture_channels['offset']

        self.texture_layers = [
            ('color',    functools.partial(self.slice_function, start=0, end=color_end)),
            ('opacity',  functools.partial(self.slice_function, start=color_end, end=opacity_end)),
            ('scaling',  functools.partial(self.slice_function, start=opacity_end, end=scaling_end)),
            ('rotation', functools.partial(self.slice_function, start=scaling_end, end=rotation_end)),
            ('offset',   functools.partial(self.slice_function, start=rotation_end, end=offset_end))
        ]

        decode_ch = sum(self.texture_channels.values())

        if decode_type == 'UNet':
            self.tex_ch = 11
            self.tex_size = 512
            self.const = nn.Parameter(
                torch.FloatTensor(1, self.tex_ch, self.tex_size, self.tex_size).uniform_(-1, 1)
            )
            self.decoder = UNet(
                self.tex_ch,
                decode_ch
            )

        elif decode_type == 'decode_only':
            self.tex_ch = 512
            self.tex_size = 8
            self.const = nn.Parameter(
                torch.FloatTensor(1, self.tex_ch, self.tex_size, self.tex_size).uniform_(-1, 1)
            )
            self.decoder = UNetDecoder(
                self.tex_ch,
                decode_ch
            )

        elif decode_type == 'feature_map':
            self.tex_ch = 11
            self.tex_size = 512
            self.const = nn.Parameter(
                torch.FloatTensor(1, self.tex_ch, self.tex_size, self.tex_size).uniform_(-1, 1)
            )
            self.decoder = FeatureMap(
                self.tex_ch,
                decode_ch,
            )

        else:
            raise NotImplementedError(f"Unsupported decode_type: {decode_type}")

    @torch.no_grad()
    def _parsing_avatar_model(self, avatar_model: FateAvatar):

        device = avatar_model.device

        self._prior_features_dc     = avatar_model._features_dc.data      # [-1.78, 1.78]
        self._prior_features_rest   = avatar_model._features_rest.data    # dummy
        self._prior_opacity         = avatar_model._opacity.data          # before sigmoid
        self._prior_offset          = avatar_model._offset.data           # before tanh

        self._prior_rotation        = avatar_model._rotation.data         # before normalize
        self._prior_rotation        = torch.nn.functional.normalize(self._prior_rotation)

        self._prior_scaling         = avatar_model._scaling.data          # before exp
        self._mean_scaling          = self._prior_scaling.mean().item()
        self._std_scaling           = self._prior_scaling.std().item()
        self._max_scaling           = self._mean_scaling + self._std_scaling

        self._prior_face_index      = avatar_model.face_index
        self._prior_bary_coords     = avatar_model.bary_coords

        # for more dense distribution
        self._prior_face_index      = torch.cat([self._prior_face_index, self.face_index.to(device)], dim=0)
        self._prior_bary_coords     = torch.cat([self._prior_bary_coords, self.bary_coords.to(device)], dim=0)

        prior_uvcoords_sample   = reweight_uvcoords_by_barycoords(
            uvcoords    = self.uvcoords.to(avatar_model.device),
            uvfaces     = self.uvfaces.to(avatar_model.device),
            face_index  = self._prior_face_index,
            bary_coords = self._prior_bary_coords
        )

        prior_uvcoords_sample = prior_uvcoords_sample[...,:2]

        query_points        = prior_uvcoords_sample.float()
        knn                 = knn_points(query_points, query_points, K=6)
        dists               = torch.sqrt(knn.dists[..., 1])
        mean_dist           = dists.mean()
        self.sample_radius  = mean_dist

        self._prior_uvcoords_sample = prior_uvcoords_sample

        self.face_scaling_canonical = avatar_model.face_scaling_canonical

        self.flame = avatar_model.flame
        self.register_buffer('delta_shapedirs', avatar_model.delta_shapedirs)
        self.register_buffer('delta_posedirs', avatar_model.delta_posedirs)
        self.register_buffer('delta_vertex', avatar_model.delta_vertex)
        self.shell_len       = avatar_model.shell_len

        self.cfg_model       = avatar_model.cfg_model

        self.bg_color        = avatar_model.bg_color
        self.img_res         = avatar_model.img_res

        self.device          = avatar_model.device

    def _export_avatar_model(self, texture_dict: dict=None):
        
        if texture_dict is None:
            tex_out = self.decoder(self.const)

            _, _, value_dict = self._gather_attribute(
                tex_out,
                self.texture_layers,
                uv_coords=self._prior_uvcoords_sample
                # uv_coords=_posterior_uvcoords_sample,
                # tatoo_dict = tatoo_dict
            )
        
        else:
            value_dict = self._gather_attribute_from_texture_dict(
                texture_dict,
                uv_coords=self._prior_uvcoords_sample
            )

        decode_features_dc = value_dict['color']
        decode_opacity     = value_dict['opacity']
        decode_scaling     = value_dict['scaling']
        decode_rotation    = value_dict['rotation']
        decode_offset      = value_dict['offset']

        output = {
            # ----- value list ----- #
            'decode_color':     decode_features_dc.permute(1, 0, 2),
            'decode_opacity':   decode_opacity[0],
            'decode_scaling':   decode_scaling[0],
            'decode_rotation':  decode_rotation[0],
            'decode_offset':    decode_offset[0],
        }

        self.avatar_model._features_dc.data = output['decode_color'].data
        self.avatar_model._opacity.data     = output['decode_opacity'].data
        self.avatar_model._offset.data      = output['decode_offset'].data
        self.avatar_model._rotation.data    = output['decode_rotation'].data
        self.avatar_model._scaling.data     = output['decode_scaling'].data

        self.avatar_model.face_index        = self._prior_face_index
        self.avatar_model.bary_coords       = self._prior_bary_coords

        return self.avatar_model

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

        tex_out = self.decoder(self.const)

        # _posterior_uvcoords_sample = self.add_sample_noise(self._prior_uvcoords_sample)

        texture_dict, act_texture_dict, value_dict = self._gather_attribute(
            tex_out,
            self.texture_layers,
            uv_coords=self._prior_uvcoords_sample
            # uv_coords=_posterior_uvcoords_sample,
            # tatoo_dict = tatoo_dict
        )

        verts, _, _ = self.flame.forward_with_delta_blendshape(
            expression_params   = expression,
            full_pose           = flame_pose,
            delta_shapedirs     = self.delta_shapedirs if self.cfg_model.delta_blendshape else None,
            delta_posedirs      = self.delta_posedirs if self.cfg_model.delta_blendshape else None,
            delta_vertex        = self.delta_vertex if self.cfg_model.delta_vertex else None
        )

        verts_orig, _, _ = self.flame.forward(expression_params   = expression,
                                              full_pose           = flame_pose,)
        
        face_orien_mat, face_scaling    = compute_face_orientation(verts, self.faces, return_scale=True)
        face_normals                    = compute_face_normals(verts, self.faces)

        scaling_ratio       = face_scaling / self.face_scaling_canonical
        flame_scaling_ratio = scaling_ratio[:, self._prior_face_index]

        flame_orien_mat     = face_orien_mat[:, self._prior_face_index]
        flame_orien_quat    = matrix_to_quaternion(flame_orien_mat)
        flame_normals       = face_normals[:, self._prior_face_index]

        pos_val = reweight_verts_by_barycoords(
            verts         = verts,
            faces         = self.faces,
            face_index    = self._prior_face_index,
            bary_coords   = self._prior_bary_coords
        )
        
        #-------------------------------    render gaussian     -------------------------------#

        gaussian = GaussianModel(sh_degree = 0)

        render_image = []
        viewspace_points = []
        visibility_filter = []
        radii = []

        decode_features_dc = value_dict['color']
        decode_opacity     = value_dict['opacity']
        decode_scaling     = value_dict['scaling']
        decode_rotation    = value_dict['rotation']
        decode_offset      = value_dict['offset']

        prior_attr_dict = {
            'color':    self._prior_features_dc,
            'opacity':  self._prior_opacity,
            'scaling':  self._prior_scaling,
            'rotation': self._prior_rotation,
            'offset':   self._prior_offset
        }

        attribute_list = prior_attr_dict.keys()

        # dummy loop
        for bs_ in range(bs):

            decode_attr_dict = {
                'color':    decode_features_dc[bs_],
                'opacity':  decode_opacity[bs_],
                'scaling':  decode_scaling[bs_],
                'rotation': decode_rotation[bs_],
                'offset':   decode_offset[bs_]
            }

            gaussian_attr_dict = {}

            for attr in attribute_list:
                if attr in self.bake_attribute:
                    gaussian_attr_dict.update({attr: decode_attr_dict[attr]})
                else:
                    gaussian_attr_dict.update({attr: prior_attr_dict[attr]})

            gaussian._features_dc   = gaussian_attr_dict['color'].contiguous().unsqueeze(1)
            gaussian._features_rest = gaussian_attr_dict['opacity'].contiguous()
            gaussian._opacity       = decode_opacity[bs_].contiguous()
            gaussian._scaling       = (gaussian_attr_dict['scaling'] + torch.log(flame_scaling_ratio[bs_])).contiguous() if self.cfg_model.resize_scale else self._scaling[bs_]
            gaussian._rotation      = quaternion_multiply(flame_orien_quat[bs_], gaussian_attr_dict['rotation']).contiguous()
            gaussian._xyz           = pos_val[bs_] + flame_normals[bs_] * self.shell_len * torch.tanh(gaussian_attr_dict['offset']).contiguous()

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
            'scale': torch.exp(decode_scaling),
            'raw_rot': quaternion_to_axis_angle(decode_rotation),
            # ----- gaussian maintainer ----- #
            'viewspace_points': viewspace_points,   # List
            'visibility_filter': visibility_filter, # List
            'radii': radii, # List
            'bs': bs,
            # ----- mesh loss ----- #
            'verts': verts,
            'verts_orig': verts_orig,
            'faces': self.faces,
            # ----- texture dict ----- #
            'texture_dict': texture_dict,
            'act_texture_dict': act_texture_dict,
            # ----- reg loss ----- #
            'decode_color': decode_features_dc.permute(1, 0, 2),
            'decode_opacity': decode_opacity[0],
            'decode_scaling': decode_scaling[0],
            'decode_rotation': decode_rotation[0],
            'decode_offset': decode_offset[0],
            'prior_features_dc': self._prior_features_dc,
            'prior_opacity': self._prior_opacity,
            'prior_scaling': self._prior_scaling,
            'prior_rotation': self._prior_rotation,
            'prior_offset': self._prior_offset,
        }

        return output
    
    @torch.no_grad()
    def visualization(self, input):
        output = self.forward(input)
        return output
    
    def add_sample_noise(self, uv_coords):

        uv_coords_ = uv_coords.squeeze(0)
        N = uv_coords_.shape[0]

        theta = 2 * torch.pi * torch.rand(N).to(self.device)
        r     = self.sample_radius * torch.rand(N).to(self.device)

        sampled_x = uv_coords_[:, 0] + r * torch.cos(theta)
        sampled_y = uv_coords_[:, 1] + r * torch.sin(theta)

        sampled_points = torch.stack((sampled_x, sampled_y), dim=1).unsqueeze(0)

        return sampled_points

    @torch.no_grad()
    def render_from_texture_dict(self, input, input_texture_dict = None):

        if input_texture_dict is None:
            input_texture_dict = self.texture_dict_cache

        cam_pose = input["cam_pose"].clone()
        fovx = input["fovx"][0]
        fovy = input["fovy"][0]
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]

        camera = Camera(R=R, T=T, FoVx=fovx, FoVy=fovy, img_res=self.img_res)

        flame_pose = input["flame_pose"]
        expression = input["expression"]
        bs = flame_pose.shape[0]    # 1, essentially

        value_dict = self._gather_attribute_from_texture_dict(
            input_texture_dict,
            uv_coords=self._prior_uvcoords_sample
        )

        verts, _, _ = self.flame.forward_with_delta_blendshape(
            expression_params   = expression,
            full_pose           = flame_pose,
            delta_shapedirs     = self.delta_shapedirs if self.cfg_model.delta_blendshape else None,
            delta_posedirs      = self.delta_posedirs if self.cfg_model.delta_blendshape else None,
            delta_vertex        = self.delta_vertex if self.cfg_model.delta_vertex else None
        )

        verts_orig, _, _ = self.flame.forward(expression_params   = expression,
                                              full_pose           = flame_pose,)
        
        face_orien_mat, face_scaling    = compute_face_orientation(verts, self.faces, return_scale=True)
        face_normals                    = compute_face_normals(verts, self.faces)

        scaling_ratio       = face_scaling / self.face_scaling_canonical
        flame_scaling_ratio = scaling_ratio[:, self._prior_face_index]

        flame_orien_mat     = face_orien_mat[:, self._prior_face_index]
        flame_orien_quat    = matrix_to_quaternion(flame_orien_mat)
        flame_normals       = face_normals[:, self._prior_face_index]

        pos_val = reweight_verts_by_barycoords(
            verts         = verts,
            faces         = self.faces,
            face_index    = self._prior_face_index,
            bary_coords   = self._prior_bary_coords
        )
        
        #-------------------------------    render gaussian     -------------------------------#

        gaussian = GaussianModel(sh_degree = 0)

        render_image = []
        viewspace_points = []
        visibility_filter = []
        radii = []

        decode_features_dc = value_dict['color']
        decode_opacity     = value_dict['opacity']
        decode_scaling     = value_dict['scaling']
        decode_rotation    = value_dict['rotation']
        decode_offset      = value_dict['offset']

        prior_attr_dict = {
            'color':    self._prior_features_dc,
            'opacity':  self._prior_opacity,
            'scaling':  self._prior_scaling,
            'rotation': self._prior_rotation,
            'offset':   self._prior_offset
        }

        attribute_list = prior_attr_dict.keys()

        # dummy loop
        for bs_ in range(bs):
            decode_attr_dict = {
                'color':    decode_features_dc[bs_],
                'opacity':  decode_opacity[bs_],
                'scaling':  decode_scaling[bs_],
                'rotation': decode_rotation[bs_],
                'offset':   decode_offset[bs_]
            }

            gaussian_attr_dict = {}

            for attr in attribute_list:
                if attr in self.bake_attribute:
                    gaussian_attr_dict.update({attr: decode_attr_dict[attr]})
                else:
                    gaussian_attr_dict.update({attr: prior_attr_dict[attr]})

            gaussian._features_dc   = gaussian_attr_dict['color'].contiguous().unsqueeze(1)
            gaussian._features_rest = gaussian_attr_dict['opacity'].contiguous()
            gaussian._opacity       = decode_opacity[bs_].contiguous()
            gaussian._scaling       = (gaussian_attr_dict['scaling'] + torch.log(flame_scaling_ratio[bs_])).contiguous() if self.cfg_model.resize_scale else self._scaling[bs_]
            gaussian._rotation      = quaternion_multiply(flame_orien_quat[bs_], gaussian_attr_dict['rotation']).contiguous()
            gaussian._xyz           = pos_val[bs_] + flame_normals[bs_] * self.shell_len * torch.tanh(gaussian_attr_dict['offset']).contiguous()

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
            'rgb_image': render_image
        }

        return output
    






        