import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flame.FLAME    import FLAME

from pytorch3d.ops  import knn_points
from functorch      import jacfwd, vmap

from volume_rendering.camera_3dgs       import Camera
from diff_gaussian_rasterization        import GaussianRasterizationSettings, GaussianRasterizer

from tools.util import get_bg_color

#-------------------------------------------------------------------------------#

class MonoGaussianAvatar(nn.Module):
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
            https://github.com/yufan1012/MonoGaussianAvatar/tree/main

        bibtex:
            @inproceedings{chen2024monogaussianavatar,
            title       = {Monogaussianavatar: Monocular gaussian point-based head avatar},
            author      = {Chen, Yufan and Wang, Lizhen and Li, Qijing and Xiao, Hongjiang and Zhang, Shengping and Yao, Hongxun and Liu, Yebin},
            booktitle   = {ACM SIGGRAPH 2024 Conference Papers},
            pages       = {1--9},
            year        = {2024}
            }
        """

        # in monogaussianavatar, camera scene settings is heavily related to radius/scale...
        dataset_type = getattr(cfg_model, 'dataset_type', None)

        if dataset_type is None:
            raise ValueError("dataset_type is not defined in cfg_model")

        if dataset_type == 'insta':
            self.cam_scale = 3
            self.scene_scale = 1
        elif dataset_type == 'imavatar':
            self.cam_scale = 4
            self.scene_scale = 1
        else:
            raise NotImplementedError(f"Dataset type '{dataset_type}' is not implemented")
        
        self.dataset_type   = dataset_type

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
        
        self._register_networks(cfg_model)

        self._register_points(cfg_model)

        self.scale_ac       = torch.sigmoid
        self.rotations_ac   = torch.nn.functional.normalize
        self.opacity_ac     = torch.sigmoid
        self.color_ac       = torch.sigmoid
        

    def _register_flame(self, flame_path, landmark_embedding_path):

        self.flame = FLAME(
            flame_path,
            landmark_embedding_path,
            n_shape              = self.cfg_model.n_shape,
            n_exp                = self.cfg_model.n_exp,
            shape_params         = self.shape_params,
            canonical_expression = self.canonical_expression,
            canonical_pose       = self.canonical_pose,
            device               = self.device,
            factor               = 4,
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


    def _register_networks(self, conf):

        deformer_dict   = dict(conf.deformer_network)
        deformer_dict.update({'num_exp': self.cfg_model.n_exp})

        self.prune_thresh               = conf.prune_thresh
        self.geometry_network           = GeometryNetwork(**conf.geometry_network)
        self.deformer_network           = ForwardDeformer(FLAMEServer=self.flame, **deformer_dict)
        self.rendering_network          = RenderingNetwork(**conf.rendering_network)
        self.gaussian_deformer_network  = GaussianNetwork(**conf.gaussian_network)
        self.ghostbone                  = self.deformer_network.ghostbone

        if self.ghostbone:
            I = torch.eye(4).unsqueeze(0).unsqueeze(0).float().to(self.device)
            self.flame.canonical_transformations = torch.cat([I, self.flame.canonical_transformations], 1)

    def _register_points(self, conf):

        n_init_points   = conf.point_cloud.n_init_points
        max_points      = conf.point_cloud.max_points
        init_radius     = 0.5 / self.scene_scale
        
        self.pc = PointCloud(
            n_init_points  = n_init_points,
            init_radius    = init_radius,
            max_points     = max_points
        ).to(self.device)
        
        self.num_points = self.pc.points.shape[0]

        self.radius = 0.15 * (0.75 ** math.log2(self.num_points / 100)) / (self.scene_scale)

        self.visible_points = torch.zeros(self.num_points).bool().to(self.device)

    def forward(self, input):
        cam_pose = input["cam_pose"].clone()
        fovx = input["fovx"][0]
        fovy = input["fovy"][0]
        R = cam_pose[:, :3, :3]
        T = cam_pose[:, :3, 3]

        camera = Camera(
            R=R, T=T * self.cam_scale,
            FoVx=fovx, FoVy=fovy, img_res=self.img_res
        )

        flame_pose = input["flame_pose"]
        expression = input["expression"]
        bs = flame_pose.shape[0]    # 1, essentially

        n_points    = self.pc.points.shape[0]
        total_points    = n_points * bs

        verts, pose_feature, transformations = self.flame.forward(
            expression_params      = expression,
            full_pose               = flame_pose
        )
        
        if self.ghostbone:
            I = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(bs, -1, -1, -1).float().to(self.device)
            transformations = torch.cat([I, transformations], 1)

        feature_vector = self._compute_canonical_normals_and_feature_vectors()

        transformed_points, rgb_points, scale_vals, rotation_vals, opacity_vals = self.get_rbg_value_functorch(
            pnts_c          = self.pc.points,
            feature_vectors = feature_vector,
            pose_feature    = pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
            betas           = expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
            transformations = transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
        )
        
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach())
        transformed_points = transformed_points.reshape(bs, n_points, 3)
        scale       = scale_vals.reshape(transformed_points.shape[0], -1, 3)
        rotation    = rotation_vals.reshape(transformed_points.shape[0], -1, 4)
        opacity     = opacity_vals.reshape(transformed_points.shape[0], -1, 1)
        offset      = transformed_points.detach() - pnts_c_flame.detach()

        offset_scale, offset_rotation, offset_opacity, offset_color = self.gaussian_deformer_network(offset)
        scale       = scale + offset_scale
        rotation    = rotation + offset_rotation
        opacity     = opacity + offset_opacity
        rgb_points  = rgb_points.reshape(bs, n_points, 3)

        rgb_points  = rgb_points + offset_color
        rgb_points  = self.color_ac(rgb_points)
        scale       = self.scale_ac(scale)
        scale       = scale * 0.025 / (self.scene_scale)   #512
        rotation    = self.rotations_ac(rotation)
        opacity     = self.opacity_ac(opacity)

        rendering_list = []
        for idx in range(bs):

            xyz_i       = transformed_points[idx]
            color_i     = rgb_points[idx]
            scale_i     = scale[idx]
            rotation_i  = rotation[idx]
            opacity_i   = opacity[idx]

            image, visible_points   = self._render(
                camera,
                self.bg_color,
                xyz_i,
                color_i,
                scale_i,
                rotation_i,
                opacity_i
            )

            self.visible_points[visible_points] = True
            rendering_list.append(image.unsqueeze(0))
 
        rgb_image   = torch.cat(rendering_list, dim=0)
        knn_v       = self.flame.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)

        if self.dataset_type == 'insta':
            flame_shapedirs = self.flame.shapedirs[..., 300: 300 + self.cfg_model.n_exp]
        else:
            flame_shapedirs = self.flame.shapedirs

        output = {
            'rgb_image':            rgb_image,
            'bs':                   bs,
            'index_batch':          index_batch,
            'posedirs':             posedirs,
            'shapedirs':            shapedirs,
            'lbs_weights':          lbs_weights,
            'flame_posedirs':       self.flame.posedirs,
            'flame_shapedirs':      flame_shapedirs,
            'flame_lbs_weights':    self.flame.lbs_weights,
            'visible_points':       self.visible_points,
            'visible_points_idx':   visible_points
        }

        return output
    
    @torch.no_grad()
    def visualization(self, input):
        output = self.forward(input)
        return output
    
    @torch.no_grad()
    def inference(self, expression, flame_pose, camera):

        bs = flame_pose.shape[0]    # 1, essentially

        n_points    = self.pc.points.shape[0]
        total_points    = n_points * bs

        verts, pose_feature, transformations = self.flame.forward(expression_params      = expression,
                                                                full_pose               = flame_pose)
        
        if self.ghostbone:
            I = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(bs, -1, -1, -1).float().to(self.device)
            transformations = torch.cat([I, transformations], 1)

        feature_vector = self._compute_canonical_normals_and_feature_vectors()

        transformed_points, rgb_points, scale_vals, rotation_vals, opacity_vals = self.get_rbg_value_functorch(
            pnts_c          = self.pc.points,
            feature_vectors = feature_vector,
            pose_feature    = pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
            betas           = expression.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
            transformations = transformations.unsqueeze(1).expand(-1, n_points, -1, -1, -1).reshape(total_points, *transformations.shape[1:]),
        )
        
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(self.pc.points.detach())
        transformed_points = transformed_points.reshape(bs, n_points, 3)
        scale       = scale_vals.reshape(transformed_points.shape[0], -1, 3)
        rotation    = rotation_vals.reshape(transformed_points.shape[0], -1, 4)
        opacity     = opacity_vals.reshape(transformed_points.shape[0], -1, 1)
        offset      = transformed_points.detach() - pnts_c_flame.detach()

        offset_scale, offset_rotation, offset_opacity, offset_color = self.gaussian_deformer_network(offset)
        scale       = scale + offset_scale
        rotation    = rotation + offset_rotation
        opacity     = opacity + offset_opacity
        rgb_points  = rgb_points.reshape(bs, n_points, 3)

        rgb_points  = rgb_points + offset_color
        rgb_points  = self.color_ac(rgb_points)
        scale       = self.scale_ac(scale)
        scale       = scale * 0.025 / (self.scene_scale)   #512
        rotation    = self.rotations_ac(rotation)
        opacity     = self.opacity_ac(opacity)

        rendering_list = []
        for idx in range(bs):

            xyz_i       = transformed_points[idx]
            color_i     = rgb_points[idx]
            scale_i     = scale[idx]
            rotation_i  = rotation[idx]
            opacity_i   = opacity[idx]

            image, visible_points   = self._render(
                camera,
                self.bg_color,
                xyz_i,
                color_i,
                scale_i,
                rotation_i,
                opacity_i
            )

            self.visible_points[visible_points] = True
            rendering_list.append(image.unsqueeze(0))
 
        rgb_image   = torch.cat(rendering_list, dim=0)
        knn_v       = self.flame.canonical_verts.unsqueeze(0).clone()
        _, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)


        return rgb_image[0]


    def get_rbg_value_functorch(self, pnts_c, feature_vectors, pose_feature, betas, transformations):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)
        total_points = betas.shape[0]
        batch_size = int(total_points / pnts_c.shape[0])
        n_points = pnts_c.shape[0]
        # pnts_c: n_points, 3
        def _func(pnts_c, betas, transformations, pose_feature):
            pnts_c = pnts_c.unsqueeze(0)
            shapedirs, posedirs, lbs_weights, pnts_c_flame = self.deformer_network.query_weights(pnts_c)
            shapedirs = shapedirs.expand(batch_size, -1, -1)
            posedirs = posedirs.expand(batch_size, -1, -1)
            lbs_weights = lbs_weights.expand(batch_size, -1)
            pnts_c_flame = pnts_c_flame.expand(batch_size, -1)
            pnts_d = self.flame.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights)
            pnts_d = pnts_d.reshape(-1)
            return pnts_d, pnts_d

        betas = betas.reshape(batch_size, n_points, *betas.shape[1:]).transpose(0, 1)
        transformations = transformations.reshape(batch_size, n_points, *transformations.shape[1:]).transpose(0, 1)
        pose_feature = pose_feature.reshape(batch_size, n_points, *pose_feature.shape[1:]).transpose(0, 1)
        grads_batch, pnts_d = vmap(jacfwd(_func, argnums=0, has_aux=True), out_dims=(0, 0))(pnts_c, betas, transformations, pose_feature)

        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)
        feature_vectors = feature_vectors.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_points, -1)
        rgb_vals = feature_vectors[:, 0:3]
        scale_vals = feature_vectors[:, 3:6]
        rotation_vals = feature_vectors[:, 6:10]
        opacity_vals = feature_vectors[:, 10:11]
        return pnts_d, rgb_vals, scale_vals, rotation_vals, opacity_vals


    def _compute_canonical_normals_and_feature_vectors(self):
        geometry_output, scales, rotations, opacity = self.geometry_network(self.pc.points.detach())
        feature_rgb_vector = geometry_output
        feature_scale_vector = scales
        feature_rotation_vector = rotations
        feature_opacity_vector = opacity
        feature_vector = torch.concat([feature_rgb_vector, feature_rotation_vector, feature_scale_vector, feature_opacity_vector], dim=1)
        return feature_vector


    def _render(self,
                viewpoint_camera,
                bg_color,
                xyz, color, scales, rotations, opacity):

        screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height    = int(viewpoint_camera.image_height),
            image_width     = int(viewpoint_camera.image_width),
            tanfovx         = tanfovx,
            tanfovy         = tanfovy,
            bg              = bg_color,
            scale_modifier  = 1.0,
            viewmatrix      = viewpoint_camera.world_view_transform,
            projmatrix      = viewpoint_camera.full_proj_transform,
            sh_degree       = 3,
            campos          = viewpoint_camera.camera_center,
            prefiltered     = False,
            debug           = False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        render_image, radii = rasterizer(
            means3D         = xyz,
            means2D         = screenspace_points,
            shs             = None,
            colors_precomp  = color,
            opacities       = opacity,
            scales          = scales + self.radius,
            # scales          = scales,
            # scales          = torch.ones_like(scales) * self.radius,
            rotations       = rotations,
            cov3D_precomp   = None)

        n_points = self.pc.points.shape[0]
        id = torch.arange(start=0, end=n_points, step=1).to(self.device)
        visible_points = id[opacity.reshape(-1) >= self.prune_thresh]
        visible_points = visible_points[visible_points != -1]

        return render_image, visible_points
    
    def _upsample_points(self, epoch):
        cur_radius  = self.radius
        points      = self.pc.points.data

        if epoch <= 100:
            noise   = (torch.rand(*points.shape).to(self.device) - 0.5) * cur_radius
        else:
            noise   = (torch.rand(*points.shape).to(self.device) - 0.5) * 0.004

        new_points  = noise + points

        if epoch < 5:
            self.pc.upsample_points(new_points, 400)
        elif 5 <= epoch < 10:
            self.pc.upsample_points(new_points, 800)
        elif 10 <= epoch < 15:
            self.pc.upsample_points(new_points, 1600)
        elif 15 <= epoch < 20:
            self.pc.upsample_points(new_points, 3200)
        elif 20 <= epoch < 25:
            self.pc.upsample_points(new_points, 6400)
        elif 25 <= epoch < 30:
            self.pc.upsample_points(new_points, 10000)
        elif 30 <= epoch < 40:
            self.pc.upsample_points(new_points, 20000)
        elif 40 <= epoch < 50:
            self.pc.upsample_points(new_points, 40000)
        elif 50 <= epoch < 60:
            self.pc.upsample_points(new_points, 80000)
        elif epoch >= 60:
            self.pc.upsample_points(new_points, 100000)

        if epoch in [5, 10, 15, 20, 25, 30, 40, 50]:
            self.radius = 0.75 * cur_radius
        elif epoch == 60:
            self.radius = 0.9 * cur_radius
        elif epoch > 60 and epoch % 5 == 0:
            self.radius = 0.75 * cur_radius

#-------------------------------------------------------------------------------#
                    # utils functions in MonoGaussianAvatar # 
#-------------------------------------------------------------------------------#

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

#-------------------------------------------------------------------------------#

def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

#-------------------------------------------------------------------------------#

class PointCloud(nn.Module):
    def __init__(
        self,
        n_init_points,
        max_points=131072,
        init_radius=0.5,
        radius_factor=0.15
    ):
        super(PointCloud, self).__init__()
        self.radius_factor = radius_factor
        self.max_points = max_points
        self.init_radius = init_radius
        self.init(n_init_points)

    def init(self, n_init_points):
        print("current point number: ", n_init_points)
        # initialize sphere
        init_points = (torch.rand(n_init_points, 3) * 2.0 - 1.0)

        init_normals = nn.functional.normalize(init_points, dim=1)
        init_points = init_normals * self.init_radius
        self.register_parameter("points", nn.Parameter(init_points))

    def prune(self, visible_points):
        """Prune not rendered points"""
        self.points = nn.Parameter(self.points.data[visible_points])
        print(
            "Pruning points, original: {}, new: {}".format(
                len(visible_points), sum(visible_points)
            )
        )

    def upsample_points(self, new_points, target_num_points):
        num_points = self.points.shape[0]
        num_upsample = target_num_points - num_points
        if num_upsample <= 0:
            print(f"No upsampling needed, current points: {num_points}")
            return
        rnd_idx = torch.randint(0, new_points.shape[0], (num_upsample,))
        upsample_point = new_points[rnd_idx, :]
        self.points = nn.Parameter(torch.cat([self.points, upsample_point], dim=0))
        print(f"Upsampled to {self.points.shape[0]} points.")

#-------------------------------------------------------------------------------#

class GeometryNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
    ):
        super().__init__()
        dims = [d_in] + dims
        self.feature_vector_size = feature_vector_size
        self.embed_fn = None
        self.multires = multires
        self.bias = bias
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        self.softplus = nn.Softplus(beta=100)
        self.scaling_layer = nn.Sequential(nn.Linear(256, 256), self.softplus,
                                           nn.Linear(256, 3))
        self.rotations_layer = nn.Sequential(nn.Linear(256, 256), self.softplus,
                                             nn.Linear(256, 4))
        self.opacity_layer = nn.Sequential(nn.Linear(256, 256), self.softplus,
                                           nn.Linear(256, 1))
        self.scale_ac = nn.Softplus(beta=100)
        self.rotations_ac = nn.functional.normalize
        self.opacity_ac = nn.Sigmoid()

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.out_layer = nn.Sequential(nn.Linear(256, 256), self.softplus,
                                       nn.Linear(256, 256), nn.Linear(256, 3))

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)
        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 1:
                x = self.softplus(x)

        color = self.out_layer(x)
        scales = self.scaling_layer(x)
        rotations = self.rotations_layer(x)
        opacity = self.opacity_layer(x)

        return color, scales, rotations, opacity
    
#-------------------------------------------------------------------------------#

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            multires_pnts=0,
    ):
        super().__init__()

        dims = [d_in + feature_vector_size] + dims + [d_out]
        self.d_in = d_in
        self.embedview_fn = None
        self.embedpnts_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        if multires_pnts > 0:
            embedpnts_fn, input_ch_pnts = get_embedder(multires_pnts)
            self.embedpnts_fn = embedpnts_fn
            dims[0] += (input_ch_pnts - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, normals):
        x = normals

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.sigmoid(x)
        return x

#-------------------------------------------------------------------------------#

class GaussianNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            multires_pnts=0,
    ):
        super().__init__()

        dims = [d_in + feature_vector_size] + dims
        self.d_in = d_in
        self.embedview_fn = None
        self.embedpnts_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        if multires_pnts > 0:
            embedpnts_fn, input_ch_pnts = get_embedder(multires_pnts)
            self.embedpnts_fn = embedpnts_fn
            dims[0] += (input_ch_pnts - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus(beta=100)
        self.scaling_layer = nn.Sequential(nn.Linear(64, 64), self.relu,
                                           nn.Linear(64, 3))
        self.rotations_layer = nn.Sequential(nn.Linear(64, 64), self.relu,
                                             nn.Linear(64, 4))
        self.opacity_layer = nn.Sequential(nn.Linear(64, 64), self.relu,
                                           nn.Linear(64, 1))
        self.color_layer = nn.Sequential(nn.Linear(64, 64), self.relu,
                                           nn.Linear(64, 3))

    def forward(self, offset):
        x = offset

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)

        offset_s = self.scaling_layer(x)
        offset_r = self.rotations_layer(x)
        offset_o = self.opacity_layer(x)
        offset_c = self.color_layer(x)
        return offset_s, offset_r, offset_o, offset_c
    
#-------------------------------------------------------------------------------#

class ForwardDeformer(nn.Module):
    def __init__(
        self,
        FLAMEServer,
        d_in,
        dims,
        multires,
        num_exp=50,
        deform_c=False,
        weight_norm=True,
        ghostbone=False,
    ):
        super().__init__()
        self.FLAMEServer = FLAMEServer
        # pose correctives, expression blendshapes and linear blend skinning weights
        d_out = 36 * 3 + num_exp * 3
        if deform_c:
            d_out = d_out + 3
        self.num_exp = num_exp
        self.deform_c = deform_c
        dims = [d_in] + dims + [d_out]
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 2):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            torch.nn.init.constant_(lin.bias, 0.0)
            torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.blendshapes = nn.Linear(dims[self.num_layers - 2], d_out)
        self.skinning_linear = nn.Linear(dims[self.num_layers - 2], dims[self.num_layers - 2])
        self.skinning = nn.Linear(dims[self.num_layers - 2], 6 if ghostbone else 5)
        torch.nn.init.constant_(self.skinning_linear.bias, 0.0)
        torch.nn.init.normal_(self.skinning_linear.weight, 0.0, np.sqrt(2) / np.sqrt(dims[self.num_layers - 2]))
        if weight_norm:
            self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
        # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
        torch.nn.init.constant_(self.blendshapes.bias, 0.0)
        torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        torch.nn.init.constant_(self.skinning.bias, 0.0)
        torch.nn.init.constant_(self.skinning.weight, 0.0)

        self.ghostbone = ghostbone

    def query_weights(self, pnts_c, mask=None):
        if mask is not None:
            pnts_c = pnts_c[mask]
        if self.embed_fn is not None:
            x = self.embed_fn(pnts_c)
        else:
            x = pnts_c

        for l in range(0, self.num_layers - 2):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            x = self.softplus(x)

        blendshapes = self.blendshapes(x)
        posedirs = blendshapes[:, :36 * 3]
        shapedirs = blendshapes[:, 36 * 3: 36 * 3 + self.num_exp * 3]
        lbs_weights = self.skinning(self.softplus(self.skinning_linear(x)))
        # softmax implementation
        lbs_weights_exp = torch.exp(20 * lbs_weights)
        lbs_weights = lbs_weights_exp / torch.sum(lbs_weights_exp, dim=-1, keepdim=True)
        if self.deform_c:
            pnts_c_flame = pnts_c + blendshapes[:, -3:]
        else:
            pnts_c_flame = pnts_c
        return shapedirs.reshape(-1, 3, self.num_exp), posedirs.reshape(-1, 4*9, 3), lbs_weights.reshape(-1, 6 if self.ghostbone else 5), pnts_c_flame

    def forward_lbs(self, pnts_c, pose_feature, betas, transformations, mask=None):
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.query_weights(pnts_c, mask)
        pts_p = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32)
        return pts_p, pnts_c_flame