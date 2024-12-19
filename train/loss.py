from    __future__ import annotations

import  torch
from    torch import nn
import  torch.nn.functional as F
import  warnings
import  lpips

from    tools.loss_utils.dssim import d_ssim
from    tools.loss_utils.vgg_feature import VGGPerceptualLoss

from    pytorch3d.structures import     Meshes
from    pytorch3d.loss.mesh_laplacian_smoothing import   mesh_laplacian_smoothing
from    pytorch3d.loss.mesh_normal_consistency  import   mesh_normal_consistency

from    typing import Type, Union
from    dataclasses import dataclass, field

warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13")

# ------------------------------------------------------------------------------- #

def parsing_loss_param(loss_class:  Type[BaseLoss],
                       cfg_loss:    dict):
    """
    Args:
        loss_class      : Loss function class, which should have a 
                            nested `Params` class with annotated parameters.

        cfg_loss (dict): A dictionary containing the configuration for the loss 
                            function, including possible weight values and other loss parameters.

    Returns:
        BaseLoss.Params: An instance of the `Params` class of `loss_class`, initialized 
                         with the extracted parameters from `cfg_loss`.
    """
    
    param_class = loss_class.Params

    # ugly fix
    param_map = {key.replace('_weight', '_loss'): key for key in param_class.__annotations__}

    param_dict = {}
    for key, mapped_key in param_map.items():
        if key in cfg_loss:
            param_dict[mapped_key] = cfg_loss[key]
        elif key in cfg_loss.weight:
            param_dict[mapped_key] = cfg_loss['weight'][key]

    return param_class(**param_dict)

# ------------------------------------------------------------------------------- #

class BaseLoss(nn.Module):

    @dataclass
    class Params:
        loss_weight:        float

    def accumulate_gradients(self, model_output, ground_truth, cur_step=None, cur_epoch=None): # to be overridden by subclass
        raise NotImplementedError
    
    def forward(self, model_output, ground_truth, cur_step=None, cur_epoch=None):
        return self.accumulate_gradients(model_output, ground_truth, cur_step, cur_epoch)
    
# ------------------------------------------------------------------------------- #

class FateAvatarLoss(BaseLoss):

    @dataclass
    class Params:
        rgb_type:           str     = 'l1'
        rgb_weight:         float   = 1
        vgg_weight:         float   = 0
        dssim_weight:       float   = 0
        scale_weight:       float   = 0
        lpips_weight:       float   = 0
        scale_threshold:    float   = 0
        rot_weight:         float   = 0
        laplacian_weight:   float   = 0
        normal_weight:      float   = 0
        flame_weight:       float   = 0
    
    def __init__(self, params: Params):
        super().__init__()
        
        self.params = params

        self.vgg_loss               = VGGPerceptualLoss()
        self.lpips_loss             = lpips.LPIPS(net='vgg').eval()
        self.l1_loss                = nn.L1Loss(reduction='mean')
        self.l2_loss                = nn.MSELoss(reduction='mean')

        self.laplacian_matrix       = None

    def get_dssim_loss(self, rgb_values, rgb_gt):
        return d_ssim(rgb_values, rgb_gt)

    def get_vgg_loss(self, rgb_values, rgb_gt):
        return self.vgg_loss(rgb_values, rgb_gt)

    def get_rgb_loss(self, rgb_values, rgb_gt):
        if self.params.rgb_type == 'l1':
            return self.l1_loss(rgb_values, rgb_gt)
        elif self.params.rgb_type == 'l2':
            return self.l2_loss(rgb_values, rgb_gt)
    
    def get_lpips_loss(self, rgb_values, rgb_gt, normalize=True):
        return self.lpips_loss(rgb_values, rgb_gt, normalize=normalize)

    def get_laplacian_smoothing_loss(self, verts_orig, verts):
        L = self.laplacian_matrix[None, ...].detach()

        basis_lap   = L.bmm(verts_orig).detach()
        offset_lap  = L.bmm(verts)

        diff = (offset_lap - basis_lap) ** 2
        diff = diff.sum(dim=-1, keepdim=True)

        return diff.mean()

    def accumulate_gradients(self, model_outputs, ground_truth, cur_step=None, cur_epoch=None):

        render_image = model_outputs['rgb_image']   # torch.Size([1, 3, 512, 512])
        gt_image     = ground_truth['rgb']          # torch.Size([1, 3, 512, 512])

        # Initialize the loss
        loss = self.get_rgb_loss(render_image, gt_image) * self.params.rgb_weight
        out = {'loss': loss, 'rgb_loss': loss}

        # vgg loss
        if self.params.vgg_weight > 0:
            vgg_loss = self.get_vgg_loss(render_image, gt_image)
            out['vgg_loss'] = vgg_loss
            out['loss'] += vgg_loss * self.params.vgg_weight

        # dssim loss
        if self.params.dssim_weight > 0:
            dssim_loss = self.get_dssim_loss(render_image, gt_image)
            out['dssim_loss'] = dssim_loss
            out['loss'] += dssim_loss * self.params.dssim_weight

        # scale loss
        if self.params.scale_weight > 0:
            scale = model_outputs['scale']
            scale_max, _ = torch.max(scale, dim=-1)
            scale_min, _ = torch.min(scale, dim=-1)
            scale_regu = F.relu(scale_max / scale_min - self.params.scale_threshold).mean()
            out['scale_loss'] = scale_regu
            out['loss'] += scale_regu * self.params.scale_weight

        # lpips loss
        if self.params.lpips_weight > 0:
            lpips_loss = self.get_lpips_loss(render_image, gt_image).squeeze()
            out['lpips_loss'] = lpips_loss
            out['loss'] += lpips_loss * self.params.lpips_weight

        # rotation loss
        if self.params.rot_weight > 0:
            raw_rot = model_outputs['raw_rot']
            rot_loss = torch.mean(raw_rot[..., 0] ** 2) + torch.mean(raw_rot[..., 2] ** 2)
            out['rot_loss'] = rot_loss
            out['loss'] += rot_loss * self.params.rot_weight

        # laplacian or normal loss
        if self.params.laplacian_weight > 0 or self.params.normal_weight > 0:
            verts = model_outputs['verts']  # [1, V, 3]
            faces = model_outputs['faces']  # [F, 3]
            meshes = Meshes(verts=verts, faces=faces[None, ...])

            if self.laplacian_matrix is None:
                self.laplacian_matrix = meshes.laplacian_packed().to_dense()

            if self.params.laplacian_weight > 0:
                verts = model_outputs['verts']
                verts_orig = model_outputs['verts_orig']
                laplacian_loss  = self.get_laplacian_smoothing_loss(verts_orig, verts)
                out['laplacian_loss'] = laplacian_loss
                out['loss'] += laplacian_loss * self.params.laplacian_weight

                # laplacian_loss = mesh_laplacian_smoothing(meshes)
                # out['laplacian_loss'] = laplacian_loss
                # out['loss'] += laplacian_loss * self.params.laplacian_weight

            if self.params.normal_weight > 0:
                normal_loss = mesh_normal_consistency(meshes)
                out['normal_loss'] = normal_loss
                out['loss'] += normal_loss * self.params.normal_weight

        # flame loss
        if self.params.flame_weight > 0:
            verts = model_outputs['verts']
            verts_orig = model_outputs['verts_orig']
            flame_loss = (verts - verts_orig) ** 2
            out['flame_loss'] = flame_loss.mean()
            out['loss'] += out['flame_loss'] * self.params.flame_weight

        return out

# ------------------------------------------------------------------------------- #

class FlashAvatarLoss(BaseLoss):

    @dataclass
    class Params:
        huber_weight: float = 1
        lpips_weight: float = 0.05

    def __init__(self, params: Params):
        super().__init__()
        
        self.params = params

        self.lpips_loss = lpips.LPIPS(net='vgg').eval()

    def get_huber_loss(self, rgb_values, rgb_gt, alpha):
        diff = torch.abs(rgb_values - rgb_gt)
        mask = (diff < alpha).float()
        loss = 0.5 * diff ** 2 * mask + alpha * (diff - 0.5 * alpha) * (1.0 - mask)
        return loss.mean()

    def get_lpips_loss(self, rgb_values, rgb_gt, normalize=True):
        return self.lpips_loss(rgb_values, rgb_gt, normalize=normalize)

    def accumulate_gradients(self, model_outputs, ground_truth, cur_step=None, cur_epoch=None):

        render_image = model_outputs['rgb_image']   # torch.Size([1, 3, 512, 512])
        gt_image = ground_truth['rgb']              # torch.Size([1, 3, 512, 512])

        # huber loss
        loss_huber = self.get_huber_loss(render_image, gt_image, 0.1)

        # check for 'mouth_mask' and compute mouth-specific loss
        if 'mouth_mask' in ground_truth and ground_truth['mouth_mask'] is not None:
            mouth_mask = ground_truth['mouth_mask']
            mouth_render = render_image * mouth_mask
            mouth_gt = gt_image * mouth_mask
            loss_huber += 40 * self.get_huber_loss(mouth_render, mouth_gt, 0.1)

        # initialize output
        out = {'loss': loss_huber, 'huber_loss': loss_huber}

        # compute lpips loss if weight is greater than 0
        if self.params.lpips_weight > 0:
            lpips_loss = self.get_lpips_loss(render_image, gt_image, normalize=True).squeeze()
            out['lpips_loss'] = lpips_loss

            # apply lpips weight based on step
            if cur_step is None or cur_step > 15000:
                out['loss'] += lpips_loss * self.params.lpips_weight
            else:
                out['loss'] += lpips_loss * 0  # No lpips loss during initial steps

        return out

# ------------------------------------------------------------------------------- #

class SplattingAvatarLoss(BaseLoss):

    @dataclass
    class Params:
        rgb_weight:         float = 1.0
        mse_weight:         float = 10.0
        scale_weight:       float = 1.0
        lpips_weight:       float = 0.01
        scale_threshold:    float = 10.0
        max_scaling:        float = 0.008

    def __init__(self, params: Params):
        super().__init__()
        
        self.params = params

        self.lpips_loss = lpips.LPIPS(net='vgg').eval()
        self.l1_loss    = nn.L1Loss(reduction='mean')
        self.l2_loss    = nn.MSELoss(reduction='mean')

    def get_rgb_loss(self, rgb_values, rgb_gt):
        return self.l1_loss(rgb_values, rgb_gt)
    
    def get_mse_loss(self, rgb_values, rgb_gt):
        return self.l2_loss(rgb_values, rgb_gt)

    def get_lpips_loss(self, rgb_values, rgb_gt, normalize=True):
        return self.lpips_loss(rgb_values, rgb_gt, normalize=normalize)

    def accumulate_gradients(self, model_outputs, ground_truth, cur_step=None, cur_epoch=None):

        render_image = model_outputs['rgb_image']   # torch.Size([1, 3, 512, 512])
        gt_image = ground_truth['rgb']              # torch.Size([1, 3, 512, 512])

        
        rgb_loss = self.get_rgb_loss(render_image, gt_image)
        loss = rgb_loss.clone() * self.params.rgb_weight

        # initialize the loss
        out = {'loss': loss, 'rgb_loss': rgb_loss}

        # mse loss
        if self.params.mse_weight > 0:
            mse_loss = self.get_mse_loss(render_image, gt_image)
            out['mse_loss'] = mse_loss
            out['loss'] += mse_loss * self.params.mse_weight

        # scale loss
        if self.params.scale_weight > 0:
            scale = model_outputs['scale']
            scale_max, _ = torch.max(scale, dim=-1)
            scale_min, _ = torch.min(scale, dim=-1)
            ratio = scale_max / scale_min
            thresh_idxs = (scale_max > self.params.max_scaling) & (ratio > self.params.scale_threshold)
            scale_regu = scale_max[thresh_idxs].mean() if thresh_idxs.any() else torch.tensor(0.0, device=scale_max.device)
            out['scale_loss'] = scale_regu
            out['loss'] += scale_regu * self.params.scale_weight

        # lpips loss
        if self.params.lpips_weight > 0:
            lpips_loss = self.get_lpips_loss(render_image, gt_image, normalize=True).squeeze()
            out['lpips_loss'] = lpips_loss
            out['loss'] += lpips_loss * self.params.lpips_weight

        return out
    
# ------------------------------------------------------------------------------- #

class GaussianAvatarsLoss(BaseLoss):

    @dataclass
    class Params:
        rgb_weight:         float = 0.8
        dssim_weight:       float = 0.2
        scale_weight:       float = 1.0
        xyz_weight:         float = 0.01
        threshold_scale:    float = 0.6
        threshold_xyz:      float = 1.0
    
    def __init__(self, params: Params):
        super().__init__()
        
        self.params = params

        self.l1_loss = nn.L1Loss(reduction='mean')

    def get_rgb_loss(self, rgb_values, rgb_gt):
        return self.l1_loss(rgb_values, rgb_gt)
    
    def get_dssim_loss(self, rgb_values, rgb_gt):
        return d_ssim(rgb_values, rgb_gt)

    def accumulate_gradients(self, model_outputs, ground_truth, cur_step=None, cur_epoch=None):

        render_image = model_outputs['rgb_image']   # torch.Size([1, 3, 512, 512])
        gt_image = ground_truth['rgb']              # torch.Size([1, 3, 512, 512])

        # initialize the loss
        rgb_loss = self.get_rgb_loss(render_image, gt_image)
        loss = rgb_loss.clone() * self.params.rgb_weight
        out = {'loss': loss, 'rgb_loss': rgb_loss}

        # dssim loss
        if self.params.dssim_weight > 0:
            dssim_loss = self.get_dssim_loss(render_image, gt_image)
            out['dssim_loss'] = dssim_loss
            out['loss'] += dssim_loss * self.params.dssim_weight

        # scale loss
        if self.params.scale_weight > 0:
            scale = model_outputs['scale']
            scale_regu = F.relu(scale - self.params.threshold_scale).norm(dim=1).mean()
            out['scale_loss'] = scale_regu
            out['loss'] += scale_regu * self.params.scale_weight

        # xyz loss
        if self.params.xyz_weight > 0:
            xyz = model_outputs['xyz']
            xyz_regu = F.relu(xyz.norm(dim=1) - self.params.threshold_xyz).mean()
            out['xyz_loss'] = xyz_regu
            out['loss'] += xyz_regu * self.params.xyz_weight

        return out

# ------------------------------------------------------------------------------- #

class MonoGaussianAvatarLoss(BaseLoss):

    @dataclass
    class Params:
        rgb_weight:         float = 1.0
        vgg_weight:         float = 0.1
        dssim_weight:       float = 0.25
        lbs_weight:         float = 10.0
        use_var_expression: bool = False
        var_expression:     torch.Tensor = None
        GT_lbs_milestones:  list = None
        GT_lbs_factor:      float = 0.5
        dataset_type:       str = 'insta'

    def __init__(self, params: Params):
        super().__init__()
        
        self.params = params

        self.vgg_loss   = VGGPerceptualLoss()
        self.l1_loss    = nn.L1Loss(reduction='mean')
        self.l2_loss    = nn.MSELoss(reduction='mean')

    def get_dssim_loss(self, rgb_values, rgb_gt):
        return d_ssim(rgb_values, rgb_gt)

    def get_vgg_loss(self, rgb_values, rgb_gt):
        return self.vgg_loss(rgb_values, rgb_gt)

    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss
    
    def get_gt_blendshape(self, index_batch, flame_lbs_weights, flame_posedirs, flame_shapedirs, ghostbone):
        if ghostbone:
            gt_lbs_weight = torch.zeros(len(index_batch), 6).cuda()
            gt_lbs_weight[:, 1:] = flame_lbs_weights[index_batch, :]
        else:
            gt_lbs_weight = flame_lbs_weights[index_batch, :]

        if self.params.dataset_type == 'insta':
            gt_shapedirs = flame_shapedirs[index_batch]
        else:
            gt_shapedirs = flame_shapedirs[index_batch, :, 100:]
            
        gt_posedirs = torch.transpose(flame_posedirs.reshape(36, -1, 3), 0, 1)[index_batch, :, :]

        output = {
            'gt_lbs_weights': gt_lbs_weight,
            'gt_posedirs': gt_posedirs,
            'gt_shapedirs': gt_shapedirs,
        }
        return output
    
    def get_lbs_loss(self, lbs_weight, gt_lbs_weight, use_var_expression=False):
        # the same function is used for lbs, shapedirs, posedirs.
        if use_var_expression and self.params.var_expression is not None:
            lbs_loss = torch.mean(self.l2_loss(lbs_weight, gt_lbs_weight) / self.params.var_expression / 50)
        else:
            lbs_loss = self.l2_loss(lbs_weight, gt_lbs_weight).mean()
        return lbs_loss

    def accumulate_gradients(self, model_outputs, ground_truth, cur_step=None, cur_epoch=None):

        if cur_epoch in self.params.GT_lbs_milestones:
            self.params.lbs_weight = self.params.lbs_weight * self.params.GT_lbs_factor

        render_image = model_outputs['rgb_image']   # torch.Size([1, 3, 512, 512])
        gt_image     = ground_truth['rgb']          # torch.Size([1, 3, 512, 512])

        rgb_loss    = self.get_rgb_loss(render_image, gt_image)
        loss        = rgb_loss.clone() * self.params.rgb_weight

        out = {'loss': loss, 'rgb_loss': rgb_loss}

        # vgg loss
        if self.params.vgg_weight > 0:
            vgg_loss            = self.get_vgg_loss(render_image, gt_image)
            out['vgg_loss']     = vgg_loss
            out['loss']         += vgg_loss * self.params.vgg_weight

        # dssim loss
        if self.params.dssim_weight > 0:
            dssim_loss          = self.get_dssim_loss(render_image, gt_image)
            out['dssim_loss']   = dssim_loss
            out['loss']         += dssim_loss * self.params.dssim_weight

        # lbs loss
        num_points  = model_outputs['lbs_weights'].shape[0]
        ghostbone   = model_outputs['lbs_weights'].shape[-1] == 6

        outputs     = self.get_gt_blendshape(
            model_outputs['index_batch'],
            model_outputs['flame_lbs_weights'],
            model_outputs['flame_posedirs'],
            model_outputs['flame_shapedirs'],
            ghostbone
        )
        
        lbs_loss    = self.get_lbs_loss(
            model_outputs['lbs_weights'].reshape(num_points, -1),
            outputs['gt_lbs_weights'].reshape(num_points, -1),
        )

        out['loss']     += lbs_loss * self.params.lbs_weight * 0.1
        out['lbs_loss'] = lbs_loss

        gt_posedirs     = outputs['gt_posedirs'].reshape(num_points, -1)
        posedirs_loss   = self.get_lbs_loss(
            model_outputs['posedirs'].reshape(num_points, -1) * 10,
            gt_posedirs* 10,
        )

        out['loss'] += posedirs_loss * self.params.lbs_weight * 10.0
        out['posedirs_loss'] = posedirs_loss

        gt_shapedirs = outputs['gt_shapedirs'].reshape(num_points, -1)

        if self.params.dataset_type == 'insta':
            pred_shapedirs = model_outputs['shapedirs'].reshape(num_points, -1)
        else:
            pred_shapedirs = model_outputs['shapedirs'].reshape(num_points, -1)[:, :50*3]

        shapedirs_loss = self.get_lbs_loss(
            pred_shapedirs * 10,
            gt_shapedirs * 10,
            use_var_expression=True,
        )

        out['loss'] += shapedirs_loss * self.params.lbs_weight * 10.0
        out['shapedirs_loss'] = shapedirs_loss

        return out


# ------------------------------------------------------------------------------- #

class UVDecoderLoss(BaseLoss):

    @dataclass
    class Params:
        rgb_weight:         float = 1.0
        vgg_weight:         float = 0.0
        dssim_weight:       float = 0.0
        scale_weight:       float = 0.0
        lpips_weight:       float = 0.0
        rot_weight:         float = 0.0
        laplacian_weight:   float = 0.0
        normal_weight:      float = 0.0
        flame_weight:       float = 0.0
        reg_weight:         float = 0.0
        reg_attribute:      list = field(default_factory=lambda: ['color'])
        scale_threshold:    float = 0.0
    
    def __init__(self, params: Params):
        super().__init__()
        
        self.params = params

        self.vgg_loss       = VGGPerceptualLoss()
        self.lpips_loss     = lpips.LPIPS(net='vgg').eval()
        self.l1_loss        = nn.L1Loss(reduction='mean')
        self.l2_loss        = nn.MSELoss(reduction='mean')

        self.laplacian_matrix   = None

    def get_dssim_loss(self, rgb_values, rgb_gt):
        return d_ssim(rgb_values, rgb_gt)

    def get_vgg_loss(self, rgb_values, rgb_gt):
        return self.vgg_loss(rgb_values, rgb_gt)

    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss
    
    def get_lpips_loss(self, rgb_values, rgb_gt, normalize=True):
        return self.lpips_loss(rgb_values, rgb_gt, normalize=normalize)
    
    def get_laplacian_smoothing_loss(self, verts_orig, verts):
        L = self.laplacian_matrix[None, ...].detach()

        basis_lap   = L.bmm(verts_orig).detach()
        offset_lap  = L.bmm(verts)

        diff = (offset_lap - basis_lap) ** 2
        diff = diff.sum(dim=-1, keepdim=True)

        return diff.mean()

    def accumulate_gradients(self, model_outputs, ground_truth, cur_step=None, cur_epoch=None):

        render_image = model_outputs['rgb_image']   # torch.Size([1, 3, 512, 512])
        gt_image     = ground_truth['rgb']          # torch.Size([1, 3, 512, 512])

        # initialize the loss
        rgb_loss = self.get_rgb_loss(render_image, gt_image)
        loss = rgb_loss.clone()
        out = {'loss': loss, 'rgb_loss': rgb_loss}

        # vgg loss
        if self.params.vgg_weight > 0:
            vgg_loss = self.get_vgg_loss(render_image, gt_image)
            out['vgg_loss'] = vgg_loss
            out['loss'] += vgg_loss * self.params.vgg_weight

        # dssim loss
        if self.params.dssim_weight > 0:
            dssim_loss = self.get_dssim_loss(render_image, gt_image)
            out['dssim_loss'] = dssim_loss
            out['loss'] += dssim_loss * self.params.dssim_weight

        # scale loss
        if self.params.scale_weight > 0:
            scale = model_outputs['scale']
            scale_max, _ = torch.max(scale, dim=-1)
            scale_min, _ = torch.min(scale, dim=-1)
            scale_regu = F.relu(scale_max/scale_min - self.params.scale_threshold).mean()
            out['scale_loss'] = scale_regu
            out['loss'] += scale_regu * self.params.scale_weight

        # lpips loss
        if self.params.lpips_weight > 0:
            lpips_loss = self.get_lpips_loss(render_image, gt_image, normalize=True).squeeze()
            out['lpips_loss'] = lpips_loss
            out['loss'] += lpips_loss * self.params.lpips_weight

        # rotation loss
        if self.params.rot_weight > 0:
            raw_rot  = model_outputs['raw_rot']
            rot_loss = torch.mean(raw_rot[..., 0] ** 2) + torch.mean(raw_rot[..., 2] ** 2)
            out['rot_loss'] = rot_loss
            out['loss'] += rot_loss * self.params.rot_weight

        # laplacian loss or normal loss
        if self.params.laplacian_weight > 0 or self.params.normal_weight > 0:
            verts = model_outputs['verts']  # [1, V, 3]
            faces = model_outputs['faces']  # [F, 3]

            meshes = Meshes(verts=verts, faces=faces[None, ...])

            if self.laplacian_matrix is None:
                self.laplacian_matrix = meshes.laplacian_packed().to_dense()

            if self.params.laplacian_weight > 0:
                verts = model_outputs['verts']
                verts_orig = model_outputs['verts_orig']
                laplacian_loss = self.get_laplacian_smoothing_loss(verts_orig, verts)
                out['laplacian_loss'] = laplacian_loss
                out['loss'] += laplacian_loss * self.params.laplacian_weight

                # laplacian_loss = mesh_laplacian_smoothing(meshes)
                # out['laplacian_loss'] = laplacian_loss
                # out['loss'] += laplacian_loss * self.params.laplacian_weight

            if self.params.normal_weight > 0:
                normal_loss = mesh_normal_consistency(meshes)
                out['normal_loss'] = normal_loss
                out['loss'] += normal_loss * self.params.normal_weight

        if self.params.flame_weight > 0:
            verts       = model_outputs['verts']  # [1, V, 3]
            verts_orig  = model_outputs['verts_orig']  # [1, V, 3]

            flame_loss = (verts - verts_orig) ** 2
            out['flame_loss'] = flame_loss.mean()
            out['loss'] += out['flame_loss'] * self.params.flame_weight

        # regularization loss
        if self.params.reg_weight > 0:
            
            out['reg_loss'] = 0.0

            for att in self.params.reg_attribute:

                if att == 'color':
                    reg_term = self.l2_loss(model_outputs['decode_color'], model_outputs['prior_features_dc'])
                elif att == 'opacity':
                    reg_term = self.l2_loss(model_outputs['decode_opacity'], model_outputs['prior_opacity'])
                elif att == 'scaling':
                    reg_term = self.l2_loss(model_outputs['decode_scaling'], model_outputs['prior_scaling'])
                elif att == 'rotation':
                    reg_term = self.l2_loss(model_outputs['decode_rotation'], model_outputs['prior_rotation'])
                elif att == 'offset':
                    reg_term = self.l2_loss(model_outputs['decode_offset'], model_outputs['prior_offset'])
                else:
                    reg_term = 0.
                
                out['reg_loss'] += reg_term
            
            out['loss'] += out['reg_loss'] * self.params.reg_weight

        return out
    
    
# ------------------------------------------------------------------------------- #

LossClass = Union[
    FateAvatarLoss, FlashAvatarLoss, SplattingAvatarLoss, GaussianAvatarsLoss, MonoGaussianAvatarLoss
]