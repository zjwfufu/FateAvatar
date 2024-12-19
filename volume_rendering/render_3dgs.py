import math
import torch
from diff_gaussian_rasterization    import GaussianRasterizationSettings, GaussianRasterizer

#-------------------------------------------------------------------------------#

def render(viewpoint_camera,
            pc,
            bg_color : torch.Tensor,
            scaling_modifier = 1.0,
            override_color:torch.Tensor=None,
            device='cuda'
           ):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    means3D = pc.get_xyz

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=device) + 0
    if screenspace_points.requires_grad:
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
        scale_modifier  = scaling_modifier,
        viewmatrix      = viewpoint_camera.world_view_transform,
        projmatrix      = viewpoint_camera.full_proj_transform,
        sh_degree       = pc.max_sh_degree,
        campos          = viewpoint_camera.camera_center,
        prefiltered     = False,
        debug           = False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = pc.get_scaling
    rotations = pc.get_rotation
    cov3D_precomp = None
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_features
    if override_color is None:
        colors_precomp = None
    else:
        colors_precomp = override_color
        shs = None
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    rendered_image, radii = rasterizer(
        means3D         = means3D,
        means2D         = means2D,
        shs             = shs,
        colors_precomp  = colors_precomp,
        opacities       = opacity,
        scales          = scales,
        rotations       = rotations,
        cov3D_precomp   = cov3D_precomp
    )

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}