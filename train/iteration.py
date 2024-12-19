import torch

from tools.util import EasyDict

from model.fateavatar                   import FateAvatar
from model.baseline.flashavatar         import FlashAvatar
from model.baseline.gaussianavatars     import GaussianAvatars
from model.baseline.monogaussianavatar  import MonoGaussianAvatar
from model.baseline.splattingavatar     import SplattingAvatar

from train.loss                         import (
    FateAvatarLoss,
    FlashAvatarLoss,
    GaussianAvatarsLoss,
    MonoGaussianAvatarLoss,
    SplattingAvatarLoss
)

#-------------------------------------------------------------------------------#

def iteration_step_fateavatar(input_data:       dict,
                              ground_truth:     dict,
                              model:            FateAvatar,
                              criterions:       FateAvatarLoss,
                              optimizers_group: dict,
                              cfg:              EasyDict,
                              global_step:      int,
                              cur_epoch:        int,
                              log = None,
                              **kwargs):
        
        if log is None:
            log = print

        output_data  = model(input_data)
        render_image = output_data['rgb_image']
        gt_image     = ground_truth['rgb']

        loss_output = criterions(output_data, ground_truth)

        loss = loss_output['loss']

        bs                  = output_data['bs']
        viewspace_points    = output_data['viewspace_points']
        visibility_filter   = output_data['visibility_filter']
        radii               = output_data['radii']

        #------------------------ zero grad ------------------------#
        for name, optimizer in optimizers_group.items():
            optimizer.zero_grad(set_to_none=True)

        loss.backward()

        #------------------------ do gaussian maintain ------------------------#
        for bs_ in range(bs):
            model._add_densification_stats(viewspace_points[bs_], visibility_filter[bs_])

        #------------------------ optimize ------------------------#
        for name, optimizer in optimizers_group.items():
            optimizer.step()

        # ------------------------ densify ------------------------
        if global_step % cfg.training.densify_interval == 0:

            # do uv densification
            old_num = model.num_points
            if old_num < cfg.training.max_points_num:
                
                model._uv_densify(optimizers_group['gs'],
                    increase_num = min(cfg.training.max_points_num - old_num, cfg.training.increase_num))
                
                log(f"Do UV densification, Guassian splats: {old_num} --> {model.num_points}.")
            else:
                log(f"Guassian splats: {old_num} has reached maximum number.")

        # ------------------------ prune ------------------------
        if global_step % cfg.training.prune_interval == 0:
            old_num = model.num_points
            model._prune_low_opacity_points(optimizers_group['gs'],
                                                 min_opacity = cfg.training.min_opacity)
            
            log(f"Prune low opacity points, Guassian splats: {old_num} --> {model.num_points}.")
        # ------------------------ reset opacity ------------------------
        if global_step % cfg.training.opacity_reset_interval == 0 and global_step != 0:
            model._reset_opacity(optimizers_group['gs'])

        return {'loss_output': loss_output,
                'render_image': render_image,
                'gt_image': gt_image}

#-------------------------------------------------------------------------------#

def iteration_step_flashavatar(input_data:       dict,
                              ground_truth:     dict,
                              model:            FlashAvatar,
                              criterions:       FlashAvatarLoss,
                              optimizers_group: dict,
                              cfg:              EasyDict,
                              global_step:      int,
                              cur_epoch:        int,
                              log = None,
                              **kwargs):
    
    output_data  = model(input_data)
    render_image = output_data['rgb_image']
    gt_image     = ground_truth['rgb']

    loss_output = criterions(output_data, ground_truth, global_step)

    loss = loss_output['loss']

    #------------------------ zero grad ------------------------#
    for name, optimizer in optimizers_group.items():
        optimizer.zero_grad(set_to_none=True)

    loss.backward()

    #------------------------ optimize ------------------------#
    for name, optimizer in optimizers_group.items():
        optimizer.step()

    return {'loss_output': loss_output,
            'render_image': render_image,
            'gt_image': gt_image}

#-------------------------------------------------------------------------------#

def iteration_step_gaussianavatars(input_data:       dict,
                                    ground_truth:     dict,
                                    model:            GaussianAvatars,
                                    criterions:       GaussianAvatarsLoss,
                                    optimizers_group: dict,
                                    cfg:              EasyDict,
                                    global_step:      int,
                                    cur_epoch:        int,
                                    log = None,
                                    **kwargs):

    output_data  = model(input_data)
    render_image = output_data['rgb_image']
    gt_image     = ground_truth['rgb']

    loss_output = criterions(output_data, ground_truth, global_step)

    loss = loss_output['loss']

    bs                  = output_data['bs']
    viewspace_points    = output_data['viewspace_points']
    visibility_filter   = output_data['visibility_filter']
    radii               = output_data['radii']

    #------------------------ zero grad ------------------------#
    for name, optimizer in optimizers_group.items():
        optimizer.zero_grad(set_to_none=True)

    loss.backward()

    #------------------------ do gaussian maintain ------------------------#
    for bs_ in range(bs):
        if global_step < cfg.training.densify_until_iter:
            model.max_radii2D[visibility_filter[bs_]] = torch.max(model.max_radii2D[visibility_filter[bs_]], radii[bs_][visibility_filter[bs_]])
            model._add_densification_stats(viewspace_points[bs_], visibility_filter[bs_])

            # ------------------------ densify ------------------------
            if global_step % cfg.training.densify_interval == 0:
                size_threshold = cfg.training.size_threshold if global_step > cfg.training.opacity_reset_interval else None
                model._densify_and_prune(
                    optimizers_group['gs'],
                    cfg.training.densify_grad_threshold,
                    cfg.training.min_opacity,
                    2.0,    # cameras_extent
                    size_threshold
                )

        if cfg.training.opacity_reset_interval > 0 and \
            (global_step - cfg.training.opacity_reset_start_iter) % cfg.training.opacity_reset_interval == 0:
            model._reset_opacity(optimizers_group['gs'])

    #------------------------ optimize ------------------------#
    for name, optimizer in optimizers_group.items():
        optimizer.step()

    #------------------------ increase sh degree ------------------------#
    if global_step % 1000 == 0:
        model._update_sh_degree()

     #------------------------ xyz lr scheduler ------------------------#
    if kwargs:
        xyz_scheduler_args  = kwargs['xyz_scheduler_args']

        for param_group in optimizers_group['gs'].param_groups:
            if param_group["name"] == "_xyz":
                lr = xyz_scheduler_args(global_step)
                param_group['lr'] = lr


    return {'loss_output': loss_output,
            'render_image': render_image,
            'gt_image': gt_image}

#-------------------------------------------------------------------------------#

def iteration_step_monogaussianavatar(input_data:       dict,
                                        ground_truth:     dict,
                                        model:            MonoGaussianAvatar,
                                        criterions:       MonoGaussianAvatarLoss,
                                        optimizers_group: dict,
                                        cfg:              EasyDict,
                                        global_step:      int,
                                        cur_epoch:        int,
                                        log = None,
                                        **kwargs):

    output_data  = model(input_data)
    render_image = output_data['rgb_image']
    gt_image     = ground_truth['rgb']

    loss_output = criterions(output_data, ground_truth, global_step, cur_epoch)

    loss = loss_output['loss']

    #------------------------ zero grad ------------------------#
    for name, optimizer in optimizers_group.items():
        optimizer.zero_grad(set_to_none=True)

    loss.backward()

    #------------------------ optimize ------------------------#
    for name, optimizer in optimizers_group.items():
        optimizer.step()

    # log(sum(output_data['visible_points']))
    # log(output_data['visible_points_idx'])

    return {'loss_output': loss_output,
            'render_image': render_image,
            'gt_image': gt_image}

#-------------------------------------------------------------------------------#

def iteration_step_splattingavatar(input_data:        dict,
                                    ground_truth:     dict,
                                    model:            SplattingAvatar,
                                    criterions:       SplattingAvatarLoss,
                                    optimizers_group: dict,
                                    cfg:              EasyDict,
                                    global_step:      int,
                                    cur_epoch:        int,
                                    log = None,
                                    **kwargs):
    
    output_data  = model(input_data)
    render_image = output_data['rgb_image']
    gt_image     = ground_truth['rgb']

    loss_output = criterions(output_data, ground_truth, global_step)

    loss = loss_output['loss']

    bs                  = output_data['bs']
    viewspace_points    = output_data['viewspace_points']
    visibility_filter   = output_data['visibility_filter']
    radii               = output_data['radii']

    #------------------------ zero grad ------------------------#
    for name, optimizer in optimizers_group.items():
        optimizer.zero_grad(set_to_none=True)

    loss.backward()

    #------------------------ do gaussian maintain ------------------------#
    for bs_ in range(bs):
        if global_step < cfg.training.densify_until_iter:
            model.max_radii2D[visibility_filter[bs_]] = torch.max(model.max_radii2D[visibility_filter[bs_]], radii[bs_][visibility_filter[bs_]])
            model._add_densification_stats(viewspace_points[bs_], visibility_filter[bs_])

            # ------------------------ densify ------------------------
            if global_step % cfg.training.densify_interval == 0:
                size_threshold = cfg.training.size_threshold if global_step > cfg.training.opacity_reset_interval else None
                model._densify_and_prune(
                    optimizers_group['gs'],
                    cfg.training.densify_grad_threshold,
                    cfg.training.min_opacity,
                    2.0,    # cameras_extent
                    size_threshold
                )

        if cfg.training.opacity_reset_interval > 0 and \
            (global_step - cfg.training.opacity_reset_start_iter) % cfg.training.opacity_reset_interval == 0:
            model._reset_opacity(optimizers_group['gs'])

    #------------------------ optimize ------------------------#
    for name, optimizer in optimizers_group.items():
        optimizer.step()

    #------------------------ walking on triangles ------------------------#
    if global_step % cfg.training.triangle_walk_interval == 0:
        model._walking_on_triangles(optimizers_group['gs'])

    return {'loss_output': loss_output,
            'render_image': render_image,
            'gt_image': gt_image}