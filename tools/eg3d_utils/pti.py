import os
import sys
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import PIL.Image
import copy
import imageio
import scipy
import math
import json

from tools.eg3d_utils.camera_eg3d import LookAtPoseSampler

# import clip

PANOHEAD_LIB_PATH = os.path.join(os.getcwd(), 'submodules/PanoHead')
sys.path.insert(1, PANOHEAD_LIB_PATH)

import dnnlib
import legacy
from training.dataset import ImageFolderDataset

ENABLE_DELTA_C = False

#----------------------------------------------------------------------------

def project_multi_view(
    G,
    dataset:                   ImageFolderDataset,
    device:                    torch.device,
    log_fn                     = None,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.01,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    lambda_mse                 = 0.1,
    lambda_perc                = 1,
    lambda_w_norm              = 1,
    lambda_clip                = 1,
    optimize_noise             = False,
):
    # assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if log_fn is not None:
            log_fn(*args)
        else:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Prepare CLIP loss
    # clip_model, _           = clip.load("./weights/ViT-B-32.pt", device=device)
    # text                    = clip.tokenize(['Back of human face, natural head shape.']).to(device)
    # text_features           = clip_model.encode_text(text).detach()
    # text_side               = clip.tokenize(['Side profile with rounded, natural head shape.']).to(device)
    # text_features_side      = clip_model.encode_text(text_side).detach()
    # pre_mean    = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    # pre_std     = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    # Compute w stats.
    logprint(f'++> Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples               = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    camera_lookat_point     = torch.tensor([0, 0, 0.0], device=device)
    cam2world_pose          = LookAtPoseSampler.sample(math.pi / 2, math.pi / 2, camera_lookat_point, radius=2.7, device=device)
    intrinsics              = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_samples               = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    w_samples               = G.mapping(torch.from_numpy(z_samples).to(device), c_samples.repeat(w_avg_samples,1))  # [N, L, C]
    w_samples               = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg                   = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std                   = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    w_avg = torch.tensor(w_avg, dtype=torch.float32, device=device).repeat(1, G.backbone.mapping.num_ws, 1)
    w_opt = w_avg.detach().clone()
    w_opt.requires_grad = True
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device="cpu")
    if optimize_noise:
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    else:
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

    # Load VGG16 feature detector.
    # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    # with dnnlib.util.open_url(url) as f:
    #     vgg16 = torch.jit.load(f).eval().to(device)

    with open('./weights/vgg16.pt', 'rb') as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name }

    # Init noise.
    if optimize_noise:
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    for step in range(num_steps + 1):

        idx = step % len(dataset)

        target_fname = os.path.join(dataset._path, dataset._image_fnames[idx])
        c = torch.from_numpy(dataset._get_raw_labels()[idx:idx+1]).to(device)

        # Load target image.
        target_pil = PIL.Image.open(target_fname).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.Resampling.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)

        target = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)

        # fix delta_c
        delta_c = G.t_mapping(torch.from_numpy(np.mean(z_samples, axis=0, keepdims=True)).to(device), c[:1], truncation_psi=1.0, truncation_cutoff=None, update_emas=False)
        delta_c = torch.squeeze(delta_c, 1)
        if ENABLE_DELTA_C:
            c[:,3] += delta_c[:,0]
            c[:,7] += delta_c[:,1]
            c[:,11] += delta_c[:,2]

        # Features for target image.
        target_images = target.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
        target_images_perc = (target_images + 1) * (255/2)
        if target_images_perc.shape[2] > 256:
            target_images_perc = F.interpolate(target_images_perc, size=(256, 256), mode='area')
        target_features = vgg16(target_images_perc, resize_images=False, return_lpips=True)

        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        synth_images = G.synthesis(ws, c=c, noise_mode='const')['image']

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_perc = (synth_images + 1) * (255/2)
        if synth_images_perc.shape[2] > 256:
            synth_images_perc = F.interpolate(synth_images_perc, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
        perc_loss = (target_features - synth_features).square().sum(1).mean()

        mse_loss = (target_images - synth_images).square().mean()

        w_norm_loss = (w_opt - w_avg).square().mean()

        # Noise regularization.
        reg_loss = 0.0
        if optimize_noise:
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)

        # # CLIP loss
        # c_back  = LookAtPoseSampler.sample(math.pi / 2 + math.pi, math.pi / 2,
        #                                 lookat_position = torch.tensor([0, 0, 0], device = device),
        #                                 radius=2.75, device=device)
        # intrinsics          = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
        # c_back              = torch.cat([c_back.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        # c_side_1  = LookAtPoseSampler.sample(math.pi / 2 + math.pi / 2, math.pi / 2,
        #                                 lookat_position = torch.tensor([0, 0, 0], device = device),
        #                                 radius=2.75, device=device)
        # c_side_2  = LookAtPoseSampler.sample(math.pi / 2 + 3 * math.pi / 2, math.pi / 2,
        #                                 lookat_position = torch.tensor([0, 0, 0], device = device),
        #                                 radius=2.75, device=device)
        # c_side_1  = torch.cat([c_side_1.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        # c_side_2  = torch.cat([c_side_2.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        # back_images = G.synthesis(ws, c=c_back, noise_mode='const')['image']
        # back_images = (back_images + 1) / 2     # -> [0, 1]
        # back_images = (back_images - pre_mean.view(-1, 1, 1)) / pre_std.view(-1, 1, 1)
        # back_images = F.interpolate(back_images, size=(224, 224), mode='area')

        # image_features = clip_model.encode_image(back_images)
        # clip_loss = 1 - F.cosine_similarity(image_features, text_features)

        # side_images_1   = G.synthesis(ws, c=c_side_1, noise_mode='const')['image']
        # side_images_1   = (side_images_1 + 1) / 2     # -> [0, 1]
        # side_images_1   = (side_images_1 - pre_mean.view(-1, 1, 1)) / pre_std.view(-1, 1, 1)
        # side_images_1   = F.interpolate(side_images_1, size=(224, 224), mode='area')

        # side_images_2   = G.synthesis(ws, c=c_side_2, noise_mode='const')['image']
        # side_images_2   = (side_images_2 + 1) / 2     # -> [0, 1]
        # side_images_2   = (side_images_2 - pre_mean.view(-1, 1, 1)) / pre_std.view(-1, 1, 1)
        # side_images_2   = F.interpolate(side_images_2, size=(224, 224), mode='area')

        # image_features_side_1 = clip_model.encode_image(side_images_1)
        # image_features_side_2 = clip_model.encode_image(side_images_2)

        # clip_loss_side = 1 - (F.cosine_similarity(image_features_side_1, text_features_side) + F.cosine_similarity(image_features_side_2, text_features_side)) / 2

        loss = lambda_mse * mse_loss + \
                lambda_perc * perc_loss +  \
                lambda_w_norm * w_norm_loss + \
                regularize_noise_weight * reg_loss
                # lambda_clip * clip_loss + \
                # lambda_clip * clip_loss_side
        
        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            # logprint(f'step: {step:>4d}/{num_steps} mse: {mse_loss:<4.2f} perc: {perc_loss:<4.2f} w_norm: {w_norm_loss:<4.2f}  noise: {float(reg_loss):<5.2f}  back sim: {float(1 - clip_loss):<5.2f}  side sim: {float(1 - clip_loss_side):<5.2f}')
            logprint(f'step: {step:>4d}/{num_steps} mse: {mse_loss:<4.2f} perc: {perc_loss:<4.2f} w_norm: {w_norm_loss:<4.2f}  noise: {float(reg_loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step - 1] = w_opt.detach().cpu()[0]

        # Normalize noise.
        if optimize_noise:
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

    if w_out.shape[1] == 1:
        w_out = w_out.repeat([1, G.mapping.num_ws, 1])

    return w_out

#----------------------------------------------------------------------------

def project_pti_multi_view(
    G,
    dataset:                   ImageFolderDataset,
    w_pivot:                   torch.Tensor,
    device:                    torch.device,
    log_fn                     = None,
    num_steps                  = 1000,
    initial_learning_rate      = 3e-4,
    lambda_mse                 = 0.1,
    lambda_perc                = 1,
    lambda_clip                = 0.1,
):

    def logprint(*args):
        if log_fn is not None:
            log_fn(*args)
        else:
            print(*args)

    G = copy.deepcopy(G).train().requires_grad_(True).to(device) # type: ignore

    # Prepare CLIP loss
    # clip_model, _   = clip.load("./weights/ViT-B-32.pt", device=device)
    # text            = clip.tokenize(['back of human face']).to(device)
    # text_features   = clip_model.encode_text(text).detach()
    # pre_mean    = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    # pre_std     = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    # Load VGG16 feature detector.
    # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    # with dnnlib.util.open_url(url) as f:
    #     vgg16 = torch.jit.load(f).eval().to(device)

    with open('./weights/vgg16.pt', 'rb') as f:
        vgg16 = torch.jit.load(f).eval().to(device)


    w_pivot = w_pivot.to(device).detach()
    optimizer = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999), lr=initial_learning_rate)

    out_params = []

    for step in range(num_steps + 1):

        idx = step % len(dataset)

        target_fname = os.path.join(dataset._path, dataset._image_fnames[idx])
        c = torch.from_numpy(dataset._get_raw_labels()[idx:idx+1]).to(device)

        # Load target image.
        target_pil = PIL.Image.open(target_fname).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.Resampling.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)

        target = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)

        # Features for target image.
        target_images = target.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
        target_images_perc = (target_images + 1) * (255/2)
        if target_images_perc.shape[2] > 256:
            target_images_perc = F.interpolate(target_images_perc, size=(256, 256), mode='area')
        target_features = vgg16(target_images_perc, resize_images=False, return_lpips=True)

        # Synth images from opt_w.
        synth_images = G.synthesis(w_pivot, c=c, noise_mode='const')['image']

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_perc = (synth_images + 1) * (255/2)
        if synth_images_perc.shape[2] > 256:
            synth_images_perc = F.interpolate(synth_images_perc, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
        perc_loss = (target_features - synth_features).square().sum(1).mean()

        mse_loss = (target_images - synth_images).square().mean()

        # CLIP loss
        # c_back  = LookAtPoseSampler.sample(math.pi / 2 + math.pi, math.pi / 2,
        #                                 lookat_position = torch.tensor([0, 0, 0], device = device),
        #                                 radius=2.75, device=device)
        # intrinsics          = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
        # c_back              = torch.cat([c_back.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        # back_images = G.synthesis(w_pivot, c=c_back, noise_mode='const')['image']
        # back_images = (back_images + 1) / 2     # -> [0, 1]
        # back_images = (back_images - pre_mean.view(-1, 1, 1)) / pre_std.view(-1, 1, 1)
        # back_images = F.interpolate(back_images, size=(224, 224), mode='area')

        # image_features = clip_model.encode_image(back_images)
        # clip_loss = 1 - F.cosine_similarity(image_features, text_features)

        # loss = lambda_mse * mse_loss + lambda_perc * perc_loss + lambda_clip * clip_loss
        loss = lambda_mse * mse_loss + lambda_perc * perc_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            # logprint(f'step: {step:>4d}/{num_steps} mse: {mse_loss:<4.2f} perc: {perc_loss:<4.2f}  back sim: {float(1 - clip_loss):<5.2f}')
            logprint(f'step: {step:>4d}/{num_steps} mse: {mse_loss:<4.2f} perc: {perc_loss:<4.2f}')
        
        if step == num_steps - 1 or step % 10 == 0:
            out_params.append(copy.deepcopy(G).eval().requires_grad_(False).cpu())

    return out_params

#----------------------------------------------------------------------------

def save_optimization_video(G,
                            dataset:            ImageFolderDataset,
                            video,
                            projected_w_steps:  list,
                            G_steps:            list,
                            device:             torch.device,
                            idx = 0):
    target_fname = os.path.join(dataset._path, dataset._image_fnames[idx])
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.Resampling.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    c = torch.from_numpy(dataset._get_raw_labels()[idx:idx + 1]).to(device)

    c_back = LookAtPoseSampler.sample(math.pi / 2 + math.pi, math.pi / 2,
                                       lookat_position=torch.tensor([0, 0, 0], device=device),
                                       radius=2.75, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c_back = torch.cat([c_back.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    c_side_1  = LookAtPoseSampler.sample(math.pi / 2 + math.pi / 2, math.pi / 2,
                                    lookat_position = torch.tensor([0, 0, 0], device = device),
                                    radius=2.75, device=device)
    c_side_2  = LookAtPoseSampler.sample(math.pi / 2 + 3 * math.pi / 2, math.pi / 2,
                                    lookat_position = torch.tensor([0, 0, 0], device = device),
                                    radius=2.75, device=device)
    c_side_1  = torch.cat([c_side_1.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c_side_2  = torch.cat([c_side_2.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    downsample_rate = 2
    for i in range(0, len(projected_w_steps), downsample_rate):
        projected_w = projected_w_steps[i]

        synth_image = G.synthesis(projected_w.unsqueeze(0).to(device), c=c, noise_mode='const')['image']
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        back_image = G.synthesis(projected_w.unsqueeze(0).to(device), c=c_back, noise_mode='const')['image']
        back_image = (back_image + 1) * (255 / 2)
        back_image = back_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        side_images_1   = G.synthesis(projected_w.unsqueeze(0).to(device), c=c_side_1, noise_mode='const')['image']
        side_images_1   = (side_images_1 + 1) * (255 / 2)
        side_images_1   = side_images_1.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        side_images_2   = G.synthesis(projected_w.unsqueeze(0).to(device), c=c_side_2, noise_mode='const')['image']
        side_images_2   = (side_images_2 + 1) * (255 / 2)
        side_images_2   = side_images_2.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        video.append_data(np.concatenate([target_uint8, synth_image, back_image, side_images_1, side_images_2], axis=1))

    for i in range(0, len(G_steps), downsample_rate):
        G_new = G_steps[i]
        G_new.to(device)

        synth_image = G_new.synthesis(projected_w_steps[-1].unsqueeze(0).to(device), c=c, noise_mode='const')['image']
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        back_image = G_new.synthesis(projected_w_steps[-1].unsqueeze(0).to(device), c=c_back, noise_mode='const')['image']
        back_image = (back_image + 1) * (255 / 2)
        back_image = back_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        side_images_1   = G.synthesis(projected_w.unsqueeze(0).to(device), c=c_side_1, noise_mode='const')['image']
        side_images_1   = (side_images_1 + 1) * (255 / 2)
        side_images_1   = side_images_1.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        side_images_2   = G.synthesis(projected_w.unsqueeze(0).to(device), c=c_side_2, noise_mode='const')['image']
        side_images_2   = (side_images_2 + 1) * (255 / 2)
        side_images_2   = side_images_2.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        video.append_data(np.concatenate([target_uint8, synth_image, back_image, side_images_1, side_images_2], axis=1))
        G_new.cpu()

    video.close()

#----------------------------------------------------------------------------

def gen_orbit_video(G,
                    J,
                    mp4_save_path: str,
                    save_path: str,
                    ws,
                    gs_lookat_point,
                    gs_radius,
                    w_frames = 40,
                    ele_list = [-math.pi / 6, 0, math.pi / 6],
                    device   = torch.device('cuda'),
                    rotate_type: str = 'camera',
                    rescale_scene: bool = False,
                    rescale_factor: float = 0.0):
    """
    Here, we render the results of the 3D-aware GAN inversion and also save the camera trajectory. 
    Note: To achieve the best alignment possible, the saved trajectory is based on GS.
    """

    if rotate_type == 'camera':
        # camera_lookat_point = torch.tensor([J[0, 0, 0].item(), 0, - J[0, 0, 2].item()], device=device)
        camera_lookat_point = torch.tensor([0, 0, 0], device=device)
    elif rotate_type == 'flame':
        camera_lookat_point = torch.tensor([J[0, 0, 0].item(), 0, 0], device=device)
    else:
        raise ValueError
    
    cam2world_pose      = LookAtPoseSampler.sample(math.pi / 2, math.pi / 2, camera_lookat_point, radius=2.7, device=device)
    intrinsics          = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c                   = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c                   = c.repeat(len(ws), 1)
    _                   = G.synthesis(ws[:1], c[:1]) # warm up

    # create new folder
    outdirs = os.path.dirname(mp4_save_path)
    img_save_path  = os.path.join(save_path, 'image')
    # mask_save_path = os.path.join(save_path, 'mask')
    os.makedirs(outdirs, exist_ok=True)
    os.makedirs(img_save_path, exist_ok=True)
    # os.makedirs(mask_save_path, exist_ok=True)
    # add delta_c
    z_samples   = np.random.RandomState(123).randn(10000, G.z_dim)
    delta_c     = G.t_mapping(torch.from_numpy(np.mean(z_samples, axis=0, keepdims=True)).to(device), c[:1], truncation_psi=1.0, truncation_cutoff=None, update_emas=False)
    delta_c     = torch.squeeze(delta_c, 1)

    # Render video.
    video_out = imageio.get_writer(mp4_save_path, mode='I', fps=25, codec='libx264')

    all_poses = {}
    for round, ele in enumerate(ele_list):
        for frame_idx in range(1, w_frames + 1):

            if rescale_scene:
                radius = 2.7 + rescale_factor
            else:
                radius = 2.7

            cam2world_pose = LookAtPoseSampler.sample(
                math.pi / 2 + 2 * math.pi * (frame_idx - 1) / (w_frames),
                math.pi / 2 - ele,
                camera_lookat_point,
                radius=radius,
                device=device
            )

            gs_cam2world_pose = LookAtPoseSampler.sample(
                math.pi / 2 + 2 * math.pi * (frame_idx - 1) / (w_frames),
                math.pi / 2 - ele,
                gs_lookat_point,
                radius=gs_radius,
                device=device
            )
            gs_world2cam = torch.linalg.inv(gs_cam2world_pose)

            intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
            c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            
            # fix delta_c
            if ENABLE_DELTA_C:
                c[:,3] += delta_c[:,0]
                c[:,7] += delta_c[:,1]
                c[:,11] += delta_c[:,2]
            result = G.synthesis(ws=ws, c=c[0:1], noise_mode='const')
            
            img = result['image'][0]
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            mask = result['image']
            mask = F.interpolate(mask, size=(512, 512), mode='bilinear')
            mask = (mask.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)

            image_name = f'{(frame_idx + round * w_frames):04d}'
            all_poses.update({image_name: gs_world2cam.squeeze().cpu().numpy().tolist()})
            PIL.Image.fromarray(img.permute(1, 2, 0).cpu().numpy(), 'RGB').save(f'{img_save_path}/{image_name}.png')
            # imageio.imwrite(f'{mask_save_path}/{image_name}.png', np.array(mask[0, :, :, 0].detach().cpu().numpy(), dtype=np.uint16))
            video_out.append_data(np.array(img.permute(1, 2, 0).cpu().numpy()))

    video_out.close()

    with open(os.path.join(save_path, 'trajectory.json'), 'w') as f:
        json.dump(all_poses, f, indent="\t")