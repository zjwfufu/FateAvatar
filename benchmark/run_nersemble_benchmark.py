import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import  glob
import  argparse
import  imageio
import  torch
import  numpy as np

from    common  import      load_config
from    train.loader        import Loader
from    tools.util  import load_to_gpu

from    benchmark.nersemble.dataset import NersembleBenchmarkDataset
from    benchmark.nersemble.fateavatar import FateAvatar

from nersemble_benchmark.constants import (
    BENCHMARK_MONO_FLAME_AVATAR_IDS,
    BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL,
    BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS,
    BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST
)

# python ./benchmark/run_nersemble_benchmark.py --root_path ../workshop_dataset --workspace ./nersemble_workspace/393 \
# --name FateAvatar --config ./config/fateavatar.yaml \
# --submit_dir ./nersemble_workspace/results --participant_id 393

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str,                       required=True)
    parser.add_argument('--root_path',  type=str,                       required=True)
    parser.add_argument('--workspace',  type=str,                       required=True)
    parser.add_argument('--name',       type=str,                       required=True)
    parser.add_argument('--submit_dir', type=str,                       required=True)
    parser.add_argument('--device',     type=torch.device,              default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--seed',       type=int,                       default=42)
    parser.add_argument('--participant_id', type=int, choices=[393, 404, 461, 477, 486], required=True)
    parser.add_argument('--ckpt_path',  type=str,                       required=False)
    parser.add_argument('--bg_color',   type=str,                       default='white')
    parser.add_argument('--use_full_head_resume',                       action='store_true')
    parser.add_argument('--use_baked_resume',                           action='store_true')

    opt = parser.parse_args()

    opt.model_name = "FateAvatar"

    # ----------------------------  override config ---------------------------- #
    cfg = load_config(
        opt.config,
        overrides={
            "name":         opt.name,
            "workspace":    opt.workspace,
            "bg_color":     opt.bg_color
            }
        )

    # ---------------------------------- model ----------------------------------- #
    avatar = FateAvatar(
        img_res = [512, 512],
        background_color = opt.bg_color,
        cfg_model = cfg.model,
        device = opt.device
    ).to(opt.device)

    # -------------------------- loading trained avatar -------------------------- #
    if opt.use_full_head_resume and opt.use_baked_resume:
        ckpt_dirs = os.path.join(opt.workspace, "baking_full_head", "checkpoints_baked")
        ckpt_list = sorted(glob.glob(f'{ckpt_dirs}/*.pth'))
        ckpt_path = ckpt_list[-1]
    
    elif opt.use_full_head_resume and opt.use_baked_resume is False:
        ckpt_path = os.path.join(opt.workspace, "checkpoints_fullhead", "fullhead.pth")

    elif opt.use_full_head_resume is False and opt.use_baked_resume:
        ckpt_dirs = os.path.join(opt.workspace, "baking", "checkpoints_baked")
        ckpt_list = sorted(glob.glob(f'{ckpt_dirs}/*.pth'))
        ckpt_path = ckpt_list[-1]
    
    else:
        ckpt_path = 'latest'

    if opt.ckpt_path:
        ckpt_path = opt.ckpt_path

    loader = Loader(
        opt.name,
        cfg,
        avatar,
        opt.device,
        workspace       = opt.workspace,
        use_checkpoint  = ckpt_path
    )

    # overwrite
    avatar = loader.model

    # -------------------------- pack loop -------------------------- #
    for sequence_name in BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST:
        for serial in BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS + [BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL]:
            dataset = NersembleBenchmarkDataset(
                root_path   = opt.root_path,
                participant_id = opt.participant_id,
                serial = serial,
                sequence_list = [sequence_name],
                mode = 'test'
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size      = 1,
                shuffle         = False,
                collate_fn      = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
                num_workers     = 4
            )

            render_video_frames = []

            for idx, (_, input_data, ground_truth) in enumerate(loader):
                load_to_gpu(input_data, ground_truth, opt.device)
                output_data = avatar(input_data)
                render_image = output_data['rgb_image']
                render_np   = render_image[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                render_video_frames.append((render_np * 255.).astype('uint8'))

            all_render_np = np.stack(render_video_frames, axis=0)
            video_folder = os.path.join(opt.submit_dir, str(opt.participant_id), sequence_name)
            os.makedirs(video_folder, exist_ok=True)
            imageio.mimwrite(
                f'{video_folder}/cam_{serial}.mp4',
                render_video_frames,
                fps=25,
                codec='libx264',
                quality=None,
                output_params=['-crf', '14', '-preset', 'slow']
            )



