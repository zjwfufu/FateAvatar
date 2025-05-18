import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import  yaml
import  torch
import  argparse

from    tools.util         import seed_everything, file_backup
from    tools.util         import EasyDict

from    train.trainer      import Trainer
from    benchmark.nersemble.dataset import NersembleBenchmarkDataset
from    benchmark.nersemble.fateavatar import FateAvatar

from    common  import      load_config
from    common  import      construct_model
from    common  import      construct_loss
from    common  import      construct_metrics

from nersemble_benchmark.constants import (BENCHMARK_MONO_FLAME_AVATAR_IDS,
    BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL, BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS,
    BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TRAIN, BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST
)

# python ./benchmark/train_nersemble_benchmark_mono.py --root_path ../workshop_dataset --workspace ./nersemble_workspace/393 \
#  --name FateAvatar --config ./config/fateavatar.yaml --participant_id 393

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str,                       required=True)
    parser.add_argument('--root_path',  type=str,                       required=True)
    parser.add_argument('--workspace',  type=str,                       required=True)
    parser.add_argument('--name',       type=str,                       required=True)
    parser.add_argument('--device',     type=torch.device,              default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--seed',       type=int,                       default=42)
    parser.add_argument('--bg_color',   type=str,                       default='white')
    parser.add_argument('--participant_id', type=int, choices=[393, 404, 461, 477, 486], required=True)
    parser.add_argument('--resume',     action='store_true')

    opt = parser.parse_args()

    opt.model_name = "FateAvatar"

    seed_everything(opt.seed)

    with open(opt.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = EasyDict(config)

    # ----------------------------- override config ------------------------------ #
    cfg = load_config(
        opt.config,
        overrides={
            "name":         opt.name,
            "workspace":    opt.workspace,
            "root_path":    opt.root_path,
            "bg_color":     opt.bg_color
            }
        )

    # --------------------------------- dataset ---------------------------------- #
    datasets = NersembleBenchmarkDataset(
        root_path   = opt.root_path,
        participant_id = opt.participant_id,
        serial = BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL,
        sequence_list = BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TRAIN
    )

    cfg.update({
        'n_shape':  100,
        'n_exp':    300,
        'optimize_tracking': False
    })

    # # ---------------------------- save identity info ---------------------------- #
    # save_identity_info(opt.workspace, datasets.train)

    # ---------------------------------- model ----------------------------------- #
    model = FateAvatar(
        img_res = [512, 512],
        background_color = opt.bg_color,
        cfg_model = cfg.model,
        device = opt.device
    ).to(opt.device)

    # ----------------------------------- loss ----------------------------------- #
    criterion = construct_loss(
        opt,
        cfg.loss,
        datasets
    )

    # --------------------------------- metrics ---------------------------------- #
    metrics = construct_metrics(
        opt.device
    )
    
    # --------------------------------- metrics ---------------------------------- #
    file_backup(opt.workspace, opt.config)

    # ------------------------------ setup trainer ------------------------------- #
    trainer = Trainer(
        opt.name,
        cfg,
        model,
        opt.device,
        train_dataset   = datasets,
        test_dataset    = datasets,
        criterions      = criterion,
        metrics         = metrics,
        workspace       = opt.workspace,
        use_checkpoint  = 'latest' if opt.resume else 'scratch'
    )
    
    # --------------------------------- execute ---------------------------------- #
    trainer.train(max_epochs=5)


