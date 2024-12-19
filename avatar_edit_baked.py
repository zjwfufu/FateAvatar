import  os
import  yaml
import  glob
import  torch
import  argparse

from    tools.util import seed_everything
from    tools.util import EasyDict

from    model.uv_decoder   import UVDecoder

from    train.baker       import UVEditor
from    train.loader       import Loader

from    common  import      load_config
from    common  import      construct_datasets
from    common  import      load_identity_info
from    common  import      construct_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str,           required=True)
    parser.add_argument('--workspace',  type=str,           required=True)
    parser.add_argument('--name',       type=str,           required=True)
    parser.add_argument('--root_path',  type=str,           required=True)
    parser.add_argument('--device',     type=torch.device,  default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--seed',       type=int,           default=42)
    parser.add_argument('--bg_color',   type=str,           default='white')
    parser.add_argument('--use_full_head_resume',           action='store_true')
    # for uv decoder
    parser.add_argument('--decode_type',   type=str,        default='UNet')
    parser.add_argument('--reg_weight',    type=float,      default=0.0)
    parser.add_argument('--rot_weight',    type=float,      default=0.0)
    parser.add_argument('--reg_attribute', nargs='+',       default=[]) # ['color', 'opacity', 'scaling', 'rotation', 'offset']
    parser.add_argument('--bake_attribute', nargs='+',      default=['color', 'opacity', 'scaling', 'rotation', 'offset'])

    opt = parser.parse_args()

    opt.model_name = "FateAvatar"   # neural baking only support in FateAvatar

    seed_everything(opt.seed)

    with open(opt.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = EasyDict(config)

    # ----------------------------  override config ---------------------------- #
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
    datasets, dataset_name = construct_datasets(
        opt,
        cfg
    )

    # ---------------------------- load identity info ---------------------------- #
    identity_dict = load_identity_info(opt, cfg)

    # ------------------------------- avatar model ------------------------------- #
    avatar = construct_model(
        opt,
        cfg.model,
        0.0,
        identity_dict = identity_dict
    )

    # -------------------------- loading trained avatar -------------------------- #
    if opt.use_full_head_resume:
        ckpt_dirs = os.path.join(opt.workspace, "baking_full_head", "checkpoints_baked")
        ckpt_list = sorted(glob.glob(f'{ckpt_dirs}/*.pth'))
        ckpt_path = ckpt_list[-1]

    else:
        ckpt_dirs = os.path.join(opt.workspace, "baking", "checkpoints_baked")
        ckpt_list = sorted(glob.glob(f'{ckpt_dirs}/*.pth'))
        ckpt_path = ckpt_list[-1]

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

    # ----------------------------  editor ---------------------------- #

    decoder = UVDecoder(
        avatar_model    = avatar,
        decode_type     = opt.decode_type,
        bake_type       = opt.bake_attribute,
    ).to(opt.device)

    editor = UVEditor(
        opt.name,
        cfg,
        decoder,
        avatar,
        opt.device,
        train_dataset   = datasets.train,
        test_dataset    = datasets.test,
        criterions      = None,
        metrics         = [],
        workspace       = opt.workspace,
        use_full_head_resume = opt.use_full_head_resume
    )

    # ----------------------------  editing ---------------------------- #
    editor.sticker_editing(sticker_name='lty')
    editor.style_transfer(transfer_mdoel='the_wave')

    