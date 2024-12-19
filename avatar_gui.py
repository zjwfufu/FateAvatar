import  os
import  glob
import  argparse
import  torch

from    tools.gui           import Viewer
from    train.loader        import Loader
from    train.callbacks     import ModelCallbacks

from    common  import      load_config
from    common  import      load_identity_info
from    common  import      construct_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str,                       required=True)
    parser.add_argument('--model_name', choices=ModelCallbacks.keys(),  required=True)
    parser.add_argument('--workspace',  type=str,                       required=True)
    parser.add_argument('--name',       type=str,                       required=True)
    parser.add_argument('--device',     type=torch.device,              default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--seed',       type=int,                       default=42)
    parser.add_argument('--bg_color',   type=str,                       default='white')
    parser.add_argument('--ckpt_path',  type=str,                       required=False)
    parser.add_argument('--use_full_head_resume',                       action='store_true')
    parser.add_argument('--use_baked_resume',                           action='store_true')

    opt = parser.parse_args()

    # ----------------------------  override config ---------------------------- #
    cfg = load_config(
        opt.config,
        overrides={
            "name":         opt.name,
            "workspace":    opt.workspace,
            "bg_color":     opt.bg_color
            }
        )

    # ---------------------------- load identity info ---------------------------- #
    identity_dict = load_identity_info(opt, cfg)

    # ---------------------------------- model ----------------------------------- #
    avatar = construct_model(
        opt,
        cfg.model,
        0.0,
        identity_dict = identity_dict
    )

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

    # ----------------------------  override config ---------------------------- #
    gui = Viewer(
        opt,
        cfg,
        avatar,
        loader,
        identity_dict
    )
    gui.render()