import  os
import  yaml
import  torch
import  argparse

from    tools.util import seed_everything
from    tools.util import EasyDict

from    model.uv_decoder   import UVDecoder

from    train.loss          import UVDecoderLoss
from    train.baker         import UVBaker
from    train.loader        import Loader

from    common  import      load_config
from    common  import      construct_datasets
from    common  import      load_identity_info
from    common  import      construct_model
from    common  import      construct_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',         type=str,           required=True)
    parser.add_argument('--workspace',      type=str,           required=True)
    parser.add_argument('--name',           type=str,           required=True)
    parser.add_argument('--root_path',      type=str,           required=True)
    parser.add_argument('--device',         type=torch.device,  default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--seed',           type=int,           default=42)
    parser.add_argument('--bg_color',       type=str,           default='white')
    parser.add_argument('--use_full_head_resume',               action='store_true')
    # for uv decoder
    parser.add_argument('--decode_type',   type=str,        default='UNet')
    parser.add_argument('--reg_weight',    type=float,      default=0.0)
    parser.add_argument('--rot_weight',    type=float,      default=0.1)
    parser.add_argument('--reg_attribute', nargs='+',       default=[]) # ['color', 'opacity', 'scaling', 'rotation', 'offset']
    parser.add_argument('--bake_attribute', nargs='+',      default=['color', 'opacity', 'scaling', 'rotation', 'offset'])

    opt = parser.parse_args()

    opt.model_name = "FateAvatar"   # neural baking only support in FateAvatar

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
    full_head_path = os.path.join(opt.workspace, "checkpoints_fullhead", "fullhead.pth")

    loader = Loader(
        opt.name,
        cfg,
        avatar,
        opt.device,
        workspace       = opt.workspace,
        use_checkpoint  = full_head_path if opt.use_full_head_resume else 'latest'
    )

    # overwrite
    avatar = loader.model

    # ------------------------------ assign BakeNet ------------------------------ #
    decoder = UVDecoder(
        avatar_model    = avatar,
        decode_type     = opt.decode_type,
        bake_type       = opt.bake_attribute,
    ).to(opt.device)

    cfg['loss']['weight']['reg_loss'] = opt.reg_weight

    decode_loss = {
        'rgb_weight':       cfg.loss.weight.rgb_loss,
        'vgg_weight':       cfg.loss.weight.vgg_loss,
        'dssim_weight':     cfg.loss.weight.dssim_loss,
        'scale_weight':     cfg.loss.weight.scale_loss,
        'scale_threshold':  cfg.loss.scale_threshold,
        'rot_weight':       opt.rot_weight,
        'laplacian_weight': cfg.loss.weight.laplacian_loss,
        'normal_weight':    cfg.loss.weight.normal_loss,
        'flame_weight':     cfg.loss.weight.flame_loss,
        'reg_weight':       opt.reg_weight,
        'reg_attribute':    opt.reg_attribute
    }
    decode_loss = UVDecoderLoss.Params(**decode_loss)
    criterion   = UVDecoderLoss(decode_loss).to(opt.device)

    # --------------------------------- metrics ---------------------------------- #
    metrics = construct_metrics(
        opt.device
    )
    
    # ------------------------------- neural baker ------------------------------- #
    extractor = UVBaker(
        opt.name,
        cfg,
        decoder,
        avatar,
        opt.device,
        train_dataset   = datasets.train,
        test_dataset    = datasets.test,
        criterions      = criterion,
        metrics         = metrics,
        workspace       = opt.workspace,
        use_full_head_resume = opt.use_full_head_resume
    )

    # --------------------------------- execute ---------------------------------- #
    extractor.bake(max_epochs=5)
    extractor.evaluate_epoch(name='uv_decode', mode='train')



    