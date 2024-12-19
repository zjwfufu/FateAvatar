import  yaml
import  torch
import  argparse

from    tools.util         import seed_everything, file_backup
from    tools.util         import EasyDict

from    train.completion   import PseudoGenerator
from    train.callbacks    import ModelCallbacks

from    common  import load_config
from    common  import load_identity_info
from    common  import construct_model

NOVEL_VIEW          = True
DLIB_KPS            = True
AFFINE_TRANSFORM    = True
INJECT_PRIOR        = True
PANOHEAD_INVERSION  = True
RENDER_INVERSION    = True
INVERSE_TRAMSFORM   = True
RETRIEVE_MASK       = True
HEATMAP_CHECK       = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str,                       required=True)
    parser.add_argument('--model_name', choices=ModelCallbacks.keys(),  required=True)
    parser.add_argument('--workspace',  type=str,                       required=True)
    parser.add_argument('--name',       type=str,                       required=True)
    parser.add_argument('--device',     type=torch.device,              default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--seed',       type=int,                       default=42)
    parser.add_argument('--bg_color',   type=str,                       default='white')

    opt = parser.parse_args()

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
            "bg_color":     opt.bg_color
            }
        )

    # ---------------------------- load identity info ---------------------------- #
    identity_dict = load_identity_info(opt, cfg)

    # ---------------------------------- model ----------------------------------- #
    model = construct_model(
        opt,
        cfg.model,
        0.0,
        identity_dict = identity_dict
    )
    
     # ------------------------------ setup generator ---------------------------- #
    generator = PseudoGenerator(
        opt.name,
        cfg,
        model,
        opt.device,
        workspace       = opt.workspace,
        use_checkpoints = 'latest' 
    )
    
    # --------------------------------- execute ---------------------------------- #
    if NOVEL_VIEW:
        generator.render_novel_view(orbit_frames=30)

    if DLIB_KPS:
        generator.detect_dlib_kps()

    if AFFINE_TRANSFORM:
        generator.execute_affine_transform()

    if INJECT_PRIOR:
        generator.inject_ffhq_prior()

    if PANOHEAD_INVERSION:
        generator.proceed_gan_inversion()

    if RENDER_INVERSION:
        generator.render_inversion_result(orbit_frames=30)

    if INVERSE_TRAMSFORM:
        generator.execute_inverse_transform()

    if RETRIEVE_MASK:
        generator.retrieve_image_mask()
        generator.retrieve_image_mask_modnet()
    
    if HEATMAP_CHECK:
        generator.heatmap_check()





