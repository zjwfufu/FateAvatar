import  os
import  yaml
import  torch
import  argparse

from    tools.util         import seed_everything
from    tools.util         import EasyDict

from    train.callbacks    import ModelCallbacks
from    train.loader       import Reenactor

from    common  import      load_config
from    common  import      construct_datasets
from    common  import      load_identity_info
from    common  import      construct_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str,                       required=True)
    parser.add_argument('--model_name', choices=ModelCallbacks.keys(),  required=True)
    parser.add_argument('--root_path',  type=str,                       required=True)
    parser.add_argument('--dst_path',   type=str,                       required=True)
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

    # ----------------------------  override config ---------------------------- #
    cfg = load_config(
        opt.config,
        overrides={
            "name":         opt.name,
            "workspace":    opt.workspace,
            "bg_color":     opt.bg_color,
            "dst_path":     opt.dst_path,
            }
        )

    # --------------------------------- dataset --------------------------------- #
    dataset, dataset_name = construct_datasets(
        opt,
        cfg,
        mode = "train"
    )
    dst_dataset = dataset.train

    dst_loader = torch.utils.data.DataLoader(
        dst_dataset,
        batch_size  = 1,
        shuffle     = False,
        collate_fn  = dst_dataset.collate_fn,
        num_workers = 4,
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
    # ----------------------------- setup reenactor ------------------------------ #
    reenactor = Reenactor(
        opt.name,
        cfg,
        model,
        opt.device,
        workspace       = opt.workspace,
        use_checkpoint  = 'latest'
    )
    
    # --------------------------------- execute ---------------------------------- #
    dst_exp     = dst_dataset.mean_expression
    src_exp     = identity_dict['canonical_expression']
    delta_exp   = src_exp - dst_exp

    reenactor.reenacting(
        dst_name=os.path.basename(cfg.dst_path),
        dst_loader=dst_loader,
        delta_exp=delta_exp
    )

    