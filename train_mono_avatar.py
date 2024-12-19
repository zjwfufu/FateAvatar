import  yaml
import  torch
import  argparse

from    tools.util         import seed_everything, file_backup
from    tools.util         import EasyDict

from    train.callbacks    import ModelCallbacks
from    train.trainer      import Trainer

from    common  import      load_config
from    common  import      save_identity_info
from    common  import      construct_datasets
from    common  import      construct_model
from    common  import      construct_loss
from    common  import      construct_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str,                       required=True)
    parser.add_argument('--model_name', choices=ModelCallbacks.keys(),  required=True)
    parser.add_argument('--root_path',  type=str,                       required=True)
    parser.add_argument('--workspace',  type=str,                       required=True)
    parser.add_argument('--name',       type=str,                       required=True)
    parser.add_argument('--device',     type=torch.device,              default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--seed',       type=int,                       default=42)
    parser.add_argument('--bg_color',   type=str,                       default='white')
    parser.add_argument('--resume',     action='store_true')

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
            "root_path":    opt.root_path,
            "bg_color":     opt.bg_color
            }
        )

    # --------------------------------- dataset ---------------------------------- #
    datasets, dataset_name = construct_datasets(
        opt,
        cfg
    )

    # ---------------------------- save identity info ---------------------------- #
    save_identity_info(opt.workspace, datasets.train)

    # ---------------------------------- model ----------------------------------- #
    model = construct_model(
        opt,
        cfg.model,
        cfg.dataset.canonical_pose,
        dataset = datasets.train
    )

    # ----------------------------------- loss ----------------------------------- #
    criterion = construct_loss(
        opt,
        cfg.loss,
        datasets.train
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
        train_dataset   = datasets.train,
        test_dataset    = datasets.test,
        criterions      = criterion,
        metrics         = metrics,
        workspace       = opt.workspace,
        use_checkpoint  = 'latest' if opt.resume else 'scratch'
    )
    
    # --------------------------------- execute ---------------------------------- #
    trainer.train(cfg.training.epochs[dataset_name])
    trainer.evaluate(mode='train', optim_epoch=cfg.training.epochs['finetune'])


