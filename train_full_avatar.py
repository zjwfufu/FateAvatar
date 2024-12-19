import  os
import  yaml
import  torch
import  argparse

from    tools.util import seed_everything
from    tools.util import EasyDict

from    train.callbacks    import ModelCallbacks
from    train.completor    import CompletionTrainer

from    common  import      load_config
from    common  import      construct_datasets
from    common  import      load_identity_info
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

    # --------------------------------- dataset ---------------------------------- #
    datasets, dataset_name = construct_datasets(
        opt,
        cfg
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

    # avoid overfit
    if opt.model_name == 'GaussianAvatars':
        model.active_sh_degree = 0
        model.max_sh_degree = 0

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

    # ------------------------------ setup trainer ------------------------------- #
    trainer = CompletionTrainer(
        opt.name,
        cfg,
        model,
        opt.device,
        train_dataset   = datasets.train,
        test_dataset    = datasets.test,
        criterions      = criterion,
        metrics         = metrics,
        workspace       = opt.workspace,
        use_checkpoint  = 'latest'
    )
    
    # --------------------------------- execute ---------------------------------- #
    trainer.render_dynamic_novel_view(name='raw_dynamic_novel_view', mode='eval')
    trainer.render_dynamic_fixed_view(name='raw_dynamic_fixed_view', mode='eval')

    trainer.augmentation(finetune_epoch=1)
    full_head_ckpt = os.path.join(opt.workspace, 'checkpoints_fullhead')

    trainer.evaluate_epoch(name='aug', mode='train_full_head')
    os.makedirs(full_head_ckpt, exist_ok=True)
    trainer.save_checkpoint(name='fullhead', remove_old=False, save_path=full_head_ckpt)

    trainer.render_dynamic_novel_view(name='dynamic_novel_view', mode='eval')
    trainer.render_dynamic_fixed_view(name='dynamic_fixed_view', mode='eval')


