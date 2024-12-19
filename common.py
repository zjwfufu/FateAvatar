import  os
import  yaml
import  torch
import  pickle
import  argparse
from    tools.util         import EasyDict

from    train.loss         import parsing_loss_param

from    train.callbacks    import DatasetCallbacks, ModelCallbacks, LossCallbacks

from    train.dataset      import IMAvatarDataset, InstaDataset

from    typing  import Tuple, Union, Optional, List, Dict, Any

from    train.metrics      import (
            PSNR_Meter,
            LPIPS_Meter,
            L1_Meter,
            L2_Meter,
            SSIM_Meter
)

# ------------------------------------------------------------------------------- #
def load_config(config_path: str, overrides: dict) -> EasyDict:
    """Load configuration file and apply overrides."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = EasyDict(config)
    cfg.update(overrides)
    return cfg

# ------------------------------------------------------------------------------- #
def construct_datasets(opt: argparse.Namespace, cfg: EasyDict, mode: str="both") -> Union[Tuple[EasyDict, str], Any]:
    """Initialize training and testing datasets."""

    if any(substring in opt.root_path for substring in ['imavatar']):
        dataset_name    = 'imavatar'
    elif any(substring in opt.root_path for substring in ['4dface']):
        dataset_name    = '4dface'
    elif any(substring in opt.root_path for substring in ['insta', 'debug']):
        dataset_name    = 'insta'
    else:
        raise ValueError("dataset type is invalid, check root path")

    dataset_common_params = {
        'root_path':                opt.root_path,
        'img_res':                  [512, 512],
        'use_mean_expression':      True,
        'use_var_expression':       True,
        'background_color':         cfg.bg_color,
        'is_flashavatar':           opt.model_name == 'FlashAvatar'
    }
    train_dataset = DatasetCallbacks[dataset_name](mode="train", **dataset_common_params)
    test_dataset = DatasetCallbacks[dataset_name](mode="test", **dataset_common_params)

    n_shape     = train_dataset.n_shape
    n_exp       = train_dataset.n_exp
    type_name   = train_dataset.type_name
    optimize_tracking = train_dataset.optimize_tracking

    for key in ['dataset', 'model']:
        cfg_section = cfg[key]
        cfg_section.update({
            'n_shape':      n_shape,
            'n_exp':        n_exp,
            'dataset_type': type_name
        })
        cfg[key] = cfg_section

    cfg.update({
        'n_shape':  n_shape,
        'n_exp':    n_exp,
        'optimize_tracking': optimize_tracking
    })

    if mode == "train":
        return EasyDict({"train": train_dataset}), dataset_name
    elif mode == "test":
        return EasyDict({"test": test_dataset}), dataset_name
    elif mode == "both":
        return EasyDict({
            "train": train_dataset,
            "test": test_dataset
        }), dataset_name
    else:
        raise NotImplementedError(f'Unknown mode: {mode}')
    
# ------------------------------------------------------------------------------- #
def construct_model(
    opt: argparse.Namespace, cfg_model: EasyDict,
    canonical_pose: float,
    dataset: Union[IMAvatarDataset, InstaDataset] = None,
    identity_dict:  dict = None
) -> torch.nn.Module:
    """
    Initialize a model based on the given options and configuration.
    """
    if dataset is not None:
        shape_params            = dataset.shape_params
        img_res                 = dataset.img_res
        canonical_expression    = dataset.mean_expression
    elif identity_dict is not None:
        shape_params            = identity_dict['shape_params']
        img_res                 = identity_dict['img_res']
        canonical_expression    = identity_dict['canonical_expression']
    else:
        raise ValueError("Either 'dataset' or 'identity_dict' must be provided.")
    
    return ModelCallbacks[opt.model_name](
        shape_params            = shape_params,
        img_res                 = img_res,
        canonical_expression    = canonical_expression,
        canonical_pose          = canonical_pose,
        background_color        = opt.bg_color,
        cfg_model               = cfg_model,
        device                  = opt.device
    ).to(opt.device)

# ------------------------------------------------------------------------------- #
def save_identity_info(workspace: str, dataset: Union[IMAvatarDataset, InstaDataset]) -> None:
    """
    Save identity information of the train dataset to a file.
    """
    identity_dict = {
        'shape_params':             dataset.shape_params,
        'canonical_expression':     dataset.mean_expression,
        'n_shape':                  dataset.n_shape,
        'n_exp':                    dataset.n_exp,
        'camera_rotation':          dataset.fix_cam_rot,
        'camera_translation':       dataset.data["world_mats"][0][:3, 3],
        'camera_fovx':              dataset.fovx,
        'camera_fovy':              dataset.fovy,
        'img_res':                  dataset.img_res,
        'dataset_type':             dataset.type_name
    }
    os.makedirs(workspace, exist_ok=True)
    with open(os.path.join(workspace, "identity_dict.pkl"), "wb") as f:
        pickle.dump(identity_dict, f)

# ------------------------------------------------------------------------------- #
def load_identity_info(
    opt: argparse.Namespace, 
    cfg: EasyDict, 
) -> dict:
    """
    Load the identity dictionary from a pickle file and update the configuration.
    """
    identity_dict_path = os.path.join(opt.workspace, 'identity_dict.pkl')
    with open(identity_dict_path, 'rb') as f:
        identity_dict = pickle.load(f)

    if identity_dict['dataset_type'] == 'insta':
        cfg.optimize_tracking = False
    elif identity_dict['dataset_type'] == 'imavatar':
        cfg.optimize_tracking = True
    else:
        raise ValueError(f"Unknown dataset type: {identity_dict['dataset_type']}")

    cfg.camera_rotation = identity_dict['camera_rotation']
    cfg.camera_translation = identity_dict['camera_translation']
    cfg.camera_fovx = identity_dict['camera_fovx']
    cfg.camera_fovy = identity_dict['camera_fovy']

    # Update model-related values in cfg_model
    cfg_model   = cfg.model
    cfg_model.update({
        'n_shape':  identity_dict['n_shape'],
        'n_exp':    identity_dict['n_exp']
    })
    cfg.model   = cfg_model

    return identity_dict

# ------------------------------------------------------------------------------- #
def construct_loss(
    opt: argparse.Namespace, cfg_loss: EasyDict,
    dataset: Union[IMAvatarDataset, InstaDataset]
) -> torch.nn.Module:
    """
    Initialize the loss function based on the model name and configuration.
    """
    if opt.model_name == "MonoGaussianAvatar":
        cfg_loss.update(
            {
                "use_var_expression": True,
                "var_expression": dataset.var_expression.to(opt.device),
                "dataset_type": dataset.type_name,
            }
        )
    loss_param = parsing_loss_param(LossCallbacks[opt.model_name], cfg_loss)
    return LossCallbacks[opt.model_name](loss_param).to(opt.device)

# ------------------------------------------------------------------------------- #
def construct_metrics(device: torch.device) -> List[Any]:
    """
    Initialize evaluation metrics.
    """
    return [PSNR_Meter(), LPIPS_Meter(device=device), L1_Meter(), L2_Meter(), SSIM_Meter()]