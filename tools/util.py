import torch
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import PIL
import time
import functools


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""
    
    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
            if isinstance(value, dict):
                value = EasyDict(value)
            return value
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self:
            self[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self:
            del self[name]
        else:
            super().__delattr__(name)

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict):
            value = EasyDict(value)
        super().__setitem__(key, value)


def dict2obj(d):
# if isinstance(d, list):
#     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d
    class C(object):
        pass
    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def transformations_to_rotation_matrix(transformations, lbs_weights):
    # W is num_points x (J + 1)
    W = lbs_weights
    num_points = W.shape[0]
    num_joints = W.shape[-1]
    # T: [num_points, (J + 1)] x [num_points, (J + 1), 16] --> [num_points, 16]
    T = torch.einsum('mj, mjk->mk', [W, transformations.view(-1, num_joints, 16)]).view(num_points, 4, 4)
    R = T[:, :3, :3]
    return R


def print_tree(logger, ele, deep=0):
    if isinstance(ele, dict):
        for key, value in ele.items():
            if isinstance(value, dict):
                logger('    |'*deep+'---'+key)
                print_tree(logger, value, deep+1)
            else:
                logger('    |'*deep+'---'+key + ': ' + str(value))
    else:
        logger('    |'*deep+'---'+str(ele))


def file_backup(workspace, config_path):
    from shutil import copyfile
    dir_lis = ['./tools', './model', './train', './flame',
                './mesh_rendering', './volume_rendering']
    
    os.makedirs(os.path.join(workspace, 'archive'), exist_ok=True)
    for dir_name in dir_lis:
        cur_dir = os.path.join(workspace, 'archive', dir_name)
        os.makedirs(cur_dir, exist_ok=True)
        files = os.listdir(dir_name)
        for f_name in files:
            if f_name[-3:] == '.py':
                copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

    copyfile(config_path, os.path.join(workspace, 'archive', os.path.basename(config_path)))

    
def load_to_gpu(model_input, ground_truth, device):
        for k, v in model_input.items():
            try:
                model_input[k] = v.to(device)
            except:
                model_input[k] = v
        for k, v in ground_truth.items():
            try:
                ground_truth[k] = v.to(device)
            except:
                ground_truth[k] = v


class Points:
    def __init__(self, 
                 face_index     = None, 
                 bary_coords    = None,
                 init_flag      = None,
                 shell_len      = None):
        self.face_index   = []   if face_index is None else face_index
        self.bary_coords  = []   if bary_coords is None else bary_coords
        self.init_flag    = []   if init_flag is None else init_flag
        self.shell_len    = []   if shell_len is None else shell_len


def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    
    if max_val == min_val:
        return torch.zeros_like(tensor)
    
    return (tensor - min_val) / (max_val - min_val)


def get_bg_color(bg_color:str):
    """Range between 0, 1"""
    if bg_color == 'white':
        background = torch.ones((3,), dtype=torch.float32)
    elif bg_color == "black":
        background = torch.zeros((3,), dtype=torch.float32)
    elif bg_color == "random":
        background = torch.rand((3,), dtype=torch.float32)
    else:
        raise ValueError("Invalid Color!")
    return background


def colorize_weights_map(weights, colormap='jet', min_val=None, max_val=None):
    if min_val is None:
        min_val = weights.min()
    if max_val is None:
        max_val = weights.max()

    vals = (weights - min_val) / (max_val - min_val)
    vals = vals.clamp(0, 1) * 255
    
    vals_np = vals.byte().cpu().numpy()
    
    cmap = plt.get_cmap(colormap)
    canvas = cmap(vals_np / 255.0)[..., :3]
    
    canvas_tensor = torch.from_numpy(canvas).float()
    canvas_tensor = canvas_tensor.permute(0, 3, 1, 2)
    
    return canvas_tensor


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True):
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded. B x C x ...
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    if num_encoding_functions == 0:
        return tensor
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0**0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=1)


def measure_fps(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        output = func(self, *args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else float('inf')
        return fps

    return wrapper