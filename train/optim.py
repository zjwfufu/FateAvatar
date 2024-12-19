import torch.optim as optim

from model.fateavatar                   import FateAvatar
from model.baseline.flashavatar         import FlashAvatar
from model.baseline.gaussianavatars     import GaussianAvatars
from model.baseline.monogaussianavatar  import MonoGaussianAvatar
from model.baseline.splattingavatar     import SplattingAvatar

# ------------------------------------------------------------------------------- #

def register_optimizer_group_fateavatar(model: FateAvatar, cfg):

    optimizer_group = {}

    gs_l = [
        {'params': [model._opacity],        'lr': cfg.training.opacity_lr,     "name": "opacity"},
        {'params': [model._offset],         'lr': cfg.training.offset_lr,      "name": "offset"},
        {'params': [model._features_dc],    'lr': cfg.training.feature_dc_lr,  "name": "color"},
        {'params': [model._rotation],       'lr': cfg.training.rotation_lr,    "name": "rotation"},
        {'params': [model._scaling],        'lr': cfg.training.scaling_lr,     "name": "scaling"}
    ]

    gs_optimizer = optim.Adam(gs_l, lr=0.0)

    optimizer_group.update({'gs': gs_optimizer})

    bs_l = [
        {'params': [model.delta_shapedirs], 'lr': cfg.training.delta_shapedirs_lr, 'name': "delta_shapedirs"},
        {'params': [model.delta_posedirs],  'lr': cfg.training.delta_posedirs_lr,  'name': "delta_posedirs"},
        {'params': [model.delta_vertex], 'lr': 0.0001, 'name': "delta_vertex"},
    ]

    bs_optimizer = optim.Adam(bs_l, lr=0.0)

    optimizer_group.update({'bs': bs_optimizer})

    return optimizer_group

# ------------------------------------------------------------------------------- #

def register_optimizer_group_flashavatar(model: FlashAvatar, cfg):
        
        optimizer_group = {}

        gs_l = [
            {'params': [model._opacity],        'lr': cfg.training.opacity_lr,        "name": "opacity"},
            {'params': [model._features_dc],    'lr': cfg.training.feature_dc_lr,     "name": "feature_dc"},
            {'params': [model._features_rest],  'lr': cfg.training.feature_dc_lr / 20,"name": "feature_rest"},
            {'params': [model._rotation],       'lr': cfg.training.rotation_lr,       "name": "rotation"},
            {'params': [model._scaling],        'lr': cfg.training.scaling_lr,        "name": "scaling"}
        ]

        gs_optimizer = optim.Adam(gs_l, lr=0.0)

        optimizer_group.update({'gs': gs_optimizer})

        net_l = [
                {'params': model.deformNet.parameters(), 'lr': cfg.training.deformer_lr},
            ]
        
        deformer_optimizer = optim.Adam(net_l, betas=(0.9, 0.999))

        optimizer_group.update({'deformer': deformer_optimizer})
        
        return optimizer_group

# ------------------------------------------------------------------------------- #

def register_optimizer_gruop_gaussianavatars(model: GaussianAvatars, cfg):

    optimizer_group = {}

    gs_l = [
        {'params': [model._xyz],            'lr': cfg.training.position_lr_init,  "name": "_xyz"},
        {'params': [model._opacity],        'lr': cfg.training.opacity_lr,        "name": "_opacity"},
        {'params': [model._features_dc],    'lr': cfg.training.feature_dc_lr,     "name": "_features_dc"},
        {'params': [model._features_rest],  'lr': cfg.training.feature_dc_lr / 20,"name": "_features_rest"},
        {'params': [model._rotation],       'lr': cfg.training.rotation_lr,       "name": "_rotation"},
        {'params': [model._scaling],        'lr': cfg.training.scaling_lr,        "name": "_scaling"}
    ]

    gs_optimizer = optim.Adam(gs_l, lr=0.0)

    optimizer_group.update({'gs': gs_optimizer})

    return optimizer_group

# ------------------------------------------------------------------------------- #

def register_optimizer_group_monogaussianavatar(model: MonoGaussianAvatar, cfg):

    optimizer_group = {}

    nn_l = [
            {'params': list(model.parameters()), 'lr': cfg.training.lr},
        ]
    
    nn_optimizer = optim.Adam(nn_l)

    optimizer_group.update({'nn': nn_optimizer})

    return optimizer_group
    
# ------------------------------------------------------------------------------- #

def register_optimizer_group_splattingavatar(model: SplattingAvatar, cfg):

    optimizer_group = {}
    
    gs_l = [
        {'params': [model._uvd],            'lr': cfg.training.uvd_lr,            "name": "_uvd"},
        {'params': [model._opacity],        'lr': cfg.training.opacity_lr,        "name": "_opacity"},
        {'params': [model._features_dc],    'lr': cfg.training.feature_dc_lr,     "name": "_features_dc"},
        {'params': [model._features_rest],  'lr': cfg.training.feature_dc_lr / 20,"name": "_features_rest"},
        {'params': [model._rotation],       'lr': cfg.training.rotation_lr,       "name": "_rotation"},
        {'params': [model._scaling],        'lr': cfg.training.scaling_lr,        "name": "_scaling"}
    ]

    gs_optimizer = optim.Adam(gs_l, lr=0.0)

    optimizer_group.update({'gs': gs_optimizer})

    return optimizer_group