from train.dataset                      import (
        IMAvatarDataset,
        InstaDataset
)

from train.loss                         import (
        FateAvatarLoss,
        FlashAvatarLoss,
        GaussianAvatarsLoss,
        MonoGaussianAvatarLoss,
        SplattingAvatarLoss
)

from model.fateavatar                   import FateAvatar
from model.baseline.flashavatar         import FlashAvatar
from model.baseline.gaussianavatars     import GaussianAvatars
from model.baseline.monogaussianavatar  import MonoGaussianAvatar
from model.baseline.splattingavatar     import SplattingAvatar

DatasetCallbacks = {
    '4dface':                           IMAvatarDataset,
    'imavatar':                         IMAvatarDataset,
    'insta' :                           InstaDataset
}

ModelCallbacks = {
    'FateAvatar':                       FateAvatar,
    'FlashAvatar':                      FlashAvatar,
    'GaussianAvatars':                  GaussianAvatars,
    'MonoGaussianAvatar':               MonoGaussianAvatar,
    'SplattingAvatar':                  SplattingAvatar
}

LossCallbacks = {
    'FateAvatar':                       FateAvatarLoss,
    'FlashAvatar':                      FlashAvatarLoss,
    'GaussianAvatars':                  GaussianAvatarsLoss,
    'MonoGaussianAvatar':               MonoGaussianAvatarLoss,
    'SplattingAvatar':                  SplattingAvatarLoss
}