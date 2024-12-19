from .fateavatar import FateAvatar
from .baseline.flashavatar          import FlashAvatar
from .baseline.gaussianavatars      import GaussianAvatars
from .baseline.monogaussianavatar   import MonoGaussianAvatar
from .baseline.splattingavatar      import SplattingAvatar

from typing import Union

ModelClass = Union[
    FateAvatar, FlashAvatar, SplattingAvatar, GaussianAvatars, MonoGaussianAvatar
]
