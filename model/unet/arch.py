import torch
import torch.nn as nn
import torch.nn.functional as F

#-------------------------------------------------------------------------------#

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UNet, self).__init__()
        self.in_ch      = in_ch
        self.out_ch     = out_ch
        self.bilinear   = bilinear

        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_ch)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        tex = self.outc(x)
        return tex
    
#-------------------------------------------------------------------------------#

class UNetDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UNetDecoder, self).__init__()
        self.in_ch      = in_ch
        self.out_ch     = out_ch
        self.bilinear   = bilinear

        factor = 2 if bilinear else 1
        # self.up1 = UpNoSkip(512, 512, bilinear) # 4x4 -> 8x8
        self.up2 = UpNoSkip(512, 512, bilinear) # 8x8 -> 16x16
        self.up3 = UpNoSkip(512, 256, bilinear) # 16x16 -> 32x32
        self.up4 = UpNoSkip(256, 128, bilinear) # 32x32 -> 64x64
        self.up5 = UpNoSkip(128, 64, bilinear)  # 64x64 -> 128x128
        self.up6 = UpNoSkip(64, 32, bilinear)   # 128x128 -> 256x256
        self.outc = OutConv(32, out_ch)

    def forward(self, x):
        # x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.up6(x)
        tex = self.outc(x)
        # tex = self.optim_texture
        return tex
    
#-------------------------------------------------------------------------------#

class FeatureMap(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(FeatureMap, self).__init__()

        self.optim_texture = nn.Parameter(
            torch.FloatTensor(1, out_ch, 512, 512).uniform_(-1, 1)
        )

    def forward(self, x):
        tex = self.optim_texture
        return tex

#-------------------------------------------------------------------------------#

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

#-------------------------------------------------------------------------------#

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)

#-------------------------------------------------------------------------------#

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
#-------------------------------------------------------------------------------#
    
class UpNoSkip(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        # input is CHW
        return self.conv(x1)
    
#-------------------------------------------------------------------------------#

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)