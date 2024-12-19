import  torch
import  lpips
import  numpy as np
from    tools.loss_utils.dssim import d_ssim


#-------------------------------------------------------------------------------#

class Meter:
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.V_cur = 0
        self.V = 0
        self.N = 0. + 1e-6
        self.avg = 0
        self.name = name
        self.max_name_length = max(len(self.name), 10)

    def clear(self):
        self.V = 0
        self.N = 0. + 1e-6
        self.avg = 0

    def update(self, val, n=1):
        self.V_cur = val
        self.V += val
        self.N += n
        self.avg = self.V / self.N

    def measure(self):
        return self.V / self.N

    def report(self):
        return f'{self.name.ljust(self.max_name_length)} = {self.measure():.8f}'

#-------------------------------------------------------------------------------#

class LossMeter(Meter):
    def __init__(self, name=None):
        super().__init__(name if name else 'Loss')

#-------------------------------------------------------------------------------#

class PSNR_Meter(Meter):
    def __init__(self):
        super().__init__('PSNR')

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        super().update(psnr)

    def prepare_inputs(self, *inputs):
        outputs = []
        for inp in inputs:
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)
        return outputs

#-------------------------------------------------------------------------------#

class LPIPS_Meter(Meter):
    def __init__(self, net='alex', device=None):
        super().__init__('LPIPS')
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def update(self, preds, truths):
        v = self.fn(truths, preds, normalize=True).item()   # normalize=True: [0, 1] to [-1, 1]
        super().update(v)

#-------------------------------------------------------------------------------#
    
class L1_Meter(Meter):
    def __init__(self):
        super().__init__('L1')

    def update(self, preds, truths):
        v = torch.mean(torch.abs(preds - truths)).item()
        super().update(v)

#-------------------------------------------------------------------------------#

class L2_Meter(Meter):
    def __init__(self):
        super().__init__('L2')

    def update(self, preds, truths):
        v = torch.mean(torch.square(preds - truths)).item()
        super().update(v)

#-------------------------------------------------------------------------------#

class SSIM_Meter(Meter):
    def __init__(self):
        super().__init__('SSIM')

    def update(self, preds, truths):
        v = d_ssim(preds, truths).item()
        v = 1 - v
        super().update(v)
    