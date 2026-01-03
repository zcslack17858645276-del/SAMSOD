import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIM(nn.Module):
    """
    Structural Similarity Loss
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        # img1: Prediction (Probabilities 0-1)
        # img2: Target (0 or 1)
        
        if img1.is_cuda:
            self.window = self.window.cuda(img1.get_device())
        self.window = self.window.type_as(img1)

        channel = img1.size(1)
        
        # calculate the mean
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # calculate variance and covariance
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        # SSIM
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class SODLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, ssim_weight=1.0):
        super(SODLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ssim_weight = ssim_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ssim_module = SSIM(window_size=11, size_average=True)

    def forward(self, pred_logits, target):
        """
        pred_logits: [B, 1, H, W] (SAM2 输出的原始 logits，未经过 Sigmoid)
        target:      [B, 1, H, W] (Ground Truth，0 或 1)
        """
        
        # align the size 
        if pred_logits.shape[-2:] != target.shape[-2:]:
            pred_logits = F.interpolate(pred_logits, size=target.shape[-2:], mode='bilinear', align_corners=False)

        # generate the probabilities for ssim and dice
        pred_probs = torch.sigmoid(pred_logits)

        # ============================
        # use logits to calculate BCE Loss
        # ============================
        loss_bce = self.bce_loss(pred_logits, target)

        # ============================
        # use probs to calculate SSIM Loss
        # ============================
        # calculate SSIM，Loss = 1 - SSIM
        loss_ssim = 1 - self.ssim_module(pred_probs, target)

        # ============================
        # use probs to calculate Dice Loss
        # ============================
        smooth = 1e-5
        intersection = (pred_probs * target).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        loss_dice = 1 - dice.mean()

        # ============================
        # Total Loss
        # ============================
        total_loss = (self.bce_weight * loss_bce) + \
                     (self.dice_weight * loss_dice) + \
                     (self.ssim_weight * loss_ssim)

        return total_loss, loss_bce, loss_dice, loss_ssim