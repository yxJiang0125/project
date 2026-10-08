import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp


class IntensityLoss(nn.Module):
    def forward(self, prediction, target):
        return torch.mean(torch.pow(torch.abs(prediction - target), 2))


class GradientLoss(nn.Module):
    def __init__(self, channels=3):
        super(GradientLoss, self).__init__()
        pos = torch.from_numpy(np.identity(channels, dtype=np.float32))
        neg = -1 * pos
        filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1)
        filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1)
        self.register_buffer('filter_x', filter_x)
        self.register_buffer('filter_y', filter_y)

    def forward(self, prediction, target):
        filter_x = self.filter_x
        filter_y = self.filter_y
        gen_frames_x = nn.functional.pad(prediction, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(prediction, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(target, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(target, [0, 0, 0, 1])
        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, filter_y))
        return torch.mean(torch.abs(gt_dx - gen_dx) + torch.abs(gt_dy - gen_dy))


class L2Loss(nn.Module):
    def __init__(self, eps=1e-8):
        super(L2Loss, self).__init__()
        self.eps = eps

    def forward(self, prediction, target):
        error = torch.mean(torch.pow((prediction - target), 2))
        return torch.sqrt(error + self.eps)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = (torch.mean(v1 / v2) + 1) / 2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = (ssim_map.mean() + 1) / 2 if size_average else (ssim_map.mean(1).mean(1).mean(1) + 1) / 2

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    mssim, mcs = [], []
    for _ in range(weights.size()[0]):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average,
                       full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    return torch.prod(pow1[:-1] * pow2[-1])


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


class MultiLossFunction(nn.Module):
    def __init__(self, config):
        super(MultiLossFunction, self).__init__()
        self.intensity_loss = IntensityLoss()
        self.gradient_loss = GradientLoss(channels=config.DATASET.NUM_INCHANNELS)
        self.msssim_loss = MSSSIM()
        self.l2_loss = L2Loss()

    def forward(self, prediction, target):
        inte_loss = self.intensity_loss(prediction, target)
        grad_loss = self.gradient_loss(prediction, target)
        msssim_val = (1 - self.msssim_loss(prediction, target)) / 2
        l2_loss = self.l2_loss(prediction, target)
        return inte_loss, grad_loss, msssim_val, l2_loss
