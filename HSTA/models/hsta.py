import logging
import random

import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.swin_transformer import swin_base_patch4_window7_224_in22k

from models.basic_modules import ConvBnRelu, ConvTransposeBnRelu, initialize_weights

logger = logging.getLogger(__name__)


class HybridCNN_Swin(nn.Module):
    """Hybrid encoder: ResNet-50 (local) + Swin Transformer (global)."""

    def __init__(self, pretrained=True):
        super(HybridCNN_Swin, self).__init__()

        resnet = models.resnet50(pretrained=pretrained)
        self.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
        )
        self.layer2 = resnet.layer2

        swin = swin_base_patch4_window7_224_in22k(pretrained=pretrained)
        self.swin_patch_embed = swin.patch_embed
        self.swin_pos_drop = getattr(swin, 'pos_drop', nn.Identity())
        self.swin_layers = swin.layers

        self.out_channels = {
            "x2": 512,   # resnet layer2  [B, 512, 28, 28]
            "x3": 512,   # swin stage3    [B, 512, 14, 14]
            "x7": 1024,  # swin stage4    [B, 1024, 7, 7]
        }

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        x_swin = self.swin_patch_embed(x)
        x_swin = self.swin_pos_drop(x_swin)
        x_swin = self.swin_layers[0](x_swin)
        x_swin = self.swin_layers[1](x_swin)

        x3_seq = self.swin_layers[2](x_swin)
        B3, H3, W3, C3 = x3_seq.shape
        x3 = x3_seq.permute(0, 3, 1, 2).contiguous()

        x7_seq = self.swin_layers[3](x3_seq)
        B4, H4, W4, C4 = x7_seq.shape
        x7 = x7_seq.permute(0, 3, 1, 2).contiguous()

        return x2, x3, x7


class TemporalShiftRandFold(nn.Module):
    """Random Temporal Shift (RTS) module."""

    def __init__(self, n_segment=4, min_div=8, max_div=16):
        super(TemporalShiftRandFold, self).__init__()
        self.n_segment = n_segment
        self.min_div = min_div
        self.max_div = max_div

    def forward(self, x):
        direction = random.choice(['left', 'right', 'both'])
        bz, nt, h, w = x.size()
        c = nt // self.n_segment
        x = x.view(bz, self.n_segment, c, h, w)

        fold_div = random.randint(self.min_div, self.max_div)
        fold = max(1, c // fold_div)

        out = torch.zeros_like(x)
        if direction == 'left':
            out[:, :-1, :fold] = x[:, 1:, :fold]
            out[:, :, fold:] = x[:, :, fold:]
        elif direction == 'right':
            out[:, 1:, :fold] = x[:, :-1, :fold]
            out[:, :, fold:] = x[:, :, fold:]
        else:  # both
            half = fold // 2
            out[:, :-1, :half] = x[:, 1:, :half]
            out[:, 1:, half:fold] = x[:, :-1, half:fold]
            out[:, :, fold:] = x[:, :, fold:]

        return out.view(bz, nt, h, w)


class TemporalAwareChannelAttention(nn.Module):
    """Temporal-Aware Channel Attention (part of TACAB)."""

    def __init__(self, channel, reduction=16, n_segment=4):
        super().__init__()
        self.n_segment = n_segment
        self.temporal_pool = nn.AdaptiveAvgPool3d(1)
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, 1),
        )
        self.motion_conv = nn.Conv3d(channel, channel // 4,
                                     kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        T = self.n_segment
        if B < T or B % T != 0:
            return self.sigmoid(torch.ones_like(x))

        x_3d = x.view(B // T, T, C, H, W).permute(0, 2, 1, 3, 4)

        temporal_global = self.temporal_pool(x_3d)
        temporal_weights = self.temporal_conv(temporal_global).squeeze(-1).squeeze(-1).squeeze(-1)

        motion_feat = self.motion_conv(x_3d)
        motion_importance = motion_feat.mean(dim=[2, 3, 4], keepdim=True)

        combined_weights = (
            temporal_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            + motion_importance.repeat(1, 4, 1, 1, 1).view(B // T, C, 1, 1, 1)
        )

        attention_map = combined_weights.repeat(1, 1, T, H, W)
        attention_map = attention_map.permute(0, 2, 1, 3, 4).contiguous().view(B, C, H, W)
        return self.sigmoid(attention_map)


class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride,
                              kernel_size=kernel_size, bias=True)
        self.norm = norm
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_feat)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y


class RCAB(nn.Module):
    """Residual Channel Attention Block with optional Temporal-Aware Channel Attention (TACAB)."""

    def __init__(self, in_feat, out_feat, kernel_size, reduction,
                 norm=False, act=nn.ReLU(True), downscale=False,
                 n_segment=4, use_temporal=True):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=1, norm=norm),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
        )
        self.CA = CALayer(out_feat, reduction)
        self.sig = nn.Sigmoid()

        self.use_temporal = use_temporal
        if use_temporal:
            self.TACA = TemporalAwareChannelAttention(out_feat, reduction=reduction,
                                                      n_segment=n_segment)
        self.downscale = downscale
        if downscale:
            self.downConv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.body(x)
        ca = self.CA(out)
        out = out * self.sig(ca)

        if self.use_temporal:
            ta = self.TACA(out)
            out = out * self.sig(ta)

        if self.downscale:
            res = self.downConv(res)
        return res + out


class Network_Myself(nn.Module):
    """HSTA: Hybrid CNN-Swin Temporal Attention Network."""

    def get_name(self):
        return self.model_name

    def __init__(self, config, pretrained=True):
        super(Network_Myself, self).__init__()
        frames = config.MODEL.ENCODED_FRAMES
        self.model_name = config.MODEL.NAME
        final_conv_kernel = config.MODEL.EXTRA.FINAL_CONV_KERNEL
        logger.info(self.model_name + ' is built!')

        self.backbone = HybridCNN_Swin(pretrained=pretrained)

        self.channels = {
            "enc_x7_out": 4096,
            "enc_x3_out": 1024,
            "enc_x2_out": 512,
            "dec_up8":    2048,
            "dec_up4":    1024,
            "dec_up2":    512,
            "dec_up1":    256,
            "dec_up0":    128,
            "dec_final":  64,
        }

        self.conv_x7 = nn.Conv2d(self.backbone.out_channels["x7"] * frames,
                                 self.channels["enc_x7_out"], kernel_size=1, bias=False)
        self.conv_x3 = nn.Conv2d(self.backbone.out_channels["x3"] * frames,
                                 self.channels["enc_x3_out"], kernel_size=1, bias=False)
        self.conv_x2 = nn.Conv2d(self.backbone.out_channels["x2"] * frames,
                                 self.channels["enc_x2_out"], kernel_size=1, bias=False)

        self.tsm_rand = TemporalShiftRandFold(n_segment=frames, min_div=8, max_div=16)

        lReLU = nn.LeakyReLU(0.2, True)

        self.up8 = ConvTransposeBnRelu(self.channels["enc_x7_out"], self.channels["dec_up8"], kernel_size=2)
        self.attn8 = RCAB(self.channels["dec_up8"], self.channels["dec_up4"],
                          kernel_size=3, reduction=16, norm='BN', act=lReLU, downscale=True)

        self.up4 = ConvTransposeBnRelu(
            self.channels["dec_up4"] + self.channels["enc_x3_out"],
            self.channels["dec_up4"], kernel_size=2
        )
        self.attn4 = RCAB(self.channels["dec_up4"], self.channels["dec_up2"],
                          kernel_size=3, reduction=16, norm='BN', act=lReLU, downscale=True)

        self.up2 = ConvTransposeBnRelu(
            self.channels["dec_up2"] + self.channels["enc_x2_out"],
            self.channels["dec_up2"], kernel_size=2
        )
        self.attn2 = RCAB(self.channels["dec_up2"], self.channels["dec_up1"],
                          kernel_size=3, reduction=16, norm='BN', act=lReLU, downscale=True)

        self.up1 = ConvTransposeBnRelu(self.channels["dec_up1"], self.channels["dec_up1"], kernel_size=2)
        self.attn1 = RCAB(self.channels["dec_up1"], self.channels["dec_up0"],
                          kernel_size=3, reduction=16, norm='BN', act=lReLU, downscale=True)

        self.up0 = ConvTransposeBnRelu(self.channels["dec_up0"], self.channels["dec_final"], kernel_size=2)
        self.attn0 = RCAB(self.channels["dec_final"], self.channels["dec_final"],
                          kernel_size=3, reduction=16, norm='BN', act=lReLU, downscale=True)

        self.final = nn.Sequential(
            ConvBnRelu(self.channels["dec_final"], self.channels["dec_final"], kernel_size=3, padding=1),
            ConvBnRelu(self.channels["dec_final"], self.channels["dec_final"], kernel_size=5, padding=2),
            nn.Conv2d(self.channels["dec_final"], 3,
                      kernel_size=final_conv_kernel,
                      padding=(final_conv_kernel - 1) // 2,
                      bias=False)
        )

        initialize_weights(self.conv_x2, self.conv_x3, self.conv_x7)
        initialize_weights(self.up0, self.up1, self.up2, self.up4, self.up8)
        initialize_weights(self.attn0, self.attn1, self.attn2, self.attn4, self.attn8)
        initialize_weights(self.final)

    def forward(self, x):
        """x: list of T frame tensors, each [B, 3, H, W]"""
        x2_features, x3_features, x7_features = [], [], []
        for frame in x:
            x2_f, x3_f, x7_f = self.backbone(frame)
            x2_features.append(x2_f)
            x3_features.append(x3_f)
            x7_features.append(x7_f)

        x2_concat = torch.cat(x2_features, dim=1)
        x3_concat = torch.cat(x3_features, dim=1)
        x7_concat = torch.cat(x7_features, dim=1)

        x7s = self.conv_x7(x7_concat)
        x3s = self.conv_x3(x3_concat)
        x2s = self.conv_x2(x2_concat)

        x7s = x7s + self.tsm_rand(x7s)

        x = self.up8(x7s)
        x = self.attn8(x)

        x = self.up4(torch.cat([x3s, x], dim=1))
        x = self.attn4(x)

        x = self.up2(torch.cat([x2s, x], dim=1))
        x = self.attn2(x)

        x = self.up1(x)
        x = self.attn1(x)

        x = self.up0(x)
        x = self.attn0(x)

        return self.final(x)
