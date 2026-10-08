#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FlowNet3D v2
连续4帧输入版本
输入过去4帧，预测下一帧
保留 flow + physics loss
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data_utils_v1 import (
    save_voxel_for_visualization,
    compare_voxels_for_visualization,
)


############################################
# 1. ResidualBlock3D
############################################
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + residual

        return self.relu(out)


############################################
# 2. FlowNet3D
############################################
class FlowNet3D(nn.Module):
    """
    输入:
        past 4 frames

    输出:
        next frame prediction
        flow field
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        in_frames = config.input_frames  # 4

        ########################################
        # encoder
        ########################################
        self.encoder = nn.Sequential(
            nn.Conv3d(in_frames, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            ResidualBlock3D(32, 64),
            nn.MaxPool3d(2),

            ResidualBlock3D(64, 128),
            nn.MaxPool3d(2),

            ResidualBlock3D(128, 256),
            nn.MaxPool3d(2),

            ResidualBlock3D(256, 256),
        )

        ########################################
        # flow head
        ########################################
        self.flow_head = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 3, 3, padding=1),
        )

        ########################################
        # decoder
        ########################################
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            ResidualBlock3D(128, 128),

            nn.ConvTranspose3d(128, 64, 2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64, 64),

            nn.ConvTranspose3d(64, 32, 2, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResidualBlock3D(32, 32),

            # 输出仅下一帧
            nn.Conv3d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        """
        x:
            [B,4,D,H,W]

        return:
            pred:
                [B,1,D,H,W]

            flow:
                [B,3,d,h,w]
        """

        feat = self.encoder(x)

        flow = self.flow_head(feat)

        pred = self.decoder(feat)

        return pred, flow


############################################
# 3. divergence
############################################
def divergence(flow):
    """
    flow:
        [B,3,D,H,W]
    """

    vx = flow[:, 0:1]
    vy = flow[:, 1:2]
    vz = flow[:, 2:3]

    dx = torch.gradient(vx, dim=2)[0]
    dy = torch.gradient(vy, dim=3)[0]
    dz = torch.gradient(vz, dim=4)[0]

    return dx + dy + dz


############################################
# 4. loss
############################################
def compute_loss(pred, gt, prev, flow, config):
    """
    pred:
        预测 t+1

    gt:
        真实 t+1

    prev:
        t
    """

    eps = 1e-8

    ########################################
    # reconstruction
    ########################################
    rec = F.mse_loss(pred, gt) * config.weight_rec

    ########################################
    # flow smooth
    ########################################
    smooth_x = torch.mean((flow[:, :, 1:] - flow[:, :, :-1]) ** 2)
    smooth_y = torch.mean((flow[:, :, :, 1:] - flow[:, :, :, :-1]) ** 2)
    smooth_z = torch.mean((flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]) ** 2)

    smooth = (smooth_x + smooth_y + smooth_z) * config.weight_smooth

    ########################################
    # continuity
    ########################################
    dt = pred - prev
    div = divergence(flow)

    cont = F.mse_loss(-div, dt) * config.weight_continuous

    ########################################
    # compact
    ########################################
    pred_std = torch.std(pred)
    compact = config.weight_anomaly / (pred_std + eps)

    total = rec + smooth + cont + compact

    info = {
        "total": total.item(),
        "rec": rec.item(),
        "smooth": smooth.item(),
        "cont": cont.item(),
        "compact": compact.item(),
        "pred_std": pred_std.item(),
        "flow_norm": torch.norm(flow).item()
    }

    return total, info


############################################
# 5. detector
############################################
class LidarAnomalyDetector:

    def __init__(self, config, device='cuda'):

        self.config = config

        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu'
        )

        self.model = FlowNet3D(config).to(self.device)

    def load_checkpoint(self, path):

        ckpt = torch.load(path, map_location=self.device)

        if 'model' in ckpt:
            state = ckpt['model']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt

        new_state = {}

        for k, v in state.items():
            new_state[k.replace('module.', '')] = v

        self.model.load_state_dict(new_state, strict=False)
        self.model.eval()

        print("✓ checkpoint loaded")

    def detect_anomaly(self, voxel_seq, gt_frame=None):
        """
        voxel_seq:
            [4,D,H,W] or [B,4,D,H,W]

        gt_frame:
            [1,D,H,W] or [B,1,D,H,W]
        """

        if voxel_seq.dim() == 4:
            voxel_seq = voxel_seq.unsqueeze(0)

        voxel_seq = voxel_seq.to(self.device)

        with torch.no_grad():

            pred, flow = self.model(voxel_seq)

            if gt_frame is None:
                gt_frame = voxel_seq[:, -1:].clone()

            if gt_frame.dim() == 4:
                gt_frame = gt_frame.unsqueeze(0)

            gt_frame = gt_frame.to(self.device)

            prev = voxel_seq[:, -1:].clone()

            _, loss_info = compute_loss(
                pred,
                gt_frame,
                prev,
                flow,
                self.config
            )

            score = (
                loss_info["rec"] +
                0.2 * loss_info["cont"] +
                0.05 * loss_info["flow_norm"]
            )

        return score, pred

    def save_predictions(self, voxel_seq, gt_frame, output_dir, sample_id=0):

        os.makedirs(output_dir, exist_ok=True)

        score, pred = self.detect_anomaly(voxel_seq, gt_frame)

        pred_np = pred.squeeze().cpu().numpy()
        gt_np = gt_frame.squeeze().cpu().numpy()

        base = f"sample_{sample_id:04d}"

        np.save(os.path.join(output_dir, f"{base}_pred.npy"), pred_np)
        np.save(os.path.join(output_dir, f"{base}_gt.npy"), gt_np)

        save_voxel_for_visualization(
            pred_np,
            os.path.join(output_dir, f"{base}_pred")
        )

        save_voxel_for_visualization(
            gt_np,
            os.path.join(output_dir, f"{base}_gt")
        )

        compare_voxels_for_visualization(
            gt_np,
            pred_np,
            output_dir
        )

        return score