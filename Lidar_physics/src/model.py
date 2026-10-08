#!/usr/bin/env python
"""
网络模型定义 + 物理损失函数 + 异常检测器

纯算法层，不涉及数据加载或训练配置。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data_utils import (
    save_voxel_for_visualization,
    save_pcd_file,
    compare_voxels_for_visualization,
    PointCloudDiagnostics,
)


############################################
# 1. 残差3D块
############################################
class ResidualBlock3D(nn.Module):
    """3D残差块"""
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
        out += residual
        return self.relu(out)


############################################
# 2. FlowNet3D (物理流网络)
############################################
class FlowNet3D(nn.Module):
    """
    基于物理流的3D异常检测模型
    
    核心思路：
    - 编码器压缩体素 → 低维物理流场
    - 解码器从流场重建体素
    - 重建误差 = 异常指标
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv3d(config.input_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResidualBlock3D(32, 64),
            nn.MaxPool3d(2),              # 64→32
            ResidualBlock3D(64, 128),
            nn.MaxPool3d(2),              # 32→16
            ResidualBlock3D(128, 256),
            nn.MaxPool3d(2),              # 16→8
            ResidualBlock3D(256, 256),
        )
        
        # 流场预测头
        self.flow_head = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 3, 3, padding=1),  # 3通道流场 (vx, vy, vz)
        )
        
        # 解码器（上采样 + 重建）
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, stride=2),   # 8→16
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            ResidualBlock3D(128, 128),
            
            nn.ConvTranspose3d(128, 64, 2, stride=2),    # 16→32
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64, 64),
            
            nn.ConvTranspose3d(64, 32, 2, stride=2),     # 32→64
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResidualBlock3D(32, 32),
            
            nn.Conv3d(32, config.input_channels, 3, padding=1),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, D, H, W] 输入体素
        
        Returns:
            pred: [B, 1, D, H, W] 重建体素
            flow: [B, 3, D/8, H/8, W/8] 物理流场
        """
        # 编码
        features = self.encoder(x)
        
        # 流场
        flow = self.flow_head(features)
        
        # 解码重建
        pred = self.decoder(features)
        
        return pred, flow


############################################
# 3. 散度计算
############################################
def divergence(flow):
    """
    计算流场的散度（用于质量守恒约束）
    
    Args:
        flow: [B, 3, D, H, W] 流场
    
    Returns:
        div: [B, 1, D, H, W] 散度场
    """
    # 计算各方向的空间梯度
    dvx_dx = torch.gradient(flow[:, 0:1], dim=2)[0] if flow.shape[2] > 1 else torch.zeros_like(flow[:, 0:1])
    dvy_dy = torch.gradient(flow[:, 1:2], dim=3)[0] if flow.shape[3] > 1 else torch.zeros_like(flow[:, 0:1])
    dvz_dz = torch.gradient(flow[:, 2:3], dim=4)[0] if flow.shape[4] > 1 else torch.zeros_like(flow[:, 0:1])
    
    return dvx_dx + dvy_dy + dvz_dz


############################################
# 4. 损失函数
############################################
def compute_loss(pred, gt, prev, flow, config, verbose=False):
    """
    异常检测损失函数
    - 重建损失：学习正常模式的分布
    - 流平滑：保持物理合理性
    - 连续性：质量守恒约束
    - 异常正则化：紧凑正常模式分布
    """
    eps = 1e-8
    
    # 1. 重建损失（主要损失：学习正常帧的分布）
    rec = F.mse_loss(pred, gt) * config.weight_rec
    
    # 2. 流平滑损失（物理约束：流场不应有剧烈变化）
    flow_smooth_x = torch.mean((flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]) ** 2)
    flow_smooth_y = torch.mean((flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]) ** 2)
    flow_smooth_z = torch.mean((flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]) ** 2)
    flow_smooth = (flow_smooth_x + flow_smooth_y + flow_smooth_z) * config.weight_smooth
    
    # 3. 连续性约束（物理：质量守恒）
    dt = pred - prev
    div = divergence(flow)
    cont = F.mse_loss(-div, dt) * config.weight_continuous
    
    # 4. 异常正则化（鼓励学习紧凑的正常模式分布）
    pred_std = torch.std(pred)
    anomaly_reg = config.weight_anomaly / (pred_std + eps)  # 标准差越小，约束越大
    
    total = rec + flow_smooth + cont + anomaly_reg
    
    return total, {
        'total': total.item(),
        'rec': rec.item(),
        'smooth': flow_smooth.item(),
        'cont': cont.item(),
        'anomaly_reg': anomaly_reg.item(),
        'pred_std': pred_std.item(),
        'flow_norm': torch.norm(flow).item(),
    }


############################################
# 5. 激光雷达异常检测器
############################################
class LidarAnomalyDetector:
    """
    激光雷达异常检测器
    封装模型推理、异常评分、结果可视化
    """
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = FlowNet3D(config).to(self.device)
    
    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 处理 DDP wrapper 的 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        try:
            self.model.load_state_dict(new_state_dict)
        except RuntimeError as e:
            print(f"⚠️  权重加载部分失败: {e}")
            self.model.load_state_dict(new_state_dict, strict=False)
        
        self.model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('best_loss', 'unknown')
        print(f"✓ 模型加载成功 (epoch: {epoch}, loss: {loss})")
    
    def detect_anomaly(self, voxel_input):
        """
        检测体素中的异常
        
        Args:
            voxel_input: [1, 1, D, H, W] 或 [D, H, W]
        
        Returns:
            score: 异常分数
            reconstruction: 重建体素 [1, 1, D, H, W]
        """
        # 确保正确的形状
        if voxel_input.dim() == 3:
            voxel_input = voxel_input.unsqueeze(0).unsqueeze(0)
        elif voxel_input.dim() == 4:
            voxel_input = voxel_input.unsqueeze(0)
        
        voxel_input = voxel_input.to(self.device)
        
        with torch.no_grad():
            prediction, flow = self.model(voxel_input)
            
            # 计算异常分数
            _, loss_dict = compute_loss(
                prediction, voxel_input,
                voxel_input,  # 使用自身作为前帧
                flow, self.config
            )
            
            # 综合异常分数
            score = (
                loss_dict['rec'] * 1.0 +
                loss_dict['flow_norm'] * 0.1
            )
        
        return score, prediction
    
    def save_predictions(self, voxel_input, output_dir, sample_id=0):
        """
        保存预测结果用于可视化
        
        Args:
            voxel_input: [1, 1, D, H, W] 或 [D, H, W]
            output_dir: 输出目录
            sample_id: 样本ID
        """
        os.makedirs(output_dir, exist_ok=True)
        
        score, reconstruction = self.detect_anomaly(voxel_input)
        
        # 转为numpy
        original_np = voxel_input.squeeze().cpu().numpy() if torch.is_tensor(voxel_input) else voxel_input
        if original_np.ndim == 3:
            original_np = original_np[np.newaxis, ...]  # 确保4D
        
        predicted_np = reconstruction.squeeze().cpu().numpy()
        if predicted_np.ndim == 3:
            predicted_np = predicted_np[np.newaxis, ...]
        
        # 保存体素数据
        base_name = f"sample_{sample_id:04d}"
        
        np.save(os.path.join(output_dir, f"{base_name}_original.npy"), original_np)
        np.save(os.path.join(output_dir, f"{base_name}_pred.npy"), predicted_np)
        
        # 保存PCD格式
        try:
            # 将体素转换为点云并保存
            from data_utils import voxelize_dense  # 这里只做格式转换用
            save_voxel_for_visualization(original_np[0] if original_np.ndim == 4 else original_np,
                                         os.path.join(output_dir, f"{base_name}_original"))
            save_voxel_for_visualization(predicted_np[0] if predicted_np.ndim == 4 else predicted_np,
                                         os.path.join(output_dir, f"{base_name}_pred"))
        except Exception as e:
            print(f"PCD保存失败: {e}")
        
        # 对比可视化数据
        compare_voxels_for_visualization(
            original_np[0] if original_np.ndim == 4 else original_np,
            predicted_np[0] if predicted_np.ndim == 4 else predicted_np,
            output_dir
        )
        
        print(f"样本 {sample_id} 已保存到 {output_dir}")
        return score
