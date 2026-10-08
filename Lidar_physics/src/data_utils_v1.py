#!/usr/bin/env python
"""
数据加载、配置文件、IO工具

纯数据与配置层，不包含训练算法。
"""

import os
import json
import struct
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime


############################################
# 1. 配置管理
############################################
class Config:
    """统一配置管理"""
    def __init__(self):
        # ====== 数据 ======
        self.data_dir = "/data/jyx/Lidar/scene1"
        self.voxel_size = 64
        self.voxel_range = [-50, 50, -50, 50, -10, 10]
        self.input_frames = 4      # 输入连续帧数
        self.pred_frames = 1       # 预测帧数
        self.num_workers = 4
        self.cache_size = 20                  # 数据集缓存
        
        # ====== 模型 ======
        self.encoder_channels = [32, 64, 128, 256]
        self.decoder_channels = [256, 128, 64, 32]
        
        # ====== 训练 ======
        self.batch_size = 4
        self.epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.save_interval = 5
        
        # ====== 损失权重 ======
        self.weight_rec = 1.0
        self.weight_smooth = 0.01
        self.weight_continuous = 0.1
        self.weight_anomaly = 0.01
        
        # ====== 日志 ======
        self.log_dir = "./logs"
        self.checkpoint_dir = "./checkpoints"
        self.viz_dir = "./viz_results"
        
        # ====== DDP ======
        self.master_addr = "localhost"
        self.master_port = "12235"
        self.backend = "nccl"


############################################
# 2. 点云文件读取
############################################
def read_pcd_file(filepath):
    """
    读取PCD格式点云文件
    
    Returns:
        points: (N, 3) numpy数组
    """
    with open(filepath, 'rb') as f:
        # 读取头部
        header = ''
        while True:
            line = f.readline().decode('ascii')
            header += line
            if line.startswith('DATA'):
                break
        
        # 解析点数
        points_count = 0
        for line in header.split('\n'):
            if line.startswith('POINTS'):
                points_count = int(line.split()[1])
            elif line.startswith('FIELDS'):
                fields = line.split()[1:]
        
        if points_count == 0:
            raise ValueError(f"无法从PCD头解析点数: {filepath}")
        
        data_type = header.split('DATA ')[1].strip()
        
        if data_type == 'ascii':
            data = f.read().decode('ascii').strip().split('\n')
            points = []
            for line in data:
                if line.strip():
                    vals = [float(v) for v in line.split()]
                    points.append(vals[:3])
            return np.array(points, dtype=np.float32)
        
        elif data_type == 'binary':
            import struct
            fmt = 'f' * 4  # 假设 x, y, z, intensity
            point_size = struct.calcsize(fmt)
            points = []
            for _ in range(points_count):
                raw = f.read(point_size)
                if len(raw) < point_size:
                    break
                vals = struct.unpack(fmt, raw)
                points.append(vals[:3])
            return np.array(points, dtype=np.float32)
        
        else:
            raise ValueError(f"不支持的PCD数据类型: {data_type}")


def read_bin_file(filepath):
    """读取KITTI格式bin文件"""
    return np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_point_cloud(filepath):
    """
    通用点云加载
    
    Args:
        filepath: 点云文件路径 (.pcd, .bin, .npy, .txt)
    
    Returns:
        points: (N, 3) numpy数组
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.pcd':
        return read_pcd_file(filepath)
    elif ext == '.bin':
        return read_bin_file(filepath)
    elif ext == '.npy':
        data = np.load(filepath)
        if data.ndim == 2 and data.shape[1] >= 3:
            return data[:, :3]
        return data
    elif ext == '.txt':
        return np.loadtxt(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


############################################
# 3. 体素化
############################################
def voxelize_dense(points, voxel_size=64, voxel_range=None):
    """
    将点云转换为稠密体素网格
    
    Args:
        points: (N, 3) numpy数组
        voxel_size: 体素分辨率
        voxel_range: [xmin, xmax, ymin, ymax, zmin, zmax]
    
    Returns:
        voxel: [voxel_size, voxel_size, voxel_size] 体素网格
    """
    if voxel_range is None:
        voxel_range = [-50, 50, -50, 50, -10, 10]
    
    xmin, xmax, ymin, ymax, zmin, zmax = voxel_range
    
    # 过滤超出范围的点
    mask = (
        (points[:, 0] >= xmin) & (points[:, 0] < xmax) &
        (points[:, 1] >= ymin) & (points[:, 1] < ymax) &
        (points[:, 2] >= zmin) & (points[:, 2] < zmax)
    )
    points = points[mask]
    
    # 量化
    xs = ((points[:, 0] - xmin) / (xmax - xmin) * voxel_size).astype(np.int32)
    ys = ((points[:, 1] - ymin) / (ymax - ymin) * voxel_size).astype(np.int32)
    zs = ((points[:, 2] - zmin) / (zmax - zmin) * voxel_size).astype(np.int32)
    
    # 裁剪
    xs = np.clip(xs, 0, voxel_size - 1)
    ys = np.clip(ys, 0, voxel_size - 1)
    zs = np.clip(zs, 0, voxel_size - 1)
    
    # 构建体素（用归一化密度代替二进制占用）
    voxel = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
    np.add.at(voxel, (xs, ys, zs), 1.0)
    
    # 截断密度
    max_density = 10.0
    voxel = np.clip(voxel, 0, max_density) / max_density
    
    return voxel


def voxel_to_points(voxel, voxel_range=None, threshold=0.1):
    """
    体素反转为点云
    
    Args:
        voxel: [D, H, W] 体素
        voxel_range: 空间范围
        threshold: 密度阈值
    
    Returns:
        points: (M, 3) 点云
    """
    if voxel_range is None:
        voxel_range = [-50, 50, -50, 50, -10, 10]
    
    xmin, xmax, ymin, ymax, zmin, zmax = voxel_range
    D, H, W = voxel.shape
    
    mask = voxel > threshold
    idxs = np.argwhere(mask)
    
    xs = idxs[:, 0].astype(np.float32) / D * (xmax - xmin) + xmin + (xmax - xmin) / (2 * D)
    ys = idxs[:, 1].astype(np.float32) / H * (ymax - ymin) + ymin + (ymax - ymin) / (2 * H)
    zs = idxs[:, 2].astype(np.float32) / W * (zmax - zmin) + zmin + (zmax - zmin) / (2 * W)
    
    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


############################################
# 4. 保存PCD文件
############################################
def save_pcd_file(points, filepath):
    """将点云保存为PCD格式"""
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    
    with open(filepath, 'w') as f:
        f.write('# .PCD v0.7 - Point Cloud Data file format\n')
        f.write('VERSION 0.7\n')
        f.write('FIELDS x y z\n')
        f.write('SIZE 4 4 4\n')
        f.write('TYPE F F F\n')
        f.write('COUNT 1 1 1\n')
        f.write(f'WIDTH {n}\n')
        f.write('HEIGHT 1\n')
        f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        f.write(f'POINTS {n}\n')
        f.write('DATA ascii\n')
        for p in points:
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')


############################################
# 5. 保存体素可视化数据
############################################
def save_voxel_for_visualization(voxel, output_prefix):
    """
    将体素保存为可视化格式
    
    Args:
        voxel: [D, H, W]
        output_prefix: 输出前缀
    """
    # 保存为PCD
    points = voxel_to_points(voxel)
    save_pcd_file(points, f"{output_prefix}.pcd")
    
    # 保存为npy
    np.save(f"{output_prefix}.npy", voxel)


def compare_voxels_for_visualization(original_voxel, predicted_voxel, output_dir):
    """
    保存原始和预测体素的对比数据
    
    Args:
        original_voxel: [D, H, W]
        predicted_voxel: [D, H, W]
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 误差图
    error = np.abs(original_voxel - predicted_voxel)
    np.save(os.path.join(output_dir, "comparison_error.npy"), error)
    
    # 统计
    stats = {
        'original_mean': float(np.mean(original_voxel)),
        'original_std': float(np.std(original_voxel)),
        'predicted_mean': float(np.mean(predicted_voxel)),
        'predicted_std': float(np.std(predicted_voxel)),
        'mse': float(np.mean((original_voxel - predicted_voxel) ** 2)),
        'mae': float(np.mean(np.abs(original_voxel - predicted_voxel))),
        'max_error': float(np.max(np.abs(original_voxel - predicted_voxel))),
    }
    
    with open(os.path.join(output_dir, "comparison_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)


############################################
# 6. 体素数据集
############################################
class VoxelDataset(Dataset):
    """
    连续帧体素数据集

    输入:
        过去4帧

    输出:
        下一帧
    """

    def __init__(self, file_list, config, mode='train', cache_size=0):

        self.files = sorted(file_list)
        self.config = config
        self.mode = mode

        self.input_frames = config.input_frames
        self.pred_frames = config.pred_frames

        self.cache = {}
        self.cache_size = cache_size

        required = self.input_frames + self.pred_frames

        if len(self.files) < required:
            raise ValueError(
                f"文件数量不足，需要至少 {required} 帧"
            )

    def __len__(self):

        return len(self.files) - self.input_frames - self.pred_frames + 1

    def _load_voxel(self, filepath):

        if filepath in self.cache:
            return self.cache[filepath]

        points = load_point_cloud(filepath)

        voxel = voxelize_dense(
            points,
            self.config.voxel_size,
            self.config.voxel_range
        )

        voxel = torch.from_numpy(voxel).float()

        if len(self.cache) < self.cache_size:
            self.cache[filepath] = voxel

        return voxel

    def __getitem__(self, idx):

        ########################################
        # input sequence
        ########################################
        input_seq = []

        for i in range(self.input_frames):
            voxel = self._load_voxel(
                self.files[idx + i]
            )

            input_seq.append(voxel)

        input_seq = torch.stack(input_seq, dim=0)   # [4,D,H,W]

        ########################################
        # target
        ########################################
        target_idx = idx + self.input_frames

        target = self._load_voxel(
            self.files[target_idx]
        ).unsqueeze(0)                              # [1,D,H,W]

        return input_seq, target


############################################
# 7. 点云诊断
############################################
class PointCloudDiagnostics:
    """
    点云诊断工具
    检查点云质量、统计信息、分布
    """
    
    @staticmethod
    def analyze_pointcloud(points):
        """分析单个点云"""
        if len(points) == 0:
            return {'error': '空点云'}
        
        stats = {
            'num_points': len(points),
            'mean': points.mean(axis=0).tolist(),
            'std': points.std(axis=0).tolist(),
            'min': points.min(axis=0).tolist(),
            'max': points.max(axis=0).tolist(),
            'range': (points.max(axis=0) - points.min(axis=0)).tolist(),
        }
        
        # 密度分析
        if len(points) > 100:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(10, len(points)))
            nn.fit(points)
            dists, _ = nn.kneighbors(points)
            stats['mean_neighbor_dist'] = float(dists[:, 1:].mean())
        
        return stats
    
    @staticmethod
    def analyze_dataset(file_list, max_samples=50):
        """分析数据集"""
        results = []
        
        for i, fpath in enumerate(file_list[:max_samples]):
            try:
                points = load_point_cloud(fpath)
                stats = PointCloudDiagnostics.analyze_pointcloud(points)
                stats['file'] = fpath
                results.append(stats)
            except Exception as e:
                results.append({'file': fpath, 'error': str(e)})
        
        # 汇总
        num_points = [r.get('num_points', 0) for r in results if 'num_points' in r]
        summary = {
            'total_files': len(file_list),
            'analyzed_files': len(results),
            'avg_points': float(np.mean(num_points)) if num_points else 0,
            'min_points': int(min(num_points)) if num_points else 0,
            'max_points': int(max(num_points)) if num_points else 0,
        }
        
        return summary, results