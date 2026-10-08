#!/usr/bin/env python
"""
从体素重建点云
将体素格式转换回点云格式
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from legacy.physics_DDP import Config, load_point_cloud, voxelize_dense, LidarAnomalyDetector
from torch.utils.data import DataLoader, Dataset

def save_pcd_file(filepath, points, colors=None):
    """保存点云为 PCD 格式"""
    n_points = len(points)
    
    with open(filepath, 'w') as f:
        f.write("# .PCD v.7 - Point Cloud Data file format\n")
        f.write("VERSION .7\n")
        if colors is not None:
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE f f f u\n")
            f.write("COUNT 1 1 1 1\n")
        else:
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE f f f\n")
            f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {n_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n_points}\n")
        f.write("DATA ascii\n")
        
        for i, point in enumerate(points):
            if colors is not None:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(colors[i])}\n")
            else:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def voxel_to_pointcloud(voxel, voxel_size=0.2, grid_origin=None, threshold=0.1):
    """
    将体素转换为点云
    
    Args:
        voxel: [D, H, W] 体素数据
        voxel_size: 体素大小
        grid_origin: 网格原点 (x, y, z)
        threshold: 强度阈值，只保留 > threshold 的体素
    
    Returns:
        points: [N, 3] 点云坐标
        intensities: [N] 体素强度值
    """
    # 找到非空体素
    mask = voxel > threshold
    indices = np.where(mask)  # (D_indices, H_indices, W_indices)
    
    if len(indices[0]) == 0:
        print("No non-empty voxels found")
        return np.zeros((0, 3)), np.zeros(0)
    
    # 体素索引 → 3D坐标
    # 注意：需要反向映射到原始坐标系统
    voxel_coords = np.column_stack(indices)  # [N, 3]
    
    # 体素中心坐标
    points = voxel_coords * voxel_size
    
    # 移动到中心
    center = points.mean(axis=0)
    points = points - center
    
    # 获取强度值
    intensities = voxel[mask]
    
    return points, intensities


def reconstruct_pointcloud_from_voxel(original_pcd_path, checkpoint_path, output_dir, config=None):
    """
    从单个点云文件重建点云
    
    Args:
        original_pcd_path: 原始点云路径
        checkpoint_path: 模型checkpoint路径
        output_dir: 输出目录
        config: 配置对象
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if config is None:
        config = Config()
    
    print(f"\n{'='*70}")
    print(f"Processing: {Path(original_pcd_path).name}")
    print(f"{'='*70}")
    
    # 1. 加载原始点云
    print("\n[1/4] Loading original point cloud...")
    original_pts = load_point_cloud(original_pcd_path, verbose=False)
    print(f"  Original points: {len(original_pts)}")
    
    # 2. 体素化
    print("\n[2/4] Voxelizing...")
    voxel_original = voxelize_dense(original_pts, voxel_size=config.voxel_size, grid_size=config.grid_size)
    print(f"  Voxel shape: {voxel_original.shape}")
    print(f"  Non-zero voxels: {(voxel_original > 0).sum()}")
    
    # 3. 模型预测
    print("\n[3/4] Predicting with model...")
    detector = LidarAnomalyDetector(config)
    detector.load_checkpoint(checkpoint_path)
    
    voxel_tensor = torch.from_numpy(voxel_original).float().unsqueeze(0).unsqueeze(0)
    score, voxel_reconstructed = detector.detect_anomaly(voxel_tensor)
    voxel_predicted = voxel_reconstructed.squeeze(0).squeeze(0).numpy()
    
    print(f"  Anomaly score: {score:.4f}")
    print(f"  Original voxel range: [{voxel_original.min():.4f}, {voxel_original.max():.4f}]")
    print(f"  Predicted voxel range: [{voxel_predicted.min():.4f}, {voxel_predicted.max():.4f}]")
    
    # 4. 转换回点云
    print("\n[4/4] Converting voxels to point clouds...")
    pts_original, int_original = voxel_to_pointcloud(
        voxel_original, 
        voxel_size=config.voxel_size, 
        threshold=0.1
    )
    pts_predicted, int_predicted = voxel_to_pointcloud(
        voxel_predicted, 
        voxel_size=config.voxel_size, 
        threshold=0.1
    )
    
    print(f"  Original reconstructed points: {len(pts_original)}")
    print(f"  Predicted reconstructed points: {len(pts_predicted)}")
    
    # 5. 保存结果
    base_name = Path(original_pcd_path).stem
    
    # 保存原始点云（从体素重建）
    original_out = os.path.join(output_dir, f"{base_name}_original_from_voxel.pcd")
    save_pcd_file(original_out, pts_original)
    print(f"\n✓ Saved original voxel cloud: {original_out}")
    
    # 保存预测点云
    predicted_out = os.path.join(output_dir, f"{base_name}_predicted.pcd")
    save_pcd_file(predicted_out, pts_predicted)
    print(f"✓ Saved predicted cloud: {predicted_out}")
    
    # 保存统计信息
    stats = {
        'filename': Path(original_pcd_path).name,
        'original_points': len(original_pts),
        'original_voxel_nonzero': int((voxel_original > 0).sum()),
        'predicted_voxel_nonzero': int((voxel_predicted > 0).sum()),
        'voxel_size': config.voxel_size,
        'grid_size': config.grid_size,
        'anomaly_score': float(score),
        'voxel_error_mean': float(np.abs(voxel_original - voxel_predicted).mean()),
        'voxel_error_max': float(np.abs(voxel_original - voxel_predicted).max()),
    }
    
    import json
    stats_file = os.path.join(output_dir, f"{base_name}_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics: {stats_file}")
    
    return stats


def main():
    print("\n" + "="*70)
    print("Point Cloud Reconstruction from Voxels")
    print("="*70)
    
    config = Config()
    config.data_dir = "/data/jyx/Lidar/scene2"
    config.grid_size = 64
    
    # 获取少量文件进行演示
    scene1_dir = Path("/data/jyx/Lidar/scene2")
    pcd_files = sorted(list(scene1_dir.glob("*.pcd")))[:3]  # 只处理前3个
    
    checkpoint_path = "./checkpoints/epoch_045_loss_0.0273.pth"
    output_dir = "./reconstructed_pointclouds_s2_v1"
    
    print(f"\nProcessing {len(pcd_files)} point cloud files...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    
    all_stats = []
    for pcd_file in pcd_files:
        try:
            stats = reconstruct_pointcloud_from_voxel(
                str(pcd_file), 
                checkpoint_path, 
                output_dir, 
                config
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"\n❌ Error processing {pcd_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印汇总
    if all_stats:
        print(f"\n{'='*70}")
        print("Summary Statistics")
        print(f"{'='*70}")
        for stats in all_stats:
            print(f"\n{stats['filename']}:")
            print(f"  Original points: {stats['original_points']}")
            print(f"  Voxel nonzero: {stats['original_voxel_nonzero']}")
            print(f"  Anomaly score: {stats['anomaly_score']:.4f}")
            print(f"  Voxel error (mean): {stats['voxel_error_mean']:.6f}")
            print(f"  Voxel error (max): {stats['voxel_error_max']:.4f}")
    
    print(f"\n✅ Reconstruction complete!")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - *_original_from_voxel.pcd: Original point cloud (reconstructed from voxel)")
    print(f"  - *_predicted.pcd: Predicted point cloud (from model reconstruction)")
    print(f"  - *_stats.json: Statistics for each file")


if __name__ == "__main__":
    main()
