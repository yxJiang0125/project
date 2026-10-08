#!/usr/bin/env python
"""
可视化体素重建结果 - 原始 vs 预测对比
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from legacy.physics_DDP import Config, VoxelDataset, LidarAnomalyDetector
from torch.utils.data import DataLoader

def visualize_voxel_slices(original, predicted, error, filename, output_dir, slice_idx=None):
    """
    可视化体素的中心切片（X、Y、Z三个方向）
    
    Args:
        original: [D, H, W] 原始体素
        predicted: [D, H, W] 预测体素
        error: [D, H, W] 误差图
        filename: 文件名（用于输出文件名）
        output_dir: 输出目录
        slice_idx: 切片索引（默认取中心）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    D, H, W = original.shape
    if slice_idx is None:
        slice_idx = D // 2
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # 三个方向的切片
    directions = [
        ('XY平面（Z={})'.format(slice_idx), original[slice_idx], predicted[slice_idx], error[slice_idx]),
        ('XZ平面（Y={})'.format(H//2), original[:, H//2, :], predicted[:, H//2, :], error[:, H//2, :]),
        ('YZ平面（X={})'.format(W//2), original[:, :, W//2], predicted[:, :, W//2], error[:, :, W//2]),
    ]
    
    for row, (title, orig_slice, pred_slice, err_slice) in enumerate(directions):
        # 原始体素
        ax1 = fig.add_subplot(gs[row, 0])
        im1 = ax1.imshow(orig_slice, cmap='viridis')
        ax1.set_title(f'{title}\n原始体素')
        ax1.set_xlabel('宽度 (W)')
        ax1.set_ylabel('高度 (H)')
        plt.colorbar(im1, ax=ax1, label='强度')
        
        # 预测体素
        ax2 = fig.add_subplot(gs[row, 1])
        im2 = ax2.imshow(pred_slice, cmap='viridis')
        ax2.set_title(f'预测体素')
        ax2.set_xlabel('宽度 (W)')
        ax2.set_ylabel('高度 (H)')
        plt.colorbar(im2, ax=ax2, label='强度')
        
        # 原始和预测的差异
        ax3 = fig.add_subplot(gs[row, 2])
        im3 = ax3.imshow(np.abs(orig_slice - pred_slice), cmap='hot')
        ax3.set_title(f'绝对误差')
        ax3.set_xlabel('宽度 (W)')
        ax3.set_ylabel('高度 (H)')
        plt.colorbar(im3, ax=ax3, label='误差')
        
        # 误差分布直方图
        ax4 = fig.add_subplot(gs[row, 3])
        err_flat = err_slice.flatten()
        ax4.hist(err_flat, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.set_title(f'误差分布')
        ax4.set_xlabel('误差值')
        ax4.set_ylabel('频数')
        ax4.grid(True, alpha=0.3)
    
    # 总体信息
    fig.suptitle(f'体素重建对比：{filename}\n'
                 f'原始范围: [{original.min():.2f}, {original.max():.2f}] | '
                 f'预测范围: [{predicted.min():.2f}, {predicted.max():.2f}] | '
                 f'平均误差: {error.mean():.4f} | '
                 f'最大误差: {error.max():.4f}',
                 fontsize=12, fontweight='bold')
    
    # 保存
    output_file = os.path.join(output_dir, f'{Path(filename).stem}_slices.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"✓ 已保存: {output_file}")
    plt.close()


def visualize_3d_voxel(original, predicted, filename, output_dir, threshold=0.1):
    """
    3D体素可视化（显示非零体素的分布）
    
    Args:
        original: [D, H, W] 原始体素
        predicted: [D, H, W] 预测体素
        filename: 文件名
        output_dir: 输出目录
        threshold: 只显示强度大于threshold的体素
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取非零体素位置
    orig_mask = original > threshold
    pred_mask = predicted > threshold
    
    orig_coords = np.where(orig_mask)
    pred_coords = np.where(pred_mask)
    
    # 创建3D图
    fig = plt.figure(figsize=(14, 6))
    
    # 原始体素
    ax1 = fig.add_subplot(121, projection='3d')
    if len(orig_coords[0]) > 0:
        scatter1 = ax1.scatter(orig_coords[0], orig_coords[1], orig_coords[2], 
                              c=original[orig_mask], cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter1, ax=ax1, label='强度')
    ax1.set_xlabel('X (D)')
    ax1.set_ylabel('Y (H)')
    ax1.set_zlabel('Z (W)')
    ax1.set_title('原始体素')
    
    # 预测体素
    ax2 = fig.add_subplot(122, projection='3d')
    if len(pred_coords[0]) > 0:
        scatter2 = ax2.scatter(pred_coords[0], pred_coords[1], pred_coords[2],
                              c=predicted[pred_mask], cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter2, ax=ax2, label='强度')
    ax2.set_xlabel('X (D)')
    ax2.set_ylabel('Y (H)')
    ax2.set_zlabel('Z (W)')
    ax2.set_title('预测体素')
    
    fig.suptitle(f'3D体素对比：{filename}\n'
                 f'原始非空体素: {len(orig_coords[0])} | 预测非空体素: {len(pred_coords[0])}',
                 fontsize=12, fontweight='bold')
    
    # 保存
    output_file = os.path.join(output_dir, f'{Path(filename).stem}_3d.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"✓ 已保存: {output_file}")
    plt.close()


def visualize_statistics(results_list, output_dir):
    """
    绘制统计汇总图表
    
    Args:
        results_list: 包含所有结果的列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scores = np.array([r['score'] for r in results_list])
    mses = np.array([r['mse'] for r in results_list])
    max_errors = np.array([r['max_error'] for r in results_list])
    
    fig = plt.figure(figsize=(14, 10))
    
    # 1. 异常分数分布
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(scores, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.axhline(y=scores.mean(), color='red', linestyle='--', label=f'均值: {scores.mean():.4f}')
    ax1.fill_between(range(len(scores)), scores - scores.std(), scores + scores.std(), 
                      alpha=0.2, color='steelblue')
    ax1.set_xlabel('样本索引')
    ax1.set_ylabel('异常分数')
    ax1.set_title('异常分数变化趋势')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. MSE分布
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(mses, 'o-', linewidth=2, markersize=8, color='green')
    ax2.axhline(y=mses.mean(), color='red', linestyle='--', label=f'均值: {mses.mean():.6f}')
    ax2.fill_between(range(len(mses)), mses - mses.std(), mses + mses.std(),
                      alpha=0.2, color='green')
    ax2.set_xlabel('样本索引')
    ax2.set_ylabel('平均误差 (MSE)')
    ax2.set_title('平均误差变化趋势')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. MaxError分布
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(max_errors, 'o-', linewidth=2, markersize=8, color='orange')
    ax3.axhline(y=max_errors.mean(), color='red', linestyle='--', label=f'均值: {max_errors.mean():.2f}')
    ax3.fill_between(range(len(max_errors)), max_errors - max_errors.std(), max_errors + max_errors.std(),
                      alpha=0.2, color='orange')
    ax3.set_xlabel('样本索引')
    ax3.set_ylabel('最大误差 (MaxError)')
    ax3.set_title('最大误差变化趋势')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 异常分数直方图
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(scores, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=scores.mean(), color='red', linestyle='--', linewidth=2, label='均值')
    ax4.set_xlabel('异常分数')
    ax4.set_ylabel('频数')
    ax4.set_title('异常分数分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. MSE与异常分数的关系
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(mses, scores, s=100, alpha=0.6, c=range(len(mses)), cmap='viridis')
    ax5.set_xlabel('平均误差 (MSE)')
    ax5.set_ylabel('异常分数')
    ax5.set_title('MSE vs 异常分数')
    ax5.grid(True, alpha=0.3)
    
    # 6. MaxError与异常分数的关系
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(max_errors, scores, s=100, alpha=0.6, c=range(len(max_errors)), cmap='plasma')
    ax6.set_xlabel('最大误差 (MaxError)')
    ax6.set_ylabel('异常分数')
    ax6.set_title('MaxError vs 异常分数')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_file = os.path.join(output_dir, 'statistics_summary.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"✓ 已保存汇总统计: {output_file}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("🎨 体素重建可视化")
    print("="*70)
    
    # 配置
    config = Config()
    config.data_dir = "/data/jyx/Lidar/scene1"
    config.grid_size = 64
    
    # 获取文件
    scene1_dir = Path("/data/jyx/Lidar/scene1")
    pcd_files = sorted(list(scene1_dir.glob("*.pcd")))[:10]
    
    print(f"\n📂 选中样本数: {len(pcd_files)}")
    
    # 创建数据集
    test_dataset = VoxelDataset(pcd_files, config, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 加载模型
    print("\n🤖 加载模型...")
    detector = LidarAnomalyDetector(config)
    detector.load_checkpoint("./checkpoints/epoch_045_loss_0.0273.pth")
    
    # 输出目录
    vis_dir = "./visualization_results_v2"
    os.makedirs(vis_dir, exist_ok=True)
    
    all_results = []
    
    # 处理每个样本
    print("\n📊 开始处理样本...")
    for i, voxel in enumerate(test_loader):
        try:
            # 获取预测
            score, reconstruction = detector.detect_anomaly(voxel)
            
            # 转为numpy
            original_np = voxel.squeeze(0).squeeze(0).numpy()
            predicted_np = reconstruction.squeeze(0).squeeze(0).numpy()
            error = np.abs(original_np - predicted_np)
            
            result = {
                'index': i,
                'filename': pcd_files[i].name,
                'score': score,
                'mse': error.mean(),
                'max_error': error.max(),
            }
            all_results.append(result)
            
            print(f"\n[{i+1}/{len(test_loader)}] {pcd_files[i].name}")
            print(f"    异常分数: {score:.4f} | 平均误差: {error.mean():.6f}")
            
            # 对前5个样本进行详细可视化
            if i < 5:
                # 切片可视化
                visualize_voxel_slices(original_np, predicted_np, error, 
                                      pcd_files[i].name, vis_dir)
                
                # 3D可视化
                visualize_3d_voxel(original_np, predicted_np, 
                                  pcd_files[i].name, vis_dir)
            
        except Exception as e:
            print(f"❌ 样本 {i} 处理失败: {e}")
            continue
    
    # 绘制统计汇总
    if all_results:
        print(f"\n📈 绘制统计汇总...")
        visualize_statistics(all_results, vis_dir)
    
    print(f"\n✅ 可视化完成！所有图片保存在: {vis_dir}")
    print(f"\n📁 输出文件:")
    print(f"  - *_slices.png: 体素切片对比（前5个样本）")
    print(f"  - *_3d.png: 3D体素对比（前5个样本）")
    print(f"  - statistics_summary.png: 所有样本的统计汇总")


if __name__ == "__main__":
    main()
