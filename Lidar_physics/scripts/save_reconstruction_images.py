#!/usr/bin/env python
"""
直接保存体素重建结果为图片
使用 PIL 避免 matplotlib 兼容性问题
"""

import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from legacy.physics_DDP import Config, VoxelDataset, LidarAnomalyDetector
from torch.utils.data import DataLoader

def normalize_to_uint8(arr):
    """将数组归一化到 [0, 255] 范围作为图像"""
    arr = np.asarray(arr, dtype=np.float32)
    arr_min = arr.min()
    arr_max = arr.max()
    
    if arr_max > arr_min:
        arr_normalized = (arr - arr_min) / (arr_max - arr_min) * 255
    else:
        arr_normalized = np.zeros_like(arr)
    
    return arr_normalized.astype(np.uint8)


def create_comparison_image(original, predicted, error, filename, output_dir):
    """
    创建原始vs预测的对比图片（水平排列）
    
    Args:
        original: [D, H, W] 原始体素
        predicted: [D, H, W] 预测体素
        error: [D, H, W] 误差
        filename: 文件名
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    D, H, W = original.shape
    slice_idx = D // 2
    
    # 获取三个方向的切片
    slices = [
        (original[slice_idx], predicted[slice_idx], error[slice_idx]),
        (original[:, H//2, :], predicted[:, H//2, :], error[:, H//2, :]),
        (original[:, :, W//2], predicted[:, :, W//2], error[:, :, W//2]),
    ]
    
    # 创建复合图片
    img_height = 64
    img_width = 64
    gap = 10
    
    # 总宽度: 3列(原始、预测、误差) * 64 + gap * 2
    # 总高度: 3行(XY、XZ、YZ) * 64 + gap * 2
    total_width = img_width * 3 + gap * 4
    total_height = img_height * 3 + gap * 4
    
    composite = Image.new('L', (total_width, total_height), color=128)
    
    y_offset = gap
    
    # 绘制三个方向
    for orig_slice, pred_slice, err_slice in slices:
        # 归一化
        orig_img_data = normalize_to_uint8(orig_slice)
        pred_img_data = normalize_to_uint8(pred_slice)
        err_img_data = normalize_to_uint8(err_slice)
        
        # 创建图片
        orig_img = Image.fromarray(orig_img_data, mode='L').resize((img_width, img_height))
        pred_img = Image.fromarray(pred_img_data, mode='L').resize((img_width, img_height))
        err_img = Image.fromarray(err_img_data, mode='L').resize((img_width, img_height))
        
        # 粘贴到复合图片
        x_offset = gap
        composite.paste(orig_img, (x_offset, y_offset))
        composite.paste(pred_img, (x_offset + img_width + gap, y_offset))
        composite.paste(err_img, (x_offset + img_width * 2 + gap * 2, y_offset))
        
        y_offset += img_height + gap
    
    # 保存
    output_file = os.path.join(output_dir, f'{Path(filename).stem}_comparison.png')
    composite.save(output_file)
    print(f"✓ saved: {output_file}")


def create_heatmap_images(original, predicted, error, filename, output_dir):
    """
    创建热力图图片（彩色版本）
    
    Args:
        original: [D, H, W] 原始体素
        predicted: [D, H, W] 预测体素
        error: [D, H, W] 误差
        filename: 文件名
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    D, H, W = original.shape
    slice_idx = D // 2
    
    # 获取中心切片
    orig_slice = original[slice_idx]
    pred_slice = predicted[slice_idx]
    err_slice = error[slice_idx]
    
    # 转为彩色热力图
    def array_to_colormap(arr):
        """转换数组为彩色图（使用简单的颜色映射）"""
        arr = np.asarray(arr, dtype=np.float32)
        arr_min = arr.min()
        arr_max = arr.max()
        
        if arr_max > arr_min:
            normalized = (arr - arr_min) / (arr_max - arr_min)
        else:
            normalized = np.zeros_like(arr)
        
        # 简单热力图：蓝-青-黄-红
        h, w = normalized.shape
        img_array = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 应用简单的颜色映射
        for i in range(h):
            for j in range(w):
                val = normalized[i, j]
                if val < 0.25:
                    r = 0
                    g = 0
                    b = int(val * 255 / 0.25)
                elif val < 0.5:
                    r = 0
                    g = int((val - 0.25) * 255 / 0.25)
                    b = 255
                elif val < 0.75:
                    r = int((val - 0.5) * 255 / 0.25)
                    g = 255
                    b = int((1 - (val - 0.5) / 0.25) * 255)
                else:
                    r = 255
                    g = int((1 - (val - 0.75) / 0.25) * 255)
                    b = 0
                
                img_array[i, j] = [r, g, b]
        
        return img_array
    
    # 创建热力图
    orig_heatmap = array_to_colormap(orig_slice)
    pred_heatmap = array_to_colormap(pred_slice)
    err_heatmap = array_to_colormap(err_slice)
    
    # 转为PIL图像
    orig_img = Image.fromarray(orig_heatmap).resize((128, 128))
    pred_img = Image.fromarray(pred_heatmap).resize((128, 128))
    err_img = Image.fromarray(err_heatmap).resize((128, 128))
    
    # 水平拼接
    composite = Image.new('RGB', (128*3 + 20, 128), color=(255, 255, 255))
    composite.paste(orig_img, (10, 0))
    composite.paste(pred_img, (138, 0))
    composite.paste(err_img, (266, 0))
    
    # 保存
    output_file = os.path.join(output_dir, f'{Path(filename).stem}_heatmap.png')
    composite.save(output_file)
    print(f"✓ saved: {output_file}")


def create_error_distribution_image(error, filename, output_dir):
    """
    创建误差分布直方图
    
    Args:
        error: [D, H, W] 误差数组
        filename: 文件名
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    error_flat = error.flatten()
    
    # 创建直方图
    hist, bin_edges = np.histogram(error_flat, bins=50)
    
    # 创建图片
    img_width = 400
    img_height = 300
    hist_img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(hist_img)
    
    # 绘制坐标轴
    margin = 50
    plot_width = img_width - margin * 2
    plot_height = img_height - margin * 2
    
    # X轴、Y轴
    draw.line([(margin, img_height - margin), (img_width - margin, img_height - margin)], fill='black', width=2)
    draw.line([(margin, margin), (margin, img_height - margin)], fill='black', width=2)
    
    # 归一化直方图
    hist_max = hist.max()
    if hist_max > 0:
        hist_norm = hist / hist_max
    else:
        hist_norm = hist
    
    # 绘制柱状图
    bar_width = plot_width / len(hist)
    for i, (h, count) in enumerate(zip(bin_edges[:-1], hist_norm)):
        x1 = margin + i * bar_width
        x2 = x1 + bar_width - 2
        y1 = img_height - margin
        y2 = img_height - margin - count * plot_height
        
        # 根据高度选择颜色
        if count < 0.3:
            color = 'green'
        elif count < 0.6:
            color = 'yellow'
        else:
            color = 'red'
        
        draw.rectangle([x1, y2, x2, y1], fill=color, outline='black')
    
    # 保存
    output_file = os.path.join(output_dir, f'{Path(filename).stem}_error_dist.png')
    hist_img.save(output_file)
    print(f"✓ saved: {output_file}")


def main():
    print("\n" + "="*70)
    print("Saving voxel reconstruction images")
    print("="*70)
    
    # 配置
    config = Config()
    config.data_dir = "/data/jyx/Lidar/scene1"
    config.grid_size = 64
    
    # 获取文件
    scene1_dir = Path("/data/jyx/Lidar/scene1")
    pcd_files = sorted(list(scene1_dir.glob("*.pcd")))[:10]
    
    print(f"\nTotal samples: {len(pcd_files)}")
    
    # 创建数据集
    test_dataset = VoxelDataset(pcd_files, config, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 加载模型
    print("\nLoading model...")
    detector = LidarAnomalyDetector(config)
    detector.load_checkpoint("./checkpoints/epoch_030_loss_0.3828.pth")
    
    # 输出目录
    img_dir = "./reconstruction_images"
    os.makedirs(img_dir, exist_ok=True)
    
    # 处理每个样本
    print("\nProcessing samples...")
    for i, voxel in enumerate(test_loader):
        try:
            # 获取预测
            score, reconstruction = detector.detect_anomaly(voxel)
            
            # 转为numpy
            original_np = voxel.squeeze(0).squeeze(0).numpy()
            predicted_np = reconstruction.squeeze(0).squeeze(0).numpy()
            error = np.abs(original_np - predicted_np)
            
            print(f"[{i+1}/{len(test_loader)}] {pcd_files[i].name}: score={score:.4f}, MSE={error.mean():.6f}, MaxErr={error.max():.4f}")
            
            # 前5个样本保存详细对比图
            if i < 5:
                create_comparison_image(original_np, predicted_np, error, pcd_files[i].name, img_dir)
                create_heatmap_images(original_np, predicted_np, error, pcd_files[i].name, img_dir)
                create_error_distribution_image(error, pcd_files[i].name, img_dir)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Output directory: {img_dir}")
    print(f"Generated files:")
    print(f"  - *_comparison.png: Original vs Predicted comparison (3 slices)")
    print(f"  - *_heatmap.png: Color heatmap")
    print(f"  - *_error_dist.png: Error distribution histogram")


if __name__ == "__main__":
    main()
