#!/usr/bin/env python3
"""
激光点云异常检测 - 预测结果可视化工具
支持体素数据可视化和点云对比
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
import json

# 尝试导入可选依赖
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("⚠️  Open3D未安装，使用matplotlib进行基础可视化")

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False


class VoxelVisualizer:
    """体素数据可视化器"""

    def __init__(self, voxel_size=0.1, grid_size=64):
        self.voxel_size = voxel_size
        self.grid_size = grid_size

    def voxel_to_points(self, voxel_grid, threshold=0.01):
        """
        将体素网格转换为点云

        Args:
            voxel_grid: [D, H, W] 或 [1, D, H, W] 体素数据
            threshold: 体素值阈值，超过此值才转换为点

        Returns:
            points: [N, 3] 点云坐标
            values: [N] 点云值
        """
        if voxel_grid.ndim == 4:
            voxel_grid = voxel_grid.squeeze(0)

        # 获取非零体素的索引
        indices = np.where(voxel_grid > threshold)
        points = np.array(indices).T  # [N, 3]

        # 转换为世界坐标
        points_world = points * self.voxel_size

        # 获取对应的值
        values = voxel_grid[indices]

        return points_world, values

    def create_point_cloud(self, points, values=None, colors=None):
        """创建点云对象"""
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            elif values is not None:
                # 根据值创建颜色映射
                normalized_values = (values - values.min()) / (values.max() - values.min() + 1e-8)
                colors = plt.cm.viridis(normalized_values)[:, :3]
                pcd.colors = o3d.utility.Vector3dVector(colors)

            return pcd
        else:
            return {'points': points, 'values': values, 'colors': colors}

    def visualize_matplotlib(self, points_list, titles=None, values_list=None):
        """使用matplotlib进行3D可视化"""
        n_plots = len(points_list)
        if n_plots == 1:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6),
                                   subplot_kw={'projection': '3d'})

        for i, (points, ax) in enumerate(zip(points_list, axes)):
            if values_list and i < len(values_list):
                values = values_list[i]
                scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                   c=values, cmap='viridis', s=1, alpha=0.6)
                plt.colorbar(scatter, ax=ax, shrink=0.8)
            else:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c='blue', s=1, alpha=0.6)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')

            if titles and i < len(titles):
                ax.set_title(titles[i])

            # 设置等比例缩放
            max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                                points[:, 1].max()-points[:, 1].min(),
                                points[:, 2].max()-points[:, 2].min()]).max() / 2.0

            mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
            mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
            mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        return fig

    def visualize_open3d(self, point_clouds, titles=None):
        """使用Open3D进行交互式可视化"""
        if not HAS_OPEN3D:
            print("❌ Open3D不可用")
            return

        geometries = []
        for i, pcd in enumerate(point_clouds):
            geometries.append(pcd)

            # 添加坐标系
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            geometries.append(coord_frame)

        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="激光点云异常检测可视化")

        for geom in geometries:
            vis.add_geometry(geom)

        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.asarray([0.1, 0.1, 0.1])

        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)

        print("🔍 Open3D可视化窗口已打开")
        print("  - 鼠标左键旋转视角")
        print("  - 鼠标滚轮缩放")
        print("  - 鼠标右键平移")
        print("  - 按 'H' 显示帮助")
        print("  - 关闭窗口继续...")

        vis.run()
        vis.destroy_window()


def load_prediction_data(prediction_file):
    """加载预测数据"""
    if prediction_file.endswith('.npy'):
        data = np.load(prediction_file)
    elif prediction_file.endswith('.pt') or prediction_file.endswith('.pth'):
        data = torch.load(prediction_file, map_location='cpu')
        if isinstance(data, torch.Tensor):
            data = data.numpy()
    elif prediction_file.endswith('.json'):
        with open(prediction_file, 'r') as f:
            data_dict = json.load(f)
        data = np.array(data_dict['prediction'])
    else:
        raise ValueError(f"不支持的文件格式: {prediction_file}")

    return data


def main():
    parser = argparse.ArgumentParser(description='激光点云异常检测 - 预测结果可视化')
    parser.add_argument('prediction_file', help='预测结果文件路径 (.npy, .pt, .pth, .json)')
    parser.add_argument('--original', '-o', help='原始输入文件路径 (用于对比)')
    parser.add_argument('--threshold', '-t', type=float, default=0.01,
                       help='体素值阈值 (default: 0.01)')
    parser.add_argument('--voxel_size', type=float, default=0.1,
                       help='体素尺寸 (default: 0.1)')
    parser.add_argument('--grid_size', type=int, default=64,
                       help='网格尺寸 (default: 64)')
    parser.add_argument('--backend', choices=['matplotlib', 'open3d', 'auto'],
                       default='auto', help='可视化后端')
    parser.add_argument('--save_fig', help='保存matplotlib图像路径')

    args = parser.parse_args()

    # 检查文件存在
    if not os.path.exists(args.prediction_file):
        print(f"❌ 预测文件不存在: {args.prediction_file}")
        return

    if args.original and not os.path.exists(args.original):
        print(f"❌ 原始文件不存在: {args.original}")
        return

    # 初始化可视化器
    visualizer = VoxelVisualizer(args.voxel_size, args.grid_size)

    # 加载预测数据
    print(f"📂 加载预测数据: {args.prediction_file}")
    pred_data = load_prediction_data(args.prediction_file)
    print(f"   数据形状: {pred_data.shape}")

    # 转换为点云
    pred_points, pred_values = visualizer.voxel_to_points(pred_data, args.threshold)
    print(f"   点云点数: {len(pred_points)}")

    point_clouds = []
    titles = []
    values_list = []

    # 预测结果
    point_clouds.append(visualizer.create_point_cloud(pred_points, pred_values))
    titles.append("预测结果")
    values_list.append(pred_values)

    # 如果有原始数据，也加载
    if args.original:
        print(f"📂 加载原始数据: {args.original}")
        orig_data = load_prediction_data(args.original)
        orig_points, orig_values = visualizer.voxel_to_points(orig_data, args.threshold)

        point_clouds.append(visualizer.create_point_cloud(orig_points, orig_values))
        titles.append("原始输入")
        values_list.append(orig_values)

        print(f"   原始点云点数: {len(orig_points)}")

    # 选择可视化后端
    backend = args.backend
    if backend == 'auto':
        backend = 'open3d' if HAS_OPEN3D else 'matplotlib'

    print(f"🎨 使用 {backend} 进行可视化...")

    if backend == 'open3d' and HAS_OPEN3D:
        visualizer.visualize_open3d(point_clouds, titles)
    else:
        fig = visualizer.visualize_matplotlib(
            [pcd['points'] if isinstance(pcd, dict) else np.asarray(pcd.points) for pcd in point_clouds],
            titles,
            values_list
        )

        if args.save_fig:
            fig.savefig(args.save_fig, dpi=300, bbox_inches='tight')
            print(f"💾 图像已保存: {args.save_fig}")
        else:
            plt.show()


if __name__ == "__main__":
    main()