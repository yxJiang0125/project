import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from legacy import physics

# ===========================
# 核心：统一 voxel 计数
# ===========================
def voxelize_count(points, grid_size=64):
    # 中心化
    center = points.mean(axis=0)
    points = points - center

    # 归一化
    max_range = np.max(np.abs(points)) + 1e-6
    points = points / max_range

    # 映射到网格（修正：grid_size-1）
    coords = (points + 1) / 2 * (grid_size - 1)
    coords = coords.astype(np.int32)

    # 过滤越界
    valid = np.all((coords >= 0) & (coords < grid_size), axis=1)
    coords = coords[valid]

    voxel = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    if len(coords) > 0:
        indices = coords[:, 0] * grid_size * grid_size + \
                  coords[:, 1] * grid_size + \
                  coords[:, 2]

        np.add.at(voxel.ravel(), indices, 1)

    return voxel


# ===========================
# 三种 voxel 表达
# ===========================
def build_voxel_variants(points, grid_size=64):
    voxel_count = voxelize_count(points, grid_size)

    # ① occupancy
    voxel_occ = (voxel_count > 0).astype(np.float32)

    # ② raw density（无log）
    voxel_raw = voxel_count / (voxel_count.max() + 1e-6)

    # ③ log density（你的方法）
    voxel_log = np.log(voxel_count + 1)
    voxel_log = voxel_log / (voxel_log.max() + 1e-6)

    return voxel_occ, voxel_raw, voxel_log


# ===========================
# 可视化部分
# ===========================
def bev_projection(voxel, mode='max'):
    if mode == 'max':
        return voxel.max(axis=2)
    elif mode == 'sum':
        return voxel.sum(axis=2)


def visualize_slices(voxel, title="voxel"):
    cx, cy, cz = np.array(voxel.shape) // 2

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(voxel[cx, :, :])
    axs[0].set_title(f"{title} - X")

    axs[1].imshow(voxel[:, cy, :])
    axs[1].set_title(f"{title} - Y")

    axs[2].imshow(voxel[:, :, cz])
    axs[2].set_title(f"{title} - Z")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_bev(voxels_dict):
    n = len(voxels_dict)
    fig, axs = plt.subplots(1, n, figsize=(4*n, 4))

    if n == 1:
        axs = [axs]

    for i, (name, voxel) in enumerate(voxels_dict.items()):
        bev = bev_projection(voxel, mode='max')
        axs[i].imshow(bev)
        axs[i].set_title(name)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_hist(voxels_dict):
    plt.figure(figsize=(6, 4))

    for name, voxel in voxels_dict.items():
        plt.hist(voxel.flatten(), bins=50, alpha=0.5, label=name)

    plt.legend()
    plt.title("Voxel Value Distribution")
    plt.show()


# ===========================
# 主函数
# ===========================
def main(points):

    voxel_occ, voxel_raw, voxel_log = build_voxel_variants(points)

    voxels = {
        "occupancy": voxel_occ,
        "raw_density": voxel_raw,
        "log_density": voxel_log
    }

    # BEV对比（最重要）
    visualize_bev(voxels)

    # 切片（建议看 log）
    visualize_slices(voxel_log, "log_density")


    visualize_hist(voxels)


# ===========================
# 示例
# ===========================
if __name__ == "__main__":
    points = physics.load_point_cloud("/data/jyx/Lidar/scene2/2025_04_04 15-23-34_317_11.pcd")  # 替换成你的点云
    main(points)