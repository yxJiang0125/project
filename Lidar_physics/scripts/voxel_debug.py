import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import struct

############################################
# 1. 点云读取（最终稳定版🔥）
############################################
def load_point_cloud(file_path):
    file_path = str(file_path)

    def read_pcd(filepath):
        with open(filepath, 'rb') as f:
            header = {}

            # ========= 读取 header =========
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('DATA'):
                    data_type = line.split()[1]
                    break

                if line.startswith('#') or len(line) == 0:
                    continue

                key_value = line.split(None, 1)
                if len(key_value) == 2:
                    header[key_value[0]] = key_value[1]

            # ========= 打印 header =========
            print("\n========== PCD HEADER ==========")
            for k, v in header.items():
                print(f"{k}: {v}")
            print("DATA:", data_type)
            print("================================\n")

            points_num = int(header['POINTS'])
            sizes = list(map(int, header['SIZE'].split()))
            point_step = sum(sizes)

            ############################################
            # ASCII
            ############################################
            if data_type.lower() == 'ascii':
                pts = []
                for _ in range(points_num):
                    line = f.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    vals = line.split()
                    try:
                        x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                        pts.append([x, y, z])
                    except:
                        continue
                return np.array(pts, dtype=np.float32)

            ############################################
            # BINARY（关键修复🔥）
            ############################################
            else:
                raw = f.read(points_num * point_step)

                pts = np.zeros((points_num, 3), dtype=np.float32)

                offset = 0
                for i in range(points_num):
                    try:
                        # 只读取 x y z（前12字节）
                        x = struct.unpack('f', raw[offset:offset+4])[0]
                        y = struct.unpack('f', raw[offset+4:offset+8])[0]
                        z = struct.unpack('f', raw[offset+8:offset+12])[0]
                        pts[i] = [x, y, z]
                    except:
                        pts[i] = [0, 0, 0]

                    offset += point_step

                return pts

    ############################################
    # 主逻辑
    ############################################
    if file_path.endswith('.pcd'):
        pts = read_pcd(file_path)

    elif file_path.endswith('.npy'):
        pts = np.load(file_path)

    elif file_path.endswith('.txt'):
        pts = np.loadtxt(file_path)

    elif file_path.endswith('.bin'):
        pts = np.fromfile(file_path, dtype=np.float32)
        if pts.shape[0] % 4 == 0:
            pts = pts.reshape(-1, 4)[:, :3]
        else:
            pts = pts.reshape(-1, 3)

    else:
        raise ValueError(f"不支持格式: {file_path}")

    pts = np.asarray(pts, dtype=np.float32)

    print(f"\n📂 加载文件: {Path(file_path).name}")
    print(f"点数: {len(pts)}")

    return pts


############################################
# 2. 数据诊断
############################################
def diagnose(points):
    print("\n========== 点云诊断 ==========")
    print("总点数:", len(points))

    nan_mask = np.isnan(points).any(axis=1)
    inf_mask = np.isinf(points).any(axis=1)
    zero_mask = (points == 0).all(axis=1)

    print("NaN点:", nan_mask.sum())
    print("Inf点:", inf_mask.sum())
    print("全0点:", zero_mask.sum())

    valid_mask = ~(nan_mask | inf_mask)
    valid_points = points[valid_mask]

    print("有效点:", len(valid_points))

    if len(valid_points) > 0:
        print("min:", valid_points.min(axis=0))
        print("max:", valid_points.max(axis=0))
        print("mean:", valid_points.mean(axis=0))
        print("std:", valid_points.std(axis=0))

    return valid_points


############################################
# 3. 清理
############################################
def clean_points(points, max_range=190):
    print("\n========== 清理点云（详细分析） ==========")

    total = len(points)

    # 1️⃣ NaN / Inf
    nan_mask = np.isnan(points).any(axis=1)
    inf_mask = np.isinf(points).any(axis=1)
    invalid_mask = nan_mask | inf_mask

    print("NaN点:", nan_mask.sum())
    print("Inf点:", inf_mask.sum())

    # 2️⃣ 超范围
    range_mask = np.any(np.abs(points) >= max_range, axis=1)
    print("超范围点:", range_mask.sum())

    # 3️⃣ 全零
    zero_mask = (points == 0).all(axis=1)
    print("全零点:", zero_mask.sum())

    # 合并删除
    remove_mask = invalid_mask | range_mask | zero_mask

    print("\n总删除:", remove_mask.sum())
    print("删除比例: {:.2f}%".format(remove_mask.sum() / total * 100))

    # 👉 打印几个被删的点（关键🔥）
    removed_points = points[remove_mask]
    print("\n示例被删除点（前10个）：")
    print(removed_points[:10])

    # 返回保留点
    return points[~remove_mask]


############################################
# 4. 稳定体素化🔥
############################################
def voxelize(points, grid_size=32):
    print("\n========== 体素化 ==========")

    # 中心化
    center = points.mean(axis=0)
    points = points - center

    # 归一化
    max_range = np.max(np.abs(points)) + 1e-6
    points = points / max_range

    print("归一化范围:", points.min(), points.max())

    coords = (points + 1) / 2 * grid_size
    coords = coords.astype(np.int32)

    valid = np.all((coords >= 0) & (coords < grid_size), axis=1)
    coords = coords[valid]

    print("有效体素点:", len(coords))

    voxel = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    if len(coords) > 0:
        indices = coords[:, 0] * grid_size * grid_size + \
                  coords[:, 1] * grid_size + \
                  coords[:, 2]

        np.add.at(voxel.ravel(), indices, 1)

        # 🔥 log压缩（你原本就有）
        voxel = np.log(voxel + 1)

        # 🔥 归一化到 [0,1]
        voxel = voxel / (voxel.max() + 1e-6)

    ############################################
    # ✅ 非零归一化（关键🔥）
    ############################################
    mask = voxel > 0

    if np.any(mask):
        mean = voxel[mask].mean()
        std = voxel[mask].std() + 1e-6

        voxel[mask] = (voxel[mask] - mean) / std

        print("非零归一化 mean:", mean)
        print("非零归一化 std:", std)

    ############################################
    # 🔥 可选：限制范围（防止训练爆炸）
    ############################################
    voxel = np.clip(voxel, -3, 3)

    ############################################
    # 统计信息
    ############################################
    print("非零体素比例:", (voxel > 0).sum() / voxel.size)
    print("voxel std:", voxel.std())

    return voxel


############################################
# 5. 可视化
############################################
def visualize_hist(points):
    plt.hist(points.flatten(), bins=100)
    plt.title("Point Distribution")
    plt.show()

############################################
# 6. 可视化删除点（3D）
############################################
############################################
# 可视化删除点（保存为图片🔥）
############################################
def visualize_removed_points(original, cleaned, save_path="removed_points.png", max_points=50000):
    print("\n========== 可视化删除点（保存图片） ==========")

    # 找被删除的点
    original_set = set(map(tuple, original))
    cleaned_set = set(map(tuple, cleaned))
    removed = np.array(list(original_set - cleaned_set))

    print("原始点数:", len(original))
    print("保留点数:", len(cleaned))
    print("删除点数:", len(removed))

    # 防止空数组
    if len(removed) == 0:
        removed = np.zeros((0, 3))

    # 降采样（避免图片太大）
    def downsample(pts):
        if len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            return pts[idx]
        return pts

    kept_ds = downsample(cleaned)
    removed_ds = downsample(removed)

    # 创建3D图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 蓝色：保留
    ax.scatter(
        kept_ds[:, 0], kept_ds[:, 1], kept_ds[:, 2],
        s=1, c='blue', label='Kept', alpha=0.5
    )

    # 红色：删除
    if len(removed_ds) > 0:
        ax.scatter(
            removed_ds[:, 0], removed_ds[:, 1], removed_ds[:, 2],
            s=3, c='red', label='Removed', alpha=0.8
        )

    ax.set_title("Point Cloud Cleaning Result")
    ax.legend()

    # 👉 关键：自动缩放坐标轴（否则会挤在一起）
    all_pts = np.vstack([kept_ds, removed_ds]) if len(removed_ds) > 0 else kept_ds
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)

    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    # 👉 关键：保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"图片已保存到: {save_path}")

############################################
# Open3D 可视化删除点
############################################
#import open3d as o3d
'''
def visualize_removed_points_o3d(original, cleaned, max_points=300000):
    print("\n========== Open3D 可视化 ==========")

    # 找删除点
    original_set = set(map(tuple, original))
    cleaned_set = set(map(tuple, cleaned))
    removed = np.array(list(original_set - cleaned_set))

    print("原始点数:", len(original))
    print("保留点数:", len(cleaned))
    print("删除点数:", len(removed))

    # 降采样（避免卡顿）
    def downsample(pts):
        if len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            return pts[idx]
        return pts

    kept_ds = downsample(cleaned)
    removed_ds = downsample(removed)

    # 创建点云对象
    pcd_kept = o3d.geometry.PointCloud()
    pcd_kept.points = o3d.utility.Vector3dVector(kept_ds)

    pcd_removed = o3d.geometry.PointCloud()
    if len(removed_ds) > 0:
        pcd_removed.points = o3d.utility.Vector3dVector(removed_ds)

    # 设置颜色
    pcd_kept.paint_uniform_color([0, 0, 1])   # 蓝色：保留
    if len(removed_ds) > 0:
        pcd_removed.paint_uniform_color([1, 0, 0])  # 红色：删除

    # 显示
    o3d.visualization.draw_geometries(
        [pcd_kept, pcd_removed],
        window_name="Point Cloud Cleaning (Blue=Kept, Red=Removed)",
        width=1000,
        height=800
    )
'''
############################################
# 6. 主函数
############################################
def main(file_path):
    pts = load_point_cloud(file_path)

    print("\n坐标范围：")
    print("min:", pts.min(axis=0))
    print("max:", pts.max(axis=0))

    pts = diagnose(pts)

    original_pts = pts.copy()

    pts = clean_points(pts)

    visualize_hist(pts)

    visualize_removed_points(original_pts, pts)

    voxel = voxelize(pts)

    print("\nvoxel std:", voxel.std())

    np.save("voxel_debug.npy", voxel)
    print("已保存 voxel_debug.npy")


############################################
# 7. 入口
############################################
if __name__ == "__main__":
    file_path = "/data/jyx/Lidar/scene2/2025_04_04 15-23-33_418_2.pcd"
    main(file_path)