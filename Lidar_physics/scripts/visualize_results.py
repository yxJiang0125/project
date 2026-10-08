"""
Visualization of LiDAR Reconstruction Results (Paper-ready Version)

Includes:
  1. BEV + multi-view projections (XY / XZ / YZ)
  2. Voxel slice comparison (GT vs Pred vs Error)
  3. Summary statistics across samples
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# PCD Reader
# ─────────────────────────────────────────────────────────────
def read_pcd(filepath):
    with open(filepath, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('utf-8', errors='replace').strip()
            if line.startswith('DATA'):
                data_type = line.split()[1].upper()
                break
            if line.startswith('#'):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                header[parts[0]] = parts[1].strip()

        n = int(header['POINTS'])
        fields = header.get('FIELDS', 'x y z').split()
        types = header.get('TYPE', 'f f f').split()
        sizes = [int(s) for s in header.get('SIZE', '4 4 4').split()]
        counts = [int(c) for c in header.get('COUNT', '1 1 1').split()]

        if data_type == 'ASCII':
            pts = []
            for _ in range(n):
                row = f.readline().decode().strip().split()
                pts.append([float(v) for v in row[:3]])
            return np.array(pts, dtype=np.float32)

        type_map = {'F': 'f', 'I': 'i', 'U': 'u'}
        dt_list = []
        xyz_cols = []

        for fname, ftype, fsize, fcount in zip(fields, types, sizes, counts):
            base = type_map.get(ftype.upper(), 'f')
            np_type = f'<{base}{fsize}'
            for k in range(fcount):
                name = fname if fcount == 1 else f'{fname}_{k}'
                dt_list.append((name, np_type))
                if fname in ('x', 'y', 'z'):
                    xyz_cols.append(name)

        dt = np.dtype(dt_list)
        raw = f.read(n * dt.itemsize)
        data = np.frombuffer(raw, dtype=dt)
        pts = np.column_stack([data[c] for c in xyz_cols])
    return pts.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 1. Multi-view Projection
# ─────────────────────────────────────────────────────────────
def plot_bev_and_side(pts_raw, pts_gt, pts_pred, title, save_path):

    sources = [
        ('Input Point Cloud', pts_raw),
        ('GT Reconstruction (Voxel→Point)', pts_gt),
        ('Predicted Reconstruction', pts_pred),
    ]

    views = [
        ('Top View (XY)', 0, 1),
        ('Side View (XZ)', 0, 2),
        ('Front View (YZ)', 1, 2),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for row, (label, pts) in enumerate(sources):
        for col, (view, xi, yi) in enumerate(views):
            ax = axes[row][col]

            if len(pts) == 0:
                ax.text(0.5, 0.5, 'Empty', ha='center', va='center')
                continue

            x, y = pts[:, xi], pts[:, yi]

            xlo, xhi = np.percentile(x, 1), np.percentile(x, 99)
            ylo, yhi = np.percentile(y, 1), np.percentile(y, 99)

            mask = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
            x, y = x[mask], y[mask]

            h, _, _ = np.histogram2d(x, y, bins=200)
            h = np.log1p(h)

            ax.imshow(h.T, origin='lower', cmap='hot', aspect='auto')
            ax.set_title(f'{label}\n{view}', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ─────────────────────────────────────────────────────────────
# 2. Voxel Slice Comparison
# ─────────────────────────────────────────────────────────────
def load_voxel(pcd_path, grid_size=64):
    pts = read_pcd(pcd_path)
    if len(pts) == 0:
        return np.zeros((grid_size, grid_size, grid_size))

    pts = pts - pts.mean(0)
    pts = pts / (np.abs(pts).max() + 1e-6)

    coords = ((pts + 1) / 2 * grid_size).astype(int)
    valid = np.all((coords >= 0) & (coords < grid_size), axis=1)
    coords = coords[valid]

    voxel = np.zeros((grid_size, grid_size, grid_size))
    np.add.at(voxel, tuple(coords.T), 1)

    voxel = np.log1p(voxel)
    voxel /= voxel.max() + 1e-6
    return voxel


def plot_voxel_slices(vox_gt, vox_pred, title, save_path):

    D, H, W = vox_gt.shape

    slices = [
        ('XY Plane', vox_gt[D//2], vox_pred[D//2]),
        ('XZ Plane', vox_gt[:, H//2, :], vox_pred[:, H//2, :]),
        ('YZ Plane', vox_gt[:, :, W//2], vox_pred[:, :, W//2]),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle(title, fontsize=12)

    for i, (name, gt, pred) in enumerate(slices):
        err = np.abs(gt - pred)

        axes[i][0].imshow(gt, cmap='viridis')
        axes[i][0].set_title(f'{name} - GT')

        axes[i][1].imshow(pred, cmap='viridis')
        axes[i][1].set_title(f'{name} - Pred')

        axes[i][2].imshow(err, cmap='hot')
        axes[i][2].set_title(f'{name} - Error')

        axes[i][3].hist(err.ravel(), bins=50)
        axes[i][3].set_title(f'{name} - Error Dist')

    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close()


# ─────────────────────────────────────────────────────────────
# 3. Summary
# ─────────────────────────────────────────────────────────────
def plot_summary(stats_list, save_path):

    scores = [s['anomaly_score'] for s in stats_list]

    fig = plt.figure(figsize=(8,5))
    plt.bar(range(len(scores)), scores)
    plt.title('Anomaly Scores (Lower is Better)')
    plt.grid(alpha=0.3)

    plt.savefig(save_path)
    plt.close()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():

    raw_dir   = Path('/data/jyx/Lidar/scene2')
    recon_dir = Path('./reconstructed_pointclouds_s2_v1')
    out_dir   = Path('./visualization_results_final_s2_v1')
    out_dir.mkdir(exist_ok=True)

    stats_files = sorted(recon_dir.glob('*_stats.json'))
    stats_list = [json.loads(p.read_text()) for p in stats_files]

    print(f'Found {len(stats_list)} samples')

    plot_summary(stats_list, out_dir / 'summary.png')

    for stats in stats_list:

        fname = stats['filename']
        stem  = Path(fname).stem

        raw_pcd = raw_dir / fname
        gt_pcd  = recon_dir / f'{stem}_original_from_voxel.pcd'
        pred_pcd= recon_dir / f'{stem}_predicted.pcd'

        pts_raw = read_pcd(str(raw_pcd))
        pts_gt  = read_pcd(str(gt_pcd))
        pts_pred= read_pcd(str(pred_pcd))

        score = stats['anomaly_score']
        title = f'{stem} | Anomaly Score: {score:.4f}'

        plot_bev_and_side(
            pts_raw, pts_gt, pts_pred,
            title,
            out_dir / f'{stem}_bev.png'
        )

        vox_gt   = load_voxel(str(gt_pcd))
        vox_pred = load_voxel(str(pred_pcd))

        plot_voxel_slices(
            vox_gt, vox_pred,
            title,
            out_dir / f'{stem}_voxel.png'
        )

    print('Done!')


if __name__ == '__main__':
    main()