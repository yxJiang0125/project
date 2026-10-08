## ✅ 代码重构完成：`physics.py` → 模块化拆分

### 新模块结构

| 文件 | 职责 | 行数 |
|------|------|------|
| **`data_utils.py`** | Config配置类、VoxelDataset数据集、体素化、点云I/O、可视化工具、诊断工具 | ~800行 |
| **`model.py`** | FlowNet3D网络、ResidualBlock3D、物理损失函数（散度/NS方程约束）、异常检测器 | ~400行 |
| **`train.py`** | DDP工具函数、TrainingLogger、训练循环、预测/异常检测入口、CLI命令 | ~400行 |

### 依赖链（无循环依赖）
```
data_utils.py  →  model.py  →  train.py
```

### 保留的旧文件
- `physics.py` / `physics_DDP.py` / `physics_v1.py` — 保留不动
- `train_physics.py` / `reconstruct_pointcloud.py` / `save_reconstruction_images.py` / `visualize_reconstruction.py` / `voxel_visualization_fortest.py` — 仍引用旧模块，按用户要求不更新

### CLI 使用方式（train.py）
```bash
# 训练
python train.py
torchrun --nproc_per_node=4 train.py

# 异常检测测试
python train.py test checkpoints/xxx.pth file1.pcd file2.pcd

# 未来帧预测
python train.py predict checkpoints/xxx.pth 10 file1.pcd file2.pcd ...

# 预测模型训练
python train.py train_predict 5 /data/jyx/Lidar/scene1
```