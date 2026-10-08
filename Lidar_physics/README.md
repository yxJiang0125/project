# 激光点云异常检测与预测系统

基于物理约束的3D CNN，用于激光点云的异常检测和未来帧预测。

## 📁 目录结构

```
.
├── src/                 # 当前主线代码（重构后的模块化版本）
│   ├── data_utils.py    #   Config / VoxelDataset / 点云 I/O / 可视化工具
│   ├── model.py         #   FlowNet3D / 物理损失 / 异常检测器
│   ├── train.py         #   训练循环 / DDP / CLI 入口
│   ├── data_utils_v1.py # v1 实验分支：连续4帧输入（input_frames=4）
│   └── model_v1.py      #   v1：FlowNet3D 4帧版（依赖 data_utils_v1）
├── legacy/              # 重构前的单体实现（保留备查，未在主线使用）
│   ├── physics.py       #   ~1960 行单体；physics_v1.py 与其内容相同
│   ├── physics_DDP.py   #   多 GPU 变体
│   └── physics_versions.txt
├── scripts/             # 独立工具脚本（部分依赖 legacy/）
│   ├── reconstruct_pointcloud.py, save_reconstruction_images.py,
│   │   visualize_reconstruction.py, train_physics.py   # 依赖 legacy.physics_DDP
│   ├── voxel_visualization_fortest.py                  # 依赖 legacy.physics
│   └── demo_prediction.py, visualize_predictions.py, visualize_results.py,
│       voxel_debug.py, voxel_debug_autogrid.py
└── outputs/             # 历史运行产物（日志 / 检查点 / 图片 / 点云）
```

> 依赖链：`data_utils.py → model.py → train.py`（无循环依赖）。
> `scripts/` 下的脚本从 `legacy/` 导入，需在项目根目录运行，例如
> `python scripts/reconstruct_pointcloud.py`。

## 🚀 功能特性

- **异常检测**: 学习正常点云分布，检测异常模式
- **未来帧预测**: 使用历史帧序列预测未来点云
- **物理约束**: 保持空间连续性和质量守恒
- **多模态融合**: 支持与其他传感器数据融合
- **可视化输出**: 完整的体素数据可视化工具
- **分布式训练**: 支持多GPU训练

## 📊 日志功能详解

### 自动记录内容

训练过程中会自动生成以下日志文件：

```
logs/
├── config.json              # 训练配置
├── model_info.json          # 模型信息
├── training_history.json    # 完整训练历史
├── recent_batch_history.json # 最近batch记录
├── batch_logs.txt          # 实时batch日志
└── training_completion.json # 训练完成总结

checkpoints/
└── epoch_XXX_loss_XXX.pth  # 模型检查点
```

### 日志内容说明

#### 1. **实时Batch日志** (`batch_logs.txt`)
```
训练开始时间: 2024-01-15 10:30:00
================================================================================
[10:30:15] E000B000/050 | LR: 0.001000 | rec: 0.123456 | smooth: 0.000234 | cont: 0.001234 | anomaly_reg: 0.000012 | GPU: 2.3GB
[10:30:20] E000B010/050 | LR: 0.001000 | rec: 0.098765 | smooth: 0.000198 | cont: 0.000987 | anomaly_reg: 0.000009 | GPU: 2.4GB
...
```

#### 2. **训练历史** (`training_history.json`)
```json
[
  {
    "epoch": 0,
    "timestamp": "2024-01-15T10:35:00",
    "epoch_time_seconds": 45.2,
    "learning_rate": 0.001,
    "losses": {
      "total": 0.125,
      "rec": 0.110,
      "smooth": 0.008,
      "cont": 0.006,
      "anomaly_reg": 0.001
    },
    "best_loss": 0.125,
    "is_best": true
  }
]
```

#### 3. **TensorBoard可视化**
如果安装了tensorboard，会自动记录：
- 各损失分量随时间的变化
- 学习率变化
- Epoch级别的汇总指标

### 使用方法

#### 训练
```bash
cd src
python train.py
# 多 GPU
torchrun --nproc_per_node=4 train.py
```

#### 训练预测模型
```bash
# 使用10帧序列预测下一帧
python src/train.py train_predict 10
```

#### 未来帧预测
```bash
# 使用前10帧预测第11帧
python src/train.py predict checkpoints/predictive_model_seq10_best.pth 10 frame_001.pcd frame_002.pcd ... frame_011.pcd
```

#### 测试异常检测
```bash
python src/train.py test checkpoints/epoch_049_loss_0.0234.pth test_data/*.pcd
```

#### 查看日志
```bash
python src/train.py logs ./outputs/logs
```

#### 可视化结果
```bash
python scripts/visualize_predictions.py prediction_results/predicted_voxel.npy prediction_results/actual_voxel.npy
```

> 运行产物默认写入**当前工作目录**下（`./logs`、`./checkpoints`、`./viz_results`），
> 与历史产物 `outputs/` 相互独立。

#### 启动TensorBoard
```bash
tensorboard --logdir ./logs
```

## 📈 监控指标

### 损失分量说明
- **重建损失 (rec)**: 衡量模型重建正常点云的能力
- **平滑损失 (smooth)**: 确保流场空间平滑
- **连续性损失 (cont)**: 质量守恒物理约束
- **异常正则化 (anomaly_reg)**: 鼓励学习紧凑分布

### 性能指标
- **训练时间**: 每个epoch耗时统计
- **GPU内存**: 实时内存使用情况
- **学习率**: 动态学习率变化
- **最佳损失**: 历史最佳性能

## 🔧 配置参数

```python
class Config:
    # 训练配置
    epochs = 50
    learning_rate = 1e-4
    batch_size = 4

    # 损失权重
    weight_rec = 1.0          # 重建损失
    weight_smooth = 0.01       # 流平滑
    weight_continuous = 0.05   # 连续性约束
    weight_anomaly = 0.1       # 异常正则化

    # 日志配置
    log_dir = "./logs"
    checkpoint_dir = "./checkpoints"
    log_interval = 10          # 每10个batch记录一次
    save_interval = 5          # 每5个epoch保存检查点
```

## 📊 日志分析

### 使用Python分析日志
```python
import json
import matplotlib.pyplot as plt

# 加载训练历史
with open('logs/training_history.json', 'r') as f:
    history = json.load(f)

# 绘制损失曲线
epochs = [h['epoch'] for h in history]
losses = [h['losses']['total'] for h in history]

plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training Loss Curve')
plt.show()
```

### 关键指标监控
- **收敛速度**: 损失下降趋势
- **过拟合检测**: 训练/验证损失对比
- **稳定性**: 损失波动情况
- **资源使用**: GPU内存和训练时间

## 🎯 多模态融合

异常检测结果可直接用于多模态融合：

```python
# 激光异常检测
lidar_detector = LidarAnomalyDetector(config)
lidar_score, _ = lidar_detector.detect_anomaly(lidar_voxel)

# 结合其他模态
fusion_detector = MultimodalAnomalyDetector(config)
final_score, scores = fusion_detector.detect_multimodal_anomaly(
    lidar_voxel, visible_score, ir_score
)
```</content>
<parameter name="filePath">/workspace/jyx/experiments/Lidar_physics/README.md