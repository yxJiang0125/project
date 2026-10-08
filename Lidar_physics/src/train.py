#!/usr/bin/env python
"""
训练配置、DDP工具函数、训练执行

纯训练配置 + 执行层，不包含网络实现或数据定义。
"""

import os
import sys
import json
import struct
import subprocess
import numpy as np
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_utils import Config, VoxelDataset, load_point_cloud, voxelize_dense
from model import FlowNet3D, compute_loss, LidarAnomalyDetector


############################################
# 1. DDP工具函数
############################################
def setup():
    """
    分布式环境初始化
    
    Returns:
        local_rank: 本地GPU编号
    """
    # 检查是否使用torchrun启动
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        
        return local_rank
    else:
        return 0


def cleanup():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank():
    """获取当前进程的全局rank"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """获取总进程数"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_ddp_mode():
    """是否处于DDP模式"""
    return dist.is_initialized()


############################################
# 2. 训练日志记录器
############################################
class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, config):
        self.config = config
        self.start_time = datetime.now()
        self.rank = get_rank()
        
        if self.rank == 0:
            os.makedirs(config.log_dir, exist_ok=True)
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            
            # 日志文件
            self.log_file = open(
                os.path.join(config.log_dir, 'batch_logs.txt'), 'a'
            )
            self.history_file = os.path.join(config.log_dir, 'training_history.json')
            self.training_history = []
            
            # 保存配置
            config_dict = {k: v for k, v in vars(config).items() 
                          if not k.startswith('_') and not callable(v)}
            with open(os.path.join(config.log_dir, 'config.json'), 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            # 显示配置
            print(f"{'='*60}")
            print(f"训练配置")
            print(f"{'='*60}")
            print(f"  数据目录: {config.data_dir}")
            print(f"  体素大小: {config.voxel_size}")
            print(f"  批大小: {config.batch_size}")
            print(f"  训练轮数: {config.epochs}")
            print(f"  学习率: {config.learning_rate}")
            print(f"  日志目录: {config.log_dir}")
            print(f"  检查点目录: {config.checkpoint_dir}")
            print(f"{'='*60}\n")
    
    def log_batch(self, epoch, batch, losses, lr=None):
        """记录批次损失"""
        if self.rank != 0:
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        loss_str = ' '.join([f"{k}={v:.6f}" for k, v in losses.items()])
        lr_str = f' lr={lr:.2e}' if lr is not None else ''
        line = f"[{timestamp}] Epoch {epoch:3d} Batch {batch:4d} {loss_str}{lr_str}\n"
        
        self.log_file.write(line)
        self.log_file.flush()
    
    def log_epoch_summary(self, epoch, avg_losses, epoch_time, best_loss, lr=None):
        """记录epoch总结"""
        if self.rank != 0:
            return
        
        loss_str = ' '.join([f"{k}={v:.6f}" for k, v in avg_losses.items()])
        lr_str = f' lr={lr:.2e}' if lr is not None else ''
        
        print(f"Epoch {epoch:3d} | Time: {epoch_time:6.1f}s | {loss_str}{lr_str} | Best: {best_loss:.6f}")
        
        # 保存到历史记录
        history_entry = {
            'epoch': epoch,
            'losses': {k: float(v) for k, v in avg_losses.items()},
            'epoch_time_seconds': epoch_time,
            'best_loss': float(best_loss),
            'timestamp': datetime.now().isoformat()
        }
        self.training_history.append(history_entry)
        
        with open(self.history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 批量日志标记
        self.log_file.write(f"\n{'='*50}\n")
        self.log_file.write(f"EPOCH {epoch} SUMMARY:\n")
        self.log_file.write(f"  {loss_str}{lr_str}\n")
        self.log_file.write(f"  Time: {epoch_time:.1f}s\n")
        self.log_file.write(f"{'='*50}\n\n")
        self.log_file.flush()
    
    def save_checkpoint(self, model, optimizer, epoch, best_loss, model_list=None):
        """保存检查点"""
        if self.rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        path = os.path.join(
            self.config.checkpoint_dir,
            f'epoch_{epoch:03d}_loss_{best_loss:.4f}.pth'
        )
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")
        
        # 管理检查点列表
        if model_list is not None:
            model_list.append(path)
            if len(model_list) > 5:
                old_path = model_list.pop(0)
                if os.path.exists(old_path):
                    os.remove(old_path)
    
    def log_training_complete(self, total_time, best_loss, final_lr=None):
        """记录训练完成"""
        if self.rank != 0:
            return
        
        print(f"\n{'='*60}")
        print(f"  总时间: {total_time/60:.1f} min")
        print(f"  最佳损失: {best_loss:.6f}")
        if final_lr is not None:
            print(f"  最终学习率: {final_lr:.2e}")
        print(f"{'='*60}\n")
        
        completion = {
            'status': 'completed',
            'total_time_seconds': total_time,
            'best_loss': float(best_loss),
            'total_epochs': len(self.training_history),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.config.log_dir, 'training_completion.json'), 'w') as f:
            json.dump(completion, f, indent=2)
    
    def close(self):
        """关闭日志"""
        if self.rank == 0 and hasattr(self, 'log_file'):
            self.log_file.close()


############################################
# 3. 训练循环
############################################
def train_ddp(config):
    """
    分布式训练主函数
    
    支持:
        - 单GPU: python train.py
        - 多GPU: torchrun --nproc_per_node=4 train.py
    """
    local_rank = setup()
    rank = get_rank()
    world_size = get_world_size()
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    if rank == 0:
        print(f"训练模式: {'DDP' if is_ddp_mode() else '单GPU'}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
    
    # 准备数据
    if not os.path.exists(config.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {config.data_dir}")
    
    files = sorted([
        os.path.join(config.data_dir, f)
        for f in os.listdir(config.data_dir)
        if f.endswith(('.pcd', '.txt', '.bin'))
    ])
    
    if rank == 0:
        print(f"✓ 加载 {len(files)} 个点云文件")
    
    dataset = VoxelDataset(files, config, mode='train', cache_size=config.cache_size)
    
    if is_ddp_mode():
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    else:
        sampler = None
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 准备模型
    model = FlowNet3D(config).to(device)
    
    if is_ddp_mode():
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-6
    )
    
    # 日志和检查点
    logger = TrainingLogger(config)
    best_loss = float('inf')
    model_list = []
    
    # 训练循环
    for epoch in range(1, config.epochs + 1):
        if is_ddp_mode() and sampler:
            sampler.set_epoch(epoch)
        
        model.train()
        epoch_start = datetime.now()
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        for batch_idx, (voxel_curr, voxel_next) in enumerate(loader):
            voxel_curr = voxel_curr.to(device)   # [B, 1, D, H, W]
            voxel_next = voxel_next.to(device)   # [B, 1, D, H, W]
            
            optimizer.zero_grad()
            
            pred, flow = model(voxel_curr)
            
            loss, loss_dict = compute_loss(
                pred, voxel_next,
                voxel_curr,       # 前一帧
                flow,
                config,
                verbose=(rank == 0 and batch_idx == 0)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            num_batches += 1
            
            if rank == 0 and batch_idx % 20 == 0:
                logger.log_batch(epoch, batch_idx, loss_dict,
                                lr=optimizer.param_groups[0]['lr'])
        
        scheduler.step()
        
        # Epoch总结
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        avg_epoch_losses = {}
        for key in epoch_losses:
            avg_epoch_losses[key] = epoch_losses[key] / num_batches if num_batches > 0 else 0
        
        logger.log_epoch_summary(
            epoch, avg_epoch_losses, epoch_time, best_loss,
            optimizer.param_groups[0]['lr']
        )
        
        # 保存最佳模型
        if avg_epoch_losses['total'] < best_loss:
            best_loss = avg_epoch_losses['total']
            if config.save_interval > 0 and epoch % config.save_interval == 0:
                logger.save_checkpoint(model, optimizer, epoch, best_loss, model_list)
    
    # 训练完成
    total_training_time = (datetime.now() - logger.start_time).total_seconds()
    logger.log_training_complete(total_training_time, best_loss,
                                 optimizer.param_groups[0]['lr'])
    logger.close()
    cleanup()


############################################
# 4. 预测模型训练
############################################
def train_predictive_model(config, seq_length=5):
    """
    训练预测模型（使用序列预测未来帧）
    
    Args:
        config: 配置对象
        seq_length: 预测序列长度
    """
    print(f"\n{'='*60}")
    print(f"开始预测模型训练 (序列长度: {seq_length})")
    print(f"{'='*60}")
    
    local_rank = setup()
    rank = get_rank()
    world_size = get_world_size()
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # 数据集
    if not os.path.exists(config.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {config.data_dir}")
    
    files = sorted([
        os.path.join(config.data_dir, f)
        for f in os.listdir(config.data_dir)
        if f.endswith(('.pcd', '.txt', '.npy', '.bin'))
    ])
    
    if len(files) < seq_length + 1:
        raise ValueError(f"预测模式需要至少{seq_length + 1}个文件，找到: {len(files)}")
    
    if rank == 0:
        print(f"✓ 加载 {len(files)} 个点云文件")
        print(f"📊 序列长度: {seq_length}")
    
    dataset = VoxelDataset(files, config, mode='predict',
                           seq_length=seq_length, cache_size=10)
    
    if is_ddp_mode():
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=True)
    else:
        sampler = None
    
    loader = DataLoader(dataset, batch_size=config.batch_size,
                       sampler=sampler, shuffle=(sampler is None),
                       num_workers=config.num_workers, drop_last=True)
    
    # 模型
    model = FlowNet3D(config).to(device)
    
    if is_ddp_mode():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    best_loss = float('inf')
    logger = TrainingLogger(config)
    model_list = []
    
    for epoch in range(1, config.epochs + 1):
        if is_ddp_mode() and sampler:
            sampler.set_epoch(epoch)
        
        model.train()
        epoch_start = datetime.now()
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        for batch_idx, (sequence, target) in enumerate(loader):
            sequence = sequence.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            pred, flow = model(sequence[:, -1])  # 使用序列最后一帧作为输入
            
            # 使用序列最后一帧作为前帧参考
            prev_frame = sequence[:, -1]
            loss, loss_dict = compute_loss(pred, target, prev_frame, flow, config)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            num_batches += 1
            
            if rank == 0 and batch_idx % 10 == 0:
                logger.log_batch(epoch, batch_idx, loss_dict,
                                lr=optimizer.param_groups[0]['lr'])
        
        scheduler.step()
        
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        logger.log_epoch_summary(epoch, avg_losses, epoch_time, best_loss,
                                optimizer.param_groups[0]['lr'])
        
        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            if config.save_interval > 0 and epoch % config.save_interval == 0:
                logger.save_checkpoint(model, optimizer, epoch, best_loss, model_list)
    
    logger.close()
    cleanup()
    
    if rank == 0:
        print(f"\n✓ 预测模型训练完成！最佳损失: {best_loss:.4f}")
        print(f"💾 模型已保存到 {config.checkpoint_dir}")


############################################
# 5. 未来帧预测
############################################
def predict_future_frame(config, checkpoint_path, input_files, seq_length=10):
    """
    使用前N帧预测未来帧
    
    Args:
        config: 配置对象
        checkpoint_path: 模型检查点路径
        input_files: 输入文件列表（至少seq_length+1个）
        seq_length: 使用多少帧进行预测
        
    Returns:
        predicted_voxel: 预测的体素数据
        actual_voxel: 实际的体素数据（用于对比）
        mse_loss: MSE损失
    """
    print(f"\n{'='*60}")
    print("未来帧预测")
    print(f"{'='*60}")
    
    if len(input_files) < seq_length + 1:
        raise ValueError(f"需要至少{seq_length + 1}个文件，但只有{len(input_files)}个")
    
    dataset = VoxelDataset(input_files, config, mode='predict', seq_length=seq_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = FlowNet3D(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"📂 使用前{seq_length}帧预测第{seq_length+1}帧")
    print(f"📊 模型加载: {checkpoint_path}")
    
    with torch.no_grad():
        sequence, target = next(iter(loader))
        sequence = sequence.to(device)
        
        print(f"输入序列形状: {sequence.shape}")
        print(f"目标帧形状: {target.shape}")
        
        pred, flow = model(sequence[:, -1])
        
        mse_loss_value = F.mse_loss(pred, target.to(device)).item()
        print(f"📈 预测MSE损失: {mse_loss_value:.6f}")
        
        pred_np = pred.cpu().numpy().squeeze()
        target_np = target.numpy().squeeze()
        
        return pred_np, target_np, mse_loss_value


############################################
# 6. 异常检测测试
############################################
def test_anomaly_detection(config, checkpoint_path, test_files,
                           output_dir="./test_results", save_format='npy'):
    """
    测试异常检测性能并保存可视化结果
    
    Args:
        config: 配置对象
        checkpoint_path: 模型检查点路径
        test_files: 测试文件列表
        output_dir: 输出目录
        save_format: 保存格式 ('npy', 'pointcloud', 'vtk')
    """
    print(f"\n{'='*60}")
    print("🔍 开始异常检测测试")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    detector = LidarAnomalyDetector(config)
    detector.load_checkpoint(checkpoint_path)
    
    test_dataset = VoxelDataset(test_files, config, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    anomaly_scores = []
    results_list = []
    
    print(f"处理 {len(test_dataset)} 个测试样本...")
    print(f"输出目录: {output_dir}")
    
    for i, voxel in enumerate(test_loader):
        score, reconstruction = detector.detect_anomaly(voxel.squeeze(0))
        anomaly_scores.append(score)
        
        original_np = voxel.squeeze(0).squeeze(0).numpy()
        predicted_np = reconstruction.squeeze(0).squeeze(0).numpy()
        
        results_list.append({
            'index': i,
            'score': score,
            'original': original_np,
            'predicted': predicted_np,
            'error': np.abs(original_np - predicted_np)
        })
        
        if i % 10 == 0:
            print(f"  样本 {i}: 异常分数 = {score:.4f}")
        
        detector.save_predictions(voxel.squeeze(0), output_dir, sample_id=i)
    
    scores_array = np.array(anomaly_scores)
    print(f"\n{'='*60}")
    print(f"📊 异常检测结果统计:")
    print(f"{'='*60}")
    print(f"  总样本数: {len(scores_array)}")
    print(f"  平均异常分数: {scores_array.mean():.4f}")
    print(f"  最大异常分数: {scores_array.max():.4f}")
    print(f"  最小异常分数: {scores_array.min():.4f}")
    print(f"  异常分数标准差: {scores_array.std():.4f}")
    print(f"  中位数: {np.median(scores_array):.4f}")
    
    threshold_suggestion = scores_array.mean() + 2 * scores_array.std()
    print(f"  建议异常阈值: {threshold_suggestion:.4f}")
    print(f"{'='*60}\n")
    
    summary = {
        'total_samples': len(scores_array),
        'mean_score': float(scores_array.mean()),
        'max_score': float(scores_array.max()),
        'min_score': float(scores_array.min()),
        'std_score': float(scores_array.std()),
        'median_score': float(np.median(scores_array)),
        'suggested_threshold': float(threshold_suggestion),
        'anomaly_scores': [float(s) for s in anomaly_scores],
        'test_files': test_files
    }
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ 测试摘要已保存: {summary_path}\n")
    
    return anomaly_scores, results_list


############################################
# 7. 主函数
############################################
if __name__ == "__main__":
    # 启动方式：
    #   单GPU:     python train.py
    #   多GPU:     torchrun --nproc_per_node=4 train.py
    #   测试:      python train.py test <checkpoint_path> <test_files...>
    #   预测:      python train.py predict <checkpoint_path> <seq_length> <files...>
    #   预测训练:  python train.py train_predict <seq_length> [data_dir]

    config = Config()

    if len(sys.argv) == 1:
        # 训练模式
        config.data_dir = "/data/jyx/Lidar/scene1"
        if not os.path.exists(config.data_dir):
            print(f"❌ 数据目录不存在: {config.data_dir}")
            print("请修改 Config.data_dir 为正确的路径")
            exit(1)
        train_ddp(config)

    elif len(sys.argv) >= 3 and sys.argv[1] == "test":
        checkpoint_path = sys.argv[2]
        test_files = sys.argv[3:] if len(sys.argv) > 3 else None

        if test_files is None:
            print("用法: python train.py test <checkpoint_path> <test_file1> [test_file2] ...")
            exit(1)

        output_dir = "./viz_results"
        print(f"开始异常检测测试...")
        print(f"检查点: {checkpoint_path}")
        print(f"输出目录: {output_dir}")
        print(f"测试文件: {len(test_files)} 个")

        anomaly_scores, results = test_anomaly_detection(
            config, checkpoint_path, test_files,
            output_dir=output_dir
        )

        print(f"测试完成！")
        print(f"结果已保存到: {output_dir}")

    elif len(sys.argv) >= 4 and sys.argv[1] == "predict":
        checkpoint_path = sys.argv[2]
        seq_length = int(sys.argv[3])
        input_files = sys.argv[4:]

        if not input_files:
            print("用法: python train.py predict <checkpoint_path> <seq_length> <file1> [file2] ...")
            exit(1)

        print(f"开始未来帧预测...")
        print(f"检查点: {checkpoint_path}")
        print(f"序列长度: {seq_length}")
        print(f"输入文件: {len(input_files)} 个")

        predicted, actual, mse = predict_future_frame(
            config, checkpoint_path, input_files, seq_length
        )

        output_dir = "./prediction_results"
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "predicted_voxel.npy"), predicted)
        np.save(os.path.join(output_dir, "actual_voxel.npy"), actual)

        print(f"\n预测完成！")
        print(f"预测误差 (MSE): {mse:.6f}")
        print(f"结果已保存到: {output_dir}")

    elif len(sys.argv) >= 3 and sys.argv[1] == "train_predict":
        seq_length = int(sys.argv[2])
        config.data_dir = sys.argv[3] if len(sys.argv) > 3 else config.data_dir
        train_predictive_model(config, seq_length)

    else:
        print("用法:")
        print("  训练: python train.py")
        print("  测试: python train.py test <checkpoint_path> <test_file1> [test_file2] ...")
        print("  预测: python train.py predict <checkpoint_path> <seq_length> <file1> [file2] ...")
        print("  训练预测模型: python train.py train_predict <seq_length> [data_dir]")