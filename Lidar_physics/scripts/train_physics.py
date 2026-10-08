#!/usr/bin/env python
"""
用法:
    单GPU训练:     python train_physics.py
    多GPU训练:     torchrun --nproc_per_node=4 train_physics.py
    预测模型训练:   python train_physics.py train_predict <seq_length> [data_dir]

依赖: 从 physics_DDP.py 导入模型、数据集、损失函数等共享组件
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from datetime import datetime

# TensorBoard 可选
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# 从 physics_DDP 导入共享组件
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from legacy.physics_DDP import (
    Config,
    VoxelDataset,
    FlowNet3D,
    compute_loss,
    load_point_cloud,
    voxelize_dense,
)


############################################
# DDP 工具函数
############################################
def get_rank():
    """获取当前进程的全局rank，支持DDP和单GPU模式"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """获取总进程数"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_ddp_mode():
    """判断是否为DDP模式"""
    return get_world_size() > 1


def setup():
    """初始化分布式环境 (DDP)"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        return local_rank
    else:
        # 单GPU模式
        return 0


def cleanup():
    """清理分布式环境"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_memory_info():
    """获取GPU内存使用信息"""
    if not torch.cuda.is_available():
        return {}

    try:
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)    # GB

        return {
            'gpu_allocated_gb': allocated,
            'gpu_reserved_gb': reserved,
            'gpu_used_gb': allocated
        }
    except:
        return {}


############################################
# 日志和检查点管理
############################################
class TrainingLogger:
    """增强的训练日志管理 - 记录详细训练过程"""
    def __init__(self, config):
        self.config = config
        self.rank = get_rank()
        self.start_time = datetime.now()
        self.epoch_start_time = None
        self.training_history = []
        self.batch_history = []

        if self.rank == 0:
            # 创建日志目录
            Path(config.log_dir).mkdir(parents=True, exist_ok=True)
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

            # TensorBoard
            if HAS_TENSORBOARD:
                self.writer = SummaryWriter(
                    log_dir=config.log_dir,
                    comment=datetime.now().strftime("%Y%m%d_%H%M%S")
                )
            else:
                self.writer = None

            # 训练历史文件
            self.history_file = Path(config.log_dir) / "training_history.json"
            self.batch_log_file = Path(config.log_dir) / "batch_logs.txt"

            # 保存配置
            config_path = Path(config.log_dir) / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)

            # 初始化日志文件
            with open(self.batch_log_file, 'w') as f:
                f.write(f"训练开始时间: {self.start_time}\n")
                f.write("="*80 + "\n")

    def start_epoch(self, epoch):
        """开始新epoch"""
        self.epoch_start_time = datetime.now()
        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"🚀 开始 Epoch {epoch}/{self.config.epochs}")
            print(f"{'='*60}")

    def log_batch(self, epoch, batch_idx, total_batches, loss_dict, lr, memory_info=None):
        """记录每个batch的详细信息"""
        if self.rank != 0:
            return

        timestamp = datetime.now()
        batch_info = {
            'timestamp': timestamp.isoformat(),
            'epoch': epoch,
            'batch': batch_idx,
            'total_batches': total_batches,
            'learning_rate': lr,
            'losses': loss_dict,
            'memory': memory_info or {}
        }

        self.batch_history.append(batch_info)

        # 写入文件
        with open(self.batch_log_file, 'a') as f:
            time_str = timestamp.strftime("%H:%M:%S")
            loss_str = " | ".join([f"{k}: {v:.6f}" for k, v in loss_dict.items()])
            f.write(f"[{time_str}] E{epoch:02d}B{batch_idx:03d}/{total_batches:03d} | "
                   f"LR: {lr:.6f} | {loss_str}")
            if memory_info:
                mem_str = f" | GPU: {memory_info.get('gpu_used', 0):.1f}GB"
                f.write(mem_str)
            f.write("\n")

        # TensorBoard记录
        global_step = epoch * total_batches + batch_idx
        self.log_loss(global_step, loss_dict)
        self.log_lr(global_step, lr)

        # 每100个batch保存一次历史
        if len(self.batch_history) % 100 == 0:
            self._save_batch_history()

    def log_epoch_summary(self, epoch, epoch_losses, epoch_time, best_loss, lr):
        """记录epoch总结"""
        if self.rank != 0:
            return

        epoch_summary = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'epoch_time_seconds': epoch_time,
            'learning_rate': lr,
            'losses': epoch_losses,
            'best_loss': best_loss,
            'is_best': epoch_losses['total'] == best_loss
        }

        self.training_history.append(epoch_summary)

        # 打印详细总结
        print(f"\n✓ Epoch {epoch} 完成 (耗时: {epoch_time:.1f}s)")
        print(f"  ├─ 总损失: {epoch_losses['total']:.6f}")
        print(f"  ├─ 重建损失: {epoch_losses['rec']:.6f}")
        print(f"  ├─ 平滑损失: {epoch_losses['smooth']:.6f}")
        print(f"  ├─ 连续性损失: {epoch_losses['cont']:.6f}")
        #print(f"  ├─ 异常正则化: {epoch_losses['anomaly_reg']:.6f}")
        print(f"  ├─ 学习率: {lr:.6f}")
        print(f"  └─ 最佳损失: {best_loss:.6f} {'✓' if epoch_losses['total'] == best_loss else ''}")

        # TensorBoard记录epoch指标
        if self.writer:
            for key, val in epoch_losses.items():
                self.writer.add_scalar(f'Epoch_Loss/{key}', val, epoch)
            self.writer.add_scalar('Epoch_Loss/best_loss', best_loss, epoch)
            self.writer.add_scalar('Epoch_Loss/learning_rate', lr, epoch)

        # 保存训练历史
        self._save_training_history()

    def log_loss(self, step, loss_dict):
        """记录损失到TensorBoard"""
        if self.rank == 0 and self.writer:
            for key, val in loss_dict.items():
                self.writer.add_scalar(f'Loss/{key}', val, step)

    def log_lr(self, step, lr):
        """记录学习率"""
        if self.rank == 0 and self.writer:
            self.writer.add_scalar('Learning_Rate', lr, step)

    def log_model_info(self, model):
        """记录模型信息"""
        if self.rank != 0:
            return

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }

        print(f"\n📊 模型信息:")
        print(f"  ├─ 总参数量: {total_params:,}")
        print(f"  ├─ 可训练参数: {trainable_params:,}")
        print(f"  └─ 模型大小: {model_info['model_size_mb']:.2f} MB")

        # 保存到配置文件
        config_path = Path(self.config.log_dir) / "model_info.json"
        with open(config_path, 'w') as f:
            json.dump(model_info, f, indent=2)

    def log_training_complete(self, total_time, best_loss, final_lr):
        """记录训练完成信息"""
        if self.rank != 0:
            return

        completion_info = {
            'training_completed_at': datetime.now().isoformat(),
            'total_training_time_seconds': total_time,
            'best_loss': best_loss,
            'final_learning_rate': final_lr,
            'total_epochs': len(self.training_history),
            'average_epoch_time': total_time / max(1, len(self.training_history))
        }

        print(f"\n🎉 训练完成！")
        print(f"  ├─ 总训练时间: {total_time:.1f} 秒")
        print(f"  ├─ 最佳损失: {best_loss:.6f}")
        print(f"  ├─ 平均epoch时间: {completion_info['average_epoch_time']:.1f} 秒")
        print(f"  └─ 日志保存位置: {self.config.log_dir}")

        # 保存完成信息
        completion_path = Path(self.config.log_dir) / "training_completion.json"
        with open(completion_path, 'w') as f:
            json.dump(completion_info, f, indent=2)

    def _save_training_history(self):
        """保存训练历史到文件"""
        if self.rank == 0:
            with open(self.history_file, 'w') as f:
                json.dump(self.training_history, f, indent=2)

    def _save_batch_history(self):
        """保存batch历史（最近的）"""
        if self.rank == 0:
            batch_history_path = Path(self.config.log_dir) / "recent_batch_history.json"
            recent_batches = self.batch_history[-1000:] if len(self.batch_history) > 1000 else self.batch_history
            with open(batch_history_path, 'w') as f:
                json.dump(recent_batches, f, indent=2)

    def save_checkpoint(self, model, optimizer, epoch, best_loss, model_list):
        """保存检查点（仅rank0）"""
        if self.rank != 0:
            return

        ckpt_path = Path(self.config.checkpoint_dir) / \
                    f"epoch_{epoch:03d}_loss_{best_loss:.4f}.pth"

        torch.save({
            'epoch': epoch,
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'config': self.config.to_dict(),
            'training_history': self.training_history
        }, ckpt_path)

        model_list.append((best_loss, str(ckpt_path)))
        model_list.sort()

        # 保留top-k
        if len(model_list) > self.config.keep_top_k:
            _, old_path = model_list.pop()
            Path(old_path).unlink()

        print(f"💾 检查点已保存: {ckpt_path}")

    def close(self):
        """关闭日志"""
        if self.rank == 0:
            if self.writer:
                self.writer.close()

            # 最终保存所有历史
            self._save_training_history()
            self._save_batch_history()

            print(f"\n📝 训练日志已保存到: {self.config.log_dir}")


############################################
# 训练函数
############################################
def train_ddp(config):
    """
    DDP训练主函数（支持单GPU和多GPU）
    
    Args:
        config: Config 配置对象
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
        mode = "DDP" if is_ddp_mode() else "单GPU"
        print(f"\n{'='*60}")
        print(f"🚀 开始训练 ({mode} 模式)")
        print(f"{'='*60}")
    
    logger = TrainingLogger(config)
    
    # 数据集（延迟加载）
    if not os.path.exists(config.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {config.data_dir}")
    
    files = sorted([
        os.path.join(config.data_dir, f)
        for f in os.listdir(config.data_dir)
        if f.endswith(('.pcd', '.txt', '.npy', '.bin'))
    ])
    
    if len(files) < 2:
        raise ValueError(f"需要至少2个点云文件，找到: {len(files)}")
    
    if rank == 0:
        print(f"✓ 加载 {len(files)} 个点云文件")
        print(f"\n{'='*60}")
        print(f"🔍 数据质量检查 (仅检查前3个文件)...")
        print(f"{'='*60}")
        
        # 检查前几个文件的质量
        voxel_samples = []
        for i, file in enumerate(files[:3]):
            try:
                pts = load_point_cloud(file, verbose=True)
                voxel = voxelize_dense(pts, voxel_size=config.voxel_size, grid_size=config.grid_size)
                voxel_samples.append(voxel)
                print(f"✓ {Path(file).name}: {len(pts)} 个有效点 | "
                      f"体素非零占比: {(voxel > 0).sum() / (64**3) * 100:.2f}%")
            except Exception as e:
                print(f"⚠️  {Path(file).name}: {e}")
        
        # 检查初始体素
        if voxel_samples:
            print(f"\n【体素统计】")
            for i, v in enumerate(voxel_samples):
                print(f"  voxel {i}: min={v.min():.6f}, max={v.max():.6f}, "
                      f"mean={v.mean():.6f}, std={v.std():.6f}")
        
        print(f"{'='*60}\n")
    
    dataset = VoxelDataset(files, config, cache_size=10)
    
    if is_ddp_mode():
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
    else:
        sampler = None
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),  # 仅非DDP模式下shuffle
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    # 模型
    model = FlowNet3D(config).to(device)

    if is_ddp_mode():
        dist.barrier()
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=config.ddp_find_unused
        )

    # 记录模型信息
    logger.log_model_info(model)
    
    # 优化器和学习率调度
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-5
    )
    
    # 混合精度训练
    if config.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    best_loss = float('inf')
    model_list = []
    global_step = 0
    
    # 主训练循环
    for epoch in range(config.epochs):
        if is_ddp_mode() and sampler is not None:
            sampler.set_epoch(epoch)

        # 开始epoch
        logger.start_epoch(epoch)
        epoch_start_time = datetime.now()

        model.train()

        epoch_losses = {'total': 0, 'rec': 0, 'smooth': 0, 'cont': 0}
        num_batches = 0

        for batch_idx, (v_prev, v_t) in enumerate(loader):
            v_prev = v_prev.to(device, non_blocking=True)
            v_t = v_t.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 混合精度
            if config.use_amp:
                with torch.cuda.amp.autocast():
                    pred, flow = model(v_prev)

                    print("input std:", v_t.std().item())
                    print("pred std:", pred.std().item())
                    
                    loss, loss_dict = compute_loss(pred, v_t, v_prev, flow, config)

                scaler.scale(loss).backward()

                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                pred, flow = model(v_prev)
                loss, loss_dict = compute_loss(pred, v_t, v_prev, flow, config)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()

            # DDP梯度同步
            if is_ddp_mode() and dist.is_available() and dist.is_initialized():
                dist.all_reduce(loss)
                loss = loss / world_size

            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]
            num_batches += 1
            global_step += 1

            # 增强的batch日志记录
            if batch_idx % config.log_interval == 0 and rank == 0:
                memory_info = get_memory_info()
                logger.log_batch(epoch, batch_idx, len(loader), loss_dict,
                               optimizer.param_groups[0]['lr'], memory_info)

        # epoch结束
        scheduler.step()
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()

        # 计算平均损失
        avg_epoch_losses = {}
        for key in epoch_losses:
            avg_epoch_losses[key] = epoch_losses[key] / num_batches if num_batches > 0 else 0

        # 记录epoch总结
        logger.log_epoch_summary(epoch, avg_epoch_losses, epoch_time, best_loss,
                               optimizer.param_groups[0]['lr'])

        # 保存最佳模型
        if avg_epoch_losses['total'] < best_loss:
            best_loss = avg_epoch_losses['total']
            if config.save_interval > 0 and epoch % config.save_interval == 0:
                logger.save_checkpoint(model, optimizer, epoch, best_loss, model_list)

    # 训练完成
    total_training_time = (datetime.now() - logger.start_time).total_seconds()
    logger.log_training_complete(total_training_time, best_loss, optimizer.param_groups[0]['lr'])

    logger.close()
    cleanup()


def train_predictive_model(config, seq_length=5):
    """
    训练预测模型（使用序列预测未来帧）
    
    Args:
        config: 配置对象
        seq_length: 预测序列长度
    """
    print(f"\n{'='*60}")
    print(f"🚀 开始预测模型训练 (序列长度: {seq_length})")
    print(f"{'='*60}")
    
    local_rank = setup()
    rank = get_rank()
    world_size = get_world_size()
    
    # 设置设备
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
        print(f"📊 序列长度: {seq_length} (使用前{seq_length}帧预测下一帧)")
    
    dataset = VoxelDataset(files, config, mode='predict', seq_length=seq_length, cache_size=10)
    
    if is_ddp_mode():
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None
    
    loader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, 
                       shuffle=(sampler is None), num_workers=config.num_workers)
    
    # 模型
    model = FlowNet3D(config).to(device)
    
    if is_ddp_mode():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # 训练循环
    best_loss = float('inf')
    logger = TrainingLogger(config)
    
    for epoch in range(config.epochs):
        if is_ddp_mode() and sampler:
            sampler.set_epoch(epoch)
            
        model.train()
        epoch_loss = 0
        
        for batch_idx, (sequence, target) in enumerate(loader):
            sequence, target = sequence.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred, flow = model(sequence)
            
            # 对于预测，使用前一帧作为参考来计算物理约束
            prev_frame = sequence[:, -1]  # 序列的最后一帧作为参考
            loss, loss_dict = compute_loss(pred, target, prev_frame, flow, config)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}: Loss = {loss.item():.6f}")
        
        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        
        if rank == 0:
            print(f"Epoch {epoch}: Avg Loss = {avg_loss:.6f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': best_loss
                }, f"./checkpoints/predictive_model_seq{seq_length}_best.pth")
    
    logger.close()
    cleanup()
    
    if rank == 0:
        print(f"\n✓ 预测模型训练完成！最佳损失: {best_loss:.4f}")
        print(f"💾 模型已保存: ./checkpoints/predictive_model_seq{seq_length}_best.pth")


############################################
# 主入口
############################################
if __name__ == "__main__":
    """
    训练启动入口
    
    用法:
        # 标准重建训练
        python train_physics.py
        
        # 预测模型训练
        python train_physics.py train_predict <seq_length> [data_dir]
    """
    if len(sys.argv) >= 2 and sys.argv[1] == "train_predict":
        # 预测模型训练模式
        seq_length = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
        
        config = Config()
        if len(sys.argv) >= 4:
            config.data_dir = sys.argv[3]
        
        if not os.path.exists(config.data_dir):
            print(f"❌ 数据目录不存在: {config.data_dir}")
            print("请修改 Config.data_dir 或通过命令行参数指定")
            exit(1)
        
        train_predictive_model(config, seq_length)
    else:
        # 标准重建训练模式
        config = Config()
        
        if not os.path.exists(config.data_dir):
            print(f"❌ 数据目录不存在: {config.data_dir}")
            print("请修改 Config.data_dir 为正确的路径")
            exit(1)
        
        # 启动训练
        # 启动方式：
        #   单GPU:     python train_physics.py
        #   多GPU:     torchrun --nproc_per_node=4 train_physics.py
        train_ddp(config)