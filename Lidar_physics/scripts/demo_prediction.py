#!/usr/bin/env python3
"""
激光点云预测演示脚本
展示如何使用训练好的模型进行未来帧预测
"""

import os
import sys
import glob
from pathlib import Path

def demo_prediction():
    """演示预测功能"""
    print("🚀 激光点云未来帧预测演示")
    print("=" * 50)

    # 1. 检查数据
    data_dir = "/workspace/jyx/experiments/Lidar_physics"  # 假设数据在这里
    pcd_files = sorted(glob.glob(os.path.join(data_dir, "*.pcd")))

    if len(pcd_files) < 11:
        print(f"❌ 需要至少11个PCD文件，当前只有 {len(pcd_files)} 个")
        print("请将你的激光点云数据文件放到数据目录中")
        return

    print(f"✓ 找到 {len(pcd_files)} 个点云文件")

    # 2. 训练预测模型（如果还没有的话）
    checkpoint_path = "./checkpoints/predictive_model_seq10_best.pth"

    if not os.path.exists(checkpoint_path):
        print("\n📚 训练预测模型...")
        print("运行命令:")
        print(f"python physics.py train_predict 10 {data_dir}")
        print("\n或者如果你已经有训练好的异常检测模型，可以直接使用:")
        print(f"python physics.py predict <checkpoint_path> 10 {' '.join(pcd_files[:11])}")
        return

    # 3. 使用前10帧预测第11帧
    print("
🔮 使用前10帧预测第11帧..."    print("运行命令:")
    print(f"python physics.py predict {checkpoint_path} 10 {' '.join(pcd_files[:11])}")

    # 4. 可视化结果
    print("
🖼️  预测完成后，可视化结果:"    print("python visualize_predictions.py ./prediction_results/predicted_voxel.npy ./prediction_results/actual_voxel.npy")

def demo_training():
    """演示训练流程"""
    print("\n📚 完整训练流程:")
    print("1. 训练基础异常检测模型:")
    print("   python physics.py")
    print("")
    print("2. 训练序列预测模型 (使用10帧预测下一帧):")
    print("   python physics.py train_predict 10")
    print("")
    print("3. 测试异常检测:")
    print("   python physics.py test ./checkpoints/epoch_XXX_loss_X.XXX.pth file1.pcd file2.pcd")
    print("")
    print("4. 进行预测:")
    print("   python physics.py predict <checkpoint> <seq_len> file1.pcd file2.pcd ...")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        demo_training()
    else:
        demo_prediction()
        print("\n" + "=" * 50)
        print("💡 提示:")
        print("- 将你的PCD文件放到工作目录")
        print("- 运行 'python demo.py train' 查看训练流程")
        print("- 确保有足够的GPU内存进行训练")