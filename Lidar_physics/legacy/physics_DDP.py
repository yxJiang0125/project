import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from datetime import datetime
import json
import struct

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


############################################
# PCD 文件读取器（替代 open3d）
############################################
def read_pcd_file(filepath):
    """
    读取PCD格式点云文件
    支持 ASCII 和 BINARY 两种格式
    """
    with open(filepath, 'rb') as f:
        # 读取header
        header_dict = {}
        points_count = 0
        
        while True:
            line = f.readline().decode('utf-8').strip()
            if not line or line.startswith('DATA'):
                if line.startswith('DATA'):
                    data_type = line.split()[1]  # ASCII 或 BINARY
                break
            
            if line.startswith('#'):
                continue
            
            parts = line.split(None, 1)
            if len(parts) == 2:
                key, value = parts
                header_dict[key] = value
        
        if 'POINTS' not in header_dict:
            raise ValueError(f"Invalid PCD file: {filepath}")
        
        points_count = int(header_dict['POINTS'])
        
        # 读取点数据
        if data_type.upper() == 'ASCII':
            points = []
            for _ in range(points_count):
                line = f.readline().decode('utf-8').strip()
                if line:
                    try:
                        vals = [float(x) for x in line.split()[:3]]
                        points.append(vals)
                    except ValueError:
                        # 跳过无法解析的行
                        continue
            points = np.array(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)
        else:  # BINARY
            # 假设标准PCD格式：x, y, z 为 float32
            data = f.read(points_count * 12)  # 3 * 4 bytes per point
            points = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)
        
        return points


def save_voxel_for_visualization(voxel, output_path, format='npy', voxel_size=0.1, grid_origin=(0, 0, 0)):
    """
    保存体素数据为可视化格式
    
    Args:
        voxel: [H, W, D] 体素数组或 [1, H, W, D] tensor
        output_path: 输出路径（不含扩展名）
        format: 输出格式 ('npy', 'pointcloud', 'vtk')
        voxel_size: 体素大小
        grid_origin: 网格原点 (x, y, z)
    """
    # 转换为numpy
    if isinstance(voxel, torch.Tensor):
        voxel = voxel.cpu().numpy()
    
    # 去掉batch维度
    if voxel.ndim == 4:
        voxel = voxel.squeeze(0)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    if format == 'npy':
        # 保存为NumPy数组
        np.save(f"{output_path}.npy", voxel)
        print(f"✓ 体素数据已保存: {output_path}.npy (shape: {voxel.shape})")
    
    elif format == 'pointcloud':
        # 转换体素为点云
        indices = np.argwhere(voxel > 0.5)  # 找到非空体素
        if len(indices) > 0:
            # 转换索引为3D坐标
            points = indices[:, ::-1] * voxel_size + np.array(grid_origin)
            
            # 保存为PCD格式
            save_pcd_file(f"{output_path}.pcd", points)
            print(f"✓ 点云已保存: {output_path}.pcd ({len(points)} 点)")
        else:
            print(f"⚠️  体素为空，跳过点云保存")
    
    elif format == 'vtk':
        # 保存为VTK格式（.vti: 结构化网格）
        try:
            import vtk
            from vtk.util import numpy_support
            
            # 创建VTK结构化网格
            gridd = vtk.vtkImageData()
            gridd.SetSpacing(voxel_size, voxel_size, voxel_size)
            gridd.SetOrigin(grid_origin[0], grid_origin[1], grid_origin[2])
            gridd.SetDimensions(voxel.shape[0] + 1, voxel.shape[1] + 1, voxel.shape[2] + 1)
            
            # 转换体素为VTK数组
            voxel_flat = voxel.flatten(order='F')  # Fortran order for VTK
            voxel_vtk = numpy_support.numpy_to_vtk(voxel_flat, deep=True)
            voxel_vtk.SetName("density")
            gridd.GetCellData().SetScalars(voxel_vtk)
            
            # 保存
            writer = vtk.vtkXMLImageDataWriter()
            writer.SetFileName(f"{output_path}.vti")
            writer.SetInputData(gridd)
            writer.Write()
            
            print(f"✓ VTK文件已保存: {output_path}.vti")
        except ImportError:
            print("⚠️  VTK未安装，跳过VTK格式保存")


def save_pcd_file(filepath, points, colors=None):
    """
    保存点云为PCD格式
    
    Args:
        filepath: 输出路径
        points: [N, 3] 点云坐标
        colors: [N, 3] 可选的RGB颜色
    """
    n_points = len(points)
    
    with open(filepath, 'w') as f:
        # 写入header
        f.write("# .PCD v.7 - Point Cloud Data file format\n")
        f.write(f"VERSION .7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE f f f\n")
        f.write(f"COUNT 1 1 1\n")
        f.write(f"WIDTH {n_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n_points}\n")
        f.write("DATA ascii\n")
        
        # 写入点数据
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def compare_voxels_for_visualization(original, predicted, output_dir, voxel_size=0.1):
    """
    并排可视化原始和预测体素
    
    Args:
        original: 原始体素 [H, W, D]
        predicted: 预测体素 [H, W, D]
        output_dir: 输出目录
        voxel_size: 体素大小
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存原始体素
    save_voxel_for_visualization(
        original, 
        os.path.join(output_dir, 'original'),
        format='pointcloud',
        voxel_size=voxel_size
    )
    
    # 保存预测体素
    save_voxel_for_visualization(
        predicted,
        os.path.join(output_dir, 'predicted'),
        format='pointcloud',
        voxel_size=voxel_size
    )
    
    # 保存差异体素（绝对误差）
    error = np.abs(original - predicted)
    save_voxel_for_visualization(
        error,
        os.path.join(output_dir, 'error'),
        format='pointcloud',
        voxel_size=voxel_size
    )
    
    # 保存为NumPy便于进一步处理
    np.save(os.path.join(output_dir, 'original.npy'), original)
    np.save(os.path.join(output_dir, 'predicted.npy'), predicted)
    np.save(os.path.join(output_dir, 'error.npy'), error)
    
    print(f"\n✓ 对比可视化已保存到: {output_dir}")
    print(f"  - original.pcd / original.npy")
    print(f"  - predicted.pcd / predicted.npy")
    print(f"  - error.pcd / error.npy")


############################################
# 点云诊断和清理模块
############################################
class PointCloudDiagnostics:
    """点云质量检查和诊断"""
    
    @staticmethod
    def diagnose(points, file_path=""):
        """诊断点云问题并返回报告"""
        report = {
            'total_points': len(points),
            'nan_count': np.isnan(points).sum(),
            'inf_count': np.isinf(points).sum(),
            'zero_count': (points == 0).all(axis=1).sum(),
            'duplicates': 0,
        }
        
        # 计算重复点数
        if len(points) > 0:
            unique_points = len(np.unique(points.astype(np.int32), axis=0))
            report['duplicates'] = len(points) - unique_points
        
        # 计算统计信息
        valid_mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
        valid_points = points[valid_mask]
        
        if len(valid_points) > 0:
            report['valid_count'] = len(valid_points)
            report['min'] = valid_points.min(axis=0)
            report['max'] = valid_points.max(axis=0)
            report['mean'] = valid_points.mean(axis=0)
            report['std'] = valid_points.std(axis=0)
        else:
            report['valid_count'] = 0
        
        return report
    
    @staticmethod
    def print_report(report, file_path=""):
        """打印诊断报告"""
        print(f"\n{'='*60}")
        print(f"📊 点云诊断报告: {Path(file_path).name if file_path else 'Unknown'}")
        print(f"{'='*60}")
        print(f"总点数:        {report['total_points']}")
        print(f"有效点数:      {report.get('valid_count', 0)}")
        print(f"NaN点数:       {report['nan_count']}")
        print(f"Inf点数:       {report['inf_count']}")
        print(f"零点数:        {report['zero_count']}")
        print(f"重复点数:      {report['duplicates']}")
        
        if report.get('valid_count', 0) > 0:
            print(f"\n坐标范围:")
            print(f"  Min: {report['min']}")
            print(f"  Max: {report['max']}")
            print(f"  Mean: {report['mean']}")
            print(f"  Std:  {report['std']}")
        print(f"{'='*60}\n")
    
    @staticmethod
    def clean(points, max_coord=1000.0):
        """清理点云数据"""
        # 移除 NaN 和 Inf
        valid_mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
        points = points[valid_mask]
        
        # 移除坐标过大的异常点（可能是编码错误）
        if len(points) > 0:
            coord_valid = np.all(np.abs(points) < max_coord, axis=1)
            points = points[coord_valid]
        
        # 移除全零点
        if len(points) > 0:
            nonzero_mask = ~(points == 0).all(axis=1)
            points = points[nonzero_mask]
        
        return points


############################################
# 0. 配置类（超参数管理）
############################################
class Config:
    """统一管理所有超参数 - 异常检测版本"""
    def __init__(self):
        # 数据配置
        self.data_dir = "/data/jyx/Lidar/scene2"
        self.voxel_size = 0.2
        self.grid_size = 64
        
        # 训练配置
        self.epochs = 50
        self.batch_size = 4           # 每张GPU的batch size，总batch = batch_size × GPU数
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.gradient_clip = 1.0
        
        # 异常检测损失权重
        self.weight_rec = 1.0      # 重建损失（主要）
        self.weight_smooth = 0.01   # 流平滑（次要）
        self.weight_continuous = 0.05  # 连续性约束（物理）
        self.weight_anomaly = 0.1   # 异常正则化
        
        # 模型配置
        self.input_channels = 1
        self.base_channels = 16
        self.use_amp = True
        self.enable_checkpoint = False

        # 多GPU配置
        self.pin_memory = True          # 多GPU下开启，加速Host→GPU传输
        self.num_workers = 4            # 建议设为 GPU数量 × 2
        self.ddp_find_unused = False    # FlowNet3D无未使用参数，保持False

        # 异常检测阈值
        self.anomaly_threshold = 0.1  # 异常分数阈值

        # 日志与保存
        self.log_dir = "./logs"
        self.checkpoint_dir = "./checkpoints"
        self.log_interval = 10
        self.save_interval = 5  # 每5个epoch保存
        self.keep_top_k = 3
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


############################################
# 1. DDP 初始化
############################################
def is_ddp_mode():
    """检查是否运行在DDP模式"""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup():
    """初始化DDP（仅在DDP模式下）"""
    if not is_ddp_mode():
        return 0  # 单进程模式，local_rank = 0
    
    # 获取环境变量
    backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl")
    
    try:
        dist.init_process_group(backend=backend)
    except RuntimeError as e:
        print(f"⚠️  {backend} 初始化失败，尝试 gloo...")
        if backend == "nccl":
            dist.init_process_group(backend="gloo")
        else:
            raise
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    """清理DDP（仅在DDP模式下）"""
    if is_ddp_mode():
        dist.destroy_process_group()


def get_rank():
    """获取当前rank"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """获取总rank数"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


############################################
# 2. 点云读取（增强验证）
############################################
def load_point_cloud(file_path, verbose=True):
    """加载点云文件并验证"""
    file_path = str(file_path)
    
    try:
        if file_path.endswith('.pcd'):
            pts = read_pcd_file(file_path)
        elif file_path.endswith('.txt'):
            pts = np.loadtxt(file_path, dtype=np.float32)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
        elif file_path.endswith('.npy'):
            pts = np.load(file_path).astype(np.float32)
        elif file_path.endswith('.bin'):
            # 二进制格式（如KITTI）
            pts = np.fromfile(file_path, dtype=np.float32).reshape(-1, 3)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        pts = np.asarray(pts, dtype=np.float32)
        
        # 诊断原始数据
        report = PointCloudDiagnostics.diagnose(pts, file_path)
        
        if verbose and (report['nan_count'] > 0 or report['inf_count'] > 0 or report['zero_count'] > 0):
            PointCloudDiagnostics.print_report(report, file_path)
        
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
    
        
        # 只保留 x, y, z
        pts = pts[:, :3]
        
        original_count = len(pts)
        pts = PointCloudDiagnostics.clean(pts, max_coord=1000.0)
        cleaned_count = len(pts)
        
        if cleaned_count == 0:
            raise ValueError(f"清理后没有有效点: {file_path}")
        
        if cleaned_count < original_count:
            removed = original_count - cleaned_count
            if verbose:
                print(f"⚠️  已清理 {removed} 个异常点 ({removed/original_count*100:.1f}%) "
                      f"from {Path(file_path).name}")
        
        return pts
        
    except Exception as e:
        print(f"❌ 加载失败 {Path(file_path).name}: {e}")
        raise


############################################
# 3. 体素化（改进算法）
############################################
def voxelize_dense(points, voxel_size=0.2, grid_size=64):
    """
    改进的体素化算法（基于voxel_debug.py）
    - 中心化和归一化处理
    - 使用np.add.at()正确处理多点落在同一体素的情况
    - log压缩和非零归一化
    """
    # 中心化
    center = points.mean(axis=0)
    points = points - center

    # 归一化
    max_range = np.max(np.abs(points)) + 1e-6
    points = points / max_range

    coords = (points + 1) / 2 * grid_size
    coords = coords.astype(np.int32)

    valid = np.all((coords >= 0) & (coords < grid_size), axis=1)
    coords = coords[valid]

    voxel = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    if len(coords) > 0:
        indices = coords[:, 0] * grid_size * grid_size + \
                  coords[:, 1] * grid_size + \
                  coords[:, 2]

        np.add.at(voxel.ravel(), indices, 1)

        # log压缩（压缩点密度的动态范围）
        voxel = np.log(voxel + 1)

        # 归一化到 [0, 1]（体素值代表占用密度，不能为负）
        voxel = voxel / (voxel.max() + 1e-6)

    return voxel


############################################
# 4. Dataset（延迟加载，减少内存）
############################################
class VoxelDataset(Dataset):
    """
    改进的数据集类
    - 延迟加载：不在初始化时加载所有数据
    - 可选缓存：缓存热点数据
    - 支持训练/测试/预测模式
    """
    def __init__(self, files, config, mode='train', cache_size=0, seq_length=1):
        """
        Args:
            files: 文件列表
            config: 配置对象
            mode: 'train', 'test', 或 'predict'
            cache_size: 缓存多少个voxel（0表示不缓存）
            seq_length: 序列长度（用于预测模式）
        """
        self.files = sorted(files)
        self.config = config
        self.mode = mode
        self.cache_size = min(cache_size, len(self.files))
        self.seq_length = seq_length
        self.voxel_cache = {}  # {idx: voxel}
        
        # 验证数据
        if mode == 'train':
            min_files = 2
        elif mode == 'predict':
            min_files = seq_length + 1  # 需要序列+1个预测帧
        else:  # test
            min_files = 1
            
        if len(self.files) < min_files:
            raise ValueError(f"{mode}模式需要至少{min_files}个文件，但只有 {len(self.files)} 个")
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.files) - 1  # 训练需要前后帧对
        elif self.mode == 'predict':
            return len(self.files) - self.seq_length  # 预测需要序列+1帧
        else:  # test
            return len(self.files)      # 测试只需要单帧
    
    def _load_voxel(self, idx):
        """加载单个体素（带缓存）"""
        if idx in self.voxel_cache:
            return self.voxel_cache[idx]
        
        pts = load_point_cloud(self.files[idx], verbose=False)
        voxel = voxelize_dense(
            pts, 
            voxel_size=self.config.voxel_size,
            grid_size=self.config.grid_size
        )
        
        # 更新缓存
        if len(self.voxel_cache) < self.cache_size:
            self.voxel_cache[idx] = voxel
        
        return voxel
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            # 训练模式：返回前后帧对
            v_prev = self._load_voxel(idx)
            v_t = self._load_voxel(idx + 1)
            
            v_prev = torch.from_numpy(v_prev).float().unsqueeze(0)
            v_t = torch.from_numpy(v_t).float().unsqueeze(0)
            
            return v_prev, v_t
            
        elif self.mode == 'predict':
            # 预测模式：返回序列帧和目标帧
            # 输入：前seq_length帧，输出：第seq_length+1帧
            sequence = []
            for i in range(self.seq_length):
                v = self._load_voxel(idx + i)
                v = torch.from_numpy(v).float().unsqueeze(0)
                sequence.append(v)
            
            # 目标帧
            target = self._load_voxel(idx + self.seq_length)
            target = torch.from_numpy(target).float().unsqueeze(0)
            
            # 将序列堆叠为 [seq_length, 1, D, H, W]
            sequence = torch.stack(sequence, dim=0)
            
            return sequence, target
            
        else:  # test
            # 测试模式：返回单帧用于异常检测
            v_t = self._load_voxel(idx)
            v_t = torch.from_numpy(v_t).float().unsqueeze(0)
            
            return v_t


############################################
# 激光异常检测模块（多模态融合接口）
############################################
class LidarAnomalyDetector:
    """
    激光点云异常检测器
    - 基于物理约束的流网络
    - 输出异常分数用于多模态融合
    - 模块化设计便于集成
    """
    
    def __init__(self, config):
        self.config = config
        self.model = FlowNet3D(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def load_checkpoint(self, checkpoint_path):
        """加载训练好的模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        print(f"✓ 加载激光异常检测模型: {checkpoint_path}")
    
    def detect_anomaly(self, voxel_tensor, save_prediction=None):
        """
        检测单帧体素的异常分数
        
        Args:
            voxel_tensor: [1, 1, 64, 64, 64] 体素数据
            save_prediction: 保存预测结果的文件路径 (可选)
            
        Returns:
            anomaly_score: float 异常分数 (0-1, 越高越异常)
            reconstruction: [1, 1, 64, 64, 64] 重建结果
        """
        with torch.no_grad():
            voxel_tensor = voxel_tensor.to(self.device)
            
            # 使用模型进行重建预测
            reconstruction, _ = self.model(voxel_tensor)
            
            # 计算重建误差作为异常分数
            reconstruction_error = F.mse_loss(reconstruction, voxel_tensor).item()
            
            # 归一化到0-1范围
            anomaly_score = min(reconstruction_error / self.config.anomaly_threshold, 1.0)
            
            # 保存预测结果
            if save_prediction:
                pred_np = reconstruction.cpu().numpy()
                if save_prediction.endswith('.npy'):
                    np.save(save_prediction, pred_np)
                elif save_prediction.endswith('.pt') or save_prediction.endswith('.pth'):
                    torch.save(reconstruction.cpu(), save_prediction)
                elif save_prediction.endswith('.json'):
                    pred_dict = {
                        'prediction': pred_np.tolist(),
                        'anomaly_score': anomaly_score,
                        'shape': pred_np.shape,
                        'timestamp': str(datetime.now())
                    }
                    with open(save_prediction, 'w') as f:
                        json.dump(pred_dict, f, indent=2)
                print(f"💾 预测结果已保存: {save_prediction}")
            
            return anomaly_score, reconstruction.cpu()
    
    def detect_batch(self, voxel_batch):
        """
        批量检测异常分数
        
        Args:
            voxel_batch: [B, 1, 64, 64, 64] 批量体素数据
            
        Returns:
            anomaly_scores: [B] 异常分数数组
            reconstructions: [B, 1, 64, 64, 64] 重建结果
        """
        anomaly_scores = []
        reconstructions = []
        
        for voxel in voxel_batch:
            score, recon = self.detect_anomaly(voxel.unsqueeze(0))
            anomaly_scores.append(score)
            reconstructions.append(recon.squeeze(0))
        
        return torch.tensor(anomaly_scores), torch.stack(reconstructions)
    
    def save_predictions(self, voxel_tensor, output_dir, sample_id=0, formats=['npy']):
        """
        保存预测结果用于可视化
        
        Args:
            voxel_tensor: [1, 1, 64, 64, 64] 输入体素
            output_dir: 输出目录
            sample_id: 样本ID
            formats: 保存格式列表 ['npy', 'pointcloud', 'vtk']
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取预测结果
        score, reconstruction = self.detect_anomaly(voxel_tensor)
        
        base_name = f"sample_{sample_id:04d}"
        
        for fmt in formats:
            if fmt == 'npy':
                # 保存numpy数组
                np.save(os.path.join(output_dir, f"{base_name}_pred.npy"), 
                       reconstruction.numpy())
                np.save(os.path.join(output_dir, f"{base_name}_input.npy"), 
                       voxel_tensor.numpy())
                
            elif fmt == 'pointcloud':
                # 保存为点云格式 (XYZ)
                visualizer = VoxelVisualizer(self.config.voxel_size, self.config.grid_size)
                
                # 转换预测结果
                pred_points, pred_values = visualizer.voxel_to_points(reconstruction.squeeze().numpy())
                np.savetxt(os.path.join(output_dir, f"{base_name}_pred.xyz"), 
                          np.column_stack([pred_points, pred_values]),
                          fmt='%.6f', header='x y z value')
                
                # 转换输入数据
                input_points, input_values = visualizer.voxel_to_points(voxel_tensor.squeeze().numpy())
                np.savetxt(os.path.join(output_dir, f"{base_name}_input.xyz"), 
                          np.column_stack([input_points, input_values]),
                          fmt='%.6f', header='x y z value')
                
            elif fmt == 'vtk' and HAS_VTK:
                # 保存VTK格式（如果可用）
                self._save_vtk(reconstruction.numpy(), 
                              os.path.join(output_dir, f"{base_name}_pred.vtk"))
                self._save_vtk(voxel_tensor.numpy(), 
                              os.path.join(output_dir, f"{base_name}_input.vtk"))
        
        # 保存元数据
        metadata = {
            'sample_id': sample_id,
            'anomaly_score': score,
            'timestamp': str(datetime.now()),
            'voxel_size': self.config.voxel_size,
            'grid_size': self.config.grid_size
        }
        
        with open(os.path.join(output_dir, f"{base_name}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_vtk(self, voxel_data, filepath):
        """保存体素数据为VTK格式"""
        if not HAS_VTK:
            return
            
        # 移除batch和channel维度
        if voxel_data.ndim == 5:
            voxel_data = voxel_data.squeeze(0).squeeze(0)
        elif voxel_data.ndim == 4:
            voxel_data = voxel_data.squeeze(0)
            
        # 创建VTK图像数据
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(voxel_data.shape)
        image_data.SetSpacing([self.config.voxel_size] * 3)
        image_data.SetOrigin([0.0, 0.0, 0.0])
        
        # 转换为VTK数组
        vtk_array = numpy_to_vtk(voxel_data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_array.SetName("voxel_values")
        image_data.GetPointData().SetScalars(vtk_array)
        
        # 写入文件
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(filepath)
        writer.SetInputData(image_data)
        writer.Write()
    
    def detect_with_details(self, voxel_tensor, sample_id=None):
        """
        检测异常并返回详细信息用于可视化
        
        Args:
            voxel_tensor: [1, 1, 64, 64, 64] 体素数据
            sample_id: 样本ID用于标识
            
        Returns:
            dict: 包含异常分数、重建、误差等信息
        """
        with torch.no_grad():
            voxel_tensor = voxel_tensor.to(self.device)
            
            # 前向传播
            reconstruction, flow = self.model(voxel_tensor)
            
            # 计算各种误差指标
            reconstruction_error = F.mse_loss(reconstruction, voxel_tensor).item()
            anomaly_score = min(reconstruction_error / self.config.anomaly_threshold, 1.0)
            
            # 逐元素误差
            element_error = torch.abs(reconstruction - voxel_tensor).cpu()
            
            # 返回详细信息
            return {
                'sample_id': sample_id,
                'anomaly_score': anomaly_score,
                'reconstruction_error': reconstruction_error,
                'original': voxel_tensor.cpu(),
                'reconstruction': reconstruction.cpu(),
                'error': element_error,
                'flow': flow.cpu(),
                'flow_norm': torch.norm(flow).item()
            }
    
    def save_predictions(self, voxel_tensor, output_dir, sample_id=0):
        """
        保存预测结果为可视化格式
        
        Args:
            voxel_tensor: [1, 1, 64, 64, 64] 体素数据
            output_dir: 输出目录
            sample_id: 样本ID
        """
        details = self.detect_with_details(voxel_tensor, sample_id)
        
        # 转为numpy
        original_np = details['original'].squeeze().numpy()
        predicted_np = details['reconstruction'].squeeze().numpy()
        
        # 创建样本目录
        sample_dir = os.path.join(output_dir, f'sample_{sample_id:04d}')
        os.makedirs(sample_dir, exist_ok=True)
        
        # 保存对比结果
        compare_voxels_for_visualization(
            original_np,
            predicted_np,
            sample_dir,
            voxel_size=self.config.voxel_size
        )
        
        # 保存元数据
        metadata = {
            'sample_id': sample_id,
            'anomaly_score': details['anomaly_score'],
            'reconstruction_error': details['reconstruction_error'],
            'flow_norm': details['flow_norm']
        }
        
        meta_path = os.path.join(sample_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ 样本 {sample_id} 已保存到 {sample_dir}")
        return sample_dir


############################################
# 5. 模型（改进架构）
############################################
class ResidualBlock3D(nn.Module):
    """3D残差块"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        # 跳跃连接
        self.skip = nn.Identity() if in_ch == out_ch else \
                   nn.Conv3d(in_ch, out_ch, 1)
    
    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + residual)
        return out


class FlowNet3D(nn.Module):
    """
    改进的流网络
    - 添加BatchNorm正则化
    - 使用残差连接
    - 多尺度特征
    - 支持激活检查点
    - 支持序列输入用于预测
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch = config.base_channels
        
        # 编码器（保持分辨率，不下采样）
        self.encoder = nn.Sequential(
            nn.Conv3d(config.input_channels, ch, 3, padding=1),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
            
            ResidualBlock3D(ch, ch),
            
            ResidualBlock3D(ch, ch * 2),
            nn.BatchNorm3d(ch * 2),
            nn.ReLU(inplace=True),
        )
        
        # 序列融合模块（用于预测模式）
        self.sequence_fusion = nn.Sequential(
            nn.Conv3d(ch * 2, ch * 2, 1),  # 1x1卷积进行特征融合
            nn.BatchNorm3d(ch * 2),
            nn.ReLU(inplace=True),
        )
        
        # 解码器头
        self.decoder_voxel = nn.Sequential(
            nn.Conv3d(ch * 2, ch, 1),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, 1, 1)
        )
        
        self.decoder_flow = nn.Sequential(
            nn.Conv3d(ch * 2, ch, 1),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, 3, 1)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, 1, D, H, W] 或 [B, seq_len, 1, D, H, W]
        
        Returns:
            pred: [B, 1, D, H, W] 重建结果
            flow: [B, 3, D, H, W] 流场
        """
        if x.dim() == 5:
            # 单帧输入： [B, 1, D, H, W]
            feat = self.encoder(x)
            
        elif x.dim() == 6:
            # 序列输入： [B, seq_len, 1, D, H, W]
            batch_size, seq_len, c, d, h, w = x.shape
            
            # 分别编码每一帧
            encoded_frames = []
            for t in range(seq_len):
                frame = x[:, t]  # [B, 1, D, H, W]
                encoded = self.encoder(frame)  # [B, ch*2, D, H, W]
                encoded_frames.append(encoded)
            
            # 在时间维度上平均融合
            feat = torch.stack(encoded_frames, dim=0).mean(dim=0)  # [B, ch*2, D, H, W]
            
            # 额外的序列融合
            feat = self.sequence_fusion(feat)
            
        else:
            raise ValueError(f"不支持的输入维度: {x.shape}")
        
        # 如果使用激活检查点，需要重新计算
        if self.config.enable_checkpoint and self.training:
            feat = torch.utils.checkpoint.checkpoint(self.encoder, x, use_reentrant=False)
        
        pred = self.decoder_voxel(feat)
        flow = self.decoder_flow(feat)
        
        return pred, flow


############################################
# 6. 散度计算（优化版）
############################################
def divergence(flow):
    """计算流场散度 (∂u/∂x + ∂v/∂y + ∂w/∂z)"""
    # 使用pad确保输出尺寸一致
    dx = flow[:, 0:1, 1:, :, :] - flow[:, 0:1, :-1, :, :]
    dy = flow[:, 1:2, :, 1:, :] - flow[:, 1:2, :, :-1, :]
    dz = flow[:, 2:3, :, :, 1:] - flow[:, 2:3, :, :, :-1]
    
    # 填充以匹配原始大小
    dx = F.pad(dx, (0, 0, 0, 0, 0, 1), mode='replicate')
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0), mode='replicate')
    dz = F.pad(dz, (0, 1, 0, 0, 0, 0), mode='replicate')
    
    return dx + dy + dz


############################################
# 7. 损失函数（改进设计）
############################################
def compute_loss(pred, gt, prev, flow, config, verbose=False):
    """
    异常检测损失函数
    - 重建损失：学习正常模式的分布
    - 流平滑：保持物理合理性
    - 连续性：质量守恒约束
    """
    eps = 1e-8
    
    # 1. 重建损失（主要损失：学习正常帧的分布）
    rec = F.mse_loss(pred, gt) * config.weight_rec
    
    # 2. 流平滑损失（物理约束：流场不应有剧烈变化）
    flow_smooth_x = torch.mean((flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]) ** 2)
    flow_smooth_y = torch.mean((flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]) ** 2)
    flow_smooth_z = torch.mean((flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]) ** 2)
    flow_smooth = (flow_smooth_x + flow_smooth_y + flow_smooth_z) * config.weight_smooth
    
    # 3. 连续性约束（物理：质量守恒）
    dt = pred - prev
    div = divergence(flow)
    cont = F.mse_loss(-div, dt) * config.weight_continuous
    
    # 4. 异常正则化（鼓励学习紧凑的正常模式分布）
    pred_std = torch.std(pred)
    anomaly_reg = config.weight_anomaly / (pred_std + eps)  # 标准差越小，约束越大
    
    total = rec + flow_smooth + cont + anomaly_reg
    
    return total, {
        'total': total.item(),
        'rec': rec.item(),
        'smooth': flow_smooth.item(),
        'cont': cont.item(),
        'anomaly_reg': anomaly_reg.item(),
        'pred_std': pred_std.item(),
        'flow_norm': torch.norm(flow).item(),
    }


def get_memory_info():
    """获取GPU内存使用信息"""
    if not torch.cuda.is_available():
        return {}

    try:
        # 获取当前GPU内存使用情况
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)    # GB

        return {
            'gpu_allocated_gb': allocated,
            'gpu_reserved_gb': reserved,
            'gpu_used_gb': allocated  # 主要关注已分配的内存
        }
    except:
        return {}


############################################
# 8. 日志和检查点管理
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
        print(f"  ├─ 异常正则化: {epoch_losses['anomaly_reg']:.6f}")
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
            # 只保存最近的1000个batch记录
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
# 9. 训练函数（完整版）
############################################
def train_ddp(config):
    """DDP训练主函数（支持单GPU和多GPU）"""
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
    
    # ✅ 数据集（延迟加载）
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
                report = PointCloudDiagnostics.diagnose(pts, file)
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
    
    # ✅ 模型
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
    
    # 优化器：多GPU时 LR 线性 scale（Linear Scaling Rule）
    scaled_lr = config.learning_rate * world_size
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=config.weight_decay
    )

    if rank == 0:
        print(f"LR: {config.learning_rate} × {world_size} GPU = {scaled_lr:.2e}")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-5
    )
    
    # ✅ 混合精度训练
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

        epoch_losses = {'total': 0, 'rec': 0, 'smooth': 0, 'cont': 0, 'anomaly_reg': 0}
        num_batches = 0

        for batch_idx, (v_prev, v_t) in enumerate(loader):
            v_prev = v_prev.to(device, non_blocking=True)
            v_t = v_t.to(device, non_blocking=True)

            optimizer.zero_grad()

            # ✅ 混合精度
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

            # ✅ DDP梯度同步
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


############################################
# 10. 主函数
############################################
if __name__ == "__main__":
    # 启动方式：
    #   单GPU:     python physics.py
    #   多GPU:     torchrun --nproc_per_node=4 physics.py
    #   多机多卡:  torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    #                        --master_addr=<主节点IP> --master_port=29500 physics.py

    config = Config()
    config.data_dir = "/data/jyx/Lidar/scene1"

    if not os.path.exists(config.data_dir):
        print(f"数据目录不存在: {config.data_dir}")
        exit(1)

    train_ddp(config)


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
    """
    print(f"\n{'='*60}")
    print("🔮 未来帧预测")
    print(f"{'='*60}")
    
    if len(input_files) < seq_length + 1:
        raise ValueError(f"需要至少{seq_length + 1}个文件，但只有{len(input_files)}个")
    
    # 创建预测数据集
    dataset = VoxelDataset(input_files, config, mode='predict', seq_length=seq_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 加载模型
    model = FlowNet3D(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"📂 使用前{seq_length}帧预测第{seq_length+1}帧")
    print(f"📊 模型加载: {checkpoint_path}")
    
    with torch.no_grad():
        # 获取预测数据
        sequence, target = next(iter(loader))
        sequence = sequence.to(device)
        
        print(f"🔍 输入序列形状: {sequence.shape}")
        print(f"🎯 目标帧形状: {target.shape}")
        
        # 预测
        pred, flow = model(sequence)
        
        # 计算预测误差
        mse_loss = F.mse_loss(pred, target.to(device)).item()
        print(f"📈 预测MSE损失: {mse_loss:.6f}")
        
        # 转换为numpy
        pred_np = pred.cpu().numpy().squeeze()
        target_np = target.numpy().squeeze()
        
        return pred_np, target_np, mse_loss


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
    
    # 创建异常检测器
    detector = LidarAnomalyDetector(config)
    detector.load_checkpoint(checkpoint_path)
    
    # 创建测试数据集
    test_dataset = VoxelDataset(test_files, config, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    anomaly_scores = []
    results_list = []
    
    print(f"处理 {len(test_dataset)} 个测试样本...")
    print(f"输出目录: {output_dir}")
    
    for i, voxel in enumerate(test_loader):
        # 推理得到异常分数和重建结果
        score, reconstruction = detector.detect_anomaly(voxel.squeeze(0))
        anomaly_scores.append(score)
        
        # 转为numpy
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
        
        # 保存可视化结果到单独的目录
        detector.save_predictions(voxel.squeeze(0), output_dir, sample_id=i)
    
    # 统计结果
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
    
    # 建议阈值
    threshold_suggestion = scores_array.mean() + 2 * scores_array.std()
    print(f"  建议异常阈值: {threshold_suggestion:.4f}")
    print(f"{'='*60}\n")
    
    # 保存总结
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
# 11. 多模态融合接口
############################################
class MultimodalAnomalyDetector:
    """
    多模态异常检测融合器
    - 结合可见光、红外、激光异常分数
    - 加权融合策略
    """
    
    def __init__(self, config):
        self.config = config
        self.lidar_detector = LidarAnomalyDetector(config)
        # 这里可以添加其他模态的检测器
        
    def load_models(self, lidar_checkpoint):
        """加载各模态模型"""
        self.lidar_detector.load_checkpoint(lidar_checkpoint)
    
    def detect_multimodal_anomaly(self, lidar_voxel, visible_score=None, ir_score=None):
        """
        多模态异常检测
        
        Args:
            lidar_voxel: 激光体素数据
            visible_score: 可见光异常分数 (可选)
            ir_score: 红外异常分数 (可选)
            
        Returns:
            fused_score: 融合异常分数
            modality_scores: 各模态分数字典
        """
        # 激光异常检测
        lidar_score, _ = self.lidar_detector.detect_anomaly(lidar_voxel)
        
        modality_scores = {
            'lidar': lidar_score,
            'visible': visible_score,
            'ir': ir_score
        }
        
        # 简单平均融合（可以改进为更复杂的策略）
        valid_scores = [s for s in [lidar_score, visible_score, ir_score] if s is not None]
        fused_score = np.mean(valid_scores) if valid_scores else lidar_score
        
        return fused_score, modality_scores


# 使用示例
if __name__ == "__main__":
    # 训练模式
    if len(sys.argv) == 1:
        config = Config()
        if not os.path.exists(config.data_dir):
            print(f"❌ 数据目录不存在: {config.data_dir}")
            print("请修改 Config.data_dir 为正确的路径")
            exit(1)
        train_ddp(config)

    # 测试模式
    elif len(sys.argv) >= 3 and sys.argv[1] == "test":
        checkpoint_path = sys.argv[2]
        test_files = sys.argv[3:] if len(sys.argv) > 3 else None

        if test_files is None:
            print("用法: python physics.py test <checkpoint_path> <test_file1> [test_file2] ...")
            exit(1)

        config = Config()
        output_dir = "./viz_results"
        
        print(f"🔍 开始异常检测测试...")
        print(f"📂 检查点: {checkpoint_path}")
        print(f"📁 输出目录: {output_dir}")
        print(f"📄 测试文件: {len(test_files)} 个")
        
        anomaly_scores, results = test_anomaly_detection(
            config, checkpoint_path, test_files, 
            output_dir=output_dir
        )
        
        print(f"\n🎯 测试完成！")
        print(f"📊 结果已保存到: {output_dir}")
        print(f"🖼️  使用以下命令可视化:")
        print(f"   python visualize_predictions.py {output_dir}/sample_0000_pred.npy")

    # 预测模式
    elif len(sys.argv) >= 4 and sys.argv[1] == "predict":
        checkpoint_path = sys.argv[2]
        seq_length = int(sys.argv[3])
        input_files = sys.argv[4:]
        
        if not input_files:
            print("用法: python physics.py predict <checkpoint_path> <seq_length> <file1> [file2] ...")
            exit(1)
        
        config = Config()
        
        print(f"🔮 开始未来帧预测...")
        print(f"📂 检查点: {checkpoint_path}")
        print(f"📊 序列长度: {seq_length}")
        print(f"📄 输入文件: {len(input_files)} 个")
        
        predicted, actual, mse = predict_future_frame(
            config, checkpoint_path, input_files, seq_length
        )
        
        # 保存结果
        output_dir = "./prediction_results"
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, "predicted_voxel.npy"), predicted)
        np.save(os.path.join(output_dir, "actual_voxel.npy"), actual)
        
        print(f"\n🎯 预测完成！")
        print(f"📈 预测误差 (MSE): {mse:.6f}")
        print(f"💾 结果已保存到: {output_dir}")
        print(f"🖼️  可视化命令:")
        print(f"   python visualize_predictions.py {output_dir}/predicted_voxel.npy {output_dir}/actual_voxel.npy")
    
    # 训练预测模型
    elif len(sys.argv) >= 3 and sys.argv[1] == "train_predict":
        seq_length = int(sys.argv[2])
        
        config = Config()
        config.data_dir = sys.argv[3] if len(sys.argv) > 3 else config.data_dir
        
        train_predictive_model(config, seq_length)
        log_dir = sys.argv[2]
        if os.path.exists(log_dir):
            print(f"📊 日志目录: {log_dir}")
            print("\n📁 文件列表:")
            for f in os.listdir(log_dir):
                file_path = os.path.join(log_dir, f)
                size = os.path.getsize(file_path) / 1024  # KB
                print(f"  {f} ({size:.1f} KB)")

            # 显示训练历史
            history_file = os.path.join(log_dir, "training_history.json")
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                print(f"\n📈 训练历史: {len(history)} 个epoch")
                if history:
                    last_epoch = history[-1]
                    print(f"  最后epoch: {last_epoch['epoch']}")
                    print(f"  损失: {last_epoch['losses']['total']:.6f}")
                    print(f"  时间: {last_epoch['epoch_time_seconds']:.1f}s")
        else:
            print(f"❌ 日志目录不存在: {log_dir}")

    else:
        print("用法:")
        print("  训练: python physics.py")
        print("  测试: python physics.py test <checkpoint_path> <test_file1> [test_file2] ...")
        print("  预测: python physics.py predict <checkpoint_path> <seq_length> <file1> [file2] ...")
        print("  训练预测模型: python physics.py train_predict <seq_length> [data_dir]")
        print("  查看日志: python physics.py logs <log_directory>")