#!/usr/bin/env python3
"""
Street Gaussians 分块训练脚本 - 修正版
用于减少训练时的显存占用，支持长序列的分块训练和拼接
"""

import os
import sys
import torch
import argparse
import time
from typing import Dict, Any
import yaml

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_config(config_path: str):
    """加载配置文件"""
    from lib.config import cfg
    
    # 加载YAML配置
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # 更新配置
        def update_cfg(cfg_node, config_dict):
            for key, value in config_dict.items():
                if hasattr(cfg_node, key):
                    if isinstance(value, dict):
                        update_cfg(getattr(cfg_node, key), value)
                    else:
                        setattr(cfg_node, key, value)
        
        update_cfg(cfg, config_dict)
    
    return cfg

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Street Gaussians Chunked Training")
    
    parser.add_argument("--config", default="configs/chunked_training.yaml", type=str,
                       help="Path to config file")
    parser.add_argument("--source_path", required=True, type=str,
                       help="Path to dataset")
    parser.add_argument("--chunk_size", default=3, type=int,
                       help="Number of frames per training chunk")
    parser.add_argument("--overlap_size", default=1, type=int,
                       help="Number of overlapping frames between chunks")
    parser.add_argument("--contribution_threshold", default=0.15, type=float,
                       help="Threshold for pruning low-contribution gaussians")
    parser.add_argument("--max_memory_gb", default=8.0, type=float,
                       help="Maximum GPU memory to use (GB)")
    parser.add_argument("--output_dir", default="./output_chunked", type=str,
                       help="Override output directory")
    parser.add_argument("--iterations", default=5000, type=int,
                       help="Training iterations per chunk")
    
    return parser.parse_args()

def setup_cuda_and_memory(max_memory_gb: float):
    """设置CUDA和内存管理"""
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU训练")
        return False
    
    # 设置CUDA设备
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    
    # 获取GPU信息
    device_props = torch.cuda.get_device_properties(0)
    total_memory_gb = device_props.total_memory / (1024**3)
    
    print(f"GPU设备信息:")
    print(f"  设备名称: {device_props.name}")
    print(f"  总内存: {total_memory_gb:.2f} GB")
    print(f"  设置最大使用: {max_memory_gb:.2f} GB")
    
    # 设置内存分配比例
    if max_memory_gb < total_memory_gb:
        memory_fraction = max_memory_gb / total_memory_gb
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        print(f"  内存使用比例: {memory_fraction:.2f}")
    
    # 设置内存管理策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    
    return True

def monitor_memory_usage(stage: str = ""):
    """监控内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        
        print(f"内存使用 {stage}:")
        print(f"  已分配: {allocated:.2f} GB")
        print(f"  已保留: {reserved:.2f} GB") 
        print(f"  峰值使用: {max_allocated:.2f} GB")
    else:
        print("CUDA不可用，无法监控GPU内存")

def create_simple_chunked_trainer(chunk_size: int, overlap_size: int):
    """创建简化的分块训练器"""
    
    class SimpleChunkInfo:
        def __init__(self, chunk_id, cameras, model_path):
            self.chunk_id = chunk_id
            self.cameras = cameras
            self.model_path = model_path
    
    class SimpleChunkedTrainer:
        def __init__(self, chunk_size, overlap_size):
            self.chunk_size = chunk_size
            self.overlap_size = overlap_size
            self.chunk_models = {}
        
        def create_chunks(self, train_cameras):
            """创建训练块"""
            chunks = []
            total_frames = len(train_cameras)
            
            start_idx = 0
            chunk_id = 0
            
            while start_idx < total_frames:
                end_idx = min(start_idx + self.chunk_size, total_frames)
                chunk_cameras = train_cameras[start_idx:end_idx]
                
                if len(chunk_cameras) > 0:
                    chunk = SimpleChunkInfo(
                        chunk_id=chunk_id,
                        cameras=chunk_cameras,
                        model_path=os.path.join(cfg.model_path, f"chunk_{chunk_id}")
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                start_idx = end_idx - self.overlap_size
                
            print(f"创建了 {len(chunks)} 个训练块，每块约 {self.chunk_size} 帧")
            return chunks
        
        def train_chunk(self, chunk, dataset, iterations):
            """训练单个块"""
            from lib.models.street_gaussian_model import StreetGaussianModel
            from lib.models.street_gaussian_renderer import StreetGaussianRenderer
            from lib.utils.loss_utils import l1_loss, ssim
            from tqdm import tqdm
            import random
            
            print(f"\n训练块 {chunk.chunk_id} (共{len(chunk.cameras)}帧)...")
            
            # 确保输出目录存在
            os.makedirs(chunk.model_path, exist_ok=True)
            
            # 创建高斯模型
            gaussians = StreetGaussianModel(dataset.scene_info.metadata)
            
            # 从点云初始化
            point_cloud = dataset.scene_info.point_cloud
            scene_radius = dataset.scene_info.metadata['scene_radius']
            gaussians.create_from_pcd(point_cloud, scene_radius)
            gaussians.training_setup()
            
            # 将模型移动到GPU
            if torch.cuda.is_available():
                gaussians = gaussians.cuda()
            
            # 创建渲染器
            renderer = StreetGaussianRenderer()
            
            # 训练循环
            viewpoint_stack = chunk.cameras.copy()
            
            for iteration in tqdm(range(1, iterations + 1), desc=f"训练块{chunk.chunk_id}"):
                # 随机选择相机
                if len(viewpoint_stack) == 0:
                    viewpoint_stack = chunk.cameras.copy()
                
                viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
                
                # 确保相机数据在GPU上
                if torch.cuda.is_available():
                    viewpoint_cam.original_image = viewpoint_cam.original_image.cuda()
                
                # 渲染
                render_pkg = renderer.render(viewpoint_cam, gaussians)
                image = render_pkg["rgb"]
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]
                
                # 计算损失
                gt_image = viewpoint_cam.original_image
                mask = torch.ones_like(gt_image[0:1]).bool()
                
                if torch.cuda.is_available():
                    mask = mask.cuda()
                
                loss = l1_loss(image, gt_image, mask)
                loss += 0.2 * (1.0 - ssim(image, gt_image, mask=mask))
                
                # 反向传播
                loss.backward()
                
                # 更新学习率
                gaussians.update_learning_rate(iteration)
                
                # 密集化
                if iteration < iterations * 0.8:  # 前80%的迭代进行密集化
                    gaussians.set_max_radii2D(radii, visibility_filter)
                    gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter)
                    
                    if iteration > 500 and iteration % 100 == 0:
                        gaussians.densify_and_prune(
                            max_grad=0.0002,
                            min_opacity=0.005,
                            prune_big_points=iteration > 1000
                        )
                
                # 重置不透明度
                if iteration % 1500 == 0:
                    gaussians.reset_opacity()
                
                # 更新优化器
                gaussians.update_optimizer()
            
            # 保存模型
            model_path = os.path.join(chunk.model_path, "model.pth")
            state_dict = gaussians.save_state_dict(is_final=True)
            torch.save(state_dict, model_path)
            
            print(f"块 {chunk.chunk_id} 训练完成，高斯基元数量: {gaussians.get_xyz.shape[0]}")
            
            self.chunk_models[chunk.chunk_id] = gaussians
            return gaussians
    
    return SimpleChunkedTrainer(chunk_size, overlap_size)

def main():
    """主函数"""
    args = parse_arguments()
    
    print("=" * 80)
    print("Street Gaussians - 分块训练 (修正版)")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"数据路径: {args.source_path}")
    print(f"块大小: {args.chunk_size}")
    print(f"重叠大小: {args.overlap_size}")
    print(f"最大内存: {args.max_memory_gb} GB")
    print(f"训练迭代: {args.iterations}")
    print("=" * 80)
    
    # 检查数据路径
    if not os.path.exists(args.source_path):
        print(f"错误: 数据路径 {args.source_path} 不存在!")
        return
    
    # 加载配置
    cfg = load_config(args.config)
    cfg.source_path = args.source_path
    cfg.model_path = args.output_dir
    
    # 设置CUDA和内存
    cuda_available = setup_cuda_and_memory(args.max_memory_gb)
    if not cuda_available:
        print("继续使用CPU训练...")
    
    # 创建输出目录
    os.makedirs(cfg.model_path, exist_ok=True)
    
    try:
        # 加载数据集
        print("\n正在加载数据集...")
        from lib.datasets.dataset import Dataset
        
        dataset = Dataset()
        train_cameras = dataset.getTrainCameras()[1]  # 获取分辨率为1的相机
        
        print(f"数据集加载完成，总帧数: {len(train_cameras)}")
        monitor_memory_usage("数据集加载后")
        
        # 创建分块训练器
        trainer = create_simple_chunked_trainer(args.chunk_size, args.overlap_size)
        chunks = trainer.create_chunks(train_cameras)
        
        # 训练每个块
        start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            print(f"\n开始训练第 {i+1}/{len(chunks)} 个块...")
            
            trainer.train_chunk(chunk, dataset, args.iterations)
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            monitor_memory_usage(f"块 {chunk.chunk_id} 训练后")
        
        training_time = time.time() - start_time
        
        print(f"\n" + "=" * 80)
        print("训练完成!")
        print(f"总训练时间: {training_time / 60:.1f} 分钟")
        print(f"训练了 {len(chunks)} 个块")
        print(f"结果保存在: {cfg.model_path}")
        print("=" * 80)
        
        # 最终内存统计
        monitor_memory_usage("最终")
        
    except Exception as e:
        print(f"\n训练失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()