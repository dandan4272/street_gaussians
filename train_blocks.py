#!/usr/bin/env python3
"""
Street Gaussian 分块训练主脚本

使用示例:
python train_blocks.py --config configs/waymo_block_config.yaml --block_size 50 --overlap_margin 5
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# 添加项目路径到系统路径
sys.path.append(str(Path(__file__).parent))

from lib.datasets.dataset import Dataset
from lib.training.block_trainer import BlockTrainer, BlockConfig, create_spatial_blocks
from lib.training.block_aggregator import GaussianBlockAggregator
from lib.config import cfg
from lib.utils.general_utils import safe_state

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Street Gaussian Block Training')
    
    # 基本参数
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--source_path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to save models')
    parser.add_argument('--exp_name', type=str, default='block_experiment',
                       help='Experiment name')
    
    # 分块参数
    parser.add_argument('--block_size', type=float, default=50.0,
                       help='Size of each block in meters')
    parser.add_argument('--overlap_margin', type=float, default=5.0,
                       help='Overlap margin between blocks in meters')
    parser.add_argument('--manual_blocks', type=str, default=None,
                       help='Path to manual block configuration JSON file')
    
    # 训练参数
    parser.add_argument('--iterations', type=int, default=None,
                       help='Training iterations per block (default: use config)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and only do aggregation')
    parser.add_argument('--skip_aggregation', action='store_true',
                       help='Skip aggregation and only do training')
    
    # 聚合参数
    parser.add_argument('--spatial_threshold', type=float, default=0.1,
                       help='Spatial threshold for deduplication in meters')
    parser.add_argument('--opacity_threshold', type=float, default=0.1,
                       help='Opacity threshold for removing low-opacity gaussians')
    
    # 其他参数
    parser.add_argument('--scene_bounds', type=float, nargs=4, default=None,
                       help='Manual scene bounds: min_x max_x min_y max_y')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing block checkpoints')
    
    return parser.parse_args()

def estimate_scene_bounds(dataset: Dataset) -> tuple:
    """估算场景边界"""
    print("Estimating scene bounds from camera positions...")
    
    all_cameras = []
    if dataset.scene_info.train_cameras:
        all_cameras.extend(dataset.scene_info.train_cameras)
    if dataset.scene_info.test_cameras:
        all_cameras.extend(dataset.scene_info.test_cameras)
    
    if not all_cameras:
        raise ValueError("No cameras found in dataset")
    
    # 收集所有相机位置
    positions = []
    for camera in all_cameras:
        positions.append(camera.T[:2])  # 只使用x, y坐标
    
    positions = np.array(positions)
    
    # 计算边界，添加一些缓冲区
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)
    
    # 添加10%的缓冲区
    x_margin = (max_x - min_x) * 0.1
    y_margin = (max_y - min_y) * 0.1
    
    bounds = (min_x - x_margin, max_x + x_margin, min_y - y_margin, max_y + y_margin)
    
    print(f"Estimated scene bounds: {bounds}")
    print(f"Scene size: {max_x - min_x:.1f}m x {max_y - min_y:.1f}m")
    
    return bounds

def load_manual_blocks(json_path: str) -> list:
    """从JSON文件加载手动定义的块配置"""
    import json
    
    with open(json_path, 'r') as f:
        blocks_data = json.load(f)
    
    block_configs = []
    for block_data in blocks_data:
        config = BlockConfig(
            block_id=block_data['block_id'],
            spatial_bounds=tuple(block_data['spatial_bounds']),
            overlap_margin=block_data.get('overlap_margin', 2.0)
        )
        block_configs.append(config)
    
    print(f"Loaded {len(block_configs)} manual block configurations")
    return block_configs

def main():
    args = parse_args()
    
    # 设置配置
    cfg.merge_from_file(args.config)
    cfg.source_path = args.source_path
    cfg.model_path = args.model_path
    cfg.exp_name = args.exp_name
    cfg.resume = args.resume
    
    # 确保输出目录存在
    os.makedirs(cfg.model_path, exist_ok=True)
    
    # 初始化随机种子
    safe_state(cfg.train.quiet if hasattr(cfg.train, 'quiet') else False)
    
    print(f"=== Street Gaussian Block Training ===")
    print(f"Config: {args.config}")
    print(f"Source: {args.source_path}")
    print(f"Output: {cfg.model_path}")
    print(f"Experiment: {args.exp_name}")
    
    # 加载数据集
    print("\nLoading dataset...")
    dataset = Dataset()
    
    # 创建或加载块配置
    if args.manual_blocks:
        print(f"Loading manual block configuration from {args.manual_blocks}")
        block_configs = load_manual_blocks(args.manual_blocks)
    else:
        # 自动创建块配置
        if args.scene_bounds:
            scene_bounds = tuple(args.scene_bounds)
            print(f"Using manual scene bounds: {scene_bounds}")
        else:
            scene_bounds = estimate_scene_bounds(dataset)
        
        block_configs = create_spatial_blocks(
            scene_bounds=scene_bounds,
            block_size=args.block_size,
            overlap_margin=args.overlap_margin
        )
    
    # 创建分块训练器
    print("\nInitializing block trainer...")
    block_trainer = BlockTrainer(dataset=dataset, block_configs=block_configs)
    
    # 保存块配置
    import json
    blocks_config_path = os.path.join(cfg.model_path, 'blocks_config.json')
    with open(blocks_config_path, 'w') as f:
        blocks_data = []
        for config in block_configs:
            blocks_data.append({
                'block_id': config.block_id,
                'spatial_bounds': config.spatial_bounds,
                'overlap_margin': config.overlap_margin,
                'center': config.center
            })
        json.dump(blocks_data, f, indent=2)
    print(f"Block configuration saved to {blocks_config_path}")
    
    # 训练阶段
    if not args.skip_training:
        print("\n=== Training Phase ===")
        try:
            block_trainer.train_all_blocks(iterations=args.iterations)
            print("All blocks training completed successfully!")
        except Exception as e:
            print(f"Training failed: {e}")
            if not args.resume:
                return 1
            else:
                print("Continuing with aggregation using existing checkpoints...")
    
    # 聚合阶段
    if not args.skip_aggregation:
        print("\n=== Aggregation Phase ===")
        try:
            # 创建聚合器
            aggregator = GaussianBlockAggregator(
                block_trainer=block_trainer,
                spatial_threshold=args.spatial_threshold,
                opacity_threshold=args.opacity_threshold
            )
            
            # 聚合模型
            aggregated_model = aggregator.aggregate_blocks()
            
            # 保存聚合后的模型
            aggregated_save_path = os.path.join(cfg.model_path, 'aggregated')
            checkpoint_path, ply_path = aggregator.save_aggregated_model(
                aggregated_model, aggregated_save_path
            )
            
            print(f"\n=== Aggregation Completed Successfully! ===")
            print(f"Aggregated model saved to: {aggregated_save_path}")
            print(f"Checkpoint: {checkpoint_path}")
            print(f"Point cloud: {ply_path}")
            
        except Exception as e:
            print(f"Aggregation failed: {e}")
            return 1
    
    print("\n=== Block Training Pipeline Completed ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())