#!/usr/bin/env python3
"""
Street Gaussian 分块可视化工具

可视化分块配置、相机分布和训练结果，帮助用户理解和调试分块训练过程。
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from lib.datasets.dataset import Dataset
    from lib.training.block_trainer import BlockConfig, BlockTrainer
    from lib.config import cfg
except ImportError as e:
    print(f"Warning: Could not import Street Gaussian modules: {e}")
    print("Some functionality may be limited.")

def load_block_config(config_path: str) -> List[BlockConfig]:
    """从JSON文件加载块配置"""
    with open(config_path, 'r') as f:
        blocks_data = json.load(f)
    
    block_configs = []
    for block_data in blocks_data:
        config = BlockConfig(
            block_id=block_data['block_id'],
            spatial_bounds=tuple(block_data['spatial_bounds']),
            overlap_margin=block_data.get('overlap_margin', 2.0)
        )
        block_configs.append(config)
    
    return block_configs

def visualize_blocks_2d(block_configs: List[BlockConfig], 
                       camera_positions: np.ndarray = None,
                       save_path: str = None,
                       title: str = "Block Configuration"):
    """可视化2D块分布"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 颜色映射
    colors = plt.cm.Set3(np.linspace(0, 1, len(block_configs)))
    
    # 绘制块边界
    for i, config in enumerate(block_configs):
        min_x, max_x, min_y, max_y = config.spatial_bounds
        margin = config.overlap_margin
        
        # 核心区域
        core_rect = patches.Rectangle(
            (min_x, min_y), max_x - min_x, max_y - min_y,
            linewidth=2, edgecolor=colors[i], facecolor=colors[i], 
            alpha=0.3, label=f'Block {config.block_id} Core'
        )
        ax.add_patch(core_rect)
        
        # 重叠区域
        overlap_rect = patches.Rectangle(
            (min_x - margin, min_y - margin), 
            (max_x - min_x) + 2*margin, (max_y - min_y) + 2*margin,
            linewidth=1, edgecolor=colors[i], facecolor='none',
            linestyle='--', alpha=0.7
        )
        ax.add_patch(overlap_rect)
        
        # 块中心点
        center_x, center_y = config.center
        ax.plot(center_x, center_y, 'o', color=colors[i], markersize=8)
        ax.text(center_x, center_y + 2, f'B{config.block_id}', 
               ha='center', va='bottom', fontweight='bold')
    
    # 绘制相机位置
    if camera_positions is not None:
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], 
                  c='red', s=20, alpha=0.6, label='Cameras', zorder=5)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    return fig, ax

def visualize_camera_assignment(block_trainer: BlockTrainer, 
                               save_path: str = None):
    """可视化相机分配结果"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 收集所有相机位置
    all_train_cameras = []
    all_test_cameras = []
    block_train_counts = []
    block_test_counts = []
    
    for dataset_cameras in [block_trainer.dataset.scene_info.train_cameras, 
                           block_trainer.dataset.scene_info.test_cameras]:
        for camera in dataset_cameras:
            pos = camera.T[:2]  # 只取x, y坐标
            if dataset_cameras == block_trainer.dataset.scene_info.train_cameras:
                all_train_cameras.append(pos)
            else:
                all_test_cameras.append(pos)
    
    all_train_cameras = np.array(all_train_cameras)
    all_test_cameras = np.array(all_test_cameras)
    
    # 统计每个块的相机数量
    for block_config in block_trainer.block_configs:
        block_id = block_config.block_id
        train_count = len(block_trainer.block_cameras[block_id]['train'])
        test_count = len(block_trainer.block_cameras[block_id]['test'])
        block_train_counts.append(train_count)
        block_test_counts.append(test_count)
    
    # 左图：训练相机分布
    visualize_blocks_2d(block_trainer.block_configs, all_train_cameras, 
                       title="Training Camera Distribution")
    
    # 右图：测试相机分布
    ax2.clear()
    visualize_blocks_2d(block_trainer.block_configs, all_test_cameras, 
                       title="Test Camera Distribution")
    
    # 添加统计信息
    stats_text = "Camera Assignment Statistics:\n"
    for i, config in enumerate(block_trainer.block_configs):
        stats_text += f"Block {config.block_id}: {block_train_counts[i]} train, {block_test_counts[i]} test\n"
    
    fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Camera assignment visualization saved to: {save_path}")
    else:
        plt.show()

def analyze_block_overlap(block_configs: List[BlockConfig]) -> Dict:
    """分析块之间的重叠情况"""
    
    overlap_analysis = {
        'total_blocks': len(block_configs),
        'overlaps': [],
        'coverage_stats': {}
    }
    
    # 计算每对块之间的重叠
    for i, config1 in enumerate(block_configs):
        for j, config2 in enumerate(block_configs):
            if i >= j:
                continue
                
            # 计算重叠区域
            min_x1, max_x1, min_y1, max_y1 = config1.spatial_bounds
            min_x2, max_x2, min_y2, max_y2 = config2.spatial_bounds
            
            # 扩展到重叠边界
            margin1, margin2 = config1.overlap_margin, config2.overlap_margin
            
            # 检查是否有重叠
            overlap_min_x = max(min_x1 - margin1, min_x2 - margin2)
            overlap_max_x = min(max_x1 + margin1, max_x2 + margin2)
            overlap_min_y = max(min_y1 - margin1, min_y2 - margin2)
            overlap_max_y = min(max_y1 + margin1, max_y2 + margin2)
            
            if overlap_min_x < overlap_max_x and overlap_min_y < overlap_max_y:
                overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)
                
                # 计算重叠比例
                area1 = (max_x1 - min_x1) * (max_y1 - min_y1)
                area2 = (max_x2 - min_x2) * (max_y2 - min_y2)
                
                overlap_info = {
                    'block1': config1.block_id,
                    'block2': config2.block_id,
                    'overlap_area': overlap_area,
                    'overlap_ratio_1': overlap_area / area1,
                    'overlap_ratio_2': overlap_area / area2,
                    'overlap_bounds': (overlap_min_x, overlap_max_x, overlap_min_y, overlap_max_y)
                }
                overlap_analysis['overlaps'].append(overlap_info)
    
    # 统计覆盖情况
    if block_configs:
        all_min_x = min(config.spatial_bounds[0] for config in block_configs)
        all_max_x = max(config.spatial_bounds[1] for config in block_configs)
        all_min_y = min(config.spatial_bounds[2] for config in block_configs)
        all_max_y = max(config.spatial_bounds[3] for config in block_configs)
        
        total_scene_area = (all_max_x - all_min_x) * (all_max_y - all_min_y)
        total_block_area = sum((config.spatial_bounds[1] - config.spatial_bounds[0]) * 
                              (config.spatial_bounds[3] - config.spatial_bounds[2]) 
                              for config in block_configs)
        
        overlap_analysis['coverage_stats'] = {
            'scene_bounds': (all_min_x, all_max_x, all_min_y, all_max_y),
            'total_scene_area': total_scene_area,
            'total_block_area': total_block_area,
            'coverage_ratio': total_block_area / total_scene_area if total_scene_area > 0 else 0
        }
    
    return overlap_analysis

def visualize_training_progress(model_paths: List[str], save_path: str = None):
    """可视化训练进度（如果有日志文件）"""
    
    # 这里可以添加训练进度的可视化
    # 比如从日志文件中读取损失、PSNR等指标
    print("Training progress visualization not implemented yet.")
    print("This would show loss curves, PSNR progression, etc. for each block.")

def generate_block_report(block_configs: List[BlockConfig], 
                         camera_assignment: Dict = None,
                         output_dir: str = "."):
    """生成块配置报告"""
    
    report_path = os.path.join(output_dir, "block_analysis_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("Street Gaussian Block Configuration Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Blocks: {len(block_configs)}\n\n")
        
        # 块详细信息
        f.write("Block Details:\n")
        f.write("-" * 40 + "\n")
        for config in block_configs:
            min_x, max_x, min_y, max_y = config.spatial_bounds
            width = max_x - min_x
            height = max_y - min_y
            area = width * height
            
            f.write(f"Block {config.block_id}:\n")
            f.write(f"  Bounds: ({min_x:.1f}, {max_x:.1f}, {min_y:.1f}, {max_y:.1f})\n")
            f.write(f"  Size: {width:.1f}m x {height:.1f}m\n")
            f.write(f"  Area: {area:.1f} m²\n")
            f.write(f"  Overlap Margin: {config.overlap_margin:.1f}m\n")
            f.write(f"  Center: ({config.center[0]:.1f}, {config.center[1]:.1f})\n")
            
            if camera_assignment and config.block_id in camera_assignment:
                train_count = len(camera_assignment[config.block_id]['train'])
                test_count = len(camera_assignment[config.block_id]['test'])
                f.write(f"  Cameras: {train_count} train, {test_count} test\n")
            
            f.write("\n")
        
        # 重叠分析
        overlap_analysis = analyze_block_overlap(block_configs)
        f.write("Overlap Analysis:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of overlapping pairs: {len(overlap_analysis['overlaps'])}\n")
        
        for overlap in overlap_analysis['overlaps']:
            f.write(f"Blocks {overlap['block1']} - {overlap['block2']}: ")
            f.write(f"{overlap['overlap_area']:.1f} m² ")
            f.write(f"({overlap['overlap_ratio_1']:.1%} / {overlap['overlap_ratio_2']:.1%})\n")
        
        # 覆盖统计
        if 'coverage_stats' in overlap_analysis:
            stats = overlap_analysis['coverage_stats']
            f.write(f"\nCoverage Statistics:\n")
            f.write(f"Total scene area: {stats['total_scene_area']:.1f} m²\n")
            f.write(f"Total block area: {stats['total_block_area']:.1f} m²\n")
            f.write(f"Coverage ratio: {stats['coverage_ratio']:.1%}\n")
    
    print(f"Block analysis report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Street Gaussian Block Visualization Tool")
    
    parser.add_argument('--block_config', type=str, required=True,
                       help='Path to blocks configuration JSON file')
    parser.add_argument('--dataset_config', type=str, default=None,
                       help='Path to dataset configuration file (for camera positions)')
    parser.add_argument('--source_path', type=str, default=None,
                       help='Path to dataset (for camera positions)')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--show_cameras', action='store_true',
                       help='Show camera positions (requires dataset_config and source_path)')
    parser.add_argument('--generate_report', action='store_true',
                       help='Generate detailed analysis report')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载块配置
    print(f"Loading block configuration from: {args.block_config}")
    block_configs = load_block_config(args.block_config)
    print(f"Loaded {len(block_configs)} blocks")
    
    # 基本块可视化
    print("Generating block visualization...")
    fig_path = os.path.join(args.output_dir, "blocks_overview.png")
    visualize_blocks_2d(block_configs, save_path=fig_path, 
                       title="Block Configuration Overview")
    
    # 相机位置可视化（如果提供了数据集信息）
    camera_positions = None
    camera_assignment = None
    
    if args.show_cameras and args.dataset_config and args.source_path:
        try:
            print("Loading dataset for camera positions...")
            cfg.merge_from_file(args.dataset_config)
            cfg.source_path = args.source_path
            
            dataset = Dataset()
            block_trainer = BlockTrainer(dataset=dataset, block_configs=block_configs)
            
            # 相机分配可视化
            print("Generating camera assignment visualization...")
            camera_fig_path = os.path.join(args.output_dir, "camera_assignment.png")
            visualize_camera_assignment(block_trainer, save_path=camera_fig_path)
            
            camera_assignment = block_trainer.block_cameras
            
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            print("Continuing without camera visualization...")
    
    # 生成分析报告
    if args.generate_report:
        print("Generating analysis report...")
        generate_block_report(block_configs, camera_assignment, args.output_dir)
    
    # 重叠分析可视化
    print("Analyzing block overlaps...")
    overlap_analysis = analyze_block_overlap(block_configs)
    print(f"Found {len(overlap_analysis['overlaps'])} overlapping block pairs")
    
    if overlap_analysis['coverage_stats']:
        stats = overlap_analysis['coverage_stats']
        print(f"Coverage ratio: {stats['coverage_ratio']:.1%}")
    
    print(f"\nAll visualizations saved to: {args.output_dir}")

if __name__ == "__main__":
    main()