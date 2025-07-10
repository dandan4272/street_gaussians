#!/usr/bin/env python3
"""
Street Gaussian 分块训练使用示例

这个脚本演示了如何使用分块训练系统来训练大场景的 Street Gaussian 模型。
适用于内存受限或超大规模场景的情况。
"""

import os
import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from lib.datasets.dataset import Dataset
from lib.training.block_trainer import BlockTrainer, BlockConfig, create_spatial_blocks
from lib.training.block_aggregator import GaussianBlockAggregator
from lib.config import cfg
from lib.utils.general_utils import safe_state

def example_waymo_block_training():
    """Waymo 数据集分块训练示例"""
    print("=== Waymo 数据集分块训练示例 ===")
    
    # 配置参数
    config_path = "configs/block_training_example.yaml"
    source_path = "/path/to/waymo/dataset"  # 请修改为实际路径
    model_path = "./outputs/waymo_block_experiment"
    
    # 场景分块参数
    block_size = 80.0  # 80米 x 80米的块
    overlap_margin = 10.0  # 10米重叠
    
    # 检查路径
    if not os.path.exists(source_path):
        print(f"警告: 数据集路径不存在: {source_path}")
        print("请修改 source_path 为实际的数据集路径")
        return
    
    # 设置配置
    cfg.merge_from_file(config_path)
    cfg.source_path = source_path
    cfg.model_path = model_path
    cfg.exp_name = "waymo_blocks"
    
    os.makedirs(model_path, exist_ok=True)
    safe_state(False)
    
    print(f"数据集路径: {source_path}")
    print(f"输出路径: {model_path}")
    print(f"块大小: {block_size}m x {block_size}m")
    print(f"重叠边界: {overlap_margin}m")
    
    # 加载数据集
    print("\n1. 加载数据集...")
    dataset = Dataset()
    
    # 估算场景边界
    print("\n2. 分析场景边界...")
    all_cameras = dataset.scene_info.train_cameras + dataset.scene_info.test_cameras
    positions = [cam.T[:2] for cam in all_cameras]
    
    import numpy as np
    positions = np.array(positions)
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)
    
    # 添加缓冲区
    margin = 20.0
    scene_bounds = (min_x - margin, max_x + margin, min_y - margin, max_y + margin)
    
    print(f"场景边界: {scene_bounds}")
    print(f"场景大小: {max_x - min_x:.1f}m x {max_y - min_y:.1f}m")
    
    # 创建分块配置
    print("\n3. 创建分块配置...")
    block_configs = create_spatial_blocks(
        scene_bounds=scene_bounds,
        block_size=block_size,
        overlap_margin=overlap_margin
    )
    
    # 保存块配置
    blocks_config_path = os.path.join(model_path, 'example_blocks_config.json')
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
    
    print(f"块配置保存到: {blocks_config_path}")
    
    # 创建分块训练器
    print("\n4. 初始化分块训练器...")
    block_trainer = BlockTrainer(dataset=dataset, block_configs=block_configs)
    
    # 显示相机分配统计
    print("\n相机分配统计:")
    for block_id in range(len(block_configs)):
        train_count = len(block_trainer.block_cameras[block_id]['train'])
        test_count = len(block_trainer.block_cameras[block_id]['test'])
        print(f"  块 {block_id}: {train_count} 训练相机, {test_count} 测试相机")
    
    # 选择性训练（训练前几个块作为示例）
    print("\n5. 开始分块训练（示例：仅训练前2个块）...")
    
    # 为了演示，我们只训练前2个块
    demo_blocks = block_configs[:2]
    
    for block_config in demo_blocks:
        print(f"\n训练块 {block_config.block_id}...")
        try:
            # 使用较少的迭代次数进行演示
            trained_model = block_trainer.train_block(block_config, iterations=5000)
            print(f"块 {block_config.block_id} 训练完成")
        except Exception as e:
            print(f"块 {block_config.block_id} 训练失败: {e}")
            continue
    
    # 聚合示例
    print("\n6. 聚合已训练的块...")
    try:
        # 创建聚合器
        aggregator = GaussianBlockAggregator(
            block_trainer=block_trainer,
            spatial_threshold=0.2,  # 20cm 空间阈值
            opacity_threshold=0.1
        )
        
        # 由于我们只训练了部分块，这里会出现警告
        # 在实际应用中，应该训练所有块
        aggregated_model = aggregator.aggregate_blocks()
        
        # 保存聚合结果
        aggregated_save_path = os.path.join(model_path, 'example_aggregated')
        checkpoint_path, ply_path = aggregator.save_aggregated_model(
            aggregated_model, aggregated_save_path
        )
        
        print(f"\n聚合完成!")
        print(f"聚合模型保存到: {aggregated_save_path}")
        
    except Exception as e:
        print(f"聚合失败: {e}")
        print("这可能是因为只训练了部分块，请训练所有块后再进行聚合")
    
    print("\n=== 示例完成 ===")

def example_custom_blocks():
    """自定义块配置示例"""
    print("\n=== 自定义块配置示例 ===")
    
    # 手动定义块配置
    custom_blocks = [
        {
            "block_id": 0,
            "spatial_bounds": [-100, -20, -80, 80],
            "overlap_margin": 8.0
        },
        {
            "block_id": 1,
            "spatial_bounds": [-28, 50, -80, 80],
            "overlap_margin": 8.0
        },
        {
            "block_id": 2,
            "spatial_bounds": [42, 120, -80, 80],
            "overlap_margin": 8.0
        }
    ]
    
    # 保存自定义块配置
    custom_config_path = "./custom_blocks_example.json"
    with open(custom_config_path, 'w') as f:
        json.dump(custom_blocks, f, indent=2)
    
    print(f"自定义块配置已保存到: {custom_config_path}")
    print("使用方法:")
    print(f"python train_blocks.py --config configs/block_training_example.yaml "
          f"--source_path /path/to/dataset --model_path ./outputs/custom "
          f"--manual_blocks {custom_config_path}")

def example_performance_tuning():
    """性能调优示例"""
    print("\n=== 性能调优建议 ===")
    
    print("1. 内存优化:")
    print("   - 减少块大小: --block_size 30")
    print("   - 增加不透明度阈值: --opacity_threshold 0.15")
    print("   - 使用更严格的空间阈值: --spatial_threshold 0.05")
    
    print("\n2. 训练效率:")
    print("   - 减少训练迭代: --iterations 10000")
    print("   - 在配置中提前停止密化: densify_until_iter: 8000")
    print("   - 增加梯度阈值以减少高斯球数量")
    
    print("\n3. 质量平衡:")
    print("   - 适中的重叠: --overlap_margin 5 到 10")
    print("   - 合理的块大小: 50-100m 取决于场景复杂度")
    print("   - 保持足够的训练迭代以确保收敛")

def main():
    """主函数"""
    print("Street Gaussian 分块训练系统示例")
    print("=" * 50)
    
    # 运行示例
    example_custom_blocks()
    example_performance_tuning()
    
    # 询问是否运行完整示例
    print("\n" + "=" * 50)
    response = input("是否运行完整的 Waymo 训练示例? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        print("注意: 请确保已正确设置数据集路径")
        example_waymo_block_training()
    else:
        print("跳过完整训练示例")
        print("\n要运行完整训练，请:")
        print("1. 修改 example_waymo_block_training() 中的数据集路径")
        print("2. 运行: python examples/run_block_training_example.py")
        
    print("\n示例脚本完成!")

if __name__ == "__main__":
    main()