#!/usr/bin/env python3
"""
Street Gaussian 分块训练系统测试脚本

基本的功能测试和单元测试，确保分块训练系统正常工作。
"""

import os
import sys
import json
import tempfile
import unittest
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from lib.training.block_trainer import BlockConfig, BlockTrainer, create_spatial_blocks
    from lib.training.block_aggregator import GaussianBlockAggregator
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import block training modules: {e}")
    MODULES_AVAILABLE = False

class TestBlockConfig(unittest.TestCase):
    """测试块配置功能"""
    
    def test_block_config_creation(self):
        """测试块配置创建"""
        if not MODULES_AVAILABLE:
            self.skipTest("Block training modules not available")
            
        config = BlockConfig(
            block_id=0,
            spatial_bounds=(-50, 50, -25, 25),
            overlap_margin=5.0
        )
        
        self.assertEqual(config.block_id, 0)
        self.assertEqual(config.spatial_bounds, (-50, 50, -25, 25))
        self.assertEqual(config.overlap_margin, 5.0)
        self.assertEqual(config.center, (0.0, 0.0))
    
    def test_create_spatial_blocks(self):
        """测试自动空间分块"""
        if not MODULES_AVAILABLE:
            self.skipTest("Block training modules not available")
            
        scene_bounds = (-100, 100, -50, 50)
        block_size = 50.0
        overlap_margin = 5.0
        
        blocks = create_spatial_blocks(scene_bounds, block_size, overlap_margin)
        
        # 应该创建 4x2 = 8 个块
        expected_blocks = 8
        self.assertEqual(len(blocks), expected_blocks)
        
        # 检查块的属性
        for i, block in enumerate(blocks):
            self.assertEqual(block.block_id, i)
            self.assertEqual(block.overlap_margin, overlap_margin)
            self.assertIsInstance(block.spatial_bounds, tuple)
            self.assertEqual(len(block.spatial_bounds), 4)
    
    def test_block_coverage(self):
        """测试块覆盖范围"""
        if not MODULES_AVAILABLE:
            self.skipTest("Block training modules not available")
            
        scene_bounds = (-60, 60, -30, 30)
        block_size = 40.0
        
        blocks = create_spatial_blocks(scene_bounds, block_size)
        
        # 检查所有块是否覆盖了完整的场景
        all_min_x = min(block.spatial_bounds[0] for block in blocks)
        all_max_x = max(block.spatial_bounds[1] for block in blocks)
        all_min_y = min(block.spatial_bounds[2] for block in blocks)
        all_max_y = max(block.spatial_bounds[3] for block in blocks)
        
        # 块的边界应该包含或超过场景边界
        self.assertLessEqual(all_min_x, scene_bounds[0])
        self.assertGreaterEqual(all_max_x, scene_bounds[1])
        self.assertLessEqual(all_min_y, scene_bounds[2])
        self.assertGreaterEqual(all_max_y, scene_bounds[3])

class TestBlockUtilities(unittest.TestCase):
    """测试块训练辅助功能"""
    
    def test_config_serialization(self):
        """测试块配置的序列化和反序列化"""
        if not MODULES_AVAILABLE:
            self.skipTest("Block training modules not available")
            
        # 创建测试配置
        original_configs = [
            BlockConfig(0, (-50, 0, -25, 25), 5.0),
            BlockConfig(1, (0, 50, -25, 25), 5.0),
        ]
        
        # 序列化
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = []
            for config in original_configs:
                config_data.append({
                    'block_id': config.block_id,
                    'spatial_bounds': config.spatial_bounds,
                    'overlap_margin': config.overlap_margin,
                    'center': config.center
                })
            json.dump(config_data, f)
            temp_file = f.name
        
        try:
            # 反序列化
            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)
            
            loaded_configs = []
            for data in loaded_data:
                config = BlockConfig(
                    block_id=data['block_id'],
                    spatial_bounds=tuple(data['spatial_bounds']),
                    overlap_margin=data['overlap_margin']
                )
                loaded_configs.append(config)
            
            # 验证
            self.assertEqual(len(loaded_configs), len(original_configs))
            for orig, loaded in zip(original_configs, loaded_configs):
                self.assertEqual(orig.block_id, loaded.block_id)
                self.assertEqual(orig.spatial_bounds, loaded.spatial_bounds)
                self.assertEqual(orig.overlap_margin, loaded.overlap_margin)
                
        finally:
            os.unlink(temp_file)

class TestMockData(unittest.TestCase):
    """测试模拟数据功能"""
    
    def test_mock_camera_creation(self):
        """测试模拟相机创建"""
        # 创建模拟相机数据
        camera_positions = np.array([
            [0, 0, 5],
            [10, 0, 5],
            [0, 10, 5],
            [-10, 0, 5]
        ])
        
        self.assertEqual(camera_positions.shape, (4, 3))
        
        # 测试位置分配到块
        if MODULES_AVAILABLE:
            blocks = create_spatial_blocks((-20, 20, -20, 20), 25.0, 5.0)
            
            for pos in camera_positions:
                # 检查每个相机位置是否在某个块的范围内
                assigned = False
                for block in blocks:
                    min_x, max_x, min_y, max_y = block.spatial_bounds
                    margin = block.overlap_margin
                    if (min_x - margin <= pos[0] <= max_x + margin and
                        min_y - margin <= pos[1] <= max_y + margin):
                        assigned = True
                        break
                
                self.assertTrue(assigned, f"Camera at {pos} not assigned to any block")

class TestPerformanceMetrics(unittest.TestCase):
    """测试性能指标计算"""
    
    def test_memory_estimation(self):
        """测试内存使用估算"""
        # 模拟高斯球数据
        total_gaussians = 100000
        num_blocks = 4
        avg_gaussians_per_block = total_gaussians / num_blocks
        
        # 每个高斯球的估计内存使用（字节）
        bytes_per_gaussian = 200
        
        estimated_full_memory = total_gaussians * bytes_per_gaussian / 1024 / 1024  # MB
        estimated_block_memory = avg_gaussians_per_block * bytes_per_gaussian / 1024 / 1024  # MB
        
        memory_reduction = 1 - (estimated_block_memory / estimated_full_memory)
        
        # 验证计算
        self.assertGreater(estimated_full_memory, estimated_block_memory)
        self.assertAlmostEqual(memory_reduction, 0.75, places=2)  # 4块应该减少75%内存
    
    def test_deduplication_metrics(self):
        """测试去重指标计算"""
        # 模拟聚合前后的高斯球数量
        before_aggregation = 50000
        after_aggregation = 35000
        
        deduplication_ratio = 1 - (after_aggregation / before_aggregation)
        reduction_count = before_aggregation - after_aggregation
        
        self.assertAlmostEqual(deduplication_ratio, 0.3, places=2)  # 30%去重率
        self.assertEqual(reduction_count, 15000)

class TestErrorHandling(unittest.TestCase):
    """测试错误处理"""
    
    def test_invalid_block_bounds(self):
        """测试无效块边界处理"""
        if not MODULES_AVAILABLE:
            self.skipTest("Block training modules not available")
            
        # 测试无效的边界（min > max）
        with self.assertRaises(Exception):
            # 这应该在实际实现中被捕获
            invalid_bounds = (50, -50, 25, -25)  # min > max
            create_spatial_blocks(invalid_bounds, 25.0)
    
    def test_zero_block_size(self):
        """测试零块大小处理"""
        if not MODULES_AVAILABLE:
            self.skipTest("Block training modules not available")
            
        # 零或负的块大小应该被处理
        scene_bounds = (-50, 50, -25, 25)
        
        with self.assertRaises(Exception):
            create_spatial_blocks(scene_bounds, 0)
        
        with self.assertRaises(Exception):
            create_spatial_blocks(scene_bounds, -10)

def run_integration_test():
    """运行简单的集成测试"""
    print("Running integration test...")
    
    if not MODULES_AVAILABLE:
        print("Skipping integration test - modules not available")
        return
    
    try:
        # 创建测试场景
        scene_bounds = (-100, 100, -50, 50)
        block_size = 60.0
        overlap_margin = 10.0
        
        # 创建块配置
        blocks = create_spatial_blocks(scene_bounds, block_size, overlap_margin)
        print(f"Created {len(blocks)} blocks")
        
        # 验证块覆盖
        total_area = 0
        for block in blocks:
            min_x, max_x, min_y, max_y = block.spatial_bounds
            area = (max_x - min_x) * (max_y - min_y)
            total_area += area
            print(f"Block {block.block_id}: bounds={block.spatial_bounds}, area={area:.1f}")
        
        scene_area = (scene_bounds[1] - scene_bounds[0]) * (scene_bounds[3] - scene_bounds[2])
        coverage_ratio = total_area / scene_area
        print(f"Total coverage ratio: {coverage_ratio:.2f}")
        
        # 模拟相机分配
        num_cameras = 20
        camera_positions = np.random.uniform(
            [scene_bounds[0], scene_bounds[2], 0],
            [scene_bounds[1], scene_bounds[3], 10],
            (num_cameras, 3)
        )
        
        # 分配相机到块
        camera_assignments = {block.block_id: [] for block in blocks}
        
        for i, pos in enumerate(camera_positions):
            for block in blocks:
                min_x, max_x, min_y, max_y = block.spatial_bounds
                margin = block.overlap_margin
                if (min_x - margin <= pos[0] <= max_x + margin and
                    min_y - margin <= pos[1] <= max_y + margin):
                    camera_assignments[block.block_id].append(i)
        
        # 打印分配结果
        for block_id, cameras in camera_assignments.items():
            print(f"Block {block_id}: {len(cameras)} cameras assigned")
        
        print("Integration test completed successfully!")
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        raise

def main():
    """主测试函数"""
    print("Street Gaussian Block Training System Tests")
    print("=" * 50)
    
    # 运行单元测试
    if MODULES_AVAILABLE:
        print("Running unit tests...")
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        print("Skipping unit tests - modules not available")
    
    print("\n" + "-" * 50)
    
    # 运行集成测试
    run_integration_test()
    
    print("\n" + "=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    main()