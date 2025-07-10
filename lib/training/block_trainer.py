import os
import torch
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.utils.camera_utils import Camera
from lib.config import cfg
from lib.utils.general_utils import safe_state
import copy

@dataclass
class BlockConfig:
    """配置单个训练块的参数"""
    block_id: int
    spatial_bounds: Tuple[float, float, float, float]  # (min_x, max_x, min_y, max_y)
    overlap_margin: float = 2.0  # 重叠边界的宽度（米）
    center: Tuple[float, float] = None  # 块的中心点
    
    def __post_init__(self):
        if self.center is None:
            min_x, max_x, min_y, max_y = self.spatial_bounds
            self.center = ((min_x + max_x) / 2, (min_y + max_y) / 2)

class BlockTrainer:
    """分块训练管理器"""
    
    def __init__(self, dataset: Dataset, block_configs: List[BlockConfig]):
        self.dataset = dataset
        self.block_configs = block_configs
        self.trained_models = {}
        self.block_cameras = {}
        
        # 为每个块分配相机
        self._assign_cameras_to_blocks()
        
    def _assign_cameras_to_blocks(self):
        """根据空间位置将相机分配到对应的块"""
        print("Assigning cameras to blocks...")
        
        for block_config in self.block_configs:
            self.block_cameras[block_config.block_id] = {
                'train': [],
                'test': []
            }
            
        # 分配训练相机
        for camera in self.dataset.scene_info.train_cameras:
            assigned_blocks = self._get_blocks_for_camera(camera)
            for block_id in assigned_blocks:
                self.block_cameras[block_id]['train'].append(camera)
                
        # 分配测试相机
        for camera in self.dataset.scene_info.test_cameras:
            assigned_blocks = self._get_blocks_for_camera(camera)
            for block_id in assigned_blocks:
                self.block_cameras[block_id]['test'].append(camera)
                
        # 打印统计信息
        for block_id in self.block_cameras:
            train_count = len(self.block_cameras[block_id]['train'])
            test_count = len(self.block_cameras[block_id]['test'])
            print(f"Block {block_id}: {train_count} train cameras, {test_count} test cameras")
    
    def _get_blocks_for_camera(self, camera) -> List[int]:
        """根据相机位置确定它属于哪些块"""
        camera_pos = camera.T  # 相机位置
        assigned_blocks = []
        
        for block_config in self.block_configs:
            min_x, max_x, min_y, max_y = block_config.spatial_bounds
            # 添加重叠边界
            margin = block_config.overlap_margin
            extended_bounds = (min_x - margin, max_x + margin, min_y - margin, max_y + margin)
            
            if (extended_bounds[0] <= camera_pos[0] <= extended_bounds[1] and 
                extended_bounds[2] <= camera_pos[1] <= extended_bounds[3]):
                assigned_blocks.append(block_config.block_id)
                
        return assigned_blocks
    
    def _filter_point_cloud_for_block(self, block_config: BlockConfig):
        """为特定块过滤点云数据"""
        original_pcd = self.dataset.scene_info.point_cloud
        
        # 获取块的边界（包含重叠区域）
        min_x, max_x, min_y, max_y = block_config.spatial_bounds
        margin = block_config.overlap_margin
        extended_bounds = (min_x - margin, max_x + margin, min_y - margin, max_y + margin)
        
        # 过滤点云
        points = original_pcd.points
        colors = original_pcd.colors
        normals = original_pcd.normals if hasattr(original_pcd, 'normals') else None
        
        # 根据空间位置过滤
        mask = ((points[:, 0] >= extended_bounds[0]) & 
                (points[:, 0] <= extended_bounds[1]) & 
                (points[:, 1] >= extended_bounds[2]) & 
                (points[:, 1] <= extended_bounds[3]))
        
        filtered_points = points[mask]
        filtered_colors = colors[mask]
        filtered_normals = normals[mask] if normals is not None else None
        
        # 创建过滤后的点云
        from lib.utils.graphics_utils import BasicPointCloud
        filtered_pcd = BasicPointCloud(
            points=filtered_points,
            colors=filtered_colors,
            normals=filtered_normals
        )
        
        print(f"Block {block_config.block_id}: filtered {len(filtered_points)} points from {len(points)} total points")
        return filtered_pcd
    
    def _create_block_dataset(self, block_config: BlockConfig) -> Dataset:
        """为特定块创建数据集"""
        # 复制原始数据集
        block_dataset = copy.deepcopy(self.dataset)
        
        # 替换点云
        block_dataset.scene_info.point_cloud = self._filter_point_cloud_for_block(block_config)
        
        # 替换相机列表
        block_dataset.scene_info.train_cameras = self.block_cameras[block_config.block_id]['train']
        block_dataset.scene_info.test_cameras = self.block_cameras[block_config.block_id]['test']
        
        # 更新场景元数据
        block_dataset.scene_info.metadata = self._update_metadata_for_block(
            block_dataset.scene_info.metadata, block_config
        )
        
        return block_dataset
    
    def _update_metadata_for_block(self, metadata: Dict, block_config: BlockConfig) -> Dict:
        """更新块的元数据"""
        block_metadata = copy.deepcopy(metadata)
        
        # 更新场景中心和半径
        min_x, max_x, min_y, max_y = block_config.spatial_bounds
        block_center = np.array([
            (min_x + max_x) / 2,
            (min_y + max_y) / 2,
            block_metadata['scene_center'][2]  # 保持原始的z坐标
        ])
        
        block_radius = max(max_x - min_x, max_y - min_y) / 2 + block_config.overlap_margin
        
        block_metadata['scene_center'] = block_center
        block_metadata['scene_radius'] = block_radius
        block_metadata['block_id'] = block_config.block_id
        block_metadata['spatial_bounds'] = block_config.spatial_bounds
        
        return block_metadata
    
    def train_block(self, block_config: BlockConfig, iterations: int = None) -> StreetGaussianModel:
        """训练单个块"""
        print(f"\n=== Training Block {block_config.block_id} ===")
        print(f"Spatial bounds: {block_config.spatial_bounds}")
        
        # 创建块数据集
        block_dataset = self._create_block_dataset(block_config)
        
        # 创建高斯模型
        gaussians = StreetGaussianModel(block_dataset.scene_info.metadata)
        
        # 创建场景
        scene = Scene(gaussians=gaussians, dataset=block_dataset)
        
        # 设置训练参数
        gaussians.training_setup()
        
        # 导入训练函数
        from lib.training.block_train_loop import train_single_block
        
        # 训练模型
        trained_model = train_single_block(
            scene=scene,
            gaussians=gaussians,
            block_config=block_config,
            iterations=iterations or cfg.train.iterations
        )
        
        # 保存训练好的模型
        self.trained_models[block_config.block_id] = trained_model
        
        # 保存检查点
        self._save_block_checkpoint(block_config.block_id, trained_model)
        
        return trained_model
    
    def train_all_blocks(self, iterations: int = None):
        """训练所有块"""
        print(f"\n=== Starting block training for {len(self.block_configs)} blocks ===")
        
        for block_config in self.block_configs:
            self.train_block(block_config, iterations)
            
        print("\n=== All blocks training completed ===")
    
    def _save_block_checkpoint(self, block_id: int, model: StreetGaussianModel):
        """保存块的检查点"""
        block_dir = os.path.join(cfg.model_path, f"block_{block_id}")
        os.makedirs(block_dir, exist_ok=True)
        
        # 保存模型状态
        checkpoint_path = os.path.join(block_dir, "model_final.pth")
        state_dict = model.save_state_dict(is_final=True)
        torch.save(state_dict, checkpoint_path)
        
        # 保存点云
        ply_path = os.path.join(block_dir, "point_cloud.ply")
        model.save_ply(ply_path)
        
        print(f"Block {block_id} checkpoint saved to {block_dir}")
    
    def load_block_checkpoint(self, block_id: int) -> StreetGaussianModel:
        """加载块的检查点"""
        block_dir = os.path.join(cfg.model_path, f"block_{block_id}")
        checkpoint_path = os.path.join(block_dir, "model_final.pth")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint for block {block_id} not found at {checkpoint_path}")
        
        # 创建块数据集和模型
        block_config = next(config for config in self.block_configs if config.block_id == block_id)
        block_dataset = self._create_block_dataset(block_config)
        gaussians = StreetGaussianModel(block_dataset.scene_info.metadata)
        
        # 加载状态
        state_dict = torch.load(checkpoint_path)
        gaussians.load_state_dict(state_dict)
        
        self.trained_models[block_id] = gaussians
        return gaussians

def create_spatial_blocks(scene_bounds: Tuple[float, float, float, float], 
                         block_size: float, 
                         overlap_margin: float = 2.0) -> List[BlockConfig]:
    """自动创建空间分块配置
    
    Args:
        scene_bounds: (min_x, max_x, min_y, max_y) 场景边界
        block_size: 每个块的大小（米）
        overlap_margin: 重叠边界宽度（米）
    """
    min_x, max_x, min_y, max_y = scene_bounds
    
    # 计算需要多少个块
    x_blocks = int(np.ceil((max_x - min_x) / block_size))
    y_blocks = int(np.ceil((max_y - min_y) / block_size))
    
    block_configs = []
    block_id = 0
    
    for i in range(x_blocks):
        for j in range(y_blocks):
            # 计算块的边界
            block_min_x = min_x + i * block_size
            block_max_x = min(min_x + (i + 1) * block_size, max_x)
            block_min_y = min_y + j * block_size
            block_max_y = min(min_y + (j + 1) * block_size, max_y)
            
            block_config = BlockConfig(
                block_id=block_id,
                spatial_bounds=(block_min_x, block_max_x, block_min_y, block_max_y),
                overlap_margin=overlap_margin
            )
            
            block_configs.append(block_config)
            block_id += 1
    
    print(f"Created {len(block_configs)} blocks ({x_blocks}x{y_blocks}) with size {block_size}m and overlap {overlap_margin}m")
    return block_configs