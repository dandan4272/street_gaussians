import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
import copy
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.gaussian_model import GaussianModel
from lib.models.gaussian_model_bkgd import GaussianModelBkgd
from lib.models.gaussian_model_actor import GaussianModelActor
from lib.training.block_trainer import BlockConfig, BlockTrainer
from lib.utils.graphics_utils import BasicPointCloud
from lib.config import cfg
from lib.utils.general_utils import quaternion_raw_multiply, matrix_to_quaternion
import trimesh

class GaussianBlockAggregator:
    """高斯分块模型聚合器"""
    
    def __init__(self, 
                 block_trainer: BlockTrainer,
                 overlap_threshold: float = 0.5,
                 spatial_threshold: float = 0.1,
                 opacity_threshold: float = 0.1):
        """
        Args:
            block_trainer: 分块训练器
            overlap_threshold: 重叠区域阈值
            spatial_threshold: 空间距离阈值（米）
            opacity_threshold: 不透明度阈值，低于此值的高斯球将被移除
        """
        self.block_trainer = block_trainer
        self.overlap_threshold = overlap_threshold
        self.spatial_threshold = spatial_threshold
        self.opacity_threshold = opacity_threshold
        
        # 加载所有训练好的模型
        self._load_all_block_models()
        
    def _load_all_block_models(self):
        """加载所有训练好的块模型"""
        print("Loading all block models...")
        self.block_models = {}
        
        for block_config in self.block_trainer.block_configs:
            try:
                model = self.block_trainer.load_block_checkpoint(block_config.block_id)
                self.block_models[block_config.block_id] = model
                print(f"Loaded block {block_config.block_id}")
            except Exception as e:
                print(f"Failed to load block {block_config.block_id}: {e}")
                
        print(f"Successfully loaded {len(self.block_models)} block models")
    
    def aggregate_blocks(self) -> StreetGaussianModel:
        """聚合所有块模型"""
        print("\n=== Starting block aggregation ===")
        
        # 创建聚合后的模型
        aggregated_model = self._create_aggregated_model()
        
        # 分别聚合不同组件
        if aggregated_model.include_background:
            print("Aggregating background models...")
            self._aggregate_background_models(aggregated_model)
            
        if aggregated_model.include_obj:
            print("Aggregating object models...")
            self._aggregate_object_models(aggregated_model)
            
        if aggregated_model.include_sky:
            print("Aggregating sky models...")
            self._aggregate_sky_models(aggregated_model)
            
        # 最终清理和优化
        print("Final cleanup...")
        self._final_cleanup(aggregated_model)
        
        print("=== Block aggregation completed ===")
        return aggregated_model
    
    def _create_aggregated_model(self) -> StreetGaussianModel:
        """创建聚合模型的框架"""
        # 使用第一个块的元数据作为基础
        first_block_id = list(self.block_models.keys())[0]
        first_model = self.block_models[first_block_id]
        
        # 创建全局元数据
        global_metadata = copy.deepcopy(first_model.metadata)
        
        # 计算全局场景边界
        all_bounds = []
        for block_config in self.block_trainer.block_configs:
            all_bounds.append(block_config.spatial_bounds)
            
        if all_bounds:
            min_x = min(bounds[0] for bounds in all_bounds)
            max_x = max(bounds[1] for bounds in all_bounds)
            min_y = min(bounds[2] for bounds in all_bounds)
            max_y = max(bounds[3] for bounds in all_bounds)
            
            global_center = np.array([
                (min_x + max_x) / 2,
                (min_y + max_y) / 2,
                global_metadata['scene_center'][2]
            ])
            global_radius = max(max_x - min_x, max_y - min_y) / 2
            
            global_metadata['scene_center'] = global_center
            global_metadata['scene_radius'] = global_radius
            global_metadata['global_bounds'] = (min_x, max_x, min_y, max_y)
        
        # 创建聚合模型
        aggregated_model = StreetGaussianModel(global_metadata)
        return aggregated_model
    
    def _aggregate_background_models(self, aggregated_model: StreetGaussianModel):
        """聚合背景模型"""
        background_data = self._collect_component_data('background')
        if not background_data:
            print("No background data to aggregate")
            return
            
        # 合并所有背景数据
        merged_data = self._merge_gaussian_data(background_data)
        
        # 去除重复点
        cleaned_data = self._remove_duplicate_gaussians(merged_data)
        
        # 创建新的背景模型
        aggregated_model.background = GaussianModelBkgd(
            model_name='background',
            scene_center=aggregated_model.metadata['scene_center'],
            scene_radius=aggregated_model.metadata['scene_radius'],
            sphere_center=aggregated_model.metadata.get('sphere_center', aggregated_model.metadata['scene_center']),
            sphere_radius=aggregated_model.metadata.get('sphere_radius', aggregated_model.metadata['scene_radius'])
        )
        
        # 设置聚合后的数据
        self._set_gaussian_model_data(aggregated_model.background, cleaned_data)
        
        print(f"Background aggregation: {len(cleaned_data['xyz'])} gaussians")
    
    def _aggregate_object_models(self, aggregated_model: StreetGaussianModel):
        """聚合对象模型"""
        # 收集所有对象
        all_objects = {}
        
        for block_id, model in self.block_models.items():
            for obj_name in model.obj_list:
                if obj_name not in all_objects:
                    all_objects[obj_name] = []
                all_objects[obj_name].append((block_id, getattr(model, obj_name)))
        
        # 为每个对象聚合数据
        for obj_name, obj_models in all_objects.items():
            print(f"Aggregating object {obj_name}...")
            
            # 收集对象数据
            obj_data = []
            for block_id, obj_model in obj_models:
                data = self._extract_gaussian_model_data(obj_model)
                # 添加块ID信息用于去重
                data['block_id'] = np.full(len(data['xyz']), block_id)
                obj_data.append(data)
            
            # 合并和清理
            merged_data = self._merge_gaussian_data(obj_data)
            cleaned_data = self._remove_duplicate_gaussians(merged_data, use_spatial_clustering=True)
            
            # 创建聚合后的对象模型
            # 使用第一个对象模型的元数据
            first_obj_model = obj_models[0][1]
            aggregated_obj = GaussianModelActor(
                model_name=obj_name,
                obj_meta=first_obj_model.obj_meta
            )
            
            # 设置数据
            self._set_gaussian_model_data(aggregated_obj, cleaned_data)
            
            # 添加到聚合模型
            setattr(aggregated_model, obj_name, aggregated_obj)
            if obj_name not in aggregated_model.obj_list:
                aggregated_model.obj_list.append(obj_name)
            
            print(f"Object {obj_name} aggregation: {len(cleaned_data['xyz'])} gaussians")
    
    def _aggregate_sky_models(self, aggregated_model: StreetGaussianModel):
        """聚合天空模型"""
        if not aggregated_model.include_sky:
            return
            
        # 对于天空模型，我们选择质量最好的一个
        # 或者可以平均多个天空模型的参数
        sky_models = []
        for block_id, model in self.block_models.items():
            if hasattr(model, 'sky_cubemap') and model.sky_cubemap is not None:
                sky_models.append(model.sky_cubemap)
        
        if sky_models:
            # 使用第一个天空模型作为基础
            # 在更复杂的实现中，可以平均多个天空模型
            aggregated_model.sky_cubemap = copy.deepcopy(sky_models[0])
            print(f"Sky aggregation: using sky model from first available block")
    
    def _collect_component_data(self, component_name: str) -> List[Dict]:
        """收集指定组件的数据"""
        component_data = []
        
        for block_id, model in self.block_models.items():
            if hasattr(model, component_name):
                component_model = getattr(model, component_name)
                data = self._extract_gaussian_model_data(component_model)
                # 添加块信息
                data['block_id'] = np.full(len(data['xyz']), block_id)
                data['block_bounds'] = next(
                    config.spatial_bounds for config in self.block_trainer.block_configs 
                    if config.block_id == block_id
                )
                component_data.append(data)
                
        return component_data
    
    def _extract_gaussian_model_data(self, model: GaussianModel) -> Dict:
        """从高斯模型中提取数据"""
        try:
            data = {
                'xyz': model.get_xyz.detach().cpu().numpy(),
                'features': model.get_features.detach().cpu().numpy(),
                'scaling': model.get_scaling.detach().cpu().numpy(),
                'rotation': model.get_rotation.detach().cpu().numpy(),
                'opacity': model.get_opacity.detach().cpu().numpy(),
            }
            
            # 如果有语义信息
            if hasattr(model, 'get_semantic'):
                data['semantic'] = model.get_semantic.detach().cpu().numpy()
                
            return data
        except Exception as e:
            print(f"Error extracting data from model: {e}")
            return {
                'xyz': np.empty((0, 3)),
                'features': np.empty((0, 0)),
                'scaling': np.empty((0, 3)),
                'rotation': np.empty((0, 4)),
                'opacity': np.empty((0, 1))
            }
    
    def _merge_gaussian_data(self, data_list: List[Dict]) -> Dict:
        """合并多个高斯数据"""
        if not data_list:
            return {}
            
        merged = {}
        
        # 获取所有键
        all_keys = set()
        for data in data_list:
            all_keys.update(data.keys())
        
        # 合并每个键的数据
        for key in all_keys:
            arrays = []
            for data in data_list:
                if key in data and len(data[key]) > 0:
                    arrays.append(data[key])
            
            if arrays:
                if key == 'block_bounds':
                    # 对于边界信息，保留所有
                    merged[key] = arrays
                else:
                    merged[key] = np.concatenate(arrays, axis=0)
            else:
                merged[key] = np.empty((0,))
                
        return merged
    
    def _remove_duplicate_gaussians(self, data: Dict, use_spatial_clustering: bool = False) -> Dict:
        """去除重复的高斯球"""
        if len(data['xyz']) == 0:
            return data
            
        print(f"Removing duplicates from {len(data['xyz'])} gaussians...")
        
        # 方法1：基于空间位置去重
        if use_spatial_clustering:
            keep_indices = self._spatial_clustering_deduplication(data)
        else:
            keep_indices = self._overlap_based_deduplication(data)
        
        # 应用过滤
        filtered_data = {}
        for key, values in data.items():
            if key in ['block_bounds']:
                filtered_data[key] = values  # 保持不变
            elif isinstance(values, np.ndarray) and len(values) > 0:
                filtered_data[key] = values[keep_indices]
            else:
                filtered_data[key] = values
                
        print(f"After deduplication: {len(filtered_data['xyz'])} gaussians")
        return filtered_data
    
    def _spatial_clustering_deduplication(self, data: Dict) -> np.ndarray:
        """基于空间聚类的去重"""
        xyz = data['xyz']
        opacity = data['opacity']
        
        # 移除低不透明度的点
        valid_mask = opacity.squeeze() > self.opacity_threshold
        
        if not np.any(valid_mask):
            return np.array([], dtype=int)
            
        valid_xyz = xyz[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_xyz) <= 1:
            return valid_indices
            
        # 使用KNN找到近邻
        nbrs = NearestNeighbors(n_neighbors=min(10, len(valid_xyz)), 
                              radius=self.spatial_threshold).fit(valid_xyz)
        
        # 找到所有近邻对
        distances, indices = nbrs.kneighbors(valid_xyz)
        
        # 标记要保留的点
        keep_mask = np.ones(len(valid_xyz), dtype=bool)
        
        for i in range(len(valid_xyz)):
            if not keep_mask[i]:
                continue
                
            # 找到距离阈值内的邻居
            neighbors = indices[i][distances[i] < self.spatial_threshold]
            
            if len(neighbors) > 1:
                # 在邻居中选择不透明度最高的
                neighbor_opacities = opacity[valid_indices[neighbors]]
                best_neighbor = neighbors[np.argmax(neighbor_opacities)]
                
                # 移除其他邻居
                for neighbor in neighbors:
                    if neighbor != best_neighbor:
                        keep_mask[neighbor] = False
        
        return valid_indices[keep_mask]
    
    def _overlap_based_deduplication(self, data: Dict) -> np.ndarray:
        """基于重叠区域的去重"""
        if 'block_id' not in data:
            # 如果没有块信息，使用基本的空间去重
            return self._basic_spatial_deduplication(data)
            
        xyz = data['xyz']
        block_ids = data['block_id']
        opacity = data['opacity']
        
        # 移除低不透明度的点
        valid_mask = opacity.squeeze() > self.opacity_threshold
        keep_indices = []
        
        # 按块分组处理
        unique_blocks = np.unique(block_ids[valid_mask])
        
        for block_id in unique_blocks:
            block_mask = (block_ids == block_id) & valid_mask
            block_indices = np.where(block_mask)[0]
            
            if len(block_indices) == 0:
                continue
                
            # 确定哪些点在重叠区域
            block_config = next(
                config for config in self.block_trainer.block_configs 
                if config.block_id == block_id
            )
            
            overlap_mask = self._get_overlap_region_mask(xyz[block_indices], block_config)
            
            # 非重叠区域的点直接保留
            non_overlap_indices = block_indices[~overlap_mask]
            keep_indices.extend(non_overlap_indices)
            
            # 重叠区域的点需要进一步处理
            overlap_indices = block_indices[overlap_mask]
            if len(overlap_indices) > 0:
                # 在重叠区域，保留不透明度最高的点
                overlap_xyz = xyz[overlap_indices]
                overlap_opacity = opacity[overlap_indices]
                
                # 使用空间聚类
                if len(overlap_xyz) > 0:
                    filtered_overlap = self._cluster_overlap_gaussians(
                        overlap_indices, overlap_xyz, overlap_opacity
                    )
                    keep_indices.extend(filtered_overlap)
        
        return np.array(keep_indices, dtype=int)
    
    def _get_overlap_region_mask(self, xyz: np.ndarray, block_config: BlockConfig) -> np.ndarray:
        """获取重叠区域的掩码"""
        min_x, max_x, min_y, max_y = block_config.spatial_bounds
        margin = block_config.overlap_margin
        
        # 检查点是否在重叠边界内
        overlap_mask = (
            (xyz[:, 0] < min_x + margin) |  # 左边界
            (xyz[:, 0] > max_x - margin) |  # 右边界
            (xyz[:, 1] < min_y + margin) |  # 下边界
            (xyz[:, 1] > max_y - margin)    # 上边界
        )
        
        return overlap_mask
    
    def _cluster_overlap_gaussians(self, indices: np.ndarray, xyz: np.ndarray, opacity: np.ndarray) -> List[int]:
        """聚类重叠区域的高斯球"""
        if len(xyz) <= 1:
            return indices.tolist()
            
        # 使用简单的贪心算法
        keep_indices = []
        remaining_indices = list(range(len(xyz)))
        
        while remaining_indices:
            # 选择不透明度最高的点
            current_idx = remaining_indices[0]
            max_opacity = opacity[current_idx]
            max_idx_in_remaining = 0
            
            for i, rem_idx in enumerate(remaining_indices):
                if opacity[rem_idx] > max_opacity:
                    max_opacity = opacity[rem_idx]
                    current_idx = rem_idx
                    max_idx_in_remaining = i
            
            # 添加到保留列表
            keep_indices.append(indices[current_idx])
            remaining_indices.pop(max_idx_in_remaining)
            
            # 移除附近的点
            current_pos = xyz[current_idx]
            remaining_indices = [
                idx for idx in remaining_indices
                if np.linalg.norm(xyz[idx] - current_pos) >= self.spatial_threshold
            ]
        
        return keep_indices
    
    def _basic_spatial_deduplication(self, data: Dict) -> np.ndarray:
        """基本的空间去重"""
        xyz = data['xyz']
        opacity = data['opacity']
        
        # 移除低不透明度的点
        valid_mask = opacity.squeeze() > self.opacity_threshold
        
        if not np.any(valid_mask):
            return np.array([], dtype=int)
            
        return np.where(valid_mask)[0]
    
    def _set_gaussian_model_data(self, model: GaussianModel, data: Dict):
        """设置高斯模型的数据"""
        if len(data['xyz']) == 0:
            print("Warning: No data to set for model")
            return
            
        # 将数据转换为tensor并设置到模型
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 设置基本属性
        model._xyz = torch.tensor(data['xyz'], dtype=torch.float32, device=device, requires_grad=True)
        model._features_dc = torch.tensor(data['features'][:, :3].reshape(-1, 1, 3), 
                                        dtype=torch.float32, device=device, requires_grad=True)
        if data['features'].shape[1] > 3:
            model._features_rest = torch.tensor(data['features'][:, 3:].reshape(-1, -1, 3), 
                                              dtype=torch.float32, device=device, requires_grad=True)
        
        model._scaling = torch.tensor(data['scaling'], dtype=torch.float32, device=device, requires_grad=True)
        model._rotation = torch.tensor(data['rotation'], dtype=torch.float32, device=device, requires_grad=True)
        model._opacity = torch.tensor(data['opacity'], dtype=torch.float32, device=device, requires_grad=True)
        
        if 'semantic' in data and len(data['semantic']) > 0:
            model._semantic = torch.tensor(data['semantic'], dtype=torch.float32, device=device, requires_grad=True)
        
        print(f"Set {len(data['xyz'])} gaussians to model")
    
    def _final_cleanup(self, aggregated_model: StreetGaussianModel):
        """最终清理和优化"""
        # 重新设置模型的内部状态
        aggregated_model.setup_functions()
        
        # 可以在这里添加额外的优化步骤
        # 例如：重新优化重叠区域，调整不透明度等
        
        print("Final cleanup completed")
    
    def save_aggregated_model(self, aggregated_model: StreetGaussianModel, save_path: str):
        """保存聚合后的模型"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型状态
        checkpoint_path = os.path.join(save_path, "aggregated_model.pth")
        state_dict = aggregated_model.save_state_dict(is_final=True)
        torch.save(state_dict, checkpoint_path)
        
        # 保存点云
        ply_path = os.path.join(save_path, "aggregated_point_cloud.ply")
        aggregated_model.save_ply(ply_path)
        
        # 保存元数据
        metadata_path = os.path.join(save_path, "metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            # 转换numpy数组为列表以便JSON序列化
            metadata_copy = copy.deepcopy(aggregated_model.metadata)
            for key, value in metadata_copy.items():
                if isinstance(value, np.ndarray):
                    metadata_copy[key] = value.tolist()
            json.dump(metadata_copy, f, indent=2)
        
        print(f"Aggregated model saved to {save_path}")
        
        return checkpoint_path, ply_path