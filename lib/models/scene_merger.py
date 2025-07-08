import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.gaussian_model import GaussianModel
from lib.models.chunked_trainer import ChunkInfo, GaussianContribution
from lib.config import cfg
from lib.utils.general_utils import quaternion_to_matrix, matrix_to_quaternion

class SceneMerger:
    """场景拼接器 - 将多个训练块拼接成完整场景"""
    
    def __init__(self, merge_threshold: float = 0.5, duplicate_threshold: float = 0.1):
        self.merge_threshold = merge_threshold
        self.duplicate_threshold = duplicate_threshold
        
    def merge_chunks(self, chunk_models: Dict[int, StreetGaussianModel], 
                    chunk_infos: List[ChunkInfo],
                    metadata: dict) -> StreetGaussianModel:
        """将多个训练块拼接成完整场景
        
        Args:
            chunk_models: 训练好的块模型字典
            chunk_infos: 块信息列表  
            metadata: 场景元数据
            
        Returns:
            merged_model: 拼接后的完整场景模型
        """
        print("\n=== Starting Scene Merging ===")
        
        # 创建新的合并模型
        merged_model = StreetGaussianModel(metadata)
        
        # 收集所有块的高斯基元数据
        all_gaussian_data = {}
        
        # 按模型组件分别处理
        for model_name in ['background']:  # 先处理背景
            if model_name in chunk_models[0].model_name_id:
                print(f"Merging {model_name} gaussians...")
                merged_data = self._merge_model_component(
                    chunk_models, chunk_infos, model_name
                )
                all_gaussian_data[model_name] = merged_data
        
        # 处理物体模型
        all_obj_names = set()
        for chunk_model in chunk_models.values():
            for model_name in chunk_model.model_name_id.keys():
                if model_name.startswith('obj_'):
                    all_obj_names.add(model_name)
                    
        for obj_name in all_obj_names:
            print(f"Merging {obj_name} gaussians...")
            merged_data = self._merge_model_component(
                chunk_models, chunk_infos, obj_name
            )
            if merged_data:  # 只有当有数据时才添加
                all_gaussian_data[obj_name] = merged_data
        
        # 创建合并后的模型组件
        self._create_merged_model(merged_model, all_gaussian_data, metadata)
        
        print(f"Scene merging completed. Total gaussians: {merged_model.get_xyz.shape[0] if hasattr(merged_model, 'get_xyz') else 0}")
        
        return merged_model
    
    def _merge_model_component(self, chunk_models: Dict[int, StreetGaussianModel],
                              chunk_infos: List[ChunkInfo], 
                              model_name: str) -> Optional[Dict]:
        """合并特定模型组件的高斯基元"""
        
        # 收集所有块中该组件的数据
        all_positions = []
        all_features_dc = []
        all_features_rest = []
        all_scaling = []
        all_rotation = []
        all_opacity = []
        all_semantic = []
        chunk_sources = []  # 记录每个高斯基元来自哪个块
        
        valid_chunks = []
        
        for chunk_id, chunk_model in chunk_models.items():
            if model_name in chunk_model.model_name_id:
                model = getattr(chunk_model, model_name)
                
                if hasattr(model, '_xyz') and model._xyz.shape[0] > 0:
                    all_positions.append(model._xyz.detach().cpu())
                    all_features_dc.append(model._features_dc.detach().cpu())
                    all_features_rest.append(model._features_rest.detach().cpu())
                    all_scaling.append(model._scaling.detach().cpu())
                    all_rotation.append(model._rotation.detach().cpu())
                    all_opacity.append(model._opacity.detach().cpu())
                    
                    if hasattr(model, '_semantic'):
                        all_semantic.append(model._semantic.detach().cpu())
                    
                    # 记录来源块
                    chunk_sources.extend([chunk_id] * model._xyz.shape[0])
                    valid_chunks.append(chunk_id)
        
        if len(all_positions) == 0:
            return None
            
        # 拼接所有数据
        positions = torch.cat(all_positions, dim=0)
        features_dc = torch.cat(all_features_dc, dim=0)
        features_rest = torch.cat(all_features_rest, dim=0)
        scaling = torch.cat(all_scaling, dim=0)
        rotation = torch.cat(all_rotation, dim=0)
        opacity = torch.cat(all_opacity, dim=0)
        
        if len(all_semantic) > 0:
            semantic = torch.cat(all_semantic, dim=0)
        else:
            semantic = None
            
        print(f"  Before deduplication: {positions.shape[0]} gaussians")
        
        # 去除重复的高斯基元
        keep_mask = self._remove_duplicates(positions, chunk_sources, chunk_infos)
        
        if keep_mask.sum() == 0:
            print(f"  Warning: All gaussians were removed during deduplication for {model_name}")
            return None
            
        # 应用去重掩码
        positions = positions[keep_mask]
        features_dc = features_dc[keep_mask]
        features_rest = features_rest[keep_mask]
        scaling = scaling[keep_mask]
        rotation = rotation[keep_mask]
        opacity = opacity[keep_mask]
        
        if semantic is not None:
            semantic = semantic[keep_mask]
            
        print(f"  After deduplication: {positions.shape[0]} gaussians")
        
        return {
            'positions': positions,
            'features_dc': features_dc,
            'features_rest': features_rest,
            'scaling': scaling,
            'rotation': rotation,
            'opacity': opacity,
            'semantic': semantic
        }
    
    def _remove_duplicates(self, positions: torch.Tensor, 
                          chunk_sources: List[int],
                          chunk_infos: List[ChunkInfo]) -> torch.Tensor:
        """移除重复的高斯基元"""
        
        positions_np = positions.numpy()
        chunk_sources_np = np.array(chunk_sources)
        
        # 使用KDTree找到距离较近的点
        tree = cKDTree(positions_np)
        
        # 找到所有距离小于阈值的点对
        close_pairs = tree.query_pairs(r=self.duplicate_threshold)
        
        # 决定保留哪些点
        keep_mask = torch.ones(len(positions), dtype=torch.bool)
        
        for i, j in close_pairs:
            if keep_mask[i] and keep_mask[j]:
                # 如果两个点来自相邻的块，保留来自重叠区域的点
                chunk_i = chunk_sources_np[i]
                chunk_j = chunk_sources_np[j]
                
                # 检查是否来自相邻块
                if abs(chunk_i - chunk_j) == 1:
                    # 保留来自较晚块的点（更完整的训练）
                    if chunk_i < chunk_j:
                        keep_mask[i] = False
                    else:
                        keep_mask[j] = False
                else:
                    # 如果不是相邻块，保留不透明度更高的点
                    opacity_i = positions[i].norm()  # 使用位置作为代理指标
                    opacity_j = positions[j].norm()
                    
                    if opacity_i > opacity_j:
                        keep_mask[j] = False
                    else:
                        keep_mask[i] = False
        
        return keep_mask
    
    def _create_merged_model(self, merged_model: StreetGaussianModel,
                           gaussian_data: Dict[str, Dict],
                           metadata: dict):
        """创建合并后的模型"""
        
        # 初始化模型结构
        merged_model.setup_functions()
        
        # 为每个模型组件设置数据
        for model_name, data in gaussian_data.items():
            if model_name in merged_model.model_name_id:
                model = getattr(merged_model, model_name)
                
                # 设置模型参数
                if data['positions'].shape[0] > 0:
                    model._xyz = nn.Parameter(data['positions'].cuda())
                    model._features_dc = nn.Parameter(data['features_dc'].cuda())
                    model._features_rest = nn.Parameter(data['features_rest'].cuda())
                    model._scaling = nn.Parameter(data['scaling'].cuda())
                    model._rotation = nn.Parameter(data['rotation'].cuda())
                    model._opacity = nn.Parameter(data['opacity'].cuda())
                    
                    if data['semantic'] is not None:
                        model._semantic = nn.Parameter(data['semantic'].cuda())
                    
                    # 初始化优化器状态
                    if hasattr(model, 'max_radii2D'):
                        model.max_radii2D = torch.zeros((data['positions'].shape[0],), device="cuda")
                    if hasattr(model, 'xyz_gradient_accum'):
                        model.xyz_gradient_accum = torch.zeros((data['positions'].shape[0], 1), device="cuda")
                    if hasattr(model, 'denom'):
                        model.denom = torch.zeros((data['positions'].shape[0], 1), device="cuda")
    
    def save_merged_model(self, merged_model: StreetGaussianModel, output_path: str):
        """保存合并后的模型"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存模型状态
        state_dict = merged_model.save_state_dict(is_final=True)
        torch.save(state_dict, output_path)
        
        # 保存PLY文件用于可视化
        ply_path = output_path.replace('.pth', '.ply')
        merged_model.save_ply(ply_path)
        
        print(f"Merged model saved to {output_path}")
        print(f"PLY file saved to {ply_path}")

class OverlapHandler:
    """处理块之间重叠区域的高斯基元"""
    
    def __init__(self, overlap_threshold: float = 0.3):
        self.overlap_threshold = overlap_threshold
        
    def handle_overlap_regions(self, chunk_models: Dict[int, StreetGaussianModel],
                              chunk_infos: List[ChunkInfo]) -> Dict[int, torch.Tensor]:
        """处理重叠区域，返回每个块应该保留的高斯基元掩码"""
        
        overlap_masks = {}
        
        for i, chunk_info in enumerate(chunk_infos):
            chunk_id = chunk_info.chunk_id
            chunk_model = chunk_models[chunk_id]
            
            # 初始化保留掩码（默认保留所有）
            total_gaussians = chunk_model.get_xyz.shape[0]
            keep_mask = torch.ones(total_gaussians, dtype=torch.bool)
            
            # 检查与相邻块的重叠
            for j, other_chunk_info in enumerate(chunk_infos):
                if i != j:
                    other_chunk_id = other_chunk_info.chunk_id
                    
                    # 检查是否有重叠帧
                    overlap_frames = self._get_overlap_frames(chunk_info, other_chunk_info)
                    
                    if len(overlap_frames) > 0:
                        # 处理重叠区域
                        overlap_mask = self._compute_overlap_mask(
                            chunk_model, chunk_models[other_chunk_id],
                            overlap_frames, chunk_info, other_chunk_info
                        )
                        keep_mask = keep_mask & (~overlap_mask)
            
            overlap_masks[chunk_id] = keep_mask
            
        return overlap_masks
    
    def _get_overlap_frames(self, chunk1: ChunkInfo, chunk2: ChunkInfo) -> List[int]:
        """获取两个块之间的重叠帧"""
        frames1 = set(chunk1.frame_indices)
        frames2 = set(chunk2.frame_indices)
        return list(frames1.intersection(frames2))
    
    def _compute_overlap_mask(self, model1: StreetGaussianModel, 
                             model2: StreetGaussianModel,
                             overlap_frames: List[int],
                             chunk1: ChunkInfo,
                             chunk2: ChunkInfo) -> torch.Tensor:
        """计算重叠区域的掩码"""
        
        positions1 = model1.get_xyz.detach().cpu().numpy()
        positions2 = model2.get_xyz.detach().cpu().numpy()
        
        # 使用KDTree找到距离较近的点
        tree2 = cKDTree(positions2)
        
        # 对于model1中的每个点，检查是否在model2中有相近的点
        distances, indices = tree2.query(positions1, k=1)
        
        # 标记距离小于阈值的点为重叠
        overlap_mask = torch.tensor(distances < self.overlap_threshold, dtype=torch.bool)
        
        # 如果chunk1的ID更小，在重叠区域保留chunk1的点
        if chunk1.chunk_id < chunk2.chunk_id:
            return torch.zeros_like(overlap_mask)  # 不移除任何点
        else:
            return overlap_mask  # 移除重叠的点