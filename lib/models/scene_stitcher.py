import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.contribution_evaluator import ContributionEvaluator
from lib.utils.camera_utils import Camera
from lib.utils.general_utils import quaternion_to_matrix, matrix_to_quaternion
from lib.models.gaussian_model import GaussianModel
import copy
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class SceneStitcher:
    """
    场景拼接器，负责将多个分块训练的高斯模型合并成一个完整的场景
    包括处理重叠区域的冗余和确保平滑过渡
    """
    
    def __init__(self, 
                 overlap_threshold: float = 0.1,
                 similarity_threshold: float = 0.05,
                 blend_region_size: float = 0.3):
        """
        Args:
            overlap_threshold: 重叠检测阈值
            similarity_threshold: 相似高斯基元检测阈值
            blend_region_size: 混合区域大小
        """
        self.overlap_threshold = overlap_threshold
        self.similarity_threshold = similarity_threshold
        self.blend_region_size = blend_region_size
        self.contribution_evaluator = ContributionEvaluator()
        
    def stitch_chunks(self, 
                     chunk_models: List[StreetGaussianModel],
                     chunk_info_list: List[Dict],
                     target_metadata: Dict) -> StreetGaussianModel:
        """
        拼接多个块的高斯模型
        
        Args:
            chunk_models: 训练好的块模型列表
            chunk_info_list: 块信息列表
            target_metadata: 目标场景的元数据
            
        Returns:
            stitched_model: 拼接后的完整模型
        """
        print(f"Starting to stitch {len(chunk_models)} chunks...")
        
        # 创建目标模型
        stitched_model = StreetGaussianModel(target_metadata)
        
        # 按时间顺序排序块
        sorted_chunks = sorted(zip(chunk_models, chunk_info_list), 
                              key=lambda x: x[1]['start_frame'])
        
        # 初始化拼接模型为第一个块
        first_model, first_info = sorted_chunks[0]
        self._initialize_stitched_model(stitched_model, first_model)
        
        print(f"Initialized with chunk 0 (frames {first_info['start_frame']}-{first_info['end_frame']})")
        
        # 逐步合并后续块
        for i, (chunk_model, chunk_info) in enumerate(sorted_chunks[1:], 1):
            print(f"\nMerging chunk {i} (frames {chunk_info['start_frame']}-{chunk_info['end_frame']})...")
            
            # 找到重叠区域
            overlap_info = self._find_overlap_region(stitched_model, chunk_model, chunk_info)
            
            # 合并块
            self._merge_chunk_into_stitched(stitched_model, chunk_model, chunk_info, overlap_info)
            
            print(f"Completed merging chunk {i}")
        
        # 后处理：去除冗余和优化
        self._post_process_stitched_model(stitched_model)
        
        print("Scene stitching completed!")
        return stitched_model
    
    def _initialize_stitched_model(self, 
                                 stitched_model: StreetGaussianModel, 
                                 first_model: StreetGaussianModel):
        """
        用第一个块初始化拼接模型
        """
        # 复制第一个模型的所有组件
        for model_name in first_model.model_name_id.keys():
            if not first_model.get_visibility(model_name):
                continue
                
            source_model = getattr(first_model, model_name)
            target_model = getattr(stitched_model, model_name)
            
            # 深拷贝所有参数
            target_model._xyz = nn.Parameter(source_model._xyz.clone().detach().requires_grad_(True))
            target_model._features_dc = nn.Parameter(source_model._features_dc.clone().detach().requires_grad_(True))
            target_model._features_rest = nn.Parameter(source_model._features_rest.clone().detach().requires_grad_(True))
            target_model._scaling = nn.Parameter(source_model._scaling.clone().detach().requires_grad_(True))
            target_model._rotation = nn.Parameter(source_model._rotation.clone().detach().requires_grad_(True))
            target_model._opacity = nn.Parameter(source_model._opacity.clone().detach().requires_grad_(True))
            target_model._semantic = nn.Parameter(source_model._semantic.clone().detach().requires_grad_(True))
            
            # 复制其他状态
            if hasattr(source_model, 'max_radii2D'):
                target_model.max_radii2D = source_model.max_radii2D.clone()
            if hasattr(source_model, 'xyz_gradient_accum'):
                target_model.xyz_gradient_accum = source_model.xyz_gradient_accum.clone()
            if hasattr(source_model, 'denom'):
                target_model.denom = source_model.denom.clone()
        
        # 复制其他组件
        if first_model.actor_pose is not None and stitched_model.actor_pose is not None:
            stitched_model.actor_pose.load_state_dict(first_model.actor_pose.save_state_dict(is_final=True))
        
        if first_model.sky_cubemap is not None and stitched_model.sky_cubemap is not None:
            stitched_model.sky_cubemap.load_state_dict(first_model.sky_cubemap.save_state_dict(is_final=True))
        
        if first_model.color_correction is not None and stitched_model.color_correction is not None:
            stitched_model.color_correction.load_state_dict(first_model.color_correction.save_state_dict(is_final=True))
    
    def _find_overlap_region(self, 
                           stitched_model: StreetGaussianModel,
                           chunk_model: StreetGaussianModel,
                           chunk_info: Dict) -> Dict:
        """
        找到重叠区域信息
        """
        overlap_info = {
            'has_overlap': chunk_info.get('overlap_start', False),
            'temporal_overlap': [],
            'spatial_overlap_regions': {}
        }
        
        if not overlap_info['has_overlap']:
            return overlap_info
        
        # 分析时间重叠
        # 这里假设重叠是基于帧ID的，实际实现需要根据具体的时间戳或帧信息
        
        # 分析空间重叠区域
        for model_name in chunk_model.model_name_id.keys():
            if not chunk_model.get_visibility(model_name) or not stitched_model.get_visibility(model_name):
                continue
                
            chunk_model_component = getattr(chunk_model, model_name)
            stitched_model_component = getattr(stitched_model, model_name)
            
            # 计算空间距离来找到重叠区域
            chunk_xyz = chunk_model_component.get_xyz
            stitched_xyz = stitched_model_component.get_xyz
            
            if len(chunk_xyz) > 0 and len(stitched_xyz) > 0:
                # 使用K-D树或其他空间数据结构来快速找到近邻
                distances = torch.cdist(chunk_xyz, stitched_xyz)
                min_distances, closest_indices = torch.min(distances, dim=1)
                
                # 找到重叠的高斯基元
                overlap_mask = min_distances < self.overlap_threshold
                overlap_indices = torch.where(overlap_mask)[0]
                
                overlap_info['spatial_overlap_regions'][model_name] = {
                    'chunk_indices': overlap_indices,
                    'stitched_indices': closest_indices[overlap_mask],
                    'distances': min_distances[overlap_mask]
                }
        
        return overlap_info
    
    def _merge_chunk_into_stitched(self, 
                                 stitched_model: StreetGaussianModel,
                                 chunk_model: StreetGaussianModel,
                                 chunk_info: Dict,
                                 overlap_info: Dict):
        """
        将块模型合并到拼接模型中
        """
        for model_name in chunk_model.model_name_id.keys():
            if not chunk_model.get_visibility(model_name):
                continue
                
            chunk_component = getattr(chunk_model, model_name)
            stitched_component = getattr(stitched_model, model_name)
            
            # 处理重叠区域
            if model_name in overlap_info.get('spatial_overlap_regions', {}):
                self._handle_overlap_region(stitched_component, chunk_component, 
                                          overlap_info['spatial_overlap_regions'][model_name])
            else:
                # 直接追加非重叠区域的高斯基元
                self._append_gaussians(stitched_component, chunk_component)
    
    def _handle_overlap_region(self, 
                             stitched_component: GaussianModel,
                             chunk_component: GaussianModel,
                             overlap_region_info: Dict):
        """
        处理重叠区域的高斯基元
        """
        chunk_indices = overlap_region_info['chunk_indices']
        stitched_indices = overlap_region_info['stitched_indices']
        distances = overlap_region_info['distances']
        
        # 策略1: 基于相似度合并相似的高斯基元
        similar_pairs = self._find_similar_gaussians(
            stitched_component, chunk_component, 
            stitched_indices, chunk_indices, distances
        )
        
        # 合并相似的高斯基元
        for stitched_idx, chunk_idx, similarity in similar_pairs:
            self._merge_similar_gaussians(stitched_component, chunk_component, 
                                        stitched_idx, chunk_idx, similarity)
        
        # 策略2: 添加不相似的高斯基元
        merged_chunk_indices = set([pair[1] for pair in similar_pairs])
        non_merged_indices = [idx for idx in chunk_indices.tolist() 
                            if idx not in merged_chunk_indices]
        
        if non_merged_indices:
            # 创建掩码并追加非合并的高斯基元
            non_merged_mask = torch.zeros(chunk_component.get_xyz.shape[0], dtype=torch.bool)
            non_merged_mask[non_merged_indices] = True
            self._append_gaussians_with_mask(stitched_component, chunk_component, non_merged_mask)
    
    def _find_similar_gaussians(self, 
                              stitched_component: GaussianModel,
                              chunk_component: GaussianModel,
                              stitched_indices: torch.Tensor,
                              chunk_indices: torch.Tensor,
                              distances: torch.Tensor) -> List[Tuple[int, int, float]]:
        """
        找到相似的高斯基元对
        """
        similar_pairs = []
        
        for i, (s_idx, c_idx, dist) in enumerate(zip(stitched_indices, chunk_indices, distances)):
            if dist > self.similarity_threshold:
                continue
                
            # 计算多维相似度
            similarity_score = self._compute_gaussian_similarity(
                stitched_component, chunk_component, s_idx.item(), c_idx.item()
            )
            
            if similarity_score > 0.7:  # 相似度阈值
                similar_pairs.append((s_idx.item(), c_idx.item(), similarity_score))
        
        return similar_pairs
    
    def _compute_gaussian_similarity(self, 
                                   model1: GaussianModel,
                                   model2: GaussianModel,
                                   idx1: int,
                                   idx2: int) -> float:
        """
        计算两个高斯基元的相似度
        """
        # 位置相似度
        pos1 = model1.get_xyz[idx1]
        pos2 = model2.get_xyz[idx2]
        pos_sim = 1.0 / (1.0 + torch.norm(pos1 - pos2).item())
        
        # 缩放相似度
        scale1 = model1.get_scaling[idx1]
        scale2 = model2.get_scaling[idx2]
        scale_sim = 1.0 / (1.0 + torch.norm(scale1 - scale2).item())
        
        # 旋转相似度
        rot1 = model1.get_rotation[idx1]
        rot2 = model2.get_rotation[idx2]
        rot_sim = torch.abs(torch.dot(rot1, rot2)).item()  # 四元数点积
        
        # 不透明度相似度
        opacity1 = model1.get_opacity[idx1]
        opacity2 = model2.get_opacity[idx2]
        opacity_sim = 1.0 / (1.0 + torch.abs(opacity1 - opacity2).item())
        
        # 加权平均
        total_similarity = (0.4 * pos_sim + 0.2 * scale_sim + 
                          0.2 * rot_sim + 0.2 * opacity_sim)
        
        return total_similarity
    
    def _merge_similar_gaussians(self, 
                               stitched_component: GaussianModel,
                               chunk_component: GaussianModel,
                               stitched_idx: int,
                               chunk_idx: int,
                               similarity: float):
        """
        合并相似的高斯基元
        """
        # 基于相似度的加权平均
        alpha = 0.5  # 可以基于similarity调整权重
        
        # 合并位置
        pos1 = stitched_component._xyz[stitched_idx]
        pos2 = chunk_component._xyz[chunk_idx]
        merged_pos = alpha * pos1 + (1 - alpha) * pos2
        stitched_component._xyz.data[stitched_idx] = merged_pos
        
        # 合并其他属性
        # 缩放
        scale1 = stitched_component._scaling[stitched_idx]
        scale2 = chunk_component._scaling[chunk_idx]
        merged_scale = alpha * scale1 + (1 - alpha) * scale2
        stitched_component._scaling.data[stitched_idx] = merged_scale
        
        # 不透明度
        opacity1 = stitched_component._opacity[stitched_idx]
        opacity2 = chunk_component._opacity[chunk_idx]
        merged_opacity = alpha * opacity1 + (1 - alpha) * opacity2
        stitched_component._opacity.data[stitched_idx] = merged_opacity
        
        # 特征
        feature_dc1 = stitched_component._features_dc[stitched_idx]
        feature_dc2 = chunk_component._features_dc[chunk_idx]
        merged_feature_dc = alpha * feature_dc1 + (1 - alpha) * feature_dc2
        stitched_component._features_dc.data[stitched_idx] = merged_feature_dc
        
        feature_rest1 = stitched_component._features_rest[stitched_idx]
        feature_rest2 = chunk_component._features_rest[chunk_idx]
        merged_feature_rest = alpha * feature_rest1 + (1 - alpha) * feature_rest2
        stitched_component._features_rest.data[stitched_idx] = merged_feature_rest
    
    def _append_gaussians(self, 
                        stitched_component: GaussianModel,
                        chunk_component: GaussianModel):
        """
        将块组件的所有高斯基元追加到拼接组件
        """
        # 获取当前张量
        current_xyz = stitched_component._xyz
        current_features_dc = stitched_component._features_dc
        current_features_rest = stitched_component._features_rest
        current_scaling = stitched_component._scaling
        current_rotation = stitched_component._rotation
        current_opacity = stitched_component._opacity
        current_semantic = stitched_component._semantic
        
        # 获取要追加的张量
        new_xyz = chunk_component._xyz
        new_features_dc = chunk_component._features_dc
        new_features_rest = chunk_component._features_rest
        new_scaling = chunk_component._scaling
        new_rotation = chunk_component._rotation
        new_opacity = chunk_component._opacity
        new_semantic = chunk_component._semantic
        
        # 拼接张量
        stitched_component._xyz = nn.Parameter(
            torch.cat([current_xyz, new_xyz], dim=0).requires_grad_(True))
        stitched_component._features_dc = nn.Parameter(
            torch.cat([current_features_dc, new_features_dc], dim=0).requires_grad_(True))
        stitched_component._features_rest = nn.Parameter(
            torch.cat([current_features_rest, new_features_rest], dim=0).requires_grad_(True))
        stitched_component._scaling = nn.Parameter(
            torch.cat([current_scaling, new_scaling], dim=0).requires_grad_(True))
        stitched_component._rotation = nn.Parameter(
            torch.cat([current_rotation, new_rotation], dim=0).requires_grad_(True))
        stitched_component._opacity = nn.Parameter(
            torch.cat([current_opacity, new_opacity], dim=0).requires_grad_(True))
        stitched_component._semantic = nn.Parameter(
            torch.cat([current_semantic, new_semantic], dim=0).requires_grad_(True))
        
        # 更新其他状态张量
        if hasattr(stitched_component, 'max_radii2D') and hasattr(chunk_component, 'max_radii2D'):
            stitched_component.max_radii2D = torch.cat([
                stitched_component.max_radii2D, chunk_component.max_radii2D
            ], dim=0)
        
        if hasattr(stitched_component, 'xyz_gradient_accum') and hasattr(chunk_component, 'xyz_gradient_accum'):
            stitched_component.xyz_gradient_accum = torch.cat([
                stitched_component.xyz_gradient_accum, chunk_component.xyz_gradient_accum
            ], dim=0)
        
        if hasattr(stitched_component, 'denom') and hasattr(chunk_component, 'denom'):
            stitched_component.denom = torch.cat([
                stitched_component.denom, chunk_component.denom
            ], dim=0)
    
    def _append_gaussians_with_mask(self, 
                                  stitched_component: GaussianModel,
                                  chunk_component: GaussianModel,
                                  mask: torch.Tensor):
        """
        根据掩码追加特定的高斯基元
        """
        # 创建临时组件只包含掩码选中的高斯基元
        temp_component = GaussianModel(chunk_component.model_name)
        
        temp_component._xyz = nn.Parameter(chunk_component._xyz[mask].clone().detach().requires_grad_(True))
        temp_component._features_dc = nn.Parameter(chunk_component._features_dc[mask].clone().detach().requires_grad_(True))
        temp_component._features_rest = nn.Parameter(chunk_component._features_rest[mask].clone().detach().requires_grad_(True))
        temp_component._scaling = nn.Parameter(chunk_component._scaling[mask].clone().detach().requires_grad_(True))
        temp_component._rotation = nn.Parameter(chunk_component._rotation[mask].clone().detach().requires_grad_(True))
        temp_component._opacity = nn.Parameter(chunk_component._opacity[mask].clone().detach().requires_grad_(True))
        temp_component._semantic = nn.Parameter(chunk_component._semantic[mask].clone().detach().requires_grad_(True))
        
        if hasattr(chunk_component, 'max_radii2D'):
            temp_component.max_radii2D = chunk_component.max_radii2D[mask].clone()
        if hasattr(chunk_component, 'xyz_gradient_accum'):
            temp_component.xyz_gradient_accum = chunk_component.xyz_gradient_accum[mask].clone()
        if hasattr(chunk_component, 'denom'):
            temp_component.denom = chunk_component.denom[mask].clone()
        
        # 追加临时组件
        self._append_gaussians(stitched_component, temp_component)
    
    def _post_process_stitched_model(self, stitched_model: StreetGaussianModel):
        """
        后处理拼接的模型：去除冗余、优化结构
        """
        print("Post-processing stitched model...")
        
        # 1. 全局去重
        self._remove_duplicate_gaussians(stitched_model)
        
        # 2. 基于贡献度剪枝
        # 这里需要一些测试相机来评估贡献度
        # 实际实现中可以使用验证集或合成的测试视角
        
        # 3. 优化密度分布
        self._optimize_density_distribution(stitched_model)
        
        print("Post-processing completed.")
    
    def _remove_duplicate_gaussians(self, stitched_model: StreetGaussianModel):
        """
        移除重复的高斯基元
        """
        for model_name in stitched_model.model_name_id.keys():
            if not stitched_model.get_visibility(model_name):
                continue
                
            component = getattr(stitched_model, model_name)
            
            if component.get_xyz.shape[0] == 0:
                continue
            
            # 使用聚类来找到重复的高斯基元
            xyz = component.get_xyz.detach().cpu().numpy()
            
            if len(xyz) > 1:
                # 使用DBSCAN或者简单的距离阈值来检测重复
                distances = cdist(xyz, xyz)
                np.fill_diagonal(distances, np.inf)
                
                # 找到距离很近的高斯基元对
                close_pairs = np.where(distances < self.similarity_threshold)
                
                # 标记要删除的高斯基元
                to_remove = set()
                for i, j in zip(close_pairs[0], close_pairs[1]):
                    if i not in to_remove and j not in to_remove:
                        # 保留不透明度更高的那个
                        opacity_i = component.get_opacity[i].item()
                        opacity_j = component.get_opacity[j].item()
                        if opacity_i > opacity_j:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                
                if to_remove:
                    keep_mask = torch.ones(len(xyz), dtype=torch.bool)
                    keep_mask[list(to_remove)] = False
                    
                    # 应用掩码
                    component._xyz = nn.Parameter(component._xyz[keep_mask].clone().detach().requires_grad_(True))
                    component._features_dc = nn.Parameter(component._features_dc[keep_mask].clone().detach().requires_grad_(True))
                    component._features_rest = nn.Parameter(component._features_rest[keep_mask].clone().detach().requires_grad_(True))
                    component._scaling = nn.Parameter(component._scaling[keep_mask].clone().detach().requires_grad_(True))
                    component._rotation = nn.Parameter(component._rotation[keep_mask].clone().detach().requires_grad_(True))
                    component._opacity = nn.Parameter(component._opacity[keep_mask].clone().detach().requires_grad_(True))
                    component._semantic = nn.Parameter(component._semantic[keep_mask].clone().detach().requires_grad_(True))
                    
                    if hasattr(component, 'max_radii2D'):
                        component.max_radii2D = component.max_radii2D[keep_mask]
                    if hasattr(component, 'xyz_gradient_accum'):
                        component.xyz_gradient_accum = component.xyz_gradient_accum[keep_mask]
                    if hasattr(component, 'denom'):
                        component.denom = component.denom[keep_mask]
                    
                    print(f"Removed {len(to_remove)} duplicate gaussians from {model_name}")
    
    def _optimize_density_distribution(self, stitched_model: StreetGaussianModel):
        """
        优化密度分布，确保场景中的高斯基元分布合理
        """
        # 这里可以实现更高级的优化策略
        # 比如基于空间密度的自适应剪枝等
        pass
    
    def save_stitched_model(self, 
                          stitched_model: StreetGaussianModel, 
                          save_path: str):
        """
        保存拼接后的模型
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存PLY文件
        ply_path = save_path.replace('.pth', '.ply')
        stitched_model.save_ply(ply_path)
        
        # 保存状态字典
        state_dict = stitched_model.save_state_dict(is_final=True)
        torch.save(state_dict, save_path)
        
        print(f"Saved stitched model to {save_path} and {ply_path}")
        
        # 打印统计信息
        total_gaussians = 0
        for model_name in stitched_model.model_name_id.keys():
            if stitched_model.get_visibility(model_name):
                component = getattr(stitched_model, model_name)
                num_gaussians = component.get_xyz.shape[0]
                total_gaussians += num_gaussians
                print(f"{model_name}: {num_gaussians} gaussians")
        
        print(f"Total: {total_gaussians} gaussians in stitched model")