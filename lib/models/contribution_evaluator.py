import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.utils.camera_utils import Camera
from lib.utils.loss_utils import l1_loss, ssim, psnr
import torch.nn.functional as F

class ContributionEvaluator:
    """
    高斯基元贡献度评估器
    评估每个高斯基元对最终渲染结果的贡献度，用于剪枝低贡献的高斯基元
    """
    
    def __init__(self, num_evaluation_frames: int = 10, contribution_threshold: float = 0.01):
        """
        Args:
            num_evaluation_frames: 用于评估的帧数
            contribution_threshold: 贡献度阈值，低于此值的高斯基元将被剪枝
        """
        self.num_evaluation_frames = num_evaluation_frames
        self.contribution_threshold = contribution_threshold
        self.contribution_scores = {}
        self.renderer = StreetGaussianRenderer()
        
    def compute_contribution_scores(self, 
                                   gaussians: StreetGaussianModel, 
                                   cameras: List[Camera],
                                   method: str = 'gradient_based') -> Dict[str, torch.Tensor]:
        """
        计算高斯基元的贡献度分数
        
        Args:
            gaussians: 高斯模型
            cameras: 评估用的相机列表
            method: 评估方法 ('gradient_based', 'visibility_based', 'hybrid')
            
        Returns:
            contribution_scores: 每个模型的贡献度分数字典
        """
        if method == 'gradient_based':
            return self._compute_gradient_based_scores(gaussians, cameras)
        elif method == 'visibility_based':
            return self._compute_visibility_based_scores(gaussians, cameras)
        elif method == 'hybrid':
            grad_scores = self._compute_gradient_based_scores(gaussians, cameras)
            vis_scores = self._compute_visibility_based_scores(gaussians, cameras)
            return self._combine_scores(grad_scores, vis_scores)
        else:
            raise ValueError(f"Unknown evaluation method: {method}")
    
    def _compute_gradient_based_scores(self, 
                                     gaussians: StreetGaussianModel, 
                                     cameras: List[Camera]) -> Dict[str, torch.Tensor]:
        """
        基于梯度的贡献度评估
        通过计算每个高斯基元参数的梯度来评估其重要性
        """
        contribution_scores = {}
        
        # 选择评估相机
        eval_cameras = self._select_evaluation_cameras(cameras)
        
        # 为每个模型组件计算贡献度
        for model_name in gaussians.model_name_id.keys():
            model: nn.Module = getattr(gaussians, model_name)
            if not gaussians.get_visibility(model_name):
                continue
                
            # 初始化贡献度分数
            num_gaussians = model.get_xyz.shape[0]
            scores = torch.zeros(num_gaussians, device='cuda')
            
            # 对每个评估相机计算梯度
            for camera in eval_cameras:
                gaussians.parse_camera(camera)
                
                # 前向传播
                render_pkg = self.renderer.render(camera, gaussians)
                gt_image = camera.original_image.cuda()
                rendered_image = render_pkg["rgb"]
                
                # 计算损失
                loss = l1_loss(rendered_image, gt_image, 
                              torch.ones_like(gt_image[0:1]).bool())
                
                # 反向传播获取梯度
                loss.backward(retain_graph=True)
                
                # 收集每个高斯基元的梯度信息
                if model_name == 'background':
                    start_idx, end_idx = gaussians.graph_gaussian_range.get('background', [0, 0])
                elif model_name in gaussians.graph_obj_list:
                    start_idx, end_idx = gaussians.graph_gaussian_range.get(model_name, [0, 0])
                else:
                    continue
                
                if start_idx < end_idx:
                    # 计算位置、旋转、缩放等参数的梯度幅度
                    xyz_grad = model._xyz.grad
                    rotation_grad = model._rotation.grad
                    scaling_grad = model._scaling.grad
                    opacity_grad = model._opacity.grad
                    
                    if xyz_grad is not None:
                        # 综合各参数的梯度计算重要性分数
                        pos_importance = torch.norm(xyz_grad, dim=1)
                        rot_importance = torch.norm(rotation_grad, dim=1) if rotation_grad is not None else 0
                        scale_importance = torch.norm(scaling_grad, dim=1) if scaling_grad is not None else 0
                        opacity_importance = torch.abs(opacity_grad.squeeze()) if opacity_grad is not None else 0
                        
                        # 加权合并不同参数的重要性
                        gaussian_importance = (0.4 * pos_importance + 
                                             0.2 * rot_importance + 
                                             0.2 * scale_importance + 
                                             0.2 * opacity_importance)
                        
                        scores += gaussian_importance
                
                # 清零梯度
                gaussians.zero_grad()
            
            # 归一化分数
            if scores.sum() > 0:
                scores = scores / scores.sum()
            
            contribution_scores[model_name] = scores
            
        return contribution_scores
    
    def _compute_visibility_based_scores(self, 
                                       gaussians: StreetGaussianModel, 
                                       cameras: List[Camera]) -> Dict[str, torch.Tensor]:
        """
        基于可见性的贡献度评估
        通过统计每个高斯基元在不同视角下的可见频率来评估重要性
        """
        contribution_scores = {}
        eval_cameras = self._select_evaluation_cameras(cameras)
        
        for model_name in gaussians.model_name_id.keys():
            model: nn.Module = getattr(gaussians, model_name)
            if not gaussians.get_visibility(model_name):
                continue
                
            num_gaussians = model.get_xyz.shape[0]
            visibility_count = torch.zeros(num_gaussians, device='cuda')
            contribution_sum = torch.zeros(num_gaussians, device='cuda')
            
            for camera in eval_cameras:
                gaussians.parse_camera(camera)
                
                with torch.no_grad():
                    render_pkg = self.renderer.render(camera, gaussians)
                    visibility_filter = render_pkg["visibility_filter"]
                    radii = render_pkg["radii"]
                    
                    # 统计可见的高斯基元
                    if model_name == 'background':
                        start_idx, end_idx = gaussians.graph_gaussian_range.get('background', [0, 0])
                    elif model_name in gaussians.graph_obj_list:
                        start_idx, end_idx = gaussians.graph_gaussian_range.get(model_name, [0, 0])
                    else:
                        continue
                    
                    if start_idx < end_idx:
                        model_visibility = visibility_filter[start_idx:end_idx]
                        model_radii = radii[start_idx:end_idx]
                        
                        # 可见性计数
                        visibility_count += model_visibility.float()
                        
                        # 基于渲染半径的贡献度
                        radii_contribution = model_radii * model_visibility.float()
                        contribution_sum += radii_contribution
            
            # 合并可见性和贡献度
            if len(eval_cameras) > 0:
                visibility_ratio = visibility_count / len(eval_cameras)
                avg_contribution = contribution_sum / (visibility_count + 1e-8)
                
                # 综合评分
                scores = visibility_ratio * avg_contribution
                
                # 归一化
                if scores.sum() > 0:
                    scores = scores / scores.sum()
                
                contribution_scores[model_name] = scores
            
        return contribution_scores
    
    def _combine_scores(self, 
                       grad_scores: Dict[str, torch.Tensor], 
                       vis_scores: Dict[str, torch.Tensor], 
                       alpha: float = 0.6) -> Dict[str, torch.Tensor]:
        """
        合并梯度和可见性评分
        
        Args:
            grad_scores: 梯度评分
            vis_scores: 可见性评分
            alpha: 梯度评分的权重
            
        Returns:
            combined_scores: 合并后的评分
        """
        combined_scores = {}
        
        for model_name in grad_scores.keys():
            if model_name in vis_scores:
                combined_scores[model_name] = (alpha * grad_scores[model_name] + 
                                             (1 - alpha) * vis_scores[model_name])
            else:
                combined_scores[model_name] = grad_scores[model_name]
        
        return combined_scores
    
    def _select_evaluation_cameras(self, cameras: List[Camera]) -> List[Camera]:
        """
        选择用于评估的相机
        
        Args:
            cameras: 所有相机列表
            
        Returns:
            eval_cameras: 评估用相机列表
        """
        if len(cameras) <= self.num_evaluation_frames:
            return cameras
        
        # 均匀采样相机
        step = len(cameras) // self.num_evaluation_frames
        eval_cameras = [cameras[i] for i in range(0, len(cameras), step)]
        eval_cameras = eval_cameras[:self.num_evaluation_frames]
        
        return eval_cameras
    
    def prune_low_contribution_gaussians(self, 
                                       gaussians: StreetGaussianModel, 
                                       contribution_scores: Dict[str, torch.Tensor],
                                       prune_ratio: float = 0.3) -> Dict[str, int]:
        """
        剪枝低贡献度的高斯基元
        
        Args:
            gaussians: 高斯模型
            contribution_scores: 贡献度分数
            prune_ratio: 剪枝比例
            
        Returns:
            pruned_counts: 每个模型被剪枝的高斯基元数量
        """
        pruned_counts = {}
        
        for model_name, scores in contribution_scores.items():
            if model_name not in gaussians.model_name_id.keys():
                continue
                
            model: nn.Module = getattr(gaussians, model_name)
            
            # 计算剪枝阈值
            sorted_scores, indices = torch.sort(scores, descending=False)
            num_to_prune = int(len(scores) * prune_ratio)
            
            if num_to_prune > 0:
                prune_threshold = sorted_scores[num_to_prune-1]
                prune_mask = scores <= prune_threshold
                
                # 应用剪枝
                model.prune_points(prune_mask)
                pruned_counts[model_name] = prune_mask.sum().item()
                
                print(f"Pruned {pruned_counts[model_name]} gaussians from {model_name} "
                      f"(threshold: {prune_threshold:.6f})")
            else:
                pruned_counts[model_name] = 0
        
        return pruned_counts
    
    def adaptive_prune_by_contribution(self, 
                                     gaussians: StreetGaussianModel, 
                                     contribution_scores: Dict[str, torch.Tensor],
                                     target_reduction_ratio: float = 0.5) -> Dict[str, int]:
        """
        基于贡献度的自适应剪枝
        
        Args:
            gaussians: 高斯模型
            contribution_scores: 贡献度分数
            target_reduction_ratio: 目标减少比例
            
        Returns:
            pruned_counts: 每个模型被剪枝的高斯基元数量
        """
        pruned_counts = {}
        
        # 收集所有高斯基元的分数
        all_scores = []
        model_ranges = {}
        current_idx = 0
        
        for model_name, scores in contribution_scores.items():
            if model_name not in gaussians.model_name_id.keys():
                continue
                
            model_ranges[model_name] = (current_idx, current_idx + len(scores))
            all_scores.append(scores)
            current_idx += len(scores)
        
        if not all_scores:
            return pruned_counts
            
        # 全局排序确定剪枝阈值
        all_scores_tensor = torch.cat(all_scores)
        sorted_scores, _ = torch.sort(all_scores_tensor)
        global_threshold_idx = int(len(sorted_scores) * target_reduction_ratio)
        global_threshold = sorted_scores[global_threshold_idx]
        
        # 对每个模型应用剪枝
        for model_name, scores in contribution_scores.items():
            if model_name not in gaussians.model_name_id.keys():
                continue
                
            model: nn.Module = getattr(gaussians, model_name)
            
            # 使用全局阈值但考虑模型特定的最小保留数量
            min_keep = max(int(len(scores) * 0.1), 10)  # 至少保留10%或10个
            prune_mask = scores < global_threshold
            
            # 确保不会剪枝太多
            if prune_mask.sum() > len(scores) - min_keep:
                sorted_scores_model, indices = torch.sort(scores, descending=True)
                prune_mask = torch.zeros_like(scores, dtype=torch.bool)
                prune_mask[indices[min_keep:]] = True
            
            if prune_mask.sum() > 0:
                model.prune_points(prune_mask)
                pruned_counts[model_name] = prune_mask.sum().item()
                
                print(f"Adaptively pruned {pruned_counts[model_name]} gaussians from {model_name}")
            else:
                pruned_counts[model_name] = 0
        
        return pruned_counts
    
    def save_contribution_scores(self, 
                               contribution_scores: Dict[str, torch.Tensor], 
                               save_path: str):
        """
        保存贡献度分数
        
        Args:
            contribution_scores: 贡献度分数
            save_path: 保存路径
        """
        scores_dict = {}
        for model_name, scores in contribution_scores.items():
            scores_dict[model_name] = scores.detach().cpu().numpy()
        
        np.savez(save_path, **scores_dict)
        print(f"Saved contribution scores to {save_path}")
    
    def load_contribution_scores(self, load_path: str) -> Dict[str, torch.Tensor]:
        """
        加载贡献度分数
        
        Args:
            load_path: 加载路径
            
        Returns:
            contribution_scores: 贡献度分数
        """
        data = np.load(load_path)
        contribution_scores = {}
        
        for model_name in data.files:
            contribution_scores[model_name] = torch.from_numpy(data[model_name]).cuda()
        
        print(f"Loaded contribution scores from {load_path}")
        return contribution_scores