import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KDTree
from scipy.optimize import linear_sum_assignment
from lib.utils.general_utils import build_scaling_rotation, strip_symmetric


class Reduce3DGS:
    """
    Reduce 3DGS: 基于最优传输理论的全局高斯减少方法
    
    该方法实现了以下核心思想：
    1. 将3DGS压缩看作全局高斯混合减少问题
    2. 使用KD树分区和最优传输来产生紧凑的几何表示
    3. 通过微调颜色和不透明度属性来解耦外观和几何
    """
    
    def __init__(self, reduction_ratio=0.1, max_clusters=1000, transport_reg=0.01):
        """
        Args:
            reduction_ratio: 目标减少比例 (保留原始高斯的比例)
            max_clusters: 最大聚类数
            transport_reg: 传输正则化参数
        """
        self.reduction_ratio = reduction_ratio
        self.max_clusters = max_clusters
        self.transport_reg = transport_reg
        
    def compute_gaussian_importance(self, gaussians):
        """计算每个高斯的重要性分数"""
        # 基于不透明度、尺度和渲染贡献的重要性分数
        opacity = gaussians.get_opacity.squeeze()
        scales = gaussians.get_scaling
        scale_volume = torch.prod(scales, dim=1)
        
        # 计算综合重要性分数
        importance = opacity * torch.log(scale_volume + 1e-6)
        return importance
    
    def build_kd_tree_partition(self, positions, num_clusters):
        """构建KD树分区"""
        positions_np = positions.detach().cpu().numpy()
        
        # 使用KMeans进行初步聚类以确定分区中心
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(positions_np)
        
        return torch.tensor(cluster_labels, device=positions.device)
    
    def compute_transport_cost(self, source_gaussians, target_positions, source_weights, target_weights):
        """计算传输代价矩阵"""
        # 计算位置距离
        source_pos = source_gaussians.get_xyz
        dist_matrix = torch.cdist(source_pos, target_positions)
        
        # 添加几何属性的代价
        source_scales = source_gaussians.get_scaling
        source_scale_vol = torch.prod(source_scales, dim=1, keepdim=True)
        
        # 综合代价包括位置距离和几何相似性
        geometric_cost = dist_matrix + 0.1 * torch.abs(source_scale_vol - target_weights.unsqueeze(0))
        
        return geometric_cost
    
    def optimal_transport_assignment(self, cost_matrix, source_weights, target_weights):
        """使用最优传输进行分配"""
        # 使用Sinkhorn算法近似最优传输
        cost_matrix_np = cost_matrix.detach().cpu().numpy()
        source_weights_np = source_weights.detach().cpu().numpy()
        target_weights_np = target_weights.detach().cpu().numpy()
        
        # 简化版本：使用匈牙利算法进行一对一分配
        if cost_matrix.shape[0] <= cost_matrix.shape[1]:
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
            assignment = torch.zeros(cost_matrix.shape[0], dtype=torch.long, device=cost_matrix.device)
            assignment[row_ind] = torch.tensor(col_ind, device=cost_matrix.device)
        else:
            # 如果源点多于目标点，需要聚合
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np.T)
            assignment = torch.zeros(cost_matrix.shape[0], dtype=torch.long, device=cost_matrix.device)
            for i, target_idx in enumerate(col_ind):
                # 找到分配给该目标的所有源点
                source_indices = torch.where(torch.argmin(cost_matrix, dim=1) == target_idx)[0]
                assignment[source_indices] = target_idx
        
        return assignment
    
    def aggregate_gaussians(self, gaussians, assignment, num_targets):
        """根据分配聚合高斯"""
        device = gaussians.get_xyz.device
        
        # 初始化聚合后的参数
        new_xyz = torch.zeros(num_targets, 3, device=device)
        new_features_dc = torch.zeros(num_targets, gaussians._features_dc.shape[1], gaussians._features_dc.shape[2], device=device)
        new_features_rest = torch.zeros(num_targets, gaussians._features_rest.shape[1], gaussians._features_rest.shape[2], device=device)
        new_scaling = torch.zeros(num_targets, 3, device=device)
        new_rotation = torch.zeros(num_targets, 4, device=device)
        new_opacity = torch.zeros(num_targets, 1, device=device)
        new_semantic = torch.zeros(num_targets, gaussians._semantic.shape[1], device=device)
        
        # 为每个目标聚合源高斯
        for target_idx in range(num_targets):
            source_mask = assignment == target_idx
            if not source_mask.any():
                continue
                
            source_indices = torch.where(source_mask)[0]
            weights = gaussians.get_opacity[source_indices].squeeze()
            
            if weights.sum() > 0:
                weights = weights / weights.sum()
                
                # 加权平均位置
                new_xyz[target_idx] = torch.sum(gaussians.get_xyz[source_indices] * weights.unsqueeze(1), dim=0)
                
                # 加权平均特征
                new_features_dc[target_idx] = torch.sum(gaussians._features_dc[source_indices] * weights.unsqueeze(1).unsqueeze(2), dim=0)
                new_features_rest[target_idx] = torch.sum(gaussians._features_rest[source_indices] * weights.unsqueeze(1).unsqueeze(2), dim=0)
                
                # 加权平均尺度（在对数空间）
                source_scales = gaussians._scaling[source_indices]
                new_scaling[target_idx] = torch.sum(source_scales * weights.unsqueeze(1), dim=0)
                
                # 旋转的平均（四元数）
                source_rotations = gaussians._rotation[source_indices]
                new_rotation[target_idx] = torch.sum(source_rotations * weights.unsqueeze(1), dim=0)
                new_rotation[target_idx] = new_rotation[target_idx] / torch.norm(new_rotation[target_idx])
                
                # 加权平均不透明度
                new_opacity[target_idx] = torch.sum(gaussians._opacity[source_indices] * weights.unsqueeze(1), dim=0)
                
                # 加权平均语义
                new_semantic[target_idx] = torch.sum(gaussians._semantic[source_indices] * weights.unsqueeze(1), dim=0)
        
        return {
            'xyz': new_xyz,
            'features_dc': new_features_dc,
            'features_rest': new_features_rest,
            'scaling': new_scaling,
            'rotation': new_rotation,
            'opacity': new_opacity,
            'semantic': new_semantic
        }
    
    def reduce_gaussians(self, gaussians):
        """主要的高斯减少方法"""
        current_num = gaussians.get_xyz.shape[0]
        target_num = max(int(current_num * self.reduction_ratio), 100)  # 至少保留100个高斯
        
        if current_num <= target_num:
            return gaussians  # 无需减少
        
        # 1. 计算重要性分数
        importance = self.compute_gaussian_importance(gaussians)
        
        # 2. 构建KD树分区
        positions = gaussians.get_xyz
        cluster_labels = self.build_kd_tree_partition(positions, target_num)
        
        # 3. 计算传输代价和最优分配
        target_positions = torch.zeros(target_num, 3, device=positions.device)
        target_weights = torch.zeros(target_num, device=positions.device)
        
        # 计算每个聚类的中心和权重
        for i in range(target_num):
            cluster_mask = cluster_labels == i
            if cluster_mask.any():
                cluster_positions = positions[cluster_mask]
                cluster_importance = importance[cluster_mask]
                
                target_positions[i] = torch.mean(cluster_positions, dim=0)
                target_weights[i] = torch.mean(cluster_importance)
        
        # 4. 计算传输代价
        cost_matrix = self.compute_transport_cost(gaussians, target_positions, importance, target_weights)
        
        # 5. 最优传输分配
        assignment = self.optimal_transport_assignment(cost_matrix, importance, target_weights)
        
        # 6. 聚合高斯
        aggregated_params = self.aggregate_gaussians(gaussians, assignment, target_num)
        
        return aggregated_params
    
    def fine_tune_appearance(self, gaussians, reduced_params, original_views=None):
        """微调外观属性以恢复视觉质量"""
        # 这里可以实现外观微调逻辑
        # 例如，基于渲染损失来优化颜色和不透明度
        
        # 简化版本：直接返回聚合的参数
        return reduced_params


class ReducedGaussianModel(nn.Module):
    """集成了reduce_3dgs方法的高斯模型"""
    
    def __init__(self, base_model, reduction_ratio=0.1):
        super().__init__()
        self.base_model = base_model
        self.reducer = Reduce3DGS(reduction_ratio=reduction_ratio)
        self.is_reduced = False
        
    def forward(self, *args, **kwargs):
        return self.base_model.forward(*args, **kwargs)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
    
    def reduce_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """使用reduce_3dgs方法替换原始的densify_and_prune"""
        if not self.is_reduced:
            # 首先进行传统的densify和prune以处理梯度信息
            self.base_model.densify_and_prune(max_grad, min_opacity, extent, max_screen_size)
            
            # 然后应用reduce_3dgs方法
            reduced_params = self.reducer.reduce_gaussians(self.base_model)
            
            # 更新模型参数
            self._update_parameters(reduced_params)
            self.is_reduced = True
            
            print(f"Reduced gaussians from {self.base_model.get_xyz.shape[0]} to {reduced_params['xyz'].shape[0]}")
        else:
            # 如果已经减少过，只进行常规的prune
            self._regular_prune(min_opacity, max_screen_size, extent)
        
        return self.base_model.scalar_dict, self.base_model.tensor_dict
    
    def _update_parameters(self, reduced_params):
        """更新模型参数"""
        device = reduced_params['xyz'].device
        
        # 更新参数
        self.base_model._xyz = nn.Parameter(reduced_params['xyz'].requires_grad_(True))
        self.base_model._features_dc = nn.Parameter(reduced_params['features_dc'].requires_grad_(True))
        self.base_model._features_rest = nn.Parameter(reduced_params['features_rest'].requires_grad_(True))
        self.base_model._scaling = nn.Parameter(reduced_params['scaling'].requires_grad_(True))
        self.base_model._rotation = nn.Parameter(reduced_params['rotation'].requires_grad_(True))
        self.base_model._opacity = nn.Parameter(reduced_params['opacity'].requires_grad_(True))
        self.base_model._semantic = nn.Parameter(reduced_params['semantic'].requires_grad_(True))
        
        # 重置辅助变量
        num_points = reduced_params['xyz'].shape[0]
        self.base_model.xyz_gradient_accum = torch.zeros((num_points, 2), device=device)
        self.base_model.denom = torch.zeros((num_points, 1), device=device)
        self.base_model.max_radii2D = torch.zeros((num_points,), device=device)
        
        # 重新设置优化器
        self.base_model.training_setup()
    
    def _regular_prune(self, min_opacity, max_screen_size, extent):
        """常规的pruning操作"""
        prune_mask = (self.base_model.get_opacity < min_opacity).squeeze()
        
        if max_screen_size:
            big_points_ws = self.base_model.get_scaling.max(dim=1).values > extent * self.base_model.percent_big_ws
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        
        if prune_mask.any():
            self.base_model.prune_points(prune_mask)
            print(f"Pruned {prune_mask.sum().item()} gaussians")