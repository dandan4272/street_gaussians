import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import json

from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.utils.camera_utils import Camera
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim
from lib.config import cfg
from lib.utils.general_utils import safe_state
from lib.utils.system_utils import searchForMaxIteration

@dataclass
class ChunkInfo:
    """存储每个训练块的信息"""
    chunk_id: int
    start_frame: int
    end_frame: int
    frame_indices: List[int]
    cameras: List[Camera]
    model_path: str
    
@dataclass
class GaussianContribution:
    """高斯基元贡献度信息"""
    gaussian_id: int
    contribution_score: float
    visibility_count: int
    rendering_frequency: float
    avg_opacity: float
    avg_scale: float

class GaussianContributionEvaluator:
    """高斯基元贡献度评估器"""
    
    def __init__(self):
        self.contribution_history = {}
        self.visibility_counts = {}
        self.opacity_history = {}
        self.scale_history = {}
        self.rendering_counts = {}
        
    def update_contribution(self, gaussians: StreetGaussianModel, 
                          visibility_filter: torch.Tensor,
                          radii: torch.Tensor,
                          iteration: int):
        """更新高斯基元的贡献度统计"""
        
        # 获取当前可见的高斯基元
        visible_gaussians = torch.where(visibility_filter)[0]
        
        # 获取高斯基元属性
        opacity = gaussians.get_opacity.squeeze()
        scaling = gaussians.get_scaling.mean(dim=1)  # 取平均缩放
        
        # 更新可见性计数
        for idx in visible_gaussians:
            idx_item = idx.item()
            if idx_item not in self.visibility_counts:
                self.visibility_counts[idx_item] = 0
                self.opacity_history[idx_item] = []
                self.scale_history[idx_item] = []
                self.rendering_counts[idx_item] = 0
                
            self.visibility_counts[idx_item] += 1
            self.opacity_history[idx_item].append(opacity[idx].item())
            self.scale_history[idx_item].append(scaling[idx].item())
            self.rendering_counts[idx_item] += 1
    
    def compute_contribution_scores(self, total_iterations: int) -> Dict[int, GaussianContribution]:
        """计算每个高斯基元的最终贡献度分数"""
        contributions = {}
        
        for gaussian_id in self.visibility_counts.keys():
            # 计算渲染频率
            rendering_freq = self.rendering_counts[gaussian_id] / total_iterations
            
            # 计算平均不透明度
            avg_opacity = np.mean(self.opacity_history[gaussian_id])
            
            # 计算平均缩放
            avg_scale = np.mean(self.scale_history[gaussian_id])
            
            # 综合贡献度分数 (可以根据需要调整权重)
            contribution_score = (
                rendering_freq * 0.4 +           # 渲染频率权重
                avg_opacity * 0.3 +              # 不透明度权重  
                (1.0 / (1.0 + avg_scale)) * 0.3  # 缩放权重(越小越重要)
            )
            
            contributions[gaussian_id] = GaussianContribution(
                gaussian_id=gaussian_id,
                contribution_score=contribution_score,
                visibility_count=self.visibility_counts[gaussian_id],
                rendering_frequency=rendering_freq,
                avg_opacity=avg_opacity,
                avg_scale=avg_scale
            )
            
        return contributions

class GaussianPruner:
    """高斯基元剔除器"""
    
    def __init__(self, contribution_threshold: float = 0.1):
        self.contribution_threshold = contribution_threshold
        
    def prune_gaussians(self, gaussians: StreetGaussianModel, 
                       contributions: Dict[int, GaussianContribution]) -> Tuple[torch.Tensor, int]:
        """根据贡献度剔除高斯基元
        
        Returns:
            keep_mask: 保留的高斯基元掩码
            pruned_count: 被剔除的高斯基元数量
        """
        total_gaussians = gaussians.get_xyz.shape[0]
        keep_mask = torch.ones(total_gaussians, dtype=torch.bool, device='cuda')
        
        # 获取所有贡献度分数
        all_scores = []
        all_indices = []
        
        for gaussian_id, contrib in contributions.items():
            if gaussian_id < total_gaussians:  # 确保索引有效
                all_scores.append(contrib.contribution_score)
                all_indices.append(gaussian_id)
        
        if len(all_scores) == 0:
            return keep_mask, 0
            
        # 计算动态阈值（保留前80%的高斯基元）
        scores_tensor = torch.tensor(all_scores)
        sorted_scores, sorted_indices = torch.sort(scores_tensor, descending=True)
        
        # 保留前80%的高斯基元
        keep_count = int(len(all_scores) * 0.8)
        keep_indices = [all_indices[i] for i in sorted_indices[:keep_count].tolist()]
        
        # 设置剔除掩码
        prune_indices = set(range(total_gaussians)) - set(keep_indices)
        for idx in prune_indices:
            if idx < total_gaussians:
                keep_mask[idx] = False
                
        pruned_count = len(prune_indices)
        
        print(f"Pruned {pruned_count} gaussians out of {total_gaussians} "
              f"({pruned_count/total_gaussians*100:.2f}%)")
              
        return keep_mask, pruned_count

class ChunkedTrainer:
    """分块训练管理器"""
    
    def __init__(self, chunk_size: int = 5, overlap_size: int = 1):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.trained_chunks = []
        self.chunk_models = {}
        self.contribution_evaluators = {}
        
    def create_chunks(self, dataset: Dataset) -> List[ChunkInfo]:
        """将数据集分割成训练块"""
        train_cameras = dataset.getTrainCameras()
        total_frames = len(train_cameras)
        
        chunks = []
        chunk_id = 0
        
        start_idx = 0
        while start_idx < total_frames:
            end_idx = min(start_idx + self.chunk_size, total_frames)
            
            # 获取帧索引
            frame_indices = list(range(start_idx, end_idx))
            chunk_cameras = [train_cameras[i] for i in frame_indices]
            
            # 创建块信息
            chunk_info = ChunkInfo(
                chunk_id=chunk_id,
                start_frame=start_idx,
                end_frame=end_idx - 1,
                frame_indices=frame_indices,
                cameras=chunk_cameras,
                model_path=os.path.join(cfg.model_path, f"chunk_{chunk_id}")
            )
            
            chunks.append(chunk_info)
            
            # 移动到下一个块（考虑重叠）
            start_idx = end_idx - self.overlap_size
            chunk_id += 1
            
        print(f"Created {len(chunks)} chunks from {total_frames} frames")
        return chunks
        
    def train_chunk(self, chunk_info: ChunkInfo, dataset: Dataset) -> StreetGaussianModel:
        """训练单个块"""
        print(f"\n=== Training Chunk {chunk_info.chunk_id} "
              f"(frames {chunk_info.start_frame}-{chunk_info.end_frame}) ===")
        
        # 创建块特定的配置
        chunk_cfg = cfg.clone()
        chunk_cfg.model_path = chunk_info.model_path
        chunk_cfg.train.iterations = cfg.train.iterations // 2  # 减少训练迭代数
        
        # 确保输出目录存在
        os.makedirs(chunk_info.model_path, exist_ok=True)
        
        # 创建高斯模型
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        
        # 修改数据集以只包含当前块的相机
        chunk_dataset = Dataset()
        chunk_dataset.scene_info = dataset.scene_info
        chunk_dataset.train_cameras = {1: chunk_info.cameras}
        chunk_dataset.test_cameras = dataset.test_cameras
        
        # 创建场景
        scene = Scene(gaussians=gaussians, dataset=chunk_dataset)
        gaussians.training_setup()
        
        # 创建渲染器和贡献度评估器
        renderer = StreetGaussianRenderer()
        evaluator = GaussianContributionEvaluator()
        
        # 训练循环
        print(f"Training chunk {chunk_info.chunk_id} for {chunk_cfg.train.iterations} iterations...")
        
        viewpoint_stack = None
        progress_bar = tqdm(range(1, chunk_cfg.train.iterations + 1))
        
        for iteration in progress_bar:
            gaussians.update_learning_rate(iteration)
            
            # 每1000次迭代增加SH度数
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
                
            # 选择随机相机
            if not viewpoint_stack:
                viewpoint_stack = chunk_info.cameras.copy()
            
            if len(viewpoint_stack) == 0:
                viewpoint_stack = chunk_info.cameras.copy()
                
            viewpoint_cam = viewpoint_stack.pop(
                np.random.randint(0, len(viewpoint_stack))
            )
            
            # 渲染
            render_pkg = renderer.render(viewpoint_cam, gaussians)
            image = render_pkg["rgb"]
            visibility_filter = render_pkg["visibility_filter"] 
            radii = render_pkg["radii"]
            
            # 更新贡献度统计
            evaluator.update_contribution(gaussians, visibility_filter, radii, iteration)
            
            # 计算损失
            gt_image = viewpoint_cam.original_image.cuda()
            mask = viewpoint_cam.guidance.get('mask', torch.ones_like(gt_image[0:1]).bool())
            mask = mask.cuda()
            
            loss = l1_loss(image, gt_image, mask)
            loss += 0.2 * (1.0 - ssim(image, gt_image, mask=mask))
            
            # 反向传播
            loss.backward()
            
            # 密集化和剪枝
            if iteration < chunk_cfg.optim.densify_until_iter:
                gaussians.set_max_radii2D(radii, visibility_filter)
                gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter)
                
                if iteration > chunk_cfg.optim.densify_from_iter and iteration % chunk_cfg.optim.densification_interval == 0:
                    gaussians.densify_and_prune(
                        max_grad=chunk_cfg.optim.densify_grad_threshold,
                        min_opacity=chunk_cfg.optim.min_opacity,
                        prune_big_points=iteration > chunk_cfg.optim.opacity_reset_interval
                    )
                    
            # 重置不透明度
            if iteration % chunk_cfg.optim.opacity_reset_interval == 0:
                gaussians.reset_opacity()
                
            # 更新优化器
            gaussians.update_optimizer()
            
            # 更新进度条
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{loss.item():.6f}",
                    "Gaussians": gaussians.get_xyz.shape[0]
                })
        
        progress_bar.close()
        
        # 保存块模型
        chunk_model_path = os.path.join(chunk_info.model_path, "chunk_model.pth")
        state_dict = gaussians.save_state_dict(is_final=True)
        state_dict['iteration'] = chunk_cfg.train.iterations
        torch.save(state_dict, chunk_model_path)
        
        # 计算和保存贡献度
        contributions = evaluator.compute_contribution_scores(chunk_cfg.train.iterations)
        contribution_path = os.path.join(chunk_info.model_path, "contributions.json")
        
        # 将贡献度转换为可序列化的格式
        contrib_dict = {}
        for gaussian_id, contrib in contributions.items():
            contrib_dict[str(gaussian_id)] = {
                'contribution_score': contrib.contribution_score,
                'visibility_count': contrib.visibility_count,
                'rendering_frequency': contrib.rendering_frequency,
                'avg_opacity': contrib.avg_opacity,
                'avg_scale': contrib.avg_scale
            }
            
        with open(contribution_path, 'w') as f:
            json.dump(contrib_dict, f, indent=2)
            
        # 剔除低贡献度的高斯基元
        pruner = GaussianPruner(contribution_threshold=0.1)
        keep_mask, pruned_count = pruner.prune_gaussians(gaussians, contributions)
        
        # 应用剪枝掩码到高斯模型
        self._apply_pruning_mask(gaussians, keep_mask)
        
        print(f"Chunk {chunk_info.chunk_id} training completed. "
              f"Final gaussian count: {gaussians.get_xyz.shape[0]}")
        
        # 保存最终的块信息
        self.chunk_models[chunk_info.chunk_id] = gaussians
        self.contribution_evaluators[chunk_info.chunk_id] = evaluator
        
        return gaussians
    
    def _apply_pruning_mask(self, gaussians: StreetGaussianModel, keep_mask: torch.Tensor):
        """应用剪枝掩码到高斯模型"""
        # 这里需要为每个模型组件应用掩码
        for model_name in gaussians.model_name_id.keys():
            model = getattr(gaussians, model_name)
            if hasattr(model, '_xyz'):
                # 应用掩码到模型参数
                current_size = model._xyz.shape[0]
                model_keep_mask = keep_mask[:current_size] if current_size <= len(keep_mask) else keep_mask
                
                if model_keep_mask.sum() > 0:  # 确保至少保留一些高斯基元
                    model._xyz = model._xyz[model_keep_mask]
                    model._features_dc = model._features_dc[model_keep_mask]
                    model._features_rest = model._features_rest[model_keep_mask]
                    model._scaling = model._scaling[model_keep_mask] 
                    model._rotation = model._rotation[model_keep_mask]
                    model._opacity = model._opacity[model_keep_mask]
                    
                    if hasattr(model, '_semantic'):
                        model._semantic = model._semantic[model_keep_mask]

    def train_all_chunks(self, dataset: Dataset) -> StreetGaussianModel:
        """训练所有块并拼接成完整场景"""
        print("Starting chunked training...")
        
        # 创建训练块
        chunks = self.create_chunks(dataset)
        
        # 训练每个块
        chunk_models = {}
        for chunk_info in chunks:
            trained_model = self.train_chunk(chunk_info, dataset)
            chunk_models[chunk_info.chunk_id] = trained_model
            
            # 清理GPU内存
            torch.cuda.empty_cache()
        
        # 拼接所有块
        from lib.models.scene_merger import SceneMerger
        merger = SceneMerger()
        merged_model = merger.merge_chunks(chunk_models, chunks, dataset.scene_info.metadata)
        
        # 保存最终模型
        final_model_path = os.path.join(cfg.model_path, "final_merged_model.pth")
        merger.save_merged_model(merged_model, final_model_path)
        
        return merged_model
    
    def get_memory_usage_stats(self) -> Dict[str, float]:
        """获取内存使用统计"""
        stats = {}
        
        if torch.cuda.is_available():
            # GPU内存统计
            stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3    # GB
            stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
        return stats