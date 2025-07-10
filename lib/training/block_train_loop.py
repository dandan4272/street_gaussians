import os
import torch
from random import randint
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.scene import Scene
from lib.config import cfg
from tqdm import tqdm
import time

def train_single_block(scene: Scene, gaussians: StreetGaussianModel, block_config, iterations: int) -> StreetGaussianModel:
    """训练单个块的函数"""
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data
    
    print(f"Training block {block_config.block_id} for {iterations} iterations")
    
    # 创建渲染器
    gaussians_renderer = StreetGaussianRenderer()
    
    # 训练统计
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    
    # 进度条
    progress_bar = tqdm(range(1, iterations + 1), desc=f"Block {block_config.block_id}")
    
    viewpoint_stack = None
    
    for iteration in range(1, iterations + 1):
        gaussians.update_learning_rate(iteration)
        
        # 每1000次迭代增加SH度数
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # 随机选择相机
        if not viewpoint_stack:
            train_cameras = scene.getTrainCameras()
            if len(train_cameras) == 0:
                print(f"Warning: No training cameras for block {block_config.block_id}")
                break
            viewpoint_stack = train_cameras.copy()
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        # 准备训练数据
        gt_image = viewpoint_cam.original_image
        mask = viewpoint_cam.guidance.get('mask', torch.ones_like(gt_image[0:1]).bool())
        gt_image = gt_image.cuda() if not gt_image.is_cuda else gt_image
        mask = mask.cuda() if not mask.is_cuda else mask
        
        # 渲染
        render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians)
        image = render_pkg["rgb"]
        acc = render_pkg['acc']
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        depth = render_pkg['depth']
        
        # 计算损失
        Ll1 = l1_loss(image, gt_image, mask)
        loss = ((1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + 
                optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=mask)))
        
        # 反向传播
        loss.backward()
        
        with torch.no_grad():
            # 更新进度条
            if iteration % 10 == 0:
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_psnr_for_log = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * ema_psnr_for_log
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.7f}",
                    "PSNR": f"{ema_psnr_for_log:.4f}"
                })
            progress_bar.update(1)
            
            # 密化和修剪
            if iteration < optim_args.densify_until_iter:
                gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys()) - set(['sky'])))
                gaussians.set_max_radii2D(radii, visibility_filter)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if iteration > optim_args.densify_from_iter and iteration % optim_args.densification_interval == 0:
                    gaussians.densify_and_prune(
                        max_grad=optim_args.densify_grad_threshold,
                        min_opacity=optim_args.min_opacity,
                        prune_big_points=iteration > optim_args.opacity_reset_interval,
                    )
            
            # 重置不透明度
            if iteration < optim_args.densify_until_iter and iteration % optim_args.opacity_reset_interval == 0:
                gaussians.reset_opacity()
            
            # 更新优化器
            if iteration < iterations:
                gaussians.update_optimizer()
    
    progress_bar.close()
    print(f"Block {block_config.block_id} training completed")
    return gaussians