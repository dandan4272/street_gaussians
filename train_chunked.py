import os
import torch
import argparse
import time
from random import randint
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.general_utils import safe_state
from lib.utils.camera_utils import Camera
from lib.utils.cfg_utils import save_cfg
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.config import cfg
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from lib.utils.system_utils import searchForMaxIteration

# 导入我们的新组件
from lib.models.chunk_trainer import ChunkTrainer
from lib.models.contribution_evaluator import ContributionEvaluator
from lib.models.scene_stitcher import SceneStitcher

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training_chunk(chunk_info, chunk_dataset, tb_writer, chunk_id):
    """
    训练单个块
    
    Args:
        chunk_info: 块信息
        chunk_dataset: 块数据集
        tb_writer: TensorBoard写入器
        chunk_id: 块ID
        
    Returns:
        trained_gaussians: 训练好的高斯模型
    """
    print(f"\n=== Training Chunk {chunk_id} ===")
    print(f"Frames: {chunk_info['start_frame']}-{chunk_info['end_frame']}")
    print(f"Number of cameras: {chunk_info['num_cameras']}")
    
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data

    start_iter = 0
    
    # 创建高斯模型和场景
    gaussians = StreetGaussianModel(chunk_dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=chunk_dataset)
    gaussians.training_setup()
    
    # 尝试加载已有的检查点
    chunk_model_dir = os.path.join(cfg.model_path, f"chunk_{chunk_id}")
    os.makedirs(chunk_model_dir, exist_ok=True)
    
    try:
        ckpt_path = os.path.join(chunk_model_dir, f'latest.pth')
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path)
            start_iter = state_dict.get('iter', 0)
            print(f'Loading chunk {chunk_id} model from {ckpt_path}')
            gaussians.load_state_dict(state_dict)
    except Exception as e:
        print(f"Could not load checkpoint for chunk {chunk_id}: {e}")
        
    gaussians_renderer = StreetGaussianRenderer()

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    
    # 调整训练迭代数（每个块使用较少的迭代）
    chunk_iterations = min(training_args.iterations // 2, 15000)  # 每个块最多15k迭代
    progress_bar = tqdm(range(start_iter, chunk_iterations))
    start_iter += 1

    viewpoint_stack = None
    
    for iteration in range(start_iter, chunk_iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代增加SH度
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择相机
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        if len(viewpoint_stack) == 0:
            viewpoint_stack = scene.getTrainCameras().copy()
            
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        # 获取图像和掩码
        gt_image = viewpoint_cam.original_image
        mask = viewpoint_cam.guidance.get('mask', torch.ones_like(gt_image[0:1]).bool())
        gt_image = gt_image.cuda(non_blocking=True) if not gt_image.is_cuda else gt_image
        mask = mask.cuda(non_blocking=True) if not mask.is_cuda else mask
        
        # 获取其他指导信息
        lidar_depth = None
        sky_mask = None
        obj_bound = None
        
        if 'lidar_depth' in viewpoint_cam.guidance:
            lidar_depth = viewpoint_cam.guidance['lidar_depth']
            lidar_depth = lidar_depth.cuda(non_blocking=True) if not lidar_depth.is_cuda else lidar_depth
            
        if 'sky_mask' in viewpoint_cam.guidance:
            sky_mask = viewpoint_cam.guidance['sky_mask']
            sky_mask = sky_mask.cuda(non_blocking=True) if not sky_mask.is_cuda else sky_mask
            
        if 'obj_bound' in viewpoint_cam.guidance:
            obj_bound = viewpoint_cam.guidance['obj_bound']
            obj_bound = obj_bound.cuda(non_blocking=True) if not obj_bound.is_cuda else obj_bound
        
        # 渲染
        render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians)
        image, acc, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["rgb"], render_pkg['acc'], render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], render_pkg["radii"]
        )
        depth = render_pkg['depth']

        scalar_dict = dict()
        
        # RGB损失
        Ll1 = l1_loss(image, gt_image, mask)
        scalar_dict['l1_loss'] = Ll1.item()
        loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + \
               optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=mask))
    
        # 天空损失
        if optim_args.lambda_sky > 0 and gaussians.include_sky and sky_mask is not None:
            acc = torch.clamp(acc, min=1e-6, max=1.-1e-6)
            sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()
            scalar_dict['sky_loss'] = sky_loss.item()
            loss += optim_args.lambda_sky * sky_loss
        
        # 物体正则化损失
        if optim_args.lambda_reg > 0 and gaussians.include_obj and iteration >= optim_args.densify_until_iter:
            render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians, parse_camera_again=False)
            image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = torch.clamp(acc_obj, min=1e-6, max=1.-1e-6)
            obj_acc_loss = torch.where(obj_bound, 
                -(acc_obj * torch.log(acc_obj) + (1. - acc_obj) * torch.log(1. - acc_obj)), 
                -torch.log(1. - acc_obj)).mean()
            scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
            loss += optim_args.lambda_reg * obj_acc_loss

        # LiDAR深度损失
        if optim_args.lambda_depth_lidar > 0 and lidar_depth is not None:            
            depth_mask = torch.logical_and((lidar_depth > 0.), mask)
            expected_depth = depth / (render_pkg['acc'] + 1e-10)  
            depth_error = torch.abs((expected_depth[depth_mask] - lidar_depth[depth_mask]))
            depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
            lidar_depth_loss = depth_error.mean()
            scalar_dict['lidar_depth_loss'] = lidar_depth_loss
            loss += optim_args.lambda_depth_lidar * lidar_depth_loss
                    
        # 颜色校正损失
        if optim_args.lambda_color_correction > 0 and gaussians.use_color_correction:
            color_correction_reg_loss = gaussians.color_correction.regularization_loss(viewpoint_cam)
            scalar_dict['color_correction_reg_loss'] = color_correction_reg_loss.item()
            loss += optim_args.lambda_color_correction * color_correction_reg_loss
                    
        scalar_dict['loss'] = loss.item()
        
        # 反向传播
        loss.backward()
        
        iter_end.record()
        
        with torch.no_grad():
            # 日志记录
            if iteration % 10 == 0:                    
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_psnr_for_log = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * ema_psnr_for_log
                progress_bar.set_postfix({
                    "Chunk": f"{chunk_id}", 
                    "Loss": f"{ema_loss_for_log:.{7}f},", 
                    "PSNR": f"{ema_psnr_for_log:.{4}f}"
                })
            progress_bar.update(1)

            # 密集化
            if iteration < optim_args.densify_until_iter:
                gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys()) - set(['sky'])))
                gaussians.set_max_radii2D(radii, visibility_filter)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                prune_big_points = iteration > optim_args.opacity_reset_interval

                if iteration > optim_args.densify_from_iter:
                    if iteration % optim_args.densification_interval == 0:
                        scalars, tensors = gaussians.densify_and_prune(
                            max_grad=optim_args.densify_grad_threshold,
                            min_opacity=optim_args.min_opacity,
                            prune_big_points=prune_big_points,
                        )
                        scalar_dict.update(scalars)
                        
            # 重置不透明度
            if iteration < optim_args.densify_until_iter:
                if iteration % optim_args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            # 优化器步骤
            if iteration < chunk_iterations:
                gaussians.update_optimizer()

            # 保存检查点
            if iteration % 5000 == 0 or iteration == chunk_iterations:
                print(f"\n[ITER {iteration}] Saving Chunk {chunk_id} Checkpoint")
                state_dict = gaussians.save_state_dict(is_final=(iteration == chunk_iterations))
                state_dict['iter'] = iteration
                ckpt_path = os.path.join(chunk_model_dir, f'iteration_{iteration}.pth')
                torch.save(state_dict, ckpt_path)
                
                # 保存为最新检查点
                latest_path = os.path.join(chunk_model_dir, 'latest.pth')
                torch.save(state_dict, latest_path)

    progress_bar.close()
    
    # 在训练结束后评估贡献度并剪枝
    print(f"\nEvaluating contributions for chunk {chunk_id}...")
    contribution_evaluator = ContributionEvaluator(num_evaluation_frames=5)
    
    eval_cameras = scene.getTrainCameras()[:10]  # 使用前10个相机评估
    contribution_scores = contribution_evaluator.compute_contribution_scores(
        gaussians, eval_cameras, method='hybrid'
    )
    
    # 适度剪枝低贡献的高斯基元
    pruned_counts = contribution_evaluator.adaptive_prune_by_contribution(
        gaussians, contribution_scores, target_reduction_ratio=0.2
    )
    
    print(f"Chunk {chunk_id} training completed!")
    print(f"Pruned gaussians: {pruned_counts}")
    
    return gaussians

def chunk_training_pipeline():
    """
    分块训练流水线主函数
    """
    print("=== Starting Chunk Training Pipeline ===")
    
    training_args = cfg.train
    data_args = cfg.data

    # 设置输出目录
    tb_writer = prepare_output_and_logger()
    
    # 加载完整数据集
    print("Loading full dataset...")
    full_dataset = Dataset()
    
    # 计算总帧数
    all_cameras = full_dataset.getTrainCameras() + full_dataset.getTestCameras()
    frame_ids = [cam.meta['frame'] for cam in all_cameras if 'frame' in cam.meta]
    total_frames = len(set(frame_ids))
    
    print(f"Total frames in dataset: {total_frames}")
    print(f"Total cameras: {len(all_cameras)}")
    
    # 创建分块训练器
    chunk_size = getattr(cfg, 'chunk_size', 5)  # 默认每块5帧
    overlap_size = getattr(cfg, 'overlap_size', 1)  # 默认重叠1帧
    
    chunk_trainer = ChunkTrainer(
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        min_chunk_frames=3,
        max_memory_usage=0.8
    )
    
    # 创建训练块
    print(f"Creating chunks with size {chunk_size} and overlap {overlap_size}...")
    chunks_info = chunk_trainer.create_chunks(total_frames, all_cameras)
    
    # 保存块信息
    chunk_trainer.save_chunk_info(cfg.model_path)
    
    # 训练每个块
    trained_models = []
    trained_chunks_info = []
    
    for i, chunk_info in enumerate(chunks_info):
        print(f"\n{'='*50}")
        print(f"Processing Chunk {i+1}/{len(chunks_info)}")
        print(f"{'='*50}")
        
        # 创建块数据集
        chunk_dataset = chunk_trainer.get_chunk_dataset(chunk_info, full_dataset)
        
        # 检查是否需要跳过（显存不足等）
        if chunk_trainer.should_reduce_chunk_size(chunk_info):
            print(f"Warning: Chunk {i} may exceed memory limits, consider reducing chunk size")
        
        # 训练块
        try:
            trained_gaussians = training_chunk(
                chunk_info, chunk_dataset, tb_writer, i
            )
            trained_models.append(trained_gaussians)
            trained_chunks_info.append(chunk_info)
            
        except torch.cuda.OutOfMemoryError:
            print(f"GPU memory exceeded for chunk {i}, skipping...")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"Error training chunk {i}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("All chunks trained successfully!")
    print(f"{'='*50}")
    
    # 拼接所有块
    if len(trained_models) > 1:
        print("\nStarting scene stitching...")
        
        stitcher = SceneStitcher(
            overlap_threshold=0.1,
            similarity_threshold=0.05,
            blend_region_size=0.3
        )
        
        # 拼接场景
        stitched_model = stitcher.stitch_chunks(
            trained_models, trained_chunks_info, full_dataset.scene_info.metadata
        )
        
        # 保存拼接后的模型
        final_model_path = os.path.join(cfg.model_path, "final_stitched_model.pth")
        stitcher.save_stitched_model(stitched_model, final_model_path)
        
        print(f"Final stitched model saved to {final_model_path}")
        
        # 最终的贡献度评估和剪枝
        print("\nFinal optimization...")
        final_evaluator = ContributionEvaluator(num_evaluation_frames=20)
        
        # 使用全数据集的一部分相机进行最终评估
        eval_cameras = all_cameras[::max(1, len(all_cameras)//20)]  # 采样20个相机
        final_contribution_scores = final_evaluator.compute_contribution_scores(
            stitched_model, eval_cameras, method='hybrid'
        )
        
        # 最终剪枝
        final_pruned_counts = final_evaluator.adaptive_prune_by_contribution(
            stitched_model, final_contribution_scores, target_reduction_ratio=0.3
        )
        
        print(f"Final pruning completed: {final_pruned_counts}")
        
        # 保存最终优化的模型
        optimized_model_path = os.path.join(cfg.model_path, "final_optimized_model.pth")
        stitcher.save_stitched_model(stitched_model, optimized_model_path)
        
    else:
        print("Only one chunk was trained successfully, no stitching needed.")
        if len(trained_models) == 1:
            single_model_path = os.path.join(cfg.model_path, "single_chunk_model.pth")
            state_dict = trained_models[0].save_state_dict(is_final=True)
            torch.save(state_dict, single_model_path)
            print(f"Single chunk model saved to {single_model_path}")

def prepare_output_and_logger():
    """
    准备输出和日志器
    """
    print("Output folder: {}".format(cfg.model_path))

    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.trained_model_dir, exist_ok=True)
    os.makedirs(cfg.record_dir, exist_ok=True)
    
    if not cfg.resume:
        os.system('rm -rf {}/*'.format(cfg.record_dir))

    with open(os.path.join(cfg.model_path, "cfg_args"), 'w') as cfg_log_f:
        viewer_arg = dict()
        viewer_arg['sh_degree'] = cfg.model.gaussian.sh_degree
        viewer_arg['white_background'] = cfg.data.white_background
        viewer_arg['source_path'] = cfg.source_path
        viewer_arg['model_path'] = cfg.model_path
        cfg_log_f.write(str(Namespace(**viewer_arg)))

    # 创建TensorBoard写入器
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.record_dir)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__ == "__main__":
    # 解析命令行参数
    parser = ArgumentParser(description="Chunked Training script parameters")
    parser.add_argument('--chunk_size', type=int, default=5, help='Number of frames per chunk')
    parser.add_argument('--overlap_size', type=int, default=1, help='Number of overlapping frames between chunks')
    parser.add_argument('--max_memory_usage', type=float, default=0.8, help='Maximum GPU memory usage ratio')
    
    args = parser.parse_args()
    
    # 设置分块训练参数
    cfg.chunk_size = args.chunk_size
    cfg.overlap_size = args.overlap_size
    cfg.max_memory_usage = args.max_memory_usage
    
    print("Chunked Training Configuration:")
    print(f"  Chunk size: {cfg.chunk_size} frames")
    print(f"  Overlap size: {cfg.overlap_size} frames")
    print(f"  Max memory usage: {cfg.max_memory_usage}")
    print(f"  Model path: {cfg.model_path}")

    # 初始化系统状态
    safe_state(cfg.train.quiet)

    # 开始分块训练
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)
    chunk_training_pipeline()

    print("\nChunked training pipeline completed!")