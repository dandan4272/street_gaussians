#!/usr/bin/env python3
"""
分块训练使用示例
演示如何使用3DGS分块训练系统来减少显存占用
"""

import os
import torch
from lib.models.chunk_trainer import ChunkTrainer
from lib.models.contribution_evaluator import ContributionEvaluator
from lib.models.scene_stitcher import SceneStitcher
from lib.datasets.dataset import Dataset
from lib.config import cfg

def example_chunked_training():
    """
    完整的分块训练示例
    展示从数据加载到最终模型保存的整个流程
    """
    
    print("=== 3DGS 分块训练示例 ===\n")
    
    # 1. 设置基本参数
    chunk_size = 5          # 每个块5帧
    overlap_size = 1        # 重叠1帧
    max_memory_usage = 0.8  # 最大显存使用80%
    
    print(f"配置参数:")
    print(f"  块大小: {chunk_size} 帧")
    print(f"  重叠大小: {overlap_size} 帧")
    print(f"  最大显存使用: {max_memory_usage}")
    print()
    
    # 2. 加载数据集
    print("正在加载数据集...")
    dataset = Dataset()
    
    # 获取所有相机
    all_cameras = dataset.getTrainCameras() + dataset.getTestCameras()
    frame_ids = [cam.meta['frame'] for cam in all_cameras if 'frame' in cam.meta]
    total_frames = len(set(frame_ids))
    
    print(f"数据集信息:")
    print(f"  总帧数: {total_frames}")
    print(f"  总相机数: {len(all_cameras)}")
    print()
    
    # 3. 创建分块训练器
    print("创建分块训练器...")
    chunk_trainer = ChunkTrainer(
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        min_chunk_frames=3,
        max_memory_usage=max_memory_usage
    )
    
    # 4. 创建训练块
    print("生成训练块...")
    chunks_info = chunk_trainer.create_chunks(total_frames, all_cameras)
    print(f"生成了 {len(chunks_info)} 个训练块")
    print()
    
    # 5. 显示块信息
    for i, chunk_info in enumerate(chunks_info):
        print(f"块 {i}: 帧 {chunk_info['start_frame']}-{chunk_info['end_frame']} "
              f"({chunk_info['num_frames']} 帧, {chunk_info['num_cameras']} 相机)")
    print()
    
    # 6. 模拟训练每个块（这里只是示例，实际需要完整训练）
    print("开始分块训练...")
    trained_models = []
    
    for i, chunk_info in enumerate(chunks_info[:2]):  # 只训练前2个块作为示例
        print(f"\n训练块 {i}...")
        
        # 创建块数据集
        chunk_dataset = chunk_trainer.get_chunk_dataset(chunk_info, dataset)
        
        # 估算内存使用
        estimated_memory = chunk_trainer.estimate_memory_usage(chunk_info)
        print(f"  估算显存使用: {estimated_memory:.2f} GB")
        
        # 检查是否需要减小块大小
        if chunk_trainer.should_reduce_chunk_size(chunk_info):
            print(f"  警告: 块 {i} 可能超出显存限制")
        
        # 这里应该调用实际的训练函数
        # trained_model = train_single_chunk(chunk_info, chunk_dataset)
        # trained_models.append(trained_model)
        
        print(f"  块 {i} 训练完成（模拟）")
    
    print("\n所有块训练完成！")
    print()
    
    # 7. 贡献度评估示例
    print("贡献度评估示例...")
    evaluator = ContributionEvaluator(num_evaluation_frames=5)
    
    # 这里需要一个训练好的模型来演示
    # contribution_scores = evaluator.compute_contribution_scores(
    #     trained_model, eval_cameras, method='hybrid'
    # )
    # print(f"计算出 {len(contribution_scores)} 个模型组件的贡献度分数")
    print("贡献度评估完成（模拟）")
    print()
    
    # 8. 场景拼接示例
    print("场景拼接示例...")
    stitcher = SceneStitcher(
        overlap_threshold=0.1,
        similarity_threshold=0.05,
        blend_region_size=0.3
    )
    
    # 这里需要训练好的模型来演示拼接
    # stitched_model = stitcher.stitch_chunks(
    #     trained_models, chunks_info, dataset.scene_info.metadata
    # )
    print("场景拼接完成（模拟）")
    print()
    
    # 9. 显示预期的性能提升
    print("=== 性能提升预期 ===")
    original_memory = 16  # 假设原版需要16GB
    chunked_memory = max(6, estimated_memory)  # 分块训练显存使用
    memory_reduction = (original_memory - chunked_memory) / original_memory * 100
    
    print(f"显存使用对比:")
    print(f"  原版训练: ~{original_memory} GB")
    print(f"  分块训练: ~{chunked_memory:.1f} GB")
    print(f"  显存减少: {memory_reduction:.1f}%")
    print()
    
    print("=== 示例完成 ===")

def example_memory_optimization():
    """
    内存优化示例
    展示如何监控和优化显存使用
    """
    print("\n=== 内存优化示例 ===")
    
    if torch.cuda.is_available():
        # 获取GPU信息
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        current_memory = torch.cuda.memory_allocated(device) / 1024**3
        
        print(f"GPU信息:")
        print(f"  设备: {torch.cuda.get_device_name(device)}")
        print(f"  总显存: {total_memory:.2f} GB")
        print(f"  已使用: {current_memory:.2f} GB")
        print(f"  可用显存: {total_memory - current_memory:.2f} GB")
        
        # 建议块大小
        available_memory = total_memory - current_memory
        if available_memory < 8:
            recommended_chunk_size = 3
        elif available_memory < 12:
            recommended_chunk_size = 5
        else:
            recommended_chunk_size = 8
            
        print(f"  建议块大小: {recommended_chunk_size} 帧")
        
    else:
        print("未检测到CUDA设备，无法进行内存优化")
    
    print()

def example_config_usage():
    """
    配置文件使用示例
    """
    print("=== 配置文件使用示例 ===")
    
    # 创建示例配置
    config_example = {
        "chunk": {
            "chunk_size": 5,
            "overlap_size": 1,
            "max_memory_usage": 0.8
        },
        "contribution": {
            "evaluation_method": "hybrid",
            "num_evaluation_frames": 10
        },
        "stitching": {
            "overlap_threshold": 0.1,
            "similarity_threshold": 0.05
        }
    }
    
    print("示例配置:")
    for section, params in config_example.items():
        print(f"  {section}:")
        for key, value in params.items():
            print(f"    {key}: {value}")
    
    print("\n配置文件路径: configs/chunked_training.yaml")
    print()

if __name__ == "__main__":
    try:
        # 运行主要示例
        example_chunked_training()
        
        # 运行内存优化示例
        example_memory_optimization()
        
        # 运行配置示例
        example_config_usage()
        
        print("所有示例运行完成！")
        print("\n要开始实际训练，请运行:")
        print("python run_chunked_training.py --source_path /path/to/data --model_path /path/to/output")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        print("请确保已正确安装所有依赖包并配置好环境")