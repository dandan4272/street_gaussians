#!/usr/bin/env python3
"""
Street Gaussians 分块训练脚本
用于减少训练时的显存占用，支持长序列的分块训练和拼接
"""

import os
import sys
import torch
import argparse
import time
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.models.chunked_trainer import ChunkedTrainer
from lib.models.scene_merger import SceneMerger
from lib.datasets.dataset import Dataset
from lib.config import cfg
from lib.utils.general_utils import safe_state

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Street Gaussians Chunked Training")
    
    parser.add_argument("--config", default="configs/default.yaml", type=str,
                       help="Path to config file")
    parser.add_argument("--chunk_size", default=5, type=int,
                       help="Number of frames per training chunk")
    parser.add_argument("--overlap_size", default=1, type=int,
                       help="Number of overlapping frames between chunks")
    parser.add_argument("--contribution_threshold", default=0.1, type=float,
                       help="Threshold for pruning low-contribution gaussians")
    parser.add_argument("--max_memory_gb", default=12.0, type=float,
                       help="Maximum GPU memory to use (GB)")
    parser.add_argument("--output_dir", default="", type=str,
                       help="Override output directory")
    parser.add_argument("--resume_chunk", default=-1, type=int,
                       help="Resume training from specific chunk (-1 to start from beginning)")
    
    return parser.parse_args()

def setup_memory_management(max_memory_gb: float):
    """设置内存管理"""
    if torch.cuda.is_available():
        # 设置GPU内存分配策略
        torch.cuda.set_per_process_memory_fraction(min(0.95, max_memory_gb / torch.cuda.get_device_properties(0).total_memory * 1024**3))
        
        # 启用内存池
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        print(f"GPU Memory Management:")
        print(f"  Device: {torch.cuda.get_device_name()}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  Max Usage: {max_memory_gb:.2f} GB")

def monitor_memory_usage(stage: str = ""):
    """监控内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"Memory Usage {stage}:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Max Allocated: {max_allocated:.2f} GB")

def save_training_summary(trainer: ChunkedTrainer, output_path: str):
    """保存训练摘要"""
    summary = {
        "chunk_size": trainer.chunk_size,
        "overlap_size": trainer.overlap_size,
        "total_chunks": len(trainer.trained_chunks),
        "chunk_details": []
    }
    
    for chunk_info in trainer.trained_chunks:
        chunk_detail = {
            "chunk_id": chunk_info.chunk_id,
            "start_frame": chunk_info.start_frame,
            "end_frame": chunk_info.end_frame,
            "frame_count": len(chunk_info.frame_indices),
            "model_path": chunk_info.model_path
        }
        summary["chunk_details"].append(chunk_detail)
    
    # 保存内存使用统计
    memory_stats = trainer.get_memory_usage_stats()
    summary["memory_stats"] = memory_stats
    
    import json
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to {output_path}")

def main():
    """主函数"""
    args = parse_arguments()
    
    print("=" * 80)
    print("Street Gaussians - Chunked Training")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Chunk Size: {args.chunk_size}")
    print(f"Overlap Size: {args.overlap_size}")
    print(f"Max Memory: {args.max_memory_gb} GB")
    print("=" * 80)
    
    # 设置输出目录
    if args.output_dir:
        cfg.model_path = args.output_dir
    
    # 初始化随机种子
    safe_state(cfg.train.quiet)
    
    # 设置内存管理
    setup_memory_management(args.max_memory_gb)
    
    # 创建输出目录
    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(os.path.join(cfg.model_path, "chunks"), exist_ok=True)
    
    # 加载数据集
    print("\nLoading dataset...")
    dataset = Dataset()
    
    total_frames = len(dataset.getTrainCameras())
    print(f"Total frames: {total_frames}")
    
    monitor_memory_usage("after dataset loading")
    
    # 创建分块训练器
    trainer = ChunkedTrainer(
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size
    )
    
    start_time = time.time()
    
    try:
        # 执行分块训练
        print("\nStarting chunked training...")
        merged_model = trainer.train_all_chunks(dataset)
        
        training_time = time.time() - start_time
        
        print(f"\n" + "=" * 80)
        print("Training Completed Successfully!")
        print(f"Total training time: {training_time / 3600:.2f} hours")
        print(f"Final model saved to: {cfg.model_path}")
        
        # 保存训练摘要
        summary_path = os.path.join(cfg.model_path, "training_summary.json")
        save_training_summary(trainer, summary_path)
        
        # 最终内存使用统计
        monitor_memory_usage("final")
        
        # 生成性能报告
        generate_performance_report(trainer, training_time, total_frames)
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def generate_performance_report(trainer: ChunkedTrainer, training_time: float, total_frames: int):
    """生成性能报告"""
    print(f"\n" + "=" * 80)
    print("Performance Report")
    print("=" * 80)
    
    # 计算统计信息
    num_chunks = len(trainer.chunk_models)
    avg_time_per_chunk = training_time / num_chunks if num_chunks > 0 else 0
    avg_time_per_frame = training_time / total_frames if total_frames > 0 else 0
    
    print(f"Training Statistics:")
    print(f"  Total Chunks: {num_chunks}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Avg Time per Chunk: {avg_time_per_chunk / 60:.2f} minutes")
    print(f"  Avg Time per Frame: {avg_time_per_frame:.2f} seconds")
    
    # 内存使用统计
    memory_stats = trainer.get_memory_usage_stats()
    if memory_stats:
        print(f"\nMemory Usage:")
        for key, value in memory_stats.items():
            print(f"  {key}: {value:.2f} GB")
    
    # 高斯基元统计
    total_gaussians = 0
    for chunk_id, model in trainer.chunk_models.items():
        if hasattr(model, 'get_xyz'):
            chunk_gaussians = model.get_xyz.shape[0]
            total_gaussians += chunk_gaussians
            print(f"  Chunk {chunk_id}: {chunk_gaussians:,} gaussians")
    
    print(f"\nTotal Gaussians across all chunks: {total_gaussians:,}")
    
    # 估算内存节省
    estimated_full_memory = total_frames * 8  # 假设每帧8GB
    actual_max_memory = memory_stats.get('gpu_max_allocated', 0)
    memory_savings = max(0, estimated_full_memory - actual_max_memory)
    
    print(f"\nMemory Efficiency:")
    print(f"  Estimated Full Training Memory: {estimated_full_memory:.2f} GB")
    print(f"  Actual Max Memory Used: {actual_max_memory:.2f} GB")
    print(f"  Estimated Memory Savings: {memory_savings:.2f} GB ({memory_savings/estimated_full_memory*100:.1f}%)")

if __name__ == "__main__":
    main()