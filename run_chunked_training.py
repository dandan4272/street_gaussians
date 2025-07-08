#!/usr/bin/env python3
"""
分块训练运行脚本
用于启动3DGS的分块训练流水线，减少显存占用并支持更长序列的训练
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run chunked 3DGS training')
    
    # 基本参数
    parser.add_argument('--source_path', type=str, required=True,
                       help='Path to the source dataset')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to save the trained model')
    parser.add_argument('--config', type=str, default='configs/chunked_training.yaml',
                       help='Path to config file')
    
    # 分块参数
    parser.add_argument('--chunk_size', type=int, default=5,
                       help='Number of frames per chunk (default: 5)')
    parser.add_argument('--overlap_size', type=int, default=1,
                       help='Number of overlapping frames between chunks (default: 1)')
    parser.add_argument('--max_memory_usage', type=float, default=0.8,
                       help='Maximum GPU memory usage ratio (default: 0.8)')
    
    # 训练参数
    parser.add_argument('--iterations', type=int, default=15000,
                       help='Number of training iterations per chunk (default: 15000)')
    parser.add_argument('--eval_method', type=str, default='hybrid',
                       choices=['gradient_based', 'visibility_based', 'hybrid'],
                       help='Contribution evaluation method (default: hybrid)')
    
    # 剪枝参数
    parser.add_argument('--chunk_prune_ratio', type=float, default=0.2,
                       help='Pruning ratio for each chunk (default: 0.2)')
    parser.add_argument('--final_prune_ratio', type=float, default=0.3,
                       help='Final pruning ratio for stitched model (default: 0.3)')
    
    # 其他参数
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from existing checkpoints')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # 验证路径
    if not os.path.exists(args.source_path):
        print(f"Error: Source path {args.source_path} does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} does not exist")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.model_path, exist_ok=True)
    
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # 构建命令
    cmd = [
        sys.executable, 'train_chunked.py',
        '--source_path', args.source_path,
        '--model_path', args.model_path,
        '--config', args.config,
        '--chunk_size', str(args.chunk_size),
        '--overlap_size', str(args.overlap_size),
        '--max_memory_usage', str(args.max_memory_usage)
    ]
    
    if args.resume:
        cmd.append('--resume')
    
    if args.quiet:
        cmd.append('--quiet')
    
    # 打印配置信息
    print("=" * 60)
    print("Chunked 3DGS Training Configuration")
    print("=" * 60)
    print(f"Source path:        {args.source_path}")
    print(f"Model path:         {args.model_path}")
    print(f"Config file:        {args.config}")
    print(f"Chunk size:         {args.chunk_size} frames")
    print(f"Overlap size:       {args.overlap_size} frames")
    print(f"Max memory usage:   {args.max_memory_usage}")
    print(f"Iterations per chunk: {args.iterations}")
    print(f"Evaluation method:  {args.eval_method}")
    print(f"Chunk prune ratio:  {args.chunk_prune_ratio}")
    print(f"Final prune ratio:  {args.final_prune_ratio}")
    print(f"GPU ID:             {args.gpu_id}")
    print(f"Resume training:    {args.resume}")
    print("=" * 60)
    
    # 运行训练
    try:
        print("Starting chunked training...")
        result = subprocess.run(cmd, env=env, check=True)
        print("Chunked training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()