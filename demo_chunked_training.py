#!/usr/bin/env python3
"""
Street Gaussians 分块训练演示脚本
快速测试分块训练系统的功能
"""

import os
import sys
import torch
import argparse
import shutil
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_demo_config(output_dir: str, data_path: str):
    """创建演示用的配置文件"""
    config_content = f"""
task: demo_chunked_training
source_path: {data_path}
exp_name: demo_experiment
to_cuda: false

# 演示用的小规模配置
chunked_training:
  enabled: true
  chunk_size: 3           # 小块用于快速演示
  overlap_size: 1
  contribution_threshold: 0.15
  max_memory_gb: 8.0      # 较低的内存要求
  prune_ratio: 0.75       # 保留75%的高斯基元

data:
  split_test: -1
  split_train: 1
  type: Waymo
  white_background: false
  extent: 10
  use_colmap: true
  filter_colmap: true
  box_scale: 1.0

model:
  gaussian:
    sh_degree: 1          # 最低SH度数
    fourier_dim: 2        # 最小傅里叶维度
    fourier_scale: 1.
    flip_prob: 0.0        # 禁用翻转以加速
  nsg:
    include_bkgd: true
    include_obj: false    # 暂时禁用物体以简化
    include_sky: false
    opt_track: false

train:
  iterations: 5000        # 快速演示用的少量迭代
  test_iterations: [2500, 5000]
  save_iterations: [5000]
  checkpoint_iterations: [5000]

optim:
  densification_interval: 300
  densify_from_iter: 500
  densify_grad_threshold: 0.0008
  densify_until_iter: 3000
  
  feature_lr: 0.001
  max_screen_size: 10
  min_opacity: 0.02
  opacity_lr: 0.03
  opacity_reset_interval: 1500
  percent_dense: 0.002
  
  position_lr_init: 0.0001
  position_lr_final: 0.00001
  position_lr_max_steps: 5000
  rotation_lr: 0.0005
  scaling_lr: 0.003

  lambda_dssim: 0.2
  lambda_sky: 0.0
  lambda_mask: 0.05
  lambda_reg: 0.05
  lambda_depth_lidar: 0.05

render:
  fps: 24
  scaling_modifier: 0.5
"""
    
    config_path = os.path.join(output_dir, "demo_config.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content.strip())
    
    return config_path

def run_demo():
    """运行演示"""
    parser = argparse.ArgumentParser(description="Street Gaussians Chunked Training Demo")
    parser.add_argument("--data_path", required=True, type=str,
                       help="Path to your dataset")
    parser.add_argument("--output_dir", default="./demo_output", type=str,
                       help="Output directory for demo results")
    parser.add_argument("--quick", action="store_true",
                       help="Run extra quick demo with minimal iterations")
    
    args = parser.parse_args()
    
    # 检查输入
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist!")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Street Gaussians 分块训练演示")
    print("=" * 60)
    print(f"数据路径: {args.data_path}")
    print(f"输出目录: {output_dir}")
    print(f"快速模式: {'是' if args.quick else '否'}")
    print("=" * 60)
    
    # 创建演示配置
    config_path = create_demo_config(str(output_dir), args.data_path)
    print(f"创建配置文件: {config_path}")
    
    # 如果是快速模式，进一步减少参数
    if args.quick:
        print("启用快速模式，减少训练参数...")
        # 这里可以进一步修改配置文件
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA，将使用CPU训练（会很慢）")
    else:
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"检测到GPU: {gpu_name}")
        print(f"GPU内存: {gpu_memory:.1f} GB")
    
    # 运行分块训练
    train_cmd = f"""
python train_chunked.py \\
    --config {config_path} \\
    --chunk_size 3 \\
    --overlap_size 1 \\
    --max_memory_gb 8.0 \\
    --output_dir {output_dir}
"""
    
    print("\n运行命令:")
    print(train_cmd.strip())
    print("\n开始训练...")
    
    # 执行训练
    os.system(train_cmd.replace('\\\n', ' ').replace('    ', ' '))
    
    # 检查结果
    check_results(output_dir)

def check_results(output_dir: Path):
    """检查训练结果"""
    print("\n" + "=" * 60)
    print("检查训练结果")
    print("=" * 60)
    
    # 检查输出文件
    expected_files = [
        "final_merged_model.pth",
        "final_merged_model.ply", 
        "training_summary.json"
    ]
    
    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {filename} (缺失)")
    
    # 检查块文件
    chunks_dir = output_dir / "chunks"
    if chunks_dir.exists():
        chunk_dirs = list(chunks_dir.glob("chunk_*"))
        print(f"\n训练块数量: {len(chunk_dirs)}")
        
        for chunk_dir in sorted(chunk_dirs):
            chunk_model = chunk_dir / "chunk_model.pth"
            chunk_contrib = chunk_dir / "contributions.json"
            
            status = "✓" if chunk_model.exists() and chunk_contrib.exists() else "✗"
            print(f"  {status} {chunk_dir.name}")
    
    # 读取训练摘要
    summary_file = output_dir / "training_summary.json"
    if summary_file.exists():
        import json
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            print(f"\n训练摘要:")
            print(f"  总块数: {summary.get('total_chunks', 'N/A')}")
            print(f"  块大小: {summary.get('chunk_size', 'N/A')}")
            print(f"  重叠大小: {summary.get('overlap_size', 'N/A')}")
            
            if 'memory_stats' in summary:
                memory = summary['memory_stats']
                print(f"  最大GPU内存: {memory.get('gpu_max_allocated', 0):.2f} GB")
                
        except Exception as e:
            print(f"无法读取训练摘要: {e}")
    
    print(f"\n演示完成！结果保存在: {output_dir}")
    print("\n下一步:")
    print("1. 查看生成的PLY文件以可视化结果")
    print("2. 检查training_summary.json了解详细统计")
    print("3. 尝试不同的参数组合优化性能")

def cleanup_demo(output_dir: str):
    """清理演示文件"""
    if os.path.exists(output_dir):
        response = input(f"删除演示输出目录 {output_dir}? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(output_dir)
            print(f"已删除 {output_dir}")
        else:
            print("保留输出文件")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示失败: {e}")
        import traceback
        traceback.print_exc()