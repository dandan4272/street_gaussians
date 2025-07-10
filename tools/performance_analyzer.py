#!/usr/bin/env python3
"""
Street Gaussian 分块训练性能分析工具

分析分块训练的性能指标，包括内存使用、训练时间、模型质量等。
"""

import os
import sys
import json
import time
import psutil
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = None
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.timestamps = []
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.timestamps = []
        
    def record_usage(self):
        """记录当前资源使用情况"""
        current_time = time.time()
        if self.start_time:
            elapsed = current_time - self.start_time
            self.timestamps.append(elapsed)
        
        # CPU内存使用
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_memory_mb = memory_info.rss / 1024 / 1024
        self.memory_usage.append(cpu_memory_mb)
        
        # GPU内存使用
        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        self.gpu_memory_usage.append(gpu_memory_mb)
        
        return {
            'cpu_memory_mb': cpu_memory_mb,
            'gpu_memory_mb': gpu_memory_mb,
            'timestamp': elapsed if self.start_time else 0
        }
        
    def get_summary(self) -> Dict:
        """获取性能总结"""
        if not self.memory_usage:
            return {}
            
        return {
            'total_time': self.timestamps[-1] if self.timestamps else 0,
            'peak_cpu_memory_mb': max(self.memory_usage),
            'avg_cpu_memory_mb': np.mean(self.memory_usage),
            'peak_gpu_memory_mb': max(self.gpu_memory_usage),
            'avg_gpu_memory_mb': np.mean(self.gpu_memory_usage),
            'memory_timeline': {
                'timestamps': self.timestamps,
                'cpu_memory': self.memory_usage,
                'gpu_memory': self.gpu_memory_usage
            }
        }

def analyze_block_training_performance(experiment_path: str) -> Dict:
    """分析分块训练性能"""
    
    analysis = {
        'experiment_path': experiment_path,
        'timestamp': datetime.now().isoformat(),
        'blocks': {},
        'aggregation': {},
        'overall': {}
    }
    
    # 分析各个块的性能
    block_dirs = [d for d in os.listdir(experiment_path) 
                  if d.startswith('block_') and os.path.isdir(os.path.join(experiment_path, d))]
    
    total_training_time = 0
    total_gaussians = 0
    
    for block_dir in sorted(block_dirs):
        block_path = os.path.join(experiment_path, block_dir)
        block_id = int(block_dir.split('_')[1])
        
        block_analysis = analyze_single_block(block_path, block_id)
        analysis['blocks'][block_id] = block_analysis
        
        if 'training_time' in block_analysis:
            total_training_time += block_analysis['training_time']
        if 'num_gaussians' in block_analysis:
            total_gaussians += block_analysis['num_gaussians']
    
    # 分析聚合性能
    aggregated_path = os.path.join(experiment_path, 'aggregated')
    if os.path.exists(aggregated_path):
        analysis['aggregation'] = analyze_aggregation_result(aggregated_path)
    
    # 整体分析
    analysis['overall'] = {
        'num_blocks': len(analysis['blocks']),
        'total_training_time': total_training_time,
        'total_gaussians_before_aggregation': total_gaussians,
        'avg_training_time_per_block': total_training_time / len(analysis['blocks']) if analysis['blocks'] else 0
    }
    
    if 'num_gaussians' in analysis['aggregation']:
        deduplication_ratio = 1 - (analysis['aggregation']['num_gaussians'] / total_gaussians) if total_gaussians > 0 else 0
        analysis['overall']['deduplication_ratio'] = deduplication_ratio
        analysis['overall']['final_gaussians'] = analysis['aggregation']['num_gaussians']
    
    return analysis

def analyze_single_block(block_path: str, block_id: int) -> Dict:
    """分析单个块的性能"""
    
    block_analysis = {
        'block_id': block_id,
        'block_path': block_path
    }
    
    # 检查模型文件
    model_file = os.path.join(block_path, 'model_final.pth')
    if os.path.exists(model_file):
        try:
            # 获取文件大小
            file_size_mb = os.path.getsize(model_file) / 1024 / 1024
            block_analysis['model_size_mb'] = file_size_mb
            
            # 尝试加载模型获取更多信息
            state_dict = torch.load(model_file, map_location='cpu')
            
            # 分析模型参数
            if 'background' in state_dict:
                bg_state = state_dict['background']
                if '_xyz' in bg_state:
                    num_gaussians = bg_state['_xyz'].shape[0]
                    block_analysis['num_gaussians'] = num_gaussians
                    
                    # 计算参数数量
                    total_params = 0
                    for key, tensor in bg_state.items():
                        if tensor.numel:
                            total_params += tensor.numel()
                    block_analysis['total_parameters'] = total_params
            
        except Exception as e:
            block_analysis['model_load_error'] = str(e)
    
    # 检查点云文件
    ply_file = os.path.join(block_path, 'point_cloud.ply')
    if os.path.exists(ply_file):
        ply_size_mb = os.path.getsize(ply_file) / 1024 / 1024
        block_analysis['pointcloud_size_mb'] = ply_size_mb
    
    # 分析训练日志（如果存在）
    log_images_dir = os.path.join(block_path, 'log_images')
    if os.path.exists(log_images_dir):
        image_files = [f for f in os.listdir(log_images_dir) if f.endswith('.jpg')]
        block_analysis['num_log_images'] = len(image_files)
        
        # 估算训练迭代数（基于图像文件名）
        if image_files:
            try:
                iterations = [int(f.split('.')[0]) for f in image_files if f.split('.')[0].isdigit()]
                if iterations:
                    block_analysis['max_iteration'] = max(iterations)
                    block_analysis['estimated_training_time'] = max(iterations) * 0.1  # 假设每迭代0.1秒
            except:
                pass
    
    return block_analysis

def analyze_aggregation_result(aggregated_path: str) -> Dict:
    """分析聚合结果"""
    
    aggregation_analysis = {
        'aggregated_path': aggregated_path
    }
    
    # 分析聚合模型
    model_file = os.path.join(aggregated_path, 'aggregated_model.pth')
    if os.path.exists(model_file):
        try:
            file_size_mb = os.path.getsize(model_file) / 1024 / 1024
            aggregation_analysis['model_size_mb'] = file_size_mb
            
            state_dict = torch.load(model_file, map_location='cpu')
            
            # 统计聚合后的高斯球数量
            total_gaussians = 0
            component_stats = {}
            
            for component_name, component_state in state_dict.items():
                if isinstance(component_state, dict) and '_xyz' in component_state:
                    num_gaussians = component_state['_xyz'].shape[0]
                    total_gaussians += num_gaussians
                    component_stats[component_name] = {
                        'num_gaussians': num_gaussians,
                        'parameters': sum(tensor.numel() for tensor in component_state.values() if hasattr(tensor, 'numel'))
                    }
            
            aggregation_analysis['num_gaussians'] = total_gaussians
            aggregation_analysis['component_stats'] = component_stats
            
        except Exception as e:
            aggregation_analysis['model_load_error'] = str(e)
    
    # 分析聚合点云
    ply_file = os.path.join(aggregated_path, 'aggregated_point_cloud.ply')
    if os.path.exists(ply_file):
        ply_size_mb = os.path.getsize(ply_file) / 1024 / 1024
        aggregation_analysis['pointcloud_size_mb'] = ply_size_mb
    
    # 分析元数据
    metadata_file = os.path.join(aggregated_path, 'metadata.json')
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            aggregation_analysis['metadata'] = metadata
        except Exception as e:
            aggregation_analysis['metadata_load_error'] = str(e)
    
    return aggregation_analysis

def generate_performance_report(analysis: Dict, output_path: str):
    """生成性能报告"""
    
    with open(output_path, 'w') as f:
        f.write("Street Gaussian Block Training Performance Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Analysis Date: {analysis['timestamp']}\n")
        f.write(f"Experiment Path: {analysis['experiment_path']}\n\n")
        
        # 整体统计
        overall = analysis['overall']
        f.write("Overall Statistics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Number of Blocks: {overall['num_blocks']}\n")
        f.write(f"Total Training Time: {overall['total_training_time']:.1f} seconds\n")
        f.write(f"Average Training Time per Block: {overall['avg_training_time_per_block']:.1f} seconds\n")
        f.write(f"Total Gaussians (before aggregation): {overall['total_gaussians_before_aggregation']}\n")
        
        if 'final_gaussians' in overall:
            f.write(f"Final Gaussians (after aggregation): {overall['final_gaussians']}\n")
            f.write(f"Deduplication Ratio: {overall['deduplication_ratio']:.1%}\n")
        
        f.write("\n")
        
        # 各块详细分析
        f.write("Block-wise Analysis:\n")
        f.write("-" * 50 + "\n")
        
        for block_id, block_data in analysis['blocks'].items():
            f.write(f"Block {block_id}:\n")
            
            if 'num_gaussians' in block_data:
                f.write(f"  Gaussians: {block_data['num_gaussians']}\n")
            if 'model_size_mb' in block_data:
                f.write(f"  Model Size: {block_data['model_size_mb']:.1f} MB\n")
            if 'pointcloud_size_mb' in block_data:
                f.write(f"  Point Cloud Size: {block_data['pointcloud_size_mb']:.1f} MB\n")
            if 'estimated_training_time' in block_data:
                f.write(f"  Estimated Training Time: {block_data['estimated_training_time']:.1f} seconds\n")
            if 'max_iteration' in block_data:
                f.write(f"  Max Iteration: {block_data['max_iteration']}\n")
            
            f.write("\n")
        
        # 聚合分析
        if analysis['aggregation']:
            agg = analysis['aggregation']
            f.write("Aggregation Analysis:\n")
            f.write("-" * 50 + "\n")
            
            if 'num_gaussians' in agg:
                f.write(f"Final Gaussians: {agg['num_gaussians']}\n")
            if 'model_size_mb' in agg:
                f.write(f"Final Model Size: {agg['model_size_mb']:.1f} MB\n")
            if 'pointcloud_size_mb' in agg:
                f.write(f"Final Point Cloud Size: {agg['pointcloud_size_mb']:.1f} MB\n")
            
            if 'component_stats' in agg:
                f.write("Component Statistics:\n")
                for comp_name, comp_stats in agg['component_stats'].items():
                    f.write(f"  {comp_name}: {comp_stats['num_gaussians']} gaussians, ")
                    f.write(f"{comp_stats['parameters']} parameters\n")
            
            f.write("\n")

def visualize_performance_metrics(analysis: Dict, output_dir: str):
    """可视化性能指标"""
    
    # 块性能对比
    if analysis['blocks']:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        block_ids = list(analysis['blocks'].keys())
        
        # 高斯球数量对比
        gaussians_counts = [analysis['blocks'][bid].get('num_gaussians', 0) for bid in block_ids]
        axes[0, 0].bar(block_ids, gaussians_counts)
        axes[0, 0].set_title('Gaussians per Block')
        axes[0, 0].set_xlabel('Block ID')
        axes[0, 0].set_ylabel('Number of Gaussians')
        
        # 模型大小对比
        model_sizes = [analysis['blocks'][bid].get('model_size_mb', 0) for bid in block_ids]
        axes[0, 1].bar(block_ids, model_sizes)
        axes[0, 1].set_title('Model Size per Block')
        axes[0, 1].set_xlabel('Block ID')
        axes[0, 1].set_ylabel('Size (MB)')
        
        # 训练时间对比
        training_times = [analysis['blocks'][bid].get('estimated_training_time', 0) for bid in block_ids]
        axes[1, 0].bar(block_ids, training_times)
        axes[1, 0].set_title('Training Time per Block')
        axes[1, 0].set_xlabel('Block ID')
        axes[1, 0].set_ylabel('Time (seconds)')
        
        # 参数数量对比
        param_counts = [analysis['blocks'][bid].get('total_parameters', 0) for bid in block_ids]
        axes[1, 1].bar(block_ids, param_counts)
        axes[1, 1].set_title('Parameters per Block')
        axes[1, 1].set_xlabel('Block ID')
        axes[1, 1].set_ylabel('Number of Parameters')
        
        plt.tight_layout()
        
        metrics_path = os.path.join(output_dir, 'performance_metrics.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        print(f"Performance metrics visualization saved to: {metrics_path}")
        plt.close()
    
    # 聚合效果可视化
    if analysis['aggregation'] and analysis['overall'].get('total_gaussians_before_aggregation', 0) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        before = analysis['overall']['total_gaussians_before_aggregation']
        after = analysis['aggregation'].get('num_gaussians', before)
        
        categories = ['Before Aggregation', 'After Aggregation']
        values = [before, after]
        colors = ['lightcoral', 'lightblue']
        
        bars = ax.bar(categories, values, color=colors)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:,}', ha='center', va='bottom')
        
        # 添加减少百分比
        reduction = (before - after) / before * 100 if before > 0 else 0
        ax.text(0.5, max(values) * 0.8, f'Reduction: {reduction:.1f}%', 
               ha='center', transform=ax.transData, fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax.set_title('Gaussian Deduplication Effect')
        ax.set_ylabel('Number of Gaussians')
        
        plt.tight_layout()
        
        dedup_path = os.path.join(output_dir, 'deduplication_effect.png')
        plt.savefig(dedup_path, dpi=300, bbox_inches='tight')
        print(f"Deduplication effect visualization saved to: {dedup_path}")
        plt.close()

def estimate_memory_savings(analysis: Dict) -> Dict:
    """估算内存节省"""
    
    if not analysis['blocks']:
        return {}
    
    # 估算单体训练的内存需求
    total_gaussians = analysis['overall'].get('total_gaussians_before_aggregation', 0)
    avg_gaussians_per_block = total_gaussians / len(analysis['blocks']) if analysis['blocks'] else 0
    
    # 假设每个高斯球需要约200字节内存（位置、旋转、缩放、不透明度、颜色等）
    bytes_per_gaussian = 200
    
    estimated_full_memory_mb = total_gaussians * bytes_per_gaussian / 1024 / 1024
    estimated_block_memory_mb = avg_gaussians_per_block * bytes_per_gaussian / 1024 / 1024
    
    memory_savings = {
        'estimated_full_training_memory_mb': estimated_full_memory_mb,
        'estimated_block_training_memory_mb': estimated_block_memory_mb,
        'memory_reduction_ratio': 1 - (estimated_block_memory_mb / estimated_full_memory_mb) if estimated_full_memory_mb > 0 else 0,
        'memory_savings_mb': estimated_full_memory_mb - estimated_block_memory_mb
    }
    
    return memory_savings

def main():
    parser = argparse.ArgumentParser(description="Street Gaussian Performance Analyzer")
    
    parser.add_argument('--experiment_path', type=str, required=True,
                       help='Path to experiment output directory')
    parser.add_argument('--output_dir', type=str, default='./performance_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--generate_plots', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Analyzing experiment: {args.experiment_path}")
    
    # 执行性能分析
    analysis = analyze_block_training_performance(args.experiment_path)
    
    # 估算内存节省
    memory_savings = estimate_memory_savings(analysis)
    analysis['memory_analysis'] = memory_savings
    
    # 生成报告
    report_path = os.path.join(args.output_dir, 'performance_report.txt')
    generate_performance_report(analysis, report_path)
    print(f"Performance report saved to: {report_path}")
    
    # 保存详细分析数据
    json_path = os.path.join(args.output_dir, 'performance_analysis.json')
    with open(json_path, 'w') as f:
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        import json
        json.dump(analysis, f, indent=2, default=convert_types)
    print(f"Detailed analysis saved to: {json_path}")
    
    # 生成可视化
    if args.generate_plots:
        print("Generating performance visualizations...")
        visualize_performance_metrics(analysis, args.output_dir)
    
    # 打印关键统计信息
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    
    overall = analysis['overall']
    print(f"Blocks Trained: {overall['num_blocks']}")
    print(f"Total Training Time: {overall['total_training_time']:.1f}s")
    print(f"Total Gaussians (before): {overall['total_gaussians_before_aggregation']:,}")
    
    if 'final_gaussians' in overall:
        print(f"Final Gaussians (after): {overall['final_gaussians']:,}")
        print(f"Deduplication: {overall['deduplication_ratio']:.1%}")
    
    if memory_savings:
        print(f"Estimated Memory Savings: {memory_savings['memory_reduction_ratio']:.1%}")
        print(f"Memory Reduction: {memory_savings['memory_savings_mb']:.1f} MB")
    
    print("="*50)

if __name__ == "__main__":
    main()