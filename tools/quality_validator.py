#!/usr/bin/env python3
"""
Street Gaussian 分块训练质量验证工具

验证分块训练和聚合结果的质量，包括渲染质量、模型一致性等。
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from lib.datasets.dataset import Dataset
    from lib.models.street_gaussian_model import StreetGaussianModel
    from lib.models.street_gaussian_renderer import StreetGaussianRenderer
    from lib.utils.loss_utils import psnr, ssim
    from lib.config import cfg
except ImportError as e:
    print(f"Warning: Could not import Street Gaussian modules: {e}")
    print("Some functionality may be limited.")

class QualityValidator:
    """质量验证器"""
    
    def __init__(self, dataset: Dataset, renderer: StreetGaussianRenderer):
        self.dataset = dataset
        self.renderer = renderer
        
    def validate_model_consistency(self, model: StreetGaussianModel) -> Dict:
        """验证模型的一致性"""
        
        consistency_report = {
            'model_structure': {},
            'parameter_distribution': {},
            'spatial_coverage': {}
        }
        
        # 检查模型结构
        try:
            # 背景模型检查
            if hasattr(model, 'background') and model.background is not None:
                bg_xyz = model.background.get_xyz
                bg_opacity = model.background.get_opacity
                
                consistency_report['model_structure']['background'] = {
                    'num_gaussians': len(bg_xyz),
                    'valid_positions': torch.isfinite(bg_xyz).all().item(),
                    'valid_opacities': torch.isfinite(bg_opacity).all().item(),
                    'opacity_range': [bg_opacity.min().item(), bg_opacity.max().item()]
                }
            
            # 对象模型检查
            if hasattr(model, 'obj_list'):
                consistency_report['model_structure']['objects'] = {}
                for obj_name in model.obj_list:
                    if hasattr(model, obj_name):
                        obj_model = getattr(model, obj_name)
                        obj_xyz = obj_model.get_xyz
                        obj_opacity = obj_model.get_opacity
                        
                        consistency_report['model_structure']['objects'][obj_name] = {
                            'num_gaussians': len(obj_xyz),
                            'valid_positions': torch.isfinite(obj_xyz).all().item(),
                            'valid_opacities': torch.isfinite(obj_opacity).all().item(),
                            'opacity_range': [obj_opacity.min().item(), obj_opacity.max().item()]
                        }
            
        except Exception as e:
            consistency_report['model_structure']['error'] = str(e)
        
        return consistency_report
    
    def validate_rendering_quality(self, model: StreetGaussianModel, 
                                 test_cameras: List = None, 
                                 max_cameras: int = 10) -> Dict:
        """验证渲染质量"""
        
        if test_cameras is None:
            test_cameras = self.dataset.scene_info.test_cameras
        
        # 限制测试相机数量以节省时间
        if len(test_cameras) > max_cameras:
            test_cameras = test_cameras[:max_cameras]
        
        quality_metrics = {
            'psnr_scores': [],
            'ssim_scores': [],
            'rendering_errors': [],
            'camera_results': {}
        }
        
        model.eval()
        
        with torch.no_grad():
            for i, camera in enumerate(test_cameras):
                try:
                    # 渲染图像
                    render_result = self.renderer.render(camera, model)
                    rendered_image = render_result['rgb']
                    
                    # 获取真实图像
                    gt_image = camera.original_image.cuda()
                    
                    # 获取mask（如果有）
                    if hasattr(camera, 'original_mask'):
                        mask = camera.original_mask.cuda().bool()
                    else:
                        mask = torch.ones_like(gt_image[0]).bool()
                    
                    # 计算质量指标
                    psnr_score = psnr(rendered_image, gt_image, mask).mean().item()
                    ssim_score = ssim(rendered_image, gt_image, mask=mask).item()
                    
                    quality_metrics['psnr_scores'].append(psnr_score)
                    quality_metrics['ssim_scores'].append(ssim_score)
                    
                    quality_metrics['camera_results'][i] = {
                        'psnr': psnr_score,
                        'ssim': ssim_score,
                        'camera_name': getattr(camera, 'image_name', f'camera_{i}')
                    }
                    
                except Exception as e:
                    error_msg = f"Camera {i}: {str(e)}"
                    quality_metrics['rendering_errors'].append(error_msg)
                    print(f"Rendering error for camera {i}: {e}")
        
        # 计算统计信息
        if quality_metrics['psnr_scores']:
            quality_metrics['summary'] = {
                'avg_psnr': np.mean(quality_metrics['psnr_scores']),
                'std_psnr': np.std(quality_metrics['psnr_scores']),
                'avg_ssim': np.mean(quality_metrics['ssim_scores']),
                'std_ssim': np.std(quality_metrics['ssim_scores']),
                'num_successful_renders': len(quality_metrics['psnr_scores']),
                'num_failed_renders': len(quality_metrics['rendering_errors'])
            }
        
        return quality_metrics
    
    def compare_block_vs_aggregated(self, block_models: Dict[int, StreetGaussianModel], 
                                   aggregated_model: StreetGaussianModel,
                                   test_cameras: List = None,
                                   max_cameras: int = 5) -> Dict:
        """比较块模型与聚合模型的质量"""
        
        if test_cameras is None:
            test_cameras = self.dataset.scene_info.test_cameras[:max_cameras]
        
        comparison_results = {
            'aggregated_quality': {},
            'block_qualities': {},
            'quality_comparison': {}
        }
        
        # 验证聚合模型质量
        print("Validating aggregated model...")
        aggregated_quality = self.validate_rendering_quality(aggregated_model, test_cameras)
        comparison_results['aggregated_quality'] = aggregated_quality
        
        # 验证各块模型质量（如果可用）
        print("Validating block models...")
        for block_id, block_model in block_models.items():
            print(f"  Validating block {block_id}...")
            try:
                block_quality = self.validate_rendering_quality(block_model, test_cameras)
                comparison_results['block_qualities'][block_id] = block_quality
            except Exception as e:
                print(f"Failed to validate block {block_id}: {e}")
        
        # 计算比较统计
        if (aggregated_quality.get('summary') and 
            comparison_results['block_qualities']):
            
            # 计算平均块质量
            block_psnrs = []
            block_ssims = []
            
            for block_quality in comparison_results['block_qualities'].values():
                if 'summary' in block_quality:
                    block_psnrs.append(block_quality['summary']['avg_psnr'])
                    block_ssims.append(block_quality['summary']['avg_ssim'])
            
            if block_psnrs:
                avg_block_psnr = np.mean(block_psnrs)
                avg_block_ssim = np.mean(block_ssims)
                
                aggregated_psnr = aggregated_quality['summary']['avg_psnr']
                aggregated_ssim = aggregated_quality['summary']['avg_ssim']
                
                comparison_results['quality_comparison'] = {
                    'avg_block_psnr': avg_block_psnr,
                    'avg_block_ssim': avg_block_ssim,
                    'aggregated_psnr': aggregated_psnr,
                    'aggregated_ssim': aggregated_ssim,
                    'psnr_difference': aggregated_psnr - avg_block_psnr,
                    'ssim_difference': aggregated_ssim - avg_block_ssim,
                    'quality_maintained': (
                        abs(aggregated_psnr - avg_block_psnr) < 2.0 and 
                        abs(aggregated_ssim - avg_block_ssim) < 0.1
                    )
                }
        
        return comparison_results

def load_models_for_validation(experiment_path: str, 
                              config_path: str,
                              source_path: str) -> Tuple[Dict, StreetGaussianModel, Dataset]:
    """加载模型用于验证"""
    
    # 设置配置
    cfg.merge_from_file(config_path)
    cfg.source_path = source_path
    cfg.mode = 'test'  # 设置为测试模式
    
    # 加载数据集
    dataset = Dataset()
    
    # 加载聚合模型
    aggregated_path = os.path.join(experiment_path, 'aggregated', 'aggregated_model.pth')
    aggregated_model = None
    
    if os.path.exists(aggregated_path):
        try:
            # 创建模型实例
            aggregated_model = StreetGaussianModel(dataset.scene_info.metadata)
            
            # 加载状态
            state_dict = torch.load(aggregated_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            aggregated_model.load_state_dict(state_dict)
            
            print(f"Loaded aggregated model from: {aggregated_path}")
            
        except Exception as e:
            print(f"Failed to load aggregated model: {e}")
    
    # 加载块模型（可选）
    block_models = {}
    block_dirs = [d for d in os.listdir(experiment_path) 
                  if d.startswith('block_') and os.path.isdir(os.path.join(experiment_path, d))]
    
    for block_dir in block_dirs:
        block_id = int(block_dir.split('_')[1])
        block_model_path = os.path.join(experiment_path, block_dir, 'model_final.pth')
        
        if os.path.exists(block_model_path):
            try:
                # 为块创建模型实例（使用相同的元数据）
                block_model = StreetGaussianModel(dataset.scene_info.metadata)
                
                # 加载状态
                state_dict = torch.load(block_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
                block_model.load_state_dict(state_dict)
                
                block_models[block_id] = block_model
                print(f"Loaded block {block_id} model")
                
            except Exception as e:
                print(f"Failed to load block {block_id} model: {e}")
    
    return block_models, aggregated_model, dataset

def generate_quality_report(validation_results: Dict, output_path: str):
    """生成质量验证报告"""
    
    with open(output_path, 'w') as f:
        f.write("Street Gaussian Block Training Quality Validation Report\n")
        f.write("=" * 65 + "\n\n")
        
        # 聚合模型质量
        if 'aggregated_quality' in validation_results:
            agg_quality = validation_results['aggregated_quality']
            f.write("Aggregated Model Quality:\n")
            f.write("-" * 40 + "\n")
            
            if 'summary' in agg_quality:
                summary = agg_quality['summary']
                f.write(f"Average PSNR: {summary['avg_psnr']:.2f} ± {summary['std_psnr']:.2f}\n")
                f.write(f"Average SSIM: {summary['avg_ssim']:.3f} ± {summary['std_ssim']:.3f}\n")
                f.write(f"Successful Renders: {summary['num_successful_renders']}\n")
                f.write(f"Failed Renders: {summary['num_failed_renders']}\n")
            
            f.write("\n")
        
        # 块模型质量对比
        if 'block_qualities' in validation_results:
            f.write("Block Model Qualities:\n")
            f.write("-" * 40 + "\n")
            
            for block_id, block_quality in validation_results['block_qualities'].items():
                if 'summary' in block_quality:
                    summary = block_quality['summary']
                    f.write(f"Block {block_id}:\n")
                    f.write(f"  PSNR: {summary['avg_psnr']:.2f} ± {summary['std_psnr']:.2f}\n")
                    f.write(f"  SSIM: {summary['avg_ssim']:.3f} ± {summary['std_ssim']:.3f}\n")
                    f.write(f"  Success Rate: {summary['num_successful_renders']}/{summary['num_successful_renders'] + summary['num_failed_renders']}\n")
            
            f.write("\n")
        
        # 质量比较
        if 'quality_comparison' in validation_results:
            comp = validation_results['quality_comparison']
            f.write("Quality Comparison:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Block PSNR: {comp['avg_block_psnr']:.2f}\n")
            f.write(f"Aggregated PSNR: {comp['aggregated_psnr']:.2f}\n")
            f.write(f"PSNR Difference: {comp['psnr_difference']:.2f}\n")
            f.write(f"Average Block SSIM: {comp['avg_block_ssim']:.3f}\n")
            f.write(f"Aggregated SSIM: {comp['aggregated_ssim']:.3f}\n")
            f.write(f"SSIM Difference: {comp['ssim_difference']:.3f}\n")
            f.write(f"Quality Maintained: {'Yes' if comp['quality_maintained'] else 'No'}\n")
            
            f.write("\n")
        
        # 模型一致性
        if 'model_consistency' in validation_results:
            consistency = validation_results['model_consistency']
            f.write("Model Consistency:\n")
            f.write("-" * 40 + "\n")
            
            if 'model_structure' in consistency:
                structure = consistency['model_structure']
                
                if 'background' in structure:
                    bg = structure['background']
                    f.write(f"Background Model:\n")
                    f.write(f"  Gaussians: {bg['num_gaussians']}\n")
                    f.write(f"  Valid Positions: {bg['valid_positions']}\n")
                    f.write(f"  Valid Opacities: {bg['valid_opacities']}\n")
                    f.write(f"  Opacity Range: {bg['opacity_range'][0]:.3f} - {bg['opacity_range'][1]:.3f}\n")
                
                if 'objects' in structure:
                    f.write(f"Object Models:\n")
                    for obj_name, obj_info in structure['objects'].items():
                        f.write(f"  {obj_name}: {obj_info['num_gaussians']} gaussians\n")

def visualize_quality_metrics(validation_results: Dict, output_dir: str):
    """可视化质量指标"""
    
    # PSNR 和 SSIM 对比
    if ('aggregated_quality' in validation_results and 
        'block_qualities' in validation_results):
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 收集数据
        block_ids = list(validation_results['block_qualities'].keys())
        block_psnrs = []
        block_ssims = []
        
        for block_id in block_ids:
            block_quality = validation_results['block_qualities'][block_id]
            if 'summary' in block_quality:
                block_psnrs.append(block_quality['summary']['avg_psnr'])
                block_ssims.append(block_quality['summary']['avg_ssim'])
        
        # 添加聚合结果
        agg_quality = validation_results['aggregated_quality']
        if 'summary' in agg_quality:
            block_ids.append('Aggregated')
            block_psnrs.append(agg_quality['summary']['avg_psnr'])
            block_ssims.append(agg_quality['summary']['avg_ssim'])
        
        # PSNR 对比
        colors = ['lightblue'] * (len(block_ids) - 1) + ['orange']
        ax1.bar(range(len(block_ids)), block_psnrs, color=colors)
        ax1.set_xticks(range(len(block_ids)))
        ax1.set_xticklabels([f'Block {bid}' if isinstance(bid, int) else bid for bid in block_ids], rotation=45)
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('PSNR Comparison')
        ax1.grid(True, alpha=0.3)
        
        # SSIM 对比
        ax2.bar(range(len(block_ids)), block_ssims, color=colors)
        ax2.set_xticks(range(len(block_ids)))
        ax2.set_xticklabels([f'Block {bid}' if isinstance(bid, int) else bid for bid in block_ids], rotation=45)
        ax2.set_ylabel('SSIM')
        ax2.set_title('SSIM Comparison')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        quality_path = os.path.join(output_dir, 'quality_comparison.png')
        plt.savefig(quality_path, dpi=300, bbox_inches='tight')
        print(f"Quality comparison visualization saved to: {quality_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Street Gaussian Quality Validator")
    
    parser.add_argument('--experiment_path', type=str, required=True,
                       help='Path to experiment output directory')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--source_path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./quality_validation',
                       help='Output directory for validation results')
    parser.add_argument('--max_test_cameras', type=int, default=10,
                       help='Maximum number of test cameras to use')
    parser.add_argument('--validate_blocks', action='store_true',
                       help='Also validate individual block models (slower)')
    parser.add_argument('--generate_plots', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading models for validation...")
    
    # 加载模型
    try:
        block_models, aggregated_model, dataset = load_models_for_validation(
            args.experiment_path, args.config, args.source_path
        )
        
        if aggregated_model is None:
            print("Error: Could not load aggregated model")
            return 1
            
    except Exception as e:
        print(f"Failed to load models: {e}")
        return 1
    
    # 创建验证器和渲染器
    try:
        from lib.models.street_gaussian_renderer import StreetGaussianRenderer
        renderer = StreetGaussianRenderer()
        validator = QualityValidator(dataset, renderer)
        
    except Exception as e:
        print(f"Failed to create validator: {e}")
        return 1
    
    validation_results = {}
    
    # 验证聚合模型一致性
    print("Validating model consistency...")
    consistency_report = validator.validate_model_consistency(aggregated_model)
    validation_results['model_consistency'] = consistency_report
    
    # 验证渲染质量
    print("Validating rendering quality...")
    quality_results = validator.validate_rendering_quality(
        aggregated_model, 
        max_cameras=args.max_test_cameras
    )
    validation_results['aggregated_quality'] = quality_results
    
    # 可选：验证块模型并比较
    if args.validate_blocks and block_models:
        print("Comparing block and aggregated models...")
        comparison_results = validator.compare_block_vs_aggregated(
            block_models, aggregated_model, max_cameras=args.max_test_cameras
        )
        validation_results.update(comparison_results)
    
    # 生成报告
    report_path = os.path.join(args.output_dir, 'quality_validation_report.txt')
    generate_quality_report(validation_results, report_path)
    print(f"Quality validation report saved to: {report_path}")
    
    # 保存详细结果
    json_path = os.path.join(args.output_dir, 'validation_results.json')
    with open(json_path, 'w') as f:
        # 转换numpy类型为Python原生类型
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(validation_results, f, indent=2, default=convert_types)
    print(f"Detailed validation results saved to: {json_path}")
    
    # 生成可视化
    if args.generate_plots:
        print("Generating quality visualizations...")
        visualize_quality_metrics(validation_results, args.output_dir)
    
    # 打印总结
    print("\n" + "="*50)
    print("QUALITY VALIDATION SUMMARY")
    print("="*50)
    
    if 'aggregated_quality' in validation_results and 'summary' in validation_results['aggregated_quality']:
        summary = validation_results['aggregated_quality']['summary']
        print(f"Aggregated Model PSNR: {summary['avg_psnr']:.2f} ± {summary['std_psnr']:.2f}")
        print(f"Aggregated Model SSIM: {summary['avg_ssim']:.3f} ± {summary['std_ssim']:.3f}")
        print(f"Render Success Rate: {summary['num_successful_renders']}/{summary['num_successful_renders'] + summary['num_failed_renders']}")
    
    if 'quality_comparison' in validation_results:
        comp = validation_results['quality_comparison']
        print(f"Quality Maintained: {'✓' if comp['quality_maintained'] else '✗'}")
        print(f"PSNR Change: {comp['psnr_difference']:+.2f} dB")
        print(f"SSIM Change: {comp['ssim_difference']:+.3f}")
    
    print("="*50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())