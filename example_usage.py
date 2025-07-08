#!/usr/bin/env python3
"""
Reduce_3DGS 使用示例

这个脚本演示了如何在您的项目中使用reduce_3dgs方法。
"""

import torch
import numpy as np
from lib.models.reduce_3dgs import Reduce3DGS, ReducedGaussianModel
from lib.models.gaussian_model import GaussianModel


def create_example_gaussian_model():
    """创建一个示例的高斯模型用于演示"""
    class ExampleGaussianModel:
        def __init__(self, num_gaussians=1000):
            # 模拟高斯参数
            self._xyz = torch.randn(num_gaussians, 3).cuda() * 10
            self._opacity = torch.sigmoid(torch.randn(num_gaussians, 1).cuda())
            self._scaling = torch.exp(torch.randn(num_gaussians, 3).cuda() * 0.5)
            self._rotation = torch.nn.functional.normalize(torch.randn(num_gaussians, 4).cuda())
            self._features_dc = torch.randn(num_gaussians, 1, 3).cuda()
            self._features_rest = torch.randn(num_gaussians, 15, 3).cuda()
            self._semantic = torch.randn(num_gaussians, 10).cuda()
            
        @property
        def get_xyz(self):
            return self._xyz
            
        @property
        def get_opacity(self):
            return self._opacity
            
        @property
        def get_scaling(self):
            return self._scaling
    
    return ExampleGaussianModel()


def example_1_basic_usage():
    """示例1: 基本的reduce_3dgs使用"""
    print("=== 示例1: 基本的reduce_3dgs使用 ===")
    
    # 创建示例高斯模型
    gaussians = create_example_gaussian_model()
    print(f"原始高斯数量: {gaussians.get_xyz.shape[0]}")
    
    # 创建reduce_3dgs实例
    reducer = Reduce3DGS(reduction_ratio=0.1)  # 保留10%
    
    # 应用减少
    reduced_params = reducer.reduce_gaussians(gaussians)
    
    print(f"减少后高斯数量: {reduced_params['xyz'].shape[0]}")
    print(f"减少比例: {reduced_params['xyz'].shape[0] / gaussians.get_xyz.shape[0]:.2%}")


def example_2_wrapped_model():
    """示例2: 使用ReducedGaussianModel包装器"""
    print("\n=== 示例2: 使用ReducedGaussianModel包装器 ===")
    
    # 创建基础模型
    base_model = create_example_gaussian_model()
    print(f"基础模型高斯数量: {base_model.get_xyz.shape[0]}")
    
    # 创建包装后的模型
    wrapped_model = ReducedGaussianModel(base_model, reduction_ratio=0.15)
    
    # 模拟训练过程中的reduce_and_prune调用
    print("执行reduce_and_prune...")
    scalars, tensors = wrapped_model.reduce_and_prune(
        max_grad=0.0002,
        min_opacity=0.005,
        extent=100.0,
        max_screen_size=None
    )
    
    print(f"减少后高斯数量: {wrapped_model.get_xyz.shape[0]}")


def example_3_parameter_tuning():
    """示例3: 参数调优示例"""
    print("\n=== 示例3: 参数调优示例 ===")
    
    gaussians = create_example_gaussian_model()
    original_count = gaussians.get_xyz.shape[0]
    
    # 测试不同的减少比例
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    for ratio in ratios:
        reducer = Reduce3DGS(reduction_ratio=ratio)
        reduced_params = reducer.reduce_gaussians(gaussians)
        final_count = reduced_params['xyz'].shape[0]
        
        print(f"减少比例 {ratio:.0%}: {original_count} -> {final_count} "
              f"(实际保留: {final_count/original_count:.1%})")


def example_4_importance_analysis():
    """示例4: 重要性分数分析"""
    print("\n=== 示例4: 重要性分数分析 ===")
    
    gaussians = create_example_gaussian_model()
    reducer = Reduce3DGS()
    
    # 计算重要性分数
    importance = reducer.compute_gaussian_importance(gaussians)
    
    print(f"重要性分数统计:")
    print(f"  最小值: {importance.min().item():.4f}")
    print(f"  最大值: {importance.max().item():.4f}")
    print(f"  平均值: {importance.mean().item():.4f}")
    print(f"  标准差: {importance.std().item():.4f}")
    
    # 分析高重要性和低重要性的高斯
    top_10_percent = torch.topk(importance, k=int(0.1 * len(importance))).indices
    bottom_10_percent = torch.topk(importance, k=int(0.1 * len(importance)), largest=False).indices
    
    print(f"\n高重要性高斯（前10%）:")
    print(f"  平均不透明度: {gaussians.get_opacity[top_10_percent].mean().item():.4f}")
    print(f"  平均尺度: {gaussians.get_scaling[top_10_percent].mean().item():.4f}")
    
    print(f"\n低重要性高斯（后10%）:")
    print(f"  平均不透明度: {gaussians.get_opacity[bottom_10_percent].mean().item():.4f}")
    print(f"  平均尺度: {gaussians.get_scaling[bottom_10_percent].mean().item():.4f}")


def example_5_integration_tips():
    """示例5: 集成提示"""
    print("\n=== 示例5: 集成提示 ===")
    
    print("在您的项目中集成reduce_3dgs的步骤:")
    print("1. 导入必要的模块:")
    print("   from lib.models.reduce_3dgs import Reduce3DGS, ReducedGaussianModel")
    
    print("\n2. 修改您的高斯模型类:")
    print("   # 在__init__中添加")
    print("   self.use_reduce_3dgs = config.get('use_reduce_3dgs', False)")
    print("   self.reduction_ratio = config.get('reduction_ratio', 0.1)")
    
    print("\n3. 包装您的模型:")
    print("   if self.use_reduce_3dgs:")
    print("       self.background = ReducedGaussianModel(background_model, self.reduction_ratio)")
    print("   else:")
    print("       self.background = background_model")
    
    print("\n4. 在训练循环中:")
    print("   if hasattr(model, 'reduce_and_prune'):")
    print("       scalars, tensors = model.reduce_and_prune(max_grad, min_opacity, extent, max_screen_size)")
    print("   else:")
    print("       scalars, tensors = model.densify_and_prune(max_grad, min_opacity, prune_big_points)")
    
    print("\n5. 配置文件设置:")
    print("   model:")
    print("     gaussian:")
    print("       use_reduce_3dgs: true")
    print("       reduction_ratio: 0.1")


def main():
    """主函数，运行所有示例"""
    print("Reduce_3DGS 使用示例")
    print("=" * 50)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，示例将在CPU上运行（可能较慢）")
        print()
    
    try:
        # 运行所有示例
        example_1_basic_usage()
        example_2_wrapped_model()
        example_3_parameter_tuning()
        example_4_importance_analysis()
        example_5_integration_tips()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("请确保已安装所有依赖项:")
        print("pip install scikit-learn scipy torch")


if __name__ == "__main__":
    main()