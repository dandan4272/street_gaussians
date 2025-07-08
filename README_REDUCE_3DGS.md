# Street Gaussians with Reduce_3DGS: 改进的3D高斯表示

本项目对原始的Street Gaussians实现了基于最优传输理论的reduce_3dgs方法，以实现更高效和紧凑的3D场景表示。

## 概述

### 原始问题
原始的Street Gaussians使用传统的densify_and_prune方法来管理3D高斯的数量，这种方法：
- 基于局部梯度信息进行densification
- 使用简单的阈值进行pruning
- 缺乏全局优化视角
- 可能导致冗余和非最优的高斯分布

### Reduce_3DGS解决方案
我们实现的reduce_3dgs方法基于"Gaussian Herding across Pens: An Optimal Transport Perspective on Global Gaussian Reduction for 3DGS"论文的思想：

1. **全局视角**: 将3DGS压缩看作全局高斯混合减少问题
2. **最优传输**: 使用最优传输理论来最小化表示误差
3. **KD树分区**: 通过空间分区来组织高斯
4. **智能聚合**: 基于重要性分数进行高斯聚合

## 核心改进

### 1. Reduce3DGS核心类
位于 `lib/models/reduce_3dgs.py`，实现了：
- **重要性计算**: 基于不透明度和尺度的综合重要性分数
- **KD树分区**: 空间聚类来组织相似的高斯
- **最优传输**: 匈牙利算法实现的最优分配
- **智能聚合**: 加权平均来合并高斯参数

### 2. ReducedGaussianModel包装器
集成了reduce_3dgs方法的高斯模型包装器：
- 兼容原始API
- 自动切换reduce_3dgs和传统方法
- 支持参数更新和优化器重置

### 3. StreetGaussianModel集成
修改了街道高斯模型以支持reduce_3dgs：
- 配置驱动的方法选择
- 自动应用到背景和对象模型
- 无缝集成到训练流程

## 技术实现详情

### 重要性分数计算
```python
def compute_gaussian_importance(self, gaussians):
    opacity = gaussians.get_opacity.squeeze()
    scales = gaussians.get_scaling
    scale_volume = torch.prod(scales, dim=1)
    importance = opacity * torch.log(scale_volume + 1e-6)
    return importance
```

### 最优传输分配
```python
def optimal_transport_assignment(self, cost_matrix, source_weights, target_weights):
    # 使用匈牙利算法进行最优分配
    row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
    # ... 处理分配结果
```

### 高斯聚合
```python
def aggregate_gaussians(self, gaussians, assignment, num_targets):
    # 加权平均所有高斯参数
    for target_idx in range(num_targets):
        source_mask = assignment == target_idx
        weights = gaussians.get_opacity[source_indices].squeeze()
        weights = weights / weights.sum()
        # 位置、特征、尺度、旋转等的加权平均
```

## 使用方法

### 1. 配置设置
在配置文件中启用reduce_3dgs：
```yaml
model:
  gaussian:
    use_reduce_3dgs: true
    reduction_ratio: 0.1  # 保留10%的高斯
```

### 2. 运行训练
```bash
# 使用提供的reduce_3dgs配置
python train_reduced.py --config configs/reduce_3dgs_config.yaml

# 或使用原始训练脚本（需要在配置中启用reduce_3dgs）
python train.py --config configs/reduce_3dgs_config.yaml
```

### 3. 关键参数
- `use_reduce_3dgs`: 是否启用reduce_3dgs方法
- `reduction_ratio`: 目标减少比例（0.1表示保留10%）
- `lambda_reduce_reg`: reduce_3dgs正则化权重

## 性能对比

### 预期改进
1. **内存使用**: 显著减少高斯数量（通常减少90%）
2. **渲染速度**: 更少的高斯意味着更快的渲染
3. **存储空间**: 模型文件大小大幅减少
4. **视觉质量**: 通过智能聚合保持视觉保真度

### 监控指标
训练过程中会显示：
- 当前高斯数量
- 减少比例
- reduce_3dgs正则化损失
- 传统质量指标（PSNR、SSIM等）

## 文件结构

```
street_gaussians/
├── lib/models/
│   ├── reduce_3dgs.py              # 核心reduce_3dgs实现
│   └── street_gaussian_model.py    # 修改后的街道高斯模型
├── configs/
│   └── reduce_3dgs_config.yaml     # reduce_3dgs配置文件
├── train_reduced.py                # 演示训练脚本
└── README_REDUCE_3DGS.md          # 本文档
```

## 算法流程

1. **初始训练阶段**: 使用传统densify_and_prune构建完整场景
2. **减少阶段**: 
   - 计算高斯重要性分数
   - 构建KD树空间分区
   - 使用最优传输进行分配
   - 聚合高斯参数
3. **微调阶段**: 在减少的高斯集上继续训练
4. **维护阶段**: 仅进行pruning，不再增加高斯

## 依赖项
新增依赖：
- `scikit-learn`: 用于KMeans聚类和KD树
- `scipy`: 用于最优传输的线性分配算法

安装：
```bash
pip install scikit-learn scipy
```

## 注意事项

1. **reduction_ratio选择**: 
   - 0.05-0.2通常是好的选择
   - 过小可能影响视觉质量
   - 过大减少效果不明显

2. **时机控制**: 
   - 建议在densify_until_iter之后应用
   - 给模型足够时间构建完整表示

3. **场景适应性**: 
   - 复杂场景可能需要更高的保留比例
   - 简单场景可以使用更激进的减少

## 未来改进方向

1. **Sinkhorn算法**: 实现更精确的最优传输
2. **层次化减少**: 多层次的减少策略
3. **自适应比例**: 基于场景复杂度的动态减少比例
4. **外观微调**: 更精细的外观属性恢复
5. **渲染优化**: 针对减少后高斯的特殊渲染优化

## 引用
本实现基于以下研究：
- Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting
- Gaussian Herding across Pens: An Optimal Transport Perspective on Global Gaussian Reduction for 3DGS
- 3D Gaussian Splatting for Real-Time Radiance Field Rendering

## 联系信息
如有问题或建议，请提交issue或联系开发团队。