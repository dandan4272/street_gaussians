# Street Gaussian 分块训练系统

本项目为 Street Gaussian 实现了分块训练功能，用于训练大规模街景场景。通过将大场景分割为多个重叠的块，独立训练后再聚合，可以有效处理内存限制并提高训练效率。

## 🔧 系统架构

### 核心组件

1. **BlockTrainer** (`lib/training/block_trainer.py`)
   - 管理场景分块和相机分配
   - 为每个块创建独立的数据集和模型
   - 协调训练过程

2. **GaussianBlockAggregator** (`lib/training/block_aggregator.py`)
   - 聚合训练好的块模型
   - 去除重叠区域的冗余高斯球
   - 合并背景、对象和天空模型

3. **分块训练循环** (`lib/training/block_train_loop.py`)
   - 针对单个块优化的训练循环
   - 基于原始训练代码适配

### 主要特性

- ✅ **空间分块**: 自动或手动定义场景分块
- ✅ **重叠处理**: 智能处理块间重叠区域
- ✅ **高斯球去重**: 移除重复和低质量的高斯球
- ✅ **多组件支持**: 分别处理背景、动态对象和天空
- ✅ **检查点管理**: 支持断点续训和模型保存
- ✅ **内存优化**: 有效管理大场景的内存使用

## 🚀 快速开始

### 1. 环境准备

确保已安装原始 Street Gaussian 的依赖项，额外需要：

```bash
pip install scikit-learn trimesh
```

### 2. 基本使用

```bash
# 使用示例配置训练
python train_blocks.py \
    --config configs/block_training_example.yaml \
    --source_path /path/to/your/dataset \
    --model_path ./outputs/block_experiment \
    --block_size 50 \
    --overlap_margin 5
```

### 3. 高级选项

```bash
# 指定场景边界
python train_blocks.py \
    --config configs/block_training_example.yaml \
    --source_path /path/to/dataset \
    --model_path ./outputs/experiment \
    --scene_bounds -100 100 -100 100 \
    --spatial_threshold 0.1 \
    --opacity_threshold 0.05

# 仅聚合（跳过训练）
python train_blocks.py \
    --config configs/block_training_example.yaml \
    --source_path /path/to/dataset \
    --model_path ./outputs/experiment \
    --skip_training

# 使用手动定义的块配置
python train_blocks.py \
    --config configs/block_training_example.yaml \
    --source_path /path/to/dataset \
    --model_path ./outputs/experiment \
    --manual_blocks custom_blocks.json
```

## 📁 输出结构

训练完成后，输出目录结构如下：

```
outputs/experiment/
├── blocks_config.json          # 块配置信息
├── block_0/                    # 第一个块
│   ├── model_final.pth        # 模型检查点
│   ├── point_cloud.ply        # 点云文件
│   └── log_images/            # 训练过程图像
├── block_1/                    # 第二个块
│   └── ...
├── aggregated/                 # 聚合结果
│   ├── aggregated_model.pth   # 最终聚合模型
│   ├── aggregated_point_cloud.ply
│   └── metadata.json          # 元数据
└── cfg_args                    # 配置参数
```

## ⚙️ 配置说明

### 分块参数

- `block_size`: 每个块的大小（米）
- `overlap_margin`: 重叠边界宽度（米）
- `scene_bounds`: 手动指定场景边界 `[min_x, max_x, min_y, max_y]`

### 聚合参数

- `spatial_threshold`: 空间去重阈值（米）
- `opacity_threshold`: 不透明度阈值
- `overlap_threshold`: 重叠区域处理阈值

### 训练参数优化

分块训练建议的配置修改：

```yaml
train:
  iterations: 15000  # 比原始训练减少

optim:
  densify_until_iter: 12000     # 提前停止密化
  min_opacity: 0.01             # 提高最小不透明度
  densify_grad_threshold: 0.0003 # 适当增加梯度阈值

model:
  nsg:
    include_sky: false          # 分块训练时关闭天空
  use_color_correction: false   # 关闭颜色校正
  use_pose_correction: false    # 关闭位姿校正
```

## 🔄 工作流程

### 1. 场景分析
- 分析相机位置估算场景边界
- 根据指定大小自动分割场景
- 为每个块分配相机和点云数据

### 2. 分块训练
- 每个块独立训练 Street Gaussian 模型
- 包含背景和动态对象的独立优化
- 保存块级别的检查点

### 3. 模型聚合
- 加载所有训练好的块模型
- 分别聚合背景、对象和天空组件
- 智能去重重叠区域的高斯球
- 生成最终的全场景模型

## 📊 性能优化

### 内存管理
- 每个块独立处理，避免整个场景同时加载
- 训练完成后及时释放块模型内存
- 支持 GPU 内存不足时的自动降级

### 训练效率
- 较小的块可以使用更快的训练设置
- 重叠区域确保块间的平滑过渡
- 支持并行训练多个块（未来版本）

### 质量保证
- 重叠区域的冗余高斯球去除
- 基于不透明度的质量过滤
- 保持动态对象的时序一致性

## 🔧 自定义块配置

### 手动定义块

创建 JSON 文件定义自定义块：

```json
[
  {
    "block_id": 0,
    "spatial_bounds": [-50, 0, -50, 50],
    "overlap_margin": 5.0
  },
  {
    "block_id": 1,
    "spatial_bounds": [-5, 50, -50, 50],
    "overlap_margin": 5.0
  }
]
```

### 块配置最佳实践

1. **块大小选择**
   - 小块 (25-50m): 更快训练，更多重叠
   - 大块 (50-100m): 更好一致性，更少重叠

2. **重叠设置**
   - 重叠过小: 可能出现接缝
   - 重叠过大: 增加冗余和计算量
   - 推荐: 5-10m 重叠

3. **相机分布**
   - 确保每个块有足够的训练相机
   - 重叠区域的相机分配给相邻块

## 🐛 故障排除

### 常见问题

1. **块训练失败**
   ```
   Block X: No training cameras
   ```
   - 检查场景边界设置
   - 确认相机位置在预期范围内

2. **聚合失败**
   ```
   No background data to aggregate
   ```
   - 检查块模型是否正确保存
   - 确认训练完成没有错误

3. **内存不足**
   ```
   CUDA out of memory
   ```
   - 减少块大小
   - 调整 `max_gaussians_per_block`
   - 使用 `clear_cache_between_blocks`

### 调试技巧

1. **可视化块分布**
   ```python
   # 查看保存的块配置
   import json
   with open('blocks_config.json') as f:
       blocks = json.load(f)
   for block in blocks:
       print(f"Block {block['block_id']}: {block['spatial_bounds']}")
   ```

2. **检查块质量**
   ```bash
   # 只训练单个块进行测试
   python train_blocks.py --config config.yaml --source_path data --model_path test --skip_aggregation
   ```

3. **渐进式聚合**
   ```bash
   # 先聚合部分块测试
   python train_blocks.py --config config.yaml --source_path data --model_path test --skip_training
   ```

## 📈 与原始训练的比较

| 方面 | 原始训练 | 分块训练 |
|------|----------|----------|
| 内存使用 | 高 | 低（按块） |
| 训练时间 | 长 | 可并行化 |
| 场景规模 | 受限 | 可扩展 |
| 模型质量 | 统一 | 需要聚合 |
| 断点续训 | 支持 | 块级支持 |

## 🔮 未来改进

- [ ] 并行块训练支持
- [ ] 更智能的块分割算法
- [ ] 自适应重叠大小
- [ ] 实时聚合监控
- [ ] 块间一致性优化
- [ ] 增量式聚合更新

## 📝 引用

如果这个分块训练系统对您的研究有帮助，请引用原始的 Street Gaussian 论文：

```bibtex
@article{street_gaussian_2024,
  title={Street Gaussians for Modeling Dynamic Urban Scenes},
  author={...},
  journal={...},
  year={2024}
}
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个分块训练系统！

## 📄 许可证

本项目遵循与原始 Street Gaussian 相同的许可证。