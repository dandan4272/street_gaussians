#!/usr/bin/env python3
"""
Street Gaussians 分块训练测试脚本
用于快速验证修正后的代码是否正常工作
"""

import os
import sys
import torch
import argparse

def test_gpu_availability():
    """测试GPU可用性"""
    print("=== GPU可用性测试 ===")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA可用")
        print(f"  设备数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name}")
            print(f"    总内存: {memory_gb:.2f} GB")
            print(f"    计算能力: {props.major}.{props.minor}")
        
        # 测试GPU操作
        try:
            torch.cuda.set_device(0)
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.matmul(test_tensor, test_tensor.T)
            del test_tensor, result
            torch.cuda.empty_cache()
            print("✓ GPU计算测试通过")
        except Exception as e:
            print(f"✗ GPU计算测试失败: {e}")
            return False
            
        return True
    else:
        print("✗ CUDA不可用")
        return False

def test_imports():
    """测试导入"""
    print("\n=== 导入测试 ===")
    
    required_modules = [
        'torch',
        'numpy', 
        'tqdm',
        'yaml'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            return False
    
    # 测试项目特定的导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from lib.config import cfg
        print("✓ lib.config")
    except Exception as e:
        print(f"✗ lib.config: {e}")
        return False
    
    try:
        from lib.datasets.dataset import Dataset
        print("✓ lib.datasets.dataset")
    except Exception as e:
        print(f"✗ lib.datasets.dataset: {e}")
        return False
    
    return True

def test_memory_settings(max_memory_gb: float = 4.0):
    """测试内存设置"""
    print(f"\n=== 内存设置测试 (限制: {max_memory_gb} GB) ===")
    
    if not torch.cuda.is_available():
        print("跳过GPU内存测试")
        return True
    
    try:
        # 获取GPU信息
        device_props = torch.cuda.get_device_properties(0)
        total_memory_gb = device_props.total_memory / (1024**3)
        
        print(f"GPU总内存: {total_memory_gb:.2f} GB")
        
        # 设置内存限制
        if max_memory_gb < total_memory_gb:
            memory_fraction = max_memory_gb / total_memory_gb
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            print(f"设置内存使用比例: {memory_fraction:.2f}")
        
        # 测试内存分配
        allocated_before = torch.cuda.memory_allocated() / (1024**3)
        print(f"分配前内存: {allocated_before:.2f} GB")
        
        # 分配一些内存测试
        test_size = min(1000, int(max_memory_gb * 100))  # 动态调整测试大小
        test_tensors = []
        
        for i in range(10):
            tensor = torch.randn(test_size, test_size).cuda()
            test_tensors.append(tensor)
            
            allocated = torch.cuda.memory_allocated() / (1024**3)
            if allocated > max_memory_gb * 0.8:  # 如果接近限制就停止
                break
        
        allocated_after = torch.cuda.memory_allocated() / (1024**3)
        print(f"分配后内存: {allocated_after:.2f} GB")
        
        # 清理内存
        del test_tensors
        torch.cuda.empty_cache()
        
        allocated_final = torch.cuda.memory_allocated() / (1024**3)
        print(f"清理后内存: {allocated_final:.2f} GB")
        
        print("✓ 内存管理测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 内存管理测试失败: {e}")
        return False

def create_minimal_test_data(output_dir: str):
    """创建最小测试数据"""
    print(f"\n=== 创建测试数据 ({output_dir}) ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建基本的文件结构
    test_files = [
        "images/frame_001.jpg",
        "images/frame_002.jpg", 
        "images/frame_003.jpg",
        "cameras.json",
        "points3D.ply"
    ]
    
    for file_path in test_files:
        full_path = os.path.join(output_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        if file_path.endswith('.jpg'):
            # 创建虚拟图像文件
            with open(full_path, 'w') as f:
                f.write("# Dummy image file for testing")
        elif file_path.endswith('.json'):
            # 创建虚拟JSON文件
            with open(full_path, 'w') as f:
                f.write('{"test": "data"}')
        elif file_path.endswith('.ply'):
            # 创建虚拟PLY文件
            with open(full_path, 'w') as f:
                f.write("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    
    print(f"✓ 测试数据创建完成")
    return True

def run_quick_test():
    """运行快速测试"""
    print("\n=== 快速训练测试 ===")
    
    # 这里可以添加一个非常简化的训练测试
    # 由于原始代码比较复杂，我们先确保基本组件能工作
    
    try:
        # 测试基本的torch操作
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
        print(f"使用设备: {device}")
        
        # 创建一些测试张量
        x = torch.randn(100, 3, device=device)
        y = torch.randn(100, 3, device=device)
        
        # 简单的计算测试
        result = torch.matmul(x, y.T)
        loss = result.mean()
        
        print(f"计算结果形状: {result.shape}")
        print(f"损失值: {loss.item():.6f}")
        
        # 清理
        del x, y, result, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✓ 基本计算测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 基本计算测试失败: {e}")
        return False

def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description="Street Gaussians 分块训练测试")
    parser.add_argument("--max_memory", default=4.0, type=float,
                       help="最大GPU内存使用 (GB)")
    parser.add_argument("--create_test_data", action="store_true",
                       help="创建测试数据")
    parser.add_argument("--test_data_dir", default="./test_data", type=str,
                       help="测试数据目录")
    
    args = parser.parse_args()
    
    print("Street Gaussians 分块训练系统测试")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        ("GPU可用性", test_gpu_availability),
        ("导入模块", test_imports),
        ("内存设置", lambda: test_memory_settings(args.max_memory)),
        ("基本计算", run_quick_test),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"测试 '{test_name}' 出现异常: {e}")
            results.append((test_name, False))
    
    # 可选：创建测试数据
    if args.create_test_data:
        create_test_data_result = create_minimal_test_data(args.test_data_dir)
        results.append(("创建测试数据", create_test_data_result))
    
    # 总结结果
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过! 系统准备就绪。")
        
        if args.create_test_data:
            print(f"\n下一步，你可以运行:")
            print(f"python train_chunked_fixed.py --source_path {args.test_data_dir} --chunk_size 2 --iterations 100")
    else:
        print("⚠️  部分测试失败，请检查系统配置。")
        
        print("\n常见问题解决方案:")
        print("1. 如果CUDA不可用，确保安装了支持CUDA的PyTorch")
        print("2. 如果导入失败，确保在正确的conda环境中")
        print("3. 如果内存测试失败，尝试减少 --max_memory 参数")

if __name__ == "__main__":
    main()