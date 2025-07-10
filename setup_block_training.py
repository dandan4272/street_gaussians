#!/usr/bin/env python3
"""
Street Gaussian 分块训练环境设置脚本

检查和安装分块训练所需的依赖，配置环境。
"""

import os
import sys
import subprocess
import importlib
import argparse
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_module_availability(module_name, package_name=None, optional=False):
    """检查模块是否可用"""
    if package_name is None:
        package_name = module_name
        
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name} is available")
        return True
    except ImportError:
        status = "⚠️" if optional else "❌"
        opt_text = " (optional)" if optional else ""
        print(f"{status} {package_name} is not available{opt_text}")
        return not optional

def check_street_gaussian_modules():
    """检查Street Gaussian原始模块"""
    print("\nChecking Street Gaussian modules...")
    
    modules_to_check = [
        ('lib.config', 'Street Gaussian config'),
        ('lib.datasets.dataset', 'Street Gaussian dataset'),
        ('lib.models.street_gaussian_model', 'Street Gaussian model'),
        ('lib.models.street_gaussian_renderer', 'Street Gaussian renderer'),
        ('lib.utils.loss_utils', 'Street Gaussian loss utils'),
    ]
    
    all_available = True
    for module, name in modules_to_check:
        if not check_module_availability(module, name):
            all_available = False
    
    return all_available

def check_block_training_modules():
    """检查分块训练模块"""
    print("\nChecking block training modules...")
    
    modules_to_check = [
        ('lib.training.block_trainer', 'Block trainer'),
        ('lib.training.block_aggregator', 'Block aggregator'),
        ('lib.training.block_train_loop', 'Block training loop'),
    ]
    
    all_available = True
    for module, name in modules_to_check:
        if not check_module_availability(module, name):
            all_available = False
    
    return all_available

def check_dependencies():
    """检查所有依赖项"""
    print("\nChecking dependencies...")
    
    # 必需的依赖
    required_deps = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'scikit-learn'),
        ('psutil', 'psutil'),
    ]
    
    # 可选的依赖
    optional_deps = [
        ('trimesh', 'trimesh'),
        ('tensorboard', 'TensorBoard'),
        ('plyfile', 'PLY file reader'),
    ]
    
    all_required_available = True
    
    print("\nRequired dependencies:")
    for module, name in required_deps:
        if not check_module_availability(module, name):
            all_required_available = False
    
    print("\nOptional dependencies:")
    for module, name in optional_deps:
        check_module_availability(module, name, optional=True)
    
    return all_required_available

def install_dependencies(auto_install=False):
    """安装缺失的依赖"""
    dependencies_to_install = []
    
    # 检查并收集需要安装的依赖
    deps = [
        ('sklearn', 'scikit-learn'),
        ('psutil', 'psutil'),
        ('trimesh', 'trimesh'),
    ]
    
    for module, package in deps:
        try:
            importlib.import_module(module)
        except ImportError:
            dependencies_to_install.append(package)
    
    if not dependencies_to_install:
        print("✅ All dependencies are already installed")
        return True
    
    print(f"\nMissing dependencies: {', '.join(dependencies_to_install)}")
    
    if not auto_install:
        response = input("Would you like to install them? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Skipping dependency installation")
            return False
    
    print("Installing dependencies...")
    
    for package in dependencies_to_install:
        try:
            print(f"Installing {package}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', package
            ], capture_output=True, text=True, check=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully installed {package}")
            else:
                print(f"❌ Failed to install {package}")
                print(result.stderr)
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error installing {package}: {e}")
            return False
    
    return True

def check_cuda_availability():
    """检查CUDA可用性"""
    print("\nChecking CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"✅ CUDA is available")
            print(f"   Devices: {device_count}")
            print(f"   Primary device: {device_name}")
            return True
        else:
            print("⚠️ CUDA is not available - training will be slower on CPU")
            return False
    except ImportError:
        print("❌ PyTorch not available - cannot check CUDA")
        return False

def verify_installation():
    """验证安装"""
    print("\nVerifying installation...")
    
    try:
        # 测试导入分块训练模块
        from lib.training.block_trainer import BlockConfig, create_spatial_blocks
        
        # 创建测试配置
        blocks = create_spatial_blocks((-10, 10, -10, 10), 15.0, 2.0)
        
        if len(blocks) > 0:
            print("✅ Block training modules are working correctly")
            print(f"   Test created {len(blocks)} blocks")
            return True
        else:
            print("❌ Block training test failed")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def create_example_config():
    """创建示例配置文件"""
    print("\nCreating example configuration...")
    
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    example_config_path = config_dir / "my_block_training.yaml"
    
    if example_config_path.exists():
        response = input(f"{example_config_path} already exists. Overwrite? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Skipping config creation")
            return
    
    config_content = """# My Block Training Configuration
# Copy from configs/block_training_example.yaml and modify as needed

task: my_block_training
source_path: /path/to/your/dataset  # CHANGE THIS
exp_name: my_experiment
to_cuda: true

data:
  type: Waymo  # or Colmap, Blender
  white_background: false
  extent: 20

model:
  gaussian:
    sh_degree: 1
    fourier_dim: 5
  nsg:
    include_bkgd: true
    include_obj: true
    include_sky: false  # Recommended for block training

train:
  iterations: 15000  # Adjust based on block size
  test_iterations: [5000, 10000, 15000]
  save_iterations: [15000]
  checkpoint_iterations: [15000]

optim:
  densify_until_iter: 12000
  min_opacity: 0.01
  densify_grad_threshold: 0.0003

# Block training specific settings
block_training:
  auto_block_size: 50.0      # Block size in meters
  overlap_margin: 5.0        # Overlap between blocks
  spatial_threshold: 0.15    # Deduplication threshold
  opacity_threshold: 0.1     # Minimum opacity to keep
"""
    
    with open(example_config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Created example configuration: {example_config_path}")
    print("   Please edit the source_path and other settings as needed")

def print_usage_instructions():
    """打印使用说明"""
    print("\n" + "="*60)
    print("SETUP COMPLETED - USAGE INSTRUCTIONS")
    print("="*60)
    
    print("\n1. Prepare your dataset:")
    print("   - Ensure your dataset is in Waymo, COLMAP, or Blender format")
    print("   - Update the source_path in your config file")
    
    print("\n2. Basic usage:")
    print("   python train_blocks.py \\")
    print("       --config configs/my_block_training.yaml \\")
    print("       --source_path /path/to/dataset \\")
    print("       --model_path ./outputs/my_experiment")
    
    print("\n3. Advanced options:")
    print("   # Custom block size and overlap")
    print("   python train_blocks.py --config ... --block_size 30 --overlap_margin 3")
    print("   ")
    print("   # Specify scene bounds manually")
    print("   python train_blocks.py --config ... --scene_bounds -100 100 -50 50")
    print("   ")
    print("   # Skip training and only aggregate existing blocks")
    print("   python train_blocks.py --config ... --skip_training")
    
    print("\n4. Visualization and analysis:")
    print("   # Visualize block configuration")
    print("   python tools/visualize_blocks.py --block_config outputs/experiment/blocks_config.json")
    print("   ")
    print("   # Analyze performance")
    print("   python tools/performance_analyzer.py --experiment_path outputs/experiment")
    print("   ")
    print("   # Validate quality")
    print("   python tools/quality_validator.py --experiment_path outputs/experiment --config configs/my_block_training.yaml --source_path /path/to/dataset")
    
    print("\n5. Testing:")
    print("   python tests/test_block_training.py")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="Setup Street Gaussian Block Training Environment")
    parser.add_argument('--install-deps', action='store_true',
                       help='Automatically install missing dependencies')
    parser.add_argument('--skip-verification', action='store_true',
                       help='Skip verification step')
    parser.add_argument('--create-config', action='store_true',
                       help='Create example configuration file')
    
    args = parser.parse_args()
    
    print("Street Gaussian Block Training Environment Setup")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查CUDA
    check_cuda_availability()
    
    # 检查依赖
    if not check_dependencies():
        if args.install_deps or input("\nWould you like to install missing dependencies? (y/N): ").lower() in ['y', 'yes']:
            if not install_dependencies(auto_install=args.install_deps):
                print("❌ Failed to install dependencies")
                sys.exit(1)
        else:
            print("❌ Missing required dependencies. Please install them manually:")
            print("   pip install scikit-learn psutil matplotlib")
            sys.exit(1)
    
    # 检查Street Gaussian模块
    sg_available = check_street_gaussian_modules()
    if not sg_available:
        print("\n⚠️ Warning: Street Gaussian modules not fully available")
        print("   Make sure you're running this from the Street Gaussian project directory")
        print("   and that the original project is properly set up")
    
    # 检查分块训练模块
    bt_available = check_block_training_modules()
    if not bt_available:
        print("\n❌ Error: Block training modules not available")
        print("   Make sure all block training files are properly placed")
        sys.exit(1)
    
    # 验证安装
    if not args.skip_verification:
        if not verify_installation():
            print("❌ Installation verification failed")
            sys.exit(1)
    
    # 创建示例配置
    if args.create_config or input("\nWould you like to create an example configuration file? (y/N): ").lower() in ['y', 'yes']:
        create_example_config()
    
    # 打印使用说明
    print_usage_instructions()
    
    print("\n✅ Setup completed successfully!")
    print("You can now use the block training system.")

if __name__ == "__main__":
    main()