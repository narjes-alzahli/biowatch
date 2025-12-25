#!/usr/bin/env python3
"""
Check system requirements for BioWatch model training.
"""

import sys
import platform

def check_python():
    """Check Python version."""
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠️  Python 3.8+ required")
        return False
    print("  ✓ Python version OK")
    return True

def check_torch():
    """Check PyTorch installation."""
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            
            # Check VRAM
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # GB
                print(f"GPU {i} VRAM: {total_memory:.1f} GB")
                
                # Recommendations (using thresholds that account for ~11.99 GB reporting)
                if total_memory >= 15.5:  # 16 GB
                    print(f"  ✓ Excellent! Can use batch_size=16, feature fusion")
                elif total_memory >= 11.5:  # 12 GB
                    print(f"  ✓ Good! Can use batch_size=8, feature fusion")
                elif total_memory >= 7.5:  # 8 GB
                    print(f"  ⚠️  OK. Use batch_size=4, early fusion")
                else:
                    print(f"  ⚠️  Limited. Use batch_size=2, early fusion, smaller input size")
        else:
            print("  ⚠️  No GPU detected. Training will be very slow on CPU.")
            print("     Consider using cloud GPU (Colab, Kaggle) or getting a GPU.")
        
        return True
    except ImportError:
        print("  ⚠️  PyTorch not installed")
        print("     Install with: pip install torch torchvision")
        return False

def check_ram():
    """Check system RAM."""
    try:
        import psutil
        ram = psutil.virtual_memory()
        total_gb = ram.total / (1024**3)
        available_gb = ram.available / (1024**3)
        
        print(f"RAM: {total_gb:.1f} GB total, {available_gb:.1f} GB available")
        
        if total_gb >= 32:
            print("  ✓ Excellent RAM")
        elif total_gb >= 16:
            print("  ✓ Good RAM")
        else:
            print("  ⚠️  Limited RAM. May need to reduce batch size or workers")
        
        return True
    except ImportError:
        print("RAM: Unable to check (install psutil for details)")
        return False

def check_storage():
    """Check available storage."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        print(f"Storage: {free_gb:.1f} GB free")
        
        if free_gb >= 100:
            print("  ✓ Plenty of storage")
        elif free_gb >= 50:
            print("  ✓ Adequate storage")
        else:
            print("  ⚠️  Limited storage. May need to clean up or add more space")
        
        return True
    except:
        print("Storage: Unable to check")
        return False

def main():
    print("=" * 60)
    print("BioWatch Model System Requirements Check")
    print("=" * 60)
    print()
    
    print("System Information:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print()
    
    print("Requirements Check:")
    print("-" * 60)
    
    checks = []
    checks.append(("Python", check_python()))
    checks.append(("PyTorch", check_torch()))
    checks.append(("RAM", check_ram()))
    checks.append(("Storage", check_storage()))
    
    print()
    print("=" * 60)
    print("Summary:")
    print("-" * 60)
    
    all_ok = all(check[1] for check in checks)
    
    if all_ok:
        print("✓ All basic requirements met!")
    else:
        print("⚠️  Some requirements missing or suboptimal")
        print()
        print("Recommendations:")
        for name, ok in checks:
            if not ok:
                if name == "PyTorch":
                    print(f"  - Install PyTorch: pip install torch torchvision")
                    print(f"    For GPU support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print()
    print("For detailed hardware requirements, see:")
    print("  models/HARDWARE_REQUIREMENTS.md")

if __name__ == '__main__':
    main()

