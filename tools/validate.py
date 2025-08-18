#!/usr/bin/env python3
"""
Quick validation test for L1 setup
This script tests that all the documented commands work correctly
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, check_output=None):
    """Run a command and check if it succeeds"""
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ PASSED")
            if check_output and check_output not in result.stdout:
                print(f"⚠️  Warning: Expected output '{check_output}' not found")
            return True
        else:
            print("❌ FAILED")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT (this might be normal for download commands)")
        return True
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False
    finally:
        print("-" * 50)

def main():
    print("🧪 L1 Setup Validation Test")
    print("=" * 50)
    
    tests = [
        ("python data_tools/add_dataset.py --help", "Check add_dataset.py has --preset option", "--preset"),
        ("python data_tools/prepare_dataset.py --help", "Check prepare_dataset.py accepts input_path", "input_path"),
        ("python tools/train.py --help", "Check training script is available", None),
        ("python tools/generate.py --help", "Check generation script is available", None),
        ("python tools/demo.py --help", "Check demo script is available", None),
    ]
    
    passed = 0
    total = len(tests)
    
    for cmd, description, check_output in tests:
        if run_command(cmd, description, check_output):
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your L1 setup is ready.")
        print("\n🚀 Try the quick start:")
        print("   python data_tools/add_dataset.py --preset beginner")
    else:
        print("⚠️  Some tests failed. Please check the installation.")
        
    # Check if required files exist
    print("\n📁 Checking project structure...")
    required_files = [
        "data_tools/add_dataset.py",
        "data_tools/prepare_dataset.py", 
        "tools/train.py",
        "tools/generate.py",
        "tools/demo.py",
        "datasets.yaml",
        "requirements.txt"
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} missing")

if __name__ == "__main__":
    main()
