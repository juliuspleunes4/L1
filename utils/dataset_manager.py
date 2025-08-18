#!/usr/bin/env python3
"""
@file       : dataset_manager.py
@author     : J.J.G. Pleunes
@date       : 07/2025
@brief      : Universal Dataset Manager for L1
@details    : This script manages datasets for L1 training, allowing users to
              download, process, and set up datasets easily.
@version    : 1.0

@license    : MIT License
Copyright (c) 2025 Julius Pleunes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys
import yaml
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import argparse

class DatasetManager:
    def __init__(self, config_file="datasets.yaml"):
        """Initialize with dataset configuration"""
        self.config_file = config_file
        self.config = self._load_config()
        self.datasets = self.config.get('datasets', {})
        self.presets = self.config.get('presets', {})
    
    def _load_config(self) -> Dict:
        """Load dataset configuration from YAML"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"❌ Configuration file {self.config_file} not found!")
            return {}
        except yaml.YAMLError as e:
            print(f"❌ Error parsing {self.config_file}: {e}")
            return {}
    
    def list_datasets(self, filter_topic: Optional[str] = None, filter_quality: Optional[str] = None):
        """List all available datasets with filtering"""
        print("📊 Available Datasets:")
        print("=" * 80)
        
        for dataset_id, dataset in self.datasets.items():
            # Apply filters
            if filter_topic and filter_topic not in dataset.get('topics', []):
                continue
            if filter_quality and dataset.get('quality') != filter_quality:
                continue
            
            print(f"\n🔹 {dataset_id}")
            print(f"   Name: {dataset['name']}")
            print(f"   Description: {dataset['description']}")
            print(f"   Quality: {dataset.get('quality', 'unknown')}")
            print(f"   Topics: {', '.join(dataset.get('topics', []))}")
            print(f"   Recommended samples: {dataset.get('recommended_samples', 'N/A'):,}")
            print(f"   Download method: {dataset.get('download_method', 'unknown')}")
    
    def list_presets(self):
        """List all available presets"""
        print("🎯 Available Presets:")
        print("=" * 60)
        
        for preset_id, preset in self.presets.items():
            print(f"\n🔸 {preset_id}")
            print(f"   Name: {preset['name']}")
            print(f"   Description: {preset['description']}")
            print(f"   Datasets: {', '.join(preset['recommended_datasets'])}")
            print(f"   Max samples: {preset['max_samples']:,}")
            print(f"   Vocab size: {preset['vocab_size']:,}")
    
    def install_dependencies(self, dataset: Dict):
        """Install required dependencies for dataset download"""
        method = dataset.get('download_method')
        
        if method == 'kagglehub':
            try:
                import kagglehub
            except ImportError:
                print("📦 Installing kagglehub...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        
        elif method == 'kaggle_api':
            try:
                import kaggle
            except ImportError:
                print("📦 Installing kaggle...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    
    def download_dataset(self, dataset_id: str) -> Optional[str]:
        """Download a specific dataset"""
        if dataset_id not in self.datasets:
            print(f"❌ Dataset '{dataset_id}' not found!")
            return None
        
        dataset = self.datasets[dataset_id]
        print(f"📥 Downloading: {dataset['name']}")
        print(f"📝 Description: {dataset['description']}")
        
        # Install dependencies
        self.install_dependencies(dataset)
        
        method = dataset.get('download_method')
        
        try:
            if method == 'kagglehub':
                return self._download_kagglehub(dataset)
            elif method == 'kaggle_api':
                return self._download_kaggle_api(dataset, dataset_id)
            else:
                print(f"❌ Unknown download method: {method}")
                return None
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def _download_kagglehub(self, dataset: Dict) -> str:
        """Download using kagglehub"""
        import kagglehub
        kagglehub_path = dataset['kagglehub_path']
        print(f"🔄 Downloading via kagglehub: {kagglehub_path}")
        
        path = kagglehub.dataset_download(kagglehub_path)
        print(f"✅ Downloaded to: {path}")
        return path
    
    def _download_kaggle_api(self, dataset: Dict, dataset_id: str) -> str:
        """Download using traditional Kaggle API"""
        kaggle_path = dataset['kaggle_path']
        print(f"🔄 Downloading via Kaggle API: {kaggle_path}")
        
        # Create download directory
        download_dir = Path("datasets") / dataset_id
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and extract
        cmd = f"kaggle datasets download -d {kaggle_path} -p {download_dir} --unzip"
        subprocess.run(cmd, shell=True, check=True)
        
        print(f"✅ Downloaded to: {download_dir}")
        return str(download_dir)
    
    def find_text_files(self, dataset_path: str, dataset: Dict) -> List[Path]:
        """Find text files in downloaded dataset"""
        dataset_dir = Path(dataset_path)
        text_files = []
        
        # Use file pattern if specified
        if 'file_pattern' in dataset:
            pattern = dataset['file_pattern']
            text_files.extend(list(dataset_dir.glob(pattern)))
            text_files.extend(list(dataset_dir.glob(f"**/{pattern}")))
        
        # Auto-detect common formats
        if dataset.get('auto_detect_format', True):
            for pattern in ["*.txt", "*.csv", "*.json", "*.jsonl"]:
                text_files.extend(list(dataset_dir.glob(pattern)))
                text_files.extend(list(dataset_dir.glob(f"**/{pattern}")))
        
        # Remove duplicates and sort by size
        text_files = list(set(text_files))
        text_files.sort(key=lambda f: f.stat().st_size, reverse=True)
        
        return text_files
    
    def process_dataset(self, dataset_id: str, dataset_path: str, max_samples: int, vocab_size: int) -> bool:
        """Process downloaded dataset for L1 training"""
        dataset = self.datasets[dataset_id]
        
        print(f"⚙️ Processing {dataset['name']}...")
        
        # Find text files
        text_files = self.find_text_files(dataset_path, dataset)
        if not text_files:
            print("❌ No text files found!")
            return False
        
        # Use the largest file (usually the main dataset)
        main_file = text_files[0]
        print(f"📄 Using file: {main_file.name} ({main_file.stat().st_size / (1024*1024):.1f} MB)")
        
        # Determine processing parameters
        file_format = self._detect_format(main_file)
        text_column = dataset.get('text_column', 'text')
        
        # Build processing command
        cmd_parts = [
            "python", "prepare_large_dataset.py",
            f'"{main_file}"',
            f"--format {file_format}",
            f"--max-samples {max_samples}",
            f"--vocab-size {vocab_size}"
        ]
        
        if file_format in ['csv', 'json']:
            if file_format == 'csv':
                cmd_parts.append(f'--text-column "{text_column}"')
            else:
                cmd_parts.append(f'--text-field "{text_column}"')
        
        cmd = " ".join(cmd_parts)
        print(f"🔧 Processing command: {cmd}")
        
        # Try processing with different column names if needed
        column_variants = [text_column, 'content', 'text', 'body', 'article', 'description']
        
        for i, col in enumerate(column_variants):
            try:
                if file_format == 'csv':
                    test_cmd = cmd.replace(f'--text-column "{text_column}"', f'--text-column "{col}"')
                elif file_format == 'json':
                    test_cmd = cmd.replace(f'--text-field "{text_column}"', f'--text-field "{col}"')
                else:
                    test_cmd = cmd
                
                print(f"🔄 Attempt {i+1}: Trying column '{col}'...")
                result = subprocess.run(test_cmd, shell=True, check=True, capture_output=True, text=True)
                print("✅ Processing successful!")
                return True
                
            except subprocess.CalledProcessError as e:
                if i < len(column_variants) - 1:
                    print(f"⚠️  Column '{col}' failed, trying next...")
                    continue
                else:
                    print(f"❌ All processing attempts failed")
                    print(f"Last error: {e.stderr}")
                    return False
        
        return False
    
    def _detect_format(self, file_path: Path) -> str:
        """Detect file format from extension"""
        suffix = file_path.suffix.lower()
        format_map = {
            '.txt': 'txt',
            '.csv': 'csv', 
            '.json': 'json',
            '.jsonl': 'jsonl'
        }
        return format_map.get(suffix, 'txt')
    
    def setup_dataset(self, dataset_id: str, max_samples: Optional[int] = None, vocab_size: Optional[int] = None) -> bool:
        """Complete dataset setup: download + process"""
        if dataset_id not in self.datasets:
            print(f"❌ Dataset '{dataset_id}' not found!")
            return False
        
        dataset = self.datasets[dataset_id]
        
        # Use recommended values if not provided
        max_samples = max_samples or dataset.get('recommended_samples', 100000)
        vocab_size = vocab_size or dataset.get('recommended_vocab', 20000)
        
        print(f"🚀 Setting up {dataset['name']}")
        print(f"🎯 Target samples: {max_samples:,}")
        print(f"📝 Vocab size: {vocab_size:,}")
        print("=" * 60)
        
        # Download dataset
        dataset_path = self.download_dataset(dataset_id)
        if not dataset_path:
            return False
        
        # Process dataset
        success = self.process_dataset(dataset_id, dataset_path, max_samples, vocab_size)
        
        if success:
            print(f"\n🎉 Dataset '{dataset_id}' setup complete!")
            print("📄 Generated files:")
            
            required_files = [
                "data/processed/train.txt",
                "data/processed/val.txt",
                "data/processed/tokenizer.json"
            ]
            
            for file_path in required_files:
                if os.path.exists(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"   ✅ {file_path} ({size_mb:.1f} MB)")
                else:
                    print(f"   ❌ {file_path} (missing)")
            
            print(f"\n🎯 Ready to train!")
            print(f"   python train_gpu.py")
            
        return success
    
    def setup_preset(self, preset_id: str) -> bool:
        """Setup multiple datasets using a preset"""
        if preset_id not in self.presets:
            print(f"❌ Preset '{preset_id}' not found!")
            return False
        
        preset = self.presets[preset_id]
        print(f"🎯 Setting up preset: {preset['name']}")
        print(f"📝 Description: {preset['description']}")
        
        datasets_to_use = preset['recommended_datasets']
        max_samples = preset['max_samples']
        vocab_size = preset['vocab_size']
        
        # For now, use the first dataset in the preset
        # TODO: Could be extended to combine multiple datasets
        primary_dataset = datasets_to_use[0]
        
        print(f"📊 Using primary dataset: {primary_dataset}")
        return self.setup_dataset(primary_dataset, max_samples, vocab_size)

def main():
    parser = argparse.ArgumentParser(description="L1 Universal Dataset Manager")
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    parser.add_argument("--list-presets", action="store_true", help="List all presets")
    parser.add_argument("--filter-topic", help="Filter datasets by topic")
    parser.add_argument("--filter-quality", help="Filter datasets by quality")
    parser.add_argument("--setup", help="Setup a specific dataset")
    parser.add_argument("--preset", help="Setup using a preset configuration")
    parser.add_argument("--samples", type=int, help="Maximum samples to process")
    parser.add_argument("--vocab-size", type=int, help="Vocabulary size")
    parser.add_argument("--config", default="datasets.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = DatasetManager(args.config)
    
    if args.list:
        manager.list_datasets(args.filter_topic, args.filter_quality)
    elif args.list_presets:
        manager.list_presets()
    elif args.setup:
        manager.setup_dataset(args.setup, args.samples, args.vocab_size)
    elif args.preset:
        manager.setup_preset(args.preset)
    else:
        print("🔧 L1 Universal Dataset Manager")
        print("=" * 50)
        print("Usage examples:")
        print("  python dataset_manager.py --list")
        print("  python dataset_manager.py --list-presets") 
        print("  python dataset_manager.py --setup wikipedia_simple")
        print("  python dataset_manager.py --preset beginner")
        print("  python dataset_manager.py --setup news_all --samples 50000")
        print("\nAdd new datasets to datasets.yaml!")

if __name__ == "__main__":
    main()
