#!/usr/bin/env python3
"""
@file       : download_preset_data.py
@author     : J.J.G. Pleunes
@date       : 08/2025
@brief      : Download and prepare datasets based on presets
@details    : This script downloads datasets according to preset configurations
              and prepares them for training with the L1 model.
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
import yaml
import argparse
from pathlib import Path
import kagglehub
from typing import Dict, List
import shutil
import requests
import zipfile
from tqdm import tqdm

class PresetDatasetDownloader:
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_config(self, config_file: str = "datasets.yaml") -> Dict:
        """Load datasets configuration"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def download_kagglehub_dataset(self, dataset_path: str, dataset_name: str) -> str:
        """Download dataset using KaggleHub"""
        print(f"=> Downloading {dataset_name} from KaggleHub...")
        try:
            download_path = kagglehub.dataset_download(dataset_path)
            print(f"   Downloaded to: {download_path}")
            return download_path
        except Exception as e:
            print(f"   Error downloading {dataset_name}: {e}")
            return None
    
    def find_text_files(self, directory: str, extensions: List[str] = ['.txt', '.csv', '.json']) -> List[str]:
        """Find text files in directory"""
        text_files = []
        for ext in extensions:
            text_files.extend(Path(directory).rglob(f"*{ext}"))
        return [str(f) for f in text_files]
    
    def combine_text_files(self, file_paths: List[str], output_file: str, max_samples: int = None):
        """Combine multiple text files into one"""
        print(f"=> Combining {len(file_paths)} files...")
        
        with open(output_file, 'w', encoding='utf-8') as outf:
            sample_count = 0
            for file_path in tqdm(file_paths, desc="Processing files"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as inf:
                        for line in inf:
                            line = line.strip()
                            if line and len(line) > 50:  # Filter short lines
                                outf.write(line + '\n')
                                sample_count += 1
                                if max_samples and sample_count >= max_samples:
                                    return sample_count
                except Exception as e:
                    print(f"   Warning: Error processing {file_path}: {e}")
                    continue
        
        return sample_count
    
    def setup_preset(self, preset_name: str, config_file: str = "datasets.yaml"):
        """Setup datasets for a specific preset"""
        config = self.load_config(config_file)
        
        if 'presets' not in config or preset_name not in config['presets']:
            available_presets = list(config.get('presets', {}).keys())
            print(f"Error: Preset '{preset_name}' not found!")
            print(f"Available presets: {', '.join(available_presets)}")
            return False
        
        preset = config['presets'][preset_name]
        datasets = config.get('datasets', {})
        
        print(f"* Setting up preset: {preset['name']}")
        print(f"  {preset['description']}")
        print(f"  Target samples: {preset['max_samples']:,}")
        print()
        
        # Prepare output file
        output_file = self.output_dir / f"{preset_name}_combined_dataset.txt"
        all_text_files = []
        
        # Download each dataset
        for dataset_id in preset['recommended_datasets']:
            if dataset_id not in datasets:
                print(f"Warning: Dataset '{dataset_id}' not found in config")
                continue
            
            dataset_config = datasets[dataset_id]
            print(f"=> Processing: {dataset_config['name']}")
            
            if dataset_config.get('download_method') == 'kagglehub':
                kagglehub_path = dataset_config.get('kagglehub_path')
                if kagglehub_path:
                    download_path = self.download_kagglehub_dataset(
                        kagglehub_path, dataset_config['name']
                    )
                    if download_path:
                        # Find text files in downloaded data
                        text_files = self.find_text_files(download_path)
                        all_text_files.extend(text_files)
                        print(f"   Found {len(text_files)} text files")
            else:
                print(f"   Warning: Download method '{dataset_config.get('download_method')}' not implemented yet")
            
            print()
        
        if all_text_files:
            print(f"=> Combining all text files into: {output_file}")
            sample_count = self.combine_text_files(
                all_text_files, 
                str(output_file), 
                preset['max_samples']
            )
            print(f"Success! Combined dataset ready: {sample_count:,} samples")
            print(f"Location: {output_file}")
            
            # Copy to standard location for training
            standard_location = self.output_dir / "combined_dataset.txt"
            shutil.copy2(output_file, standard_location)
            print(f"Copied to standard location: {standard_location}")
            
            return True
        else:
            print("Error: No text files found to combine")
            return False

def main():
    parser = argparse.ArgumentParser(description="Download datasets for L1 training presets")
    parser.add_argument("preset", 
                       help="Preset name (beginner, intermediate, advanced, conversational, technical, knowledge)")
    parser.add_argument("--config", default="datasets.yaml",
                       help="Configuration file")
    parser.add_argument("--output-dir", default="data/raw",
                       help="Output directory for downloaded data")
    
    args = parser.parse_args()
    
    downloader = PresetDatasetDownloader(args.output_dir)
    success = downloader.setup_preset(args.preset, args.config)
    
    if success:
        print(f"\nNext steps:")
        print(f"   1. Process the dataset: python prepare_large_dataset.py {args.output_dir}/combined_dataset.txt")
        print(f"   2. Start training: python train_gpu_compatible.py")
    else:
        print(f"\nFailed to setup preset '{args.preset}'")

if __name__ == "__main__":
    main()
