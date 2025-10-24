#!/usr/bin/env python3
"""
Dataset Download Script for Respiratory Disease Detection
========================================================

This script downloads and organizes multiple respiratory sound datasets
for training machine learning models. It handles data validation,
preprocessing, and organization into a unified format.
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
import gzip
import shutil
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests

class DatasetDownloader:
    """
    Download and manage respiratory sound datasets.
    
    Supported datasets:
    - HF_Lung_V1: Large-scale lung sound database
    - KAUH: King Abdullah University Hospital dataset  
    - HLS-CMDS: Heart and Lung Sounds Clinical Manikin Dataset
    - ICBHI 2017: Respiratory Sound Database
    - Custom curated datasets
    """
    
    def __init__(self, data_dir="data/datasets"):
        """
        Initialize dataset downloader.
        
        Args:
            data_dir (str): Directory to store datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            "hf_lung_v1": {
                "name": "HF_Lung_V1",
                "url": "https://gitlab.com/techsupportHF/HF_Lung_V1/-/archive/main/HF_Lung_V1-main.zip",
                "description": "9,765 lung sound files with detailed annotations",
                "size_mb": 2500,
                "format": "wav",
                "labels": ["normal", "wheeze", "crackle", "stridor", "rhonchus"],
                "license": "Creative Commons",
                "citation": "Hsu et al., PLoS ONE 2021"
            },
            "icbhi_2017": {
                "name": "ICBHI 2017 Challenge",
                "url": "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database.zip", 
                "description": "920 annotated recordings from ICBHI Challenge",
                "size_mb": 400,
                "format": "wav",
                "labels": ["normal", "crackle", "wheeze", "both"],
                "license": "Academic use",
                "citation": "Rocha et al., 2019"
            },
            "kauh_dataset": {
                "name": "KAUH Dataset",
                "url": "https://data.mendeley.com/public-files/datasets/jwyy9np4gv/files/",
                "description": "308 recordings from King Abdullah University Hospital",
                "size_mb": 150,
                "format": "wav", 
                "labels": ["normal", "asthma", "copd", "pneumonia", "bronchitis"],
                "license": "CC BY 4.0",
                "citation": "Fraiwan et al., 2021"
            },
            "hls_cmds": {
                "name": "Heart and Lung Sounds Clinical Manikin Dataset",
                "url": "https://www.kaggle.com/datasets/yasamantorabi/heart-and-lung-sounds-dataset-hls-cmds/download",
                "description": "535 recordings from clinical manikin",
                "size_mb": 80,
                "format": "wav",
                "labels": ["normal", "abnormal_heart", "abnormal_lung"],
                "license": "Public Domain",
                "citation": "Torabi et al., 2024"
            },
            "respiratory_sound_database": {
                "name": "Respiratory Sound Database", 
                "url": "https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database/download",
                "description": "920 annotated recordings of varying length",
                "size_mb": 200,
                "format": "wav",
                "labels": ["healthy", "copd", "urti", "bronchiectasis", "pneumonia"],
                "license": "Open Access",
                "citation": "Multiple contributors"
            }
        }
    
    def download_file(self, url, filename, desc=None):
        """
        Download file with progress bar.
        
        Args:
            url (str): Download URL
            filename (str): Local filename
            desc (str): Description for progress bar
        """
        if desc is None:
            desc = f"Downloading {filename}"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ Downloaded: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {url}: {str(e)}")
            return False
    
    def verify_checksum(self, filepath, expected_hash=None):
        """
        Verify file integrity using MD5 hash.
        
        Args:
            filepath (str): Path to file
            expected_hash (str): Expected MD5 hash
            
        Returns:
            bool: True if hash matches or no expected hash provided
        """
        if expected_hash is None:
            return True
        
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        calculated_hash = hash_md5.hexdigest()
        
        if calculated_hash == expected_hash:
            print(f"✅ Checksum verified: {filepath}")
            return True
        else:
            print(f"❌ Checksum mismatch for {filepath}")
            print(f"   Expected: {expected_hash}")
            print(f"   Calculated: {calculated_hash}")
            return False
    
    def extract_archive(self, archive_path, extract_to):
        """
        Extract various archive formats.
        
        Args:
            archive_path (str): Path to archive file
            extract_to (str): Directory to extract to
        """
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)
        
        try:
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            
            elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            
            elif archive_path.endswith('.gz'):
                with gzip.open(archive_path, 'rb') as f_in:
                    output_path = extract_to / Path(archive_path).stem
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            else:
                print(f"⚠️ Unsupported archive format: {archive_path}")
                return False
            
            print(f"✅ Extracted: {archive_path} -> {extract_to}")
            return True
            
        except Exception as e:
            print(f"❌ Error extracting {archive_path}: {str(e)}")
            return False
    
    def download_hf_lung_v1(self):
        """Download HF_Lung_V1 dataset."""
        dataset_info = self.datasets["hf_lung_v1"]
        dataset_dir = self.data_dir / "hf_lung_v1"
        
        print(f"\n📥 Downloading {dataset_info['name']}...")
        print(f"   Description: {dataset_info['description']}")
        print(f"   Size: ~{dataset_info['size_mb']} MB")
        
        # Download from GitLab
        archive_path = dataset_dir / "hf_lung_v1.zip"
        
        if not self.download_file(dataset_info["url"], archive_path):
            return False
        
        # Extract
        if not self.extract_archive(archive_path, dataset_dir):
            return False
        
        # Organize files
        self.organize_hf_lung_data(dataset_dir)
        
        # Cleanup
        os.remove(archive_path)
        
        return True
    
    def download_icbhi_2017(self):
        """Download ICBHI 2017 dataset."""
        dataset_info = self.datasets["icbhi_2017"]
        dataset_dir = self.data_dir / "icbhi_2017"
        
        print(f"\n📥 Downloading {dataset_info['name']}...")
        print(f"   Description: {dataset_info['description']}")
        
        # Note: This URL might require registration/authentication
        print("⚠️ ICBHI 2017 dataset requires manual download from:")
        print("   https://bhichallenge.med.auth.gr/")
        print("   Please download and place files in:", dataset_dir)
        
        # Create directory structure
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "audio_and_txt_files").mkdir(exist_ok=True)
        
        # Create instructions file
        instructions = """
ICBHI 2017 Dataset Download Instructions:

1. Visit: https://bhichallenge.med.auth.gr/
2. Register for an account (free for academic use)
3. Download ICBHI_final_database.zip
4. Extract the contents to this directory
5. Run the preprocessing script to organize the data

Expected structure after extraction:
- audio_and_txt_files/
  - *.wav (audio files)
  - *.txt (annotation files)
- patient_list_folds.txt
- demographic_info.txt
"""
        
        with open(dataset_dir / "DOWNLOAD_INSTRUCTIONS.txt", 'w') as f:
            f.write(instructions)
        
        return True
    
    def download_synthetic_datasets(self):
        """Create synthetic datasets for testing when real data unavailable."""
        print("\n🔧 Creating synthetic datasets for testing...")
        
        synthetic_dir = self.data_dir / "synthetic"
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate synthetic audio samples
        import librosa
        import soundfile as sf
        
        sample_rate = 16000
        duration = 10  # seconds
        
        # Create different types of synthetic respiratory sounds
        sound_types = {
            "normal": self.generate_normal_breathing,
            "wheeze": self.generate_wheeze_sound,
            "crackle": self.generate_crackle_sound,
            "stridor": self.generate_stridor_sound
        }
        
        for sound_type, generator_func in sound_types.items():
            type_dir = synthetic_dir / sound_type
            type_dir.mkdir(exist_ok=True)
            
            # Generate 10 samples per type
            for i in range(10):
                audio = generator_func(duration, sample_rate)
                filename = type_dir / f"{sound_type}_{i:03d}.wav"
                sf.write(filename, audio, sample_rate)
        
        # Create metadata file
        metadata = []
        for sound_type in sound_types.keys():
            for i in range(10):
                metadata.append({
                    'filename': f"{sound_type}/{sound_type}_{i:03d}.wav",
                    'label': sound_type,
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'synthetic': True
                })
        
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(synthetic_dir / "metadata.csv", index=False)
        
        print(f"✅ Created synthetic dataset with {len(metadata)} samples")
        return True
    
    def generate_normal_breathing(self, duration, sample_rate):
        """Generate synthetic normal breathing sound."""
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Base breathing rhythm (12 breaths per minute)
        breathing_freq = 12 / 60  # Hz
        
        # Generate breathing envelope
        breathing_pattern = np.sin(2 * np.pi * breathing_freq * t)
        breathing_pattern = np.maximum(0, breathing_pattern)  # Only positive
        
        # Add some noise and harmonics
        noise = np.random.normal(0, 0.1, len(t))
        harmonics = 0.3 * np.sin(4 * np.pi * breathing_freq * t)
        
        audio = breathing_pattern + harmonics + noise
        audio = audio * 0.1  # Reduce amplitude
        
        return audio.astype(np.float32)
    
    def generate_wheeze_sound(self, duration, sample_rate):
        """Generate synthetic wheeze sound."""
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Base breathing with wheeze overtone
        breathing_freq = 15 / 60  # Slightly faster breathing
        wheeze_freq = 400  # Hz - typical wheeze frequency
        
        breathing = np.sin(2 * np.pi * breathing_freq * t)
        wheeze = 0.5 * np.sin(2 * np.pi * wheeze_freq * t)
        
        # Modulate wheeze with breathing
        wheeze_modulated = wheeze * np.maximum(0, breathing)
        
        # Add base breathing and noise
        base_breathing = 0.3 * np.maximum(0, breathing)
        noise = np.random.normal(0, 0.05, len(t))
        
        audio = base_breathing + wheeze_modulated + noise
        audio = audio * 0.15
        
        return audio.astype(np.float32)
    
    def generate_crackle_sound(self, duration, sample_rate):
        """Generate synthetic crackle sound."""
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Base breathing
        breathing_freq = 14 / 60
        breathing = np.sin(2 * np.pi * breathing_freq * t)
        base_breathing = 0.2 * np.maximum(0, breathing)
        
        # Add crackle events
        audio = base_breathing.copy()
        
        # Random crackle events during inspiration
        inspiration_mask = breathing > 0
        inspiration_indices = np.where(inspiration_mask)[0]
        
        # Add 5-15 crackle events
        n_crackles = np.random.randint(5, 16)
        crackle_indices = np.random.choice(inspiration_indices, n_crackles, replace=False)
        
        for idx in crackle_indices:
            # Short, sharp transient
            crackle_duration = int(0.005 * sample_rate)  # 5ms
            end_idx = min(idx + crackle_duration, len(audio))
            
            crackle_t = np.linspace(0, 0.005, end_idx - idx)
            crackle = 0.3 * np.exp(-crackle_t * 200) * np.random.normal(0, 1, len(crackle_t))
            
            audio[idx:end_idx] += crackle
        
        # Add background noise
        noise = np.random.normal(0, 0.03, len(t))
        audio += noise
        
        return audio.astype(np.float32)
    
    def generate_stridor_sound(self, duration, sample_rate):
        """Generate synthetic stridor sound."""
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Base breathing pattern
        breathing_freq = 18 / 60  # Faster, labored breathing
        stridor_freq = 300  # Hz - typical stridor frequency
        
        breathing = np.sin(2 * np.pi * breathing_freq * t)
        
        # Stridor is more prominent during inspiration
        inspiration_mask = breathing > 0
        
        # Generate stridor sound
        stridor = np.zeros_like(t)
        stridor[inspiration_mask] = 0.4 * np.sin(2 * np.pi * stridor_freq * t[inspiration_mask])
        
        # Add harmonics
        stridor[inspiration_mask] += 0.2 * np.sin(4 * np.pi * stridor_freq * t[inspiration_mask])
        
        # Base breathing
        base_breathing = 0.2 * np.maximum(0, breathing)
        
        # Add noise
        noise = np.random.normal(0, 0.04, len(t))
        
        audio = base_breathing + stridor + noise
        audio = audio * 0.12
        
        return audio.astype(np.float32)
    
    def organize_hf_lung_data(self, dataset_dir):
        """Organize HF_Lung_V1 dataset structure."""
        print("📂 Organizing HF_Lung_V1 dataset...")
        
        # Find extracted directory
        extracted_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and 'HF_Lung' in d.name]
        
        if not extracted_dirs:
            print("⚠️ Could not find extracted HF_Lung_V1 directory")
            return False
        
        source_dir = extracted_dirs[0]
        
        # Create organized structure
        organized_dir = dataset_dir / "organized"
        organized_dir.mkdir(exist_ok=True)
        
        # Move and organize files
        for audio_file in source_dir.rglob("*.wav"):
            # Create category directory based on filename or metadata
            category = self.classify_hf_lung_file(audio_file.name)
            category_dir = organized_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # Copy file
            shutil.copy2(audio_file, category_dir / audio_file.name)
        
        print(f"✅ Organized HF_Lung_V1 dataset in {organized_dir}")
        return True
    
    def classify_hf_lung_file(self, filename):
        """Classify HF_Lung file based on filename patterns."""
        filename_lower = filename.lower()
        
        if 'wheeze' in filename_lower or 'whz' in filename_lower:
            return 'wheeze'
        elif 'crackle' in filename_lower or 'ckl' in filename_lower:
            return 'crackle'
        elif 'stridor' in filename_lower:
            return 'stridor'
        elif 'rhonchus' in filename_lower or 'rhon' in filename_lower:
            return 'rhonchus'
        elif 'normal' in filename_lower or 'healthy' in filename_lower:
            return 'normal'
        else:
            return 'unknown'
    
    def create_unified_dataset(self):
        """Create unified dataset from all downloaded datasets."""
        print("\n🔗 Creating unified dataset...")
        
        unified_dir = self.data_dir / "unified"
        unified_dir.mkdir(exist_ok=True)
        
        all_metadata = []
        
        # Process each dataset
        for dataset_name in self.datasets.keys():
            dataset_dir = self.data_dir / dataset_name
            if dataset_dir.exists():
                metadata = self.process_dataset_for_unification(dataset_name, dataset_dir)
                all_metadata.extend(metadata)
        
        # Create master metadata file
        if all_metadata:
            unified_df = pd.DataFrame(all_metadata)
            unified_df.to_csv(unified_dir / "master_metadata.csv", index=False)
            
            # Create label distribution summary
            label_counts = unified_df['label'].value_counts()
            print("\n📊 Dataset Label Distribution:")
            for label, count in label_counts.items():
                print(f"   {label}: {count} samples")
            
            # Save label distribution
            label_counts.to_csv(unified_dir / "label_distribution.csv")
            
            print(f"\n✅ Created unified dataset with {len(all_metadata)} samples")
            print(f"   Metadata saved to: {unified_dir / 'master_metadata.csv'}")
        
        return True
    
    def process_dataset_for_unification(self, dataset_name, dataset_dir):
        """Process individual dataset for unification."""
        metadata = []
        
        # Find audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        
        for audio_file in dataset_dir.rglob('*'):
            if audio_file.suffix.lower() in audio_extensions:
                # Extract metadata
                try:
                    import librosa
                    duration = librosa.get_duration(filename=str(audio_file))
                    
                    # Determine label from directory structure or filename
                    label = self.infer_label_from_path(audio_file, dataset_name)
                    
                    metadata.append({
                        'filename': str(audio_file.relative_to(self.data_dir)),
                        'dataset': dataset_name,
                        'label': label,
                        'duration': duration,
                        'file_size': audio_file.stat().st_size,
                        'format': audio_file.suffix.lower()
                    })
                    
                except Exception as e:
                    print(f"⚠️ Error processing {audio_file}: {str(e)}")
                    continue
        
        return metadata
    
    def infer_label_from_path(self, audio_path, dataset_name):
        """Infer label from file path and dataset conventions."""
        path_str = str(audio_path).lower()
        
        # Common label patterns
        label_patterns = {
            'normal': ['normal', 'healthy', 'clean'],
            'wheeze': ['wheeze', 'whz', 'asthma'],
            'crackle': ['crackle', 'ckl', 'rale', 'pneumonia'],
            'stridor': ['stridor'],
            'rhonchus': ['rhonchus', 'rhon'],
            'copd': ['copd'],
            'bronchitis': ['bronchitis', 'bronch'],
            'pneumonia': ['pneumonia'],
            'asthma': ['asthma']
        }
        
        for label, patterns in label_patterns.items():
            for pattern in patterns:
                if pattern in path_str:
                    return label
        
        # Check parent directory names
        for parent in audio_path.parents:
            parent_name = parent.name.lower()
            for label, patterns in label_patterns.items():
                for pattern in patterns:
                    if pattern in parent_name:
                        return label
        
        return 'unknown'
    
    def download_all(self):
        """Download all available datasets."""
        print("🚀 Starting comprehensive dataset download...")
        
        success_count = 0
        total_count = len(self.datasets)
        
        # Download each dataset
        for dataset_name in self.datasets.keys():
            try:
                if dataset_name == "hf_lung_v1":
                    success = self.download_hf_lung_v1()
                elif dataset_name == "icbhi_2017":
                    success = self.download_icbhi_2017()
                else:
                    print(f"\n⚠️ {dataset_name} download not implemented yet")
                    success = False
                
                if success:
                    success_count += 1
                
            except Exception as e:
                print(f"❌ Error downloading {dataset_name}: {str(e)}")
        
        # Always create synthetic datasets
        self.download_synthetic_datasets()
        
        # Create unified dataset
        self.create_unified_dataset()
        
        print(f"\n📋 Download Summary:")
        print(f"   Successfully downloaded: {success_count}/{total_count} datasets")
        print(f"   Data directory: {self.data_dir}")
        
        return success_count > 0

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download respiratory sound datasets")
    parser.add_argument("--data-dir", default="data/datasets", 
                       help="Directory to store datasets")
    parser.add_argument("--dataset", choices=["all", "hf_lung_v1", "icbhi_2017", "synthetic"],
                       default="all", help="Specific dataset to download")
    parser.add_argument("--create-unified", action="store_true",
                       help="Create unified dataset from existing downloads")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.create_unified:
        downloader.create_unified_dataset()
    elif args.dataset == "all":
        downloader.download_all()
    elif args.dataset == "synthetic":
        downloader.download_synthetic_datasets()
    elif args.dataset == "hf_lung_v1":
        downloader.download_hf_lung_v1()
    elif args.dataset == "icbhi_2017":
        downloader.download_icbhi_2017()
    
    print("\n🎉 Dataset download process completed!")

if __name__ == "__main__":
    main()