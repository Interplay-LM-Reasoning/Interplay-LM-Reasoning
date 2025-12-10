#!/usr/bin/env python3
"""
Robust Hugging Face Dataset Upload Script
Uploads .tar files with retry logic and progress tracking
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Tuple
import signal

try:
    from huggingface_hub import HfApi, HfFolder
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed. Please run:")
    print("pip install huggingface_hub tqdm")
    sys.exit(1)

class HuggingFaceUploader:
    def __init__(self, source_dir: str, repo_id: str, subfolder: str = "train_data", 
                 max_retries: int = 5, retry_delay: int = 10, timeout: int = 3600):
        self.source_dir = Path(source_dir)
        self.repo_id = repo_id
        self.subfolder = subfolder
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.api = HfApi()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'hf_upload_{int(time.time())}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Track upload status
        self.uploaded_files = []
        self.failed_files = []
        self.interrupted = False
        
        # Handle interruption gracefully
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        self.logger.info("Upload interrupted by user. Cleaning up...")
        self.interrupted = True
        self._print_summary()
        sys.exit(130)
    
    def check_authentication(self) -> bool:
        """Check if user is authenticated with Hugging Face"""
        try:
            user_info = self.api.whoami()
            self.logger.info(f"Authenticated as: {user_info['name']}")
            return True
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            self.logger.error("Please run: huggingface-cli login")
            return False
    
    def get_tar_files(self) -> List[Tuple[Path, int]]:
        """Get list of .tar files sorted by size (smallest first)"""
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {self.source_dir}")
        
        tar_files = []
        for tar_file in self.source_dir.glob("*.tar"):
            if tar_file.is_file():
                size = tar_file.stat().st_size
                tar_files.append((tar_file, size))
        
        # Sort by size (smallest first for quicker initial feedback)
        tar_files.sort(key=lambda x: x[1])
        
        self.logger.info(f"Found {len(tar_files)} .tar files")
        total_size = sum(size for _, size in tar_files)
        self.logger.info(f"Total size: {total_size / (1024**3):.2f} GB")
        
        return tar_files
    
    def upload_file_with_retry(self, file_path: Path) -> bool:
        """Upload a single file with retry logic"""
        filename = file_path.name
        file_size = file_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        for attempt in range(1, self.max_retries + 1):
            if self.interrupted:
                return False
                
            self.logger.info(f"Uploading {filename} ({file_size_mb:.1f}MB) - Attempt {attempt}/{self.max_retries}")
            
            try:
                # Create a progress bar for large files
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    def update_progress(chunk_size):
                        pbar.update(chunk_size)
                    
                    # Upload with timeout handling
                    start_time = time.time()
                    result = self.api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=f"{self.subfolder}/{filename}",
                        repo_id=self.repo_id,
                        repo_type="dataset"
                    )
                    
                    upload_time = time.time() - start_time
                    speed_mbps = (file_size_mb / upload_time) if upload_time > 0 else 0
                    
                    self.logger.info(f"✅ Successfully uploaded {filename} in {upload_time:.1f}s ({speed_mbps:.1f} MB/s)")
                    self.logger.info(f"URL: {result}")
                    return True
                    
            except KeyboardInterrupt:
                self.interrupted = True
                return False
            except Exception as e:
                self.logger.warning(f"❌ Upload attempt {attempt} failed for {filename}: {str(e)}")
                
                if attempt < self.max_retries:
                    self.logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"❌ Failed to upload {filename} after {self.max_retries} attempts")
        
        return False
    
    def _print_summary(self):
        """Print upload summary"""
        total_files = len(self.uploaded_files) + len(self.failed_files)
        self.logger.info("\n" + "="*50)
        self.logger.info("UPLOAD SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total files processed: {total_files}")
        self.logger.info(f"Successfully uploaded: {len(self.uploaded_files)}")
        self.logger.info(f"Failed uploads: {len(self.failed_files)}")
        
        if self.uploaded_files:
            self.logger.info("\n✅ Successfully uploaded:")
            for filename in self.uploaded_files:
                self.logger.info(f"  - {filename}")
        
        if self.failed_files:
            self.logger.info("\n❌ Failed uploads:")
            for filename in self.failed_files:
                self.logger.info(f"  - {filename}")
    
    def upload_all(self) -> bool:
        """Upload all .tar files from source directory"""
        self.logger.info(f"Starting upload process")
        self.logger.info(f"Source: {self.source_dir}")
        self.logger.info(f"Destination: https://huggingface.co/datasets/{self.repo_id}/tree/main/{self.subfolder}")
        
        # Check authentication
        if not self.check_authentication():
            return False
        
        # Get files to upload
        try:
            tar_files = self.get_tar_files()
        except FileNotFoundError as e:
            self.logger.error(str(e))
            return False
        
        if not tar_files:
            self.logger.warning("No .tar files found to upload")
            return True
        
        # Upload each file
        for i, (file_path, file_size) in enumerate(tar_files, 1):
            if self.interrupted:
                break
                
            self.logger.info(f"\n📁 Processing file {i}/{len(tar_files)}: {file_path.name}")
            
            if self.upload_file_with_retry(file_path):
                self.uploaded_files.append(file_path.name)
            else:
                self.failed_files.append(file_path.name)
                if self.interrupted:
                    break
        
        # Print summary
        self._print_summary()
        
        return len(self.failed_files) == 0

def main():
    parser = argparse.ArgumentParser(description="Upload .tar files to Hugging Face dataset repository")
    parser.add_argument("--source-dir", 
                       default="/netdisk/zhuoran/code/OmniReward-Factory/data/train_data",
                       help="Source directory containing .tar files")
    parser.add_argument("--repo-id", 
                       default="jinzhuoran/OmniRewardData",
                       help="Hugging Face repository ID")
    parser.add_argument("--subfolder", 
                       default="train_data",
                       help="Subfolder in the repository")
    parser.add_argument("--max-retries", 
                       type=int, default=5,
                       help="Maximum number of retry attempts")
    parser.add_argument("--retry-delay", 
                       type=int, default=10,
                       help="Delay between retries in seconds")
    parser.add_argument("--timeout", 
                       type=int, default=3600,
                       help="Timeout for each upload in seconds")
    
    args = parser.parse_args()
    
    uploader = HuggingFaceUploader(
        source_dir=args.source_dir,
        repo_id=args.repo_id,
        subfolder=args.subfolder,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        timeout=args.timeout
    )
    
    success = uploader.upload_all()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
