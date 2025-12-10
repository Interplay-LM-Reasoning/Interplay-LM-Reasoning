#!/bin/bash

# Configuration
SOURCE_DIR="/netdisk/zhuoran/code/OmniReward-Factory/data/train_data"
HF_REPO="jinzhuoran/OmniRewardData"
HF_SUBFOLDER="train_data"
MAX_RETRIES=5
RETRY_DELAY=10  # seconds
TIMEOUT=3600    # 1 hour timeout for each upload

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if huggingface_hub is installed
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v python3 &> /dev/null; then
        error "Python3 is not installed"
        exit 1
    fi
    
    if ! python3 -c "import huggingface_hub" &> /dev/null; then
        warning "huggingface_hub not found. Installing..."
        pip3 install huggingface_hub
        if [ $? -ne 0 ]; then
            error "Failed to install huggingface_hub"
            exit 1
        fi
    fi
    
    success "Dependencies check passed"
}

# Check if user is logged into Hugging Face
check_hf_login() {
    log "Checking Hugging Face authentication..."
    
    python3 -c "
from huggingface_hub import HfApi
try:
    api = HfApi()
    user = api.whoami()
    print(f'Logged in as: {user[\"name\"]}')
except Exception as e:
    print('Not logged in')
    exit(1)
"
    
    if [ $? -ne 0 ]; then
        error "Not logged into Hugging Face. Please run: huggingface-cli login"
        exit 1
    fi
    
    success "Hugging Face authentication verified"
}

# Upload a single file with retry logic
upload_file() {
    local file_path="$1"
    local filename=$(basename "$file_path")
    local attempt=1
    
    log "Starting upload of $filename (attempt $attempt/$MAX_RETRIES)"
    
    while [ $attempt -le $MAX_RETRIES ]; do
        log "Uploading $filename - Attempt $attempt/$MAX_RETRIES"
        
        # Use timeout to prevent hanging uploads
        timeout $TIMEOUT python3 -c "
import sys
from huggingface_hub import HfApi
from pathlib import Path

try:
    api = HfApi()
    result = api.upload_file(
        path_or_fileobj='$file_path',
        path_in_repo='$HF_SUBFOLDER/$filename',
        repo_id='$HF_REPO',
        repo_type='dataset'
    )
    print(f'Successfully uploaded: $filename')
    print(f'URL: {result}')
except Exception as e:
    print(f'Upload failed: {str(e)}')
    sys.exit(1)
"
        
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            success "Successfully uploaded $filename"
            return 0
        elif [ $exit_code -eq 124 ]; then
            warning "Upload of $filename timed out (attempt $attempt/$MAX_RETRIES)"
        else
            warning "Upload of $filename failed (attempt $attempt/$MAX_RETRIES)"
        fi
        
        if [ $attempt -lt $MAX_RETRIES ]; then
            log "Waiting $RETRY_DELAY seconds before retry..."
            sleep $RETRY_DELAY
        fi
        
        ((attempt++))
    done
    
    error "Failed to upload $filename after $MAX_RETRIES attempts"
    return 1
}

# Get list of .tar files and their sizes
get_tar_files() {
    log "Scanning for .tar files in $SOURCE_DIR"
    
    if [ ! -d "$SOURCE_DIR" ]; then
        error "Source directory does not exist: $SOURCE_DIR"
        exit 1
    fi
    
    # Find all .tar files and sort by size (smallest first for faster initial uploads)
    find "$SOURCE_DIR" -name "*.tar" -type f -exec ls -la {} \; | sort -k5 -n | awk '{print $9}'
}

# Create upload status file
create_status_file() {
    local status_file="upload_status_$(date +%Y%m%d_%H%M%S).log"
    echo "$status_file"
}

# Main upload function
main() {
    log "Starting Hugging Face upload process"
    log "Source: $SOURCE_DIR"
    log "Destination: https://huggingface.co/datasets/$HF_REPO/tree/main/$HF_SUBFOLDER"
    
    # Check dependencies and authentication
    check_dependencies
    check_hf_login
    
    # Get list of tar files
    local tar_files
    tar_files=$(get_tar_files)
    
    if [ -z "$tar_files" ]; then
        warning "No .tar files found in $SOURCE_DIR"
        exit 0
    fi
    
    local total_files=$(echo "$tar_files" | wc -l)
    log "Found $total_files .tar files to upload"
    
    # Create status file
    local status_file=$(create_status_file)
    log "Upload status will be logged to: $status_file"
    
    # Upload each file
    local uploaded=0
    local failed=0
    
    echo "$tar_files" | while IFS= read -r file_path; do
        if [ -f "$file_path" ]; then
            local file_size=$(stat -c%s "$file_path" 2>/dev/null || stat -f%z "$file_path" 2>/dev/null)
            local file_size_mb=$((file_size / 1024 / 1024))
            
            log "Processing: $(basename "$file_path") (${file_size_mb}MB)"
            echo "$(date): Starting upload of $(basename "$file_path")" >> "$status_file"
            
            if upload_file "$file_path"; then
                ((uploaded++))
                echo "$(date): SUCCESS - $(basename "$file_path")" >> "$status_file"
                success "Progress: $uploaded/$total_files files uploaded"
            else
                ((failed++))
                echo "$(date): FAILED - $(basename "$file_path")" >> "$status_file"
                error "Progress: $uploaded/$total_files uploaded, $failed failed"
            fi
        else
            warning "File not found: $file_path"
        fi
    done
    
    log "Upload process completed"
    log "Status file: $status_file"
    
    # Final summary
    if [ $failed -eq 0 ]; then
        success "All files uploaded successfully!"
    else
        warning "Upload completed with $failed failures. Check $status_file for details."
    fi
}

# Handle interruption
cleanup() {
    log "Upload interrupted by user"
    exit 130
}

trap cleanup INT TERM

# Run main function
main "$@"
