# Hugging Face Upload Scripts

This directory contains two robust scripts for uploading .tar files to Hugging Face datasets with retry logic for unstable connections.

## Files Created

1. **`upload_to_hf.sh`** - Bash script version
2. **`upload_to_hf.py`** - Python script version (recommended)
3. **`hf_upload_requirements.txt`** - Python dependencies

## Setup

### 1. Install Dependencies

```bash
# For Python script (recommended)
pip install -r hf_upload_requirements.txt

# Alternative: install individually
pip install huggingface_hub tqdm
```

### 2. Authenticate with Hugging Face

```bash
huggingface-cli login
```

Enter your Hugging Face token when prompted.

## Usage

### Python Script (Recommended)

```bash
# Basic usage with default settings
./upload_to_hf.py

# Custom settings
./upload_to_hf.py \
    --source-dir "/your/custom/path" \
    --repo-id "your-username/your-dataset" \
    --subfolder "custom_folder" \
    --max-retries 3 \
    --retry-delay 15
```

### Bash Script

```bash
# Run with default settings
./upload_to_hf.sh
```

## Features

### Robust Error Handling
- **Automatic Retry**: Failed uploads are retried up to 5 times by default
- **Connection Timeout**: Each upload has a timeout to prevent hanging
- **Graceful Interruption**: Can be safely interrupted with Ctrl+C
- **Progress Tracking**: Shows upload progress and status

### Smart Upload Strategy
- **Size-based Ordering**: Uploads smaller files first for quicker feedback
- **Resume Capability**: Keeps track of successful uploads
- **Detailed Logging**: Creates log files with upload status
- **Progress Bars**: Visual progress indication (Python version)

### Configuration Options
- **Source Directory**: Default `/netdisk/zhuoran/code/OmniReward-Factory/data/train_data`
- **Repository**: Default `jinzhuoran/OmniRewardData`
- **Subfolder**: Default `train_data`
- **Max Retries**: Default 5 attempts per file
- **Retry Delay**: Default 10 seconds between retries
- **Timeout**: Default 1 hour per upload

## Default Configuration

The scripts are pre-configured for your use case:
- **Source**: `/netdisk/zhuoran/code/OmniReward-Factory/data/train_data`
- **Destination**: `https://huggingface.co/datasets/jinzhuoran/OmniRewardData/tree/main/train_data`

## Monitoring Progress

### Python Script
- Real-time progress bars for each file
- Detailed logs saved to `hf_upload_<timestamp>.log`
- Console output with timestamps and status

### Bash Script
- Status updates with timestamps
- Upload status saved to `upload_status_<timestamp>.log`
- Color-coded output for easy reading

## Troubleshooting

### Authentication Issues
```bash
# Re-login to Hugging Face
huggingface-cli logout
huggingface-cli login
```

### Permission Issues
```bash
# Make scripts executable
chmod +x upload_to_hf.py
chmod +x upload_to_hf.sh
```

### Network Issues
- The scripts automatically retry failed uploads
- Increase retry delay for very unstable connections:
  ```bash
  ./upload_to_hf.py --retry-delay 30 --max-retries 10
  ```

### Large Files
- The timeout is set to 1 hour by default
- For very large files, increase timeout:
  ```bash
  ./upload_to_hf.py --timeout 7200  # 2 hours
  ```

## Example Output

```
2024-07-25 10:30:15 - INFO - Starting upload process
2024-07-25 10:30:15 - INFO - Source: /netdisk/zhuoran/code/OmniReward-Factory/data/train_data
2024-07-25 10:30:15 - INFO - Destination: https://huggingface.co/datasets/jinzhuoran/OmniRewardData/tree/main/train_data
2024-07-25 10:30:16 - INFO - Authenticated as: jinzhuoran
2024-07-25 10:30:16 - INFO - Found 5 .tar files
2024-07-25 10:30:16 - INFO - Total size: 2.34 GB

📁 Processing file 1/5: data_part1.tar
data_part1.tar: 100%|████████████| 512MB/512MB [02:15<00:00, 3.78MB/s]
2024-07-25 10:32:31 - INFO - ✅ Successfully uploaded data_part1.tar in 135.2s (3.8 MB/s)
```

## Security Notes

- Never commit your Hugging Face token to version control
- The scripts use the token stored by `huggingface-cli login`
- Log files may contain filenames but no sensitive data


