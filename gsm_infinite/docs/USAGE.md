# Usage Guide

This guide provides comprehensive instructions for using GSM-Infinite to evaluate language models and generate datasets.

## Quick Start

### 1. Basic Configuration

Navigate to the GSM-Infinite directory and configure your setup:

```bash
cd gsm-infinite
```

Edit the `config.sh` file with your settings:

```bash
#!/bin/bash

# API Configuration
backend_type='openai'  # Options: 'openai', 'gemini', 'anthropic'
SAMPLER_OPENAI_BASE_URL='http://127.0.0.1:30000/v1'  # Your API endpoint
SAMPLER_OPENAI_API_KEY='your_api_key'

# For evaluation (symbolic datasets only)
EVAL_OPENAI_BASE_URL='http://127.0.0.1:30000/v1'
EVAL_OPENAI_API_KEY='your_eval_api_key'

# Model Configuration
model_name='your_model_name'  # e.g., 'Qwen/Qwen2.5-7B-Instruct'
save_name='your_save_name'    # e.g., 'qwen-2.5-7b-instruct'

# Control what to run
run_sampling=true
run_evaluation=true
```

### 2. Run Evaluation

Execute the main script:

```bash
bash run.sh
```

This will:
- Sample predictions from your model
- Evaluate the results
- Save dataset and model output to the `datasets/` directory and final results to `results/` directory

### 3. View Results

Launch the interactive dashboard:

```bash
streamlit run app.py
```

## Configuration Options

### API Backends

GSM-Infinite supports three API backends:

#### OpenAI-Compatible APIs
```bash
backend_type='openai'
SAMPLER_OPENAI_BASE_URL='https://api.openai.com/v1'
SAMPLER_OPENAI_API_KEY='your_openai_key'
```

#### Google Gemini
```bash
backend_type='gemini'
GEMINI_API_KEY='your_gemini_key'
```

#### Anthropic Claude
```bash
backend_type='anthropic'
ANTHROPIC_API_KEY='your_anthropic_key'
```

### Dataset Configuration

Configure which datasets and context lengths to evaluate:

```bash
# Context lengths to test
lengths=( 
    "0"     # Zero noise (no additional context)
    "8k"    # 8K tokens context
    "16k"   # 16K tokens context
    "32k"   # 32K tokens context
)

# Dataset types
dataset_suffixes=( 
    "symbolic"  # Abstract mathematical operations
    "medium"    # Realistic word problems
    "hard"      # Complex multi-step problems
)
```

### Operation Range Configuration

Control the complexity range for each dataset:

```bash
declare -A ops_config

# Symbolic dataset configurations
ops_config["0_symbolic"]='{"start": 1, "end": 50, "stride": 1}'
ops_config["8k_symbolic"]='{"start": 1, "end": 30, "stride": 1}'

# Medium dataset configurations  
ops_config["0_medium"]='{"start": 2, "end": 30, "stride": 1}'
ops_config["8k_medium"]='{"start": 2, "end": 30, "stride": 1}'

# Hard dataset configurations
ops_config["0_hard"]='{"start": 2, "end": 30, "stride": 1}'
ops_config["8k_hard"]='{"start": 2, "end": 30, "stride": 1}'
```

### Sampling Parameters

Fine-tune the sampling behavior:

```bash
# Number of samples per problem
num_samples=1

# Temperature settings (different for symbolic vs realistic)
temperature_symbolic=1.0    # Higher temperature for symbolic
temperature_realistic=0.0   # Lower temperature for realistic

# Token limits
max_tokens=4096

# Batch processing
batch_size=200
limit_symbolic=100   # Examples per operation for symbolic
limit_realistic=200  # Examples per operation for realistic
```

## Advanced Usage

### Running Specific Evaluations

You can control which parts of the pipeline to run:

```bash
# Only run sampling (no evaluation)
run_sampling=true
run_evaluation=false

# Only run evaluation (skip sampling)
run_sampling=false
run_evaluation=true

# Run only symbolic evaluation
run_symbolic_evaluation=true
run_realistic_evaluation=false

# Run only realistic evaluation
run_symbolic_evaluation=false
run_realistic_evaluation=true
```

### Custom Dataset Filtering

For realistic datasets (medium/hard), you can configure the problem templates:

```bash
filter_config='[
    {"percentage": 0.4, "template": "crazy_zootopia", "mode": "normalforward"},
    {"percentage": 0.05, "template": "movie_festival_awards", "mode": "normalforward"},
    {"percentage": 0.05, "template": "teachers_in_school", "mode": "normalforward"},
    {"percentage": 0.4, "template": "crazy_zootopia", "mode": "forwardreverse"},
    {"percentage": 0.05, "template": "movie_festival_awards", "mode": "forwardreverse"},
    {"percentage": 0.05, "template": "teachers_in_school", "mode": "forwardreverse"}
]'
```

<!-- ### Direct Python Usage

You can also use GSM-Infinite programmatically:

```python
# Example: Generate symbolic dataset
from gsm_infinite.data.symbolic.generate_symbolic import generate_symbolic_dataset

dataset = generate_symbolic_dataset(
    num_ops=10,
    context_length=1000,
    num_examples=100
)

# Example: Evaluate predictions
from gsm_infinite.pred.eval_symbolic import evaluate_symbolic

results = evaluate_symbolic(
    predictions_file="path/to/predictions.json",
    model_name="eval_model"
)
``` -->

## Output Structure

After running evaluations, you'll find outputs in these directories:

```
gsm-infinite/
├── datasets/           # Generated predictions
├── results/           # Evaluation results
└── processed_results.csv  # Aggregated results for dashboard
```

### Prediction Files

Prediction files are saved as JSON with this structure:

```json
[
    {
        "solution": "The solution of this problem",
        "op": "number of operations in the solution of this problem",
        "n": "number of variables in this problem",
        "length": "context length of filler text",
        "id": "unique id of this problem",
        "d": "subset id",
        "replies": [
            "model's 1st reply",
            "model's 2nd reply",
            ...
        ]
        .....
    },
]
```

### Results Files

Evaluation results include accuracy metrics 

```txt
length: 0, op: 1, acc: 0.99
length: 0, op: 3, acc: 0.66
...
```

## Interactive Dashboard

The Streamlit dashboard provides:

- **Model Comparison**: Compare multiple models side-by-side
- **Performance Visualization**: Interactive plots of accuracy vs. complexity
- **Export Options**: Download results in various formats

### Dashboard Features

1. **Series Selection**: Add multiple model/dataset combinations
2. **Filtering**: Filter by operation count, context length, etc.
3. **Visualization**: Line plots
4. **Export**: CSV, JSON, and image exports

## Best Practices

### For Accurate Evaluation

1. **Sufficient samples**: Use multiple samples for robust statistics

2. **Proper evaluation model**: Use a capable model for answer extraction

### For Efficient Processing

1. **Batch processing**: Use appropriate batch sizes for your API limits
2. **Incremental evaluation**: Run subsets first to test configuration
3. **Resource monitoring**: Monitor API usage and costs


<!-- ## Next Steps -->

<!-- - Learn about [Data Generation](DATA_GENERATION.md) to create custom datasets -->
<!-- - Explore [Evaluation Details](EVALUATION.md) for advanced evaluation techniques
- Check the [API Reference](API_REFERENCE.md) for complete configuration options -->

