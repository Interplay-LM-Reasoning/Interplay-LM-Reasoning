# Installation Guide

This guide provides detailed instructions for setting up GSM-Infinite on your system.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## Installation Methods

### Method 1: Install from Source

```bash
# Clone the repository
git clone https://github.com/Infini-AI-Lab/gsm_infinite.git
cd gsm_infinite

# Install in development mode
pip install -e .
```

### Method 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/Infini-AI-Lab/gsm_infinite.git
cd gsm_infinite

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

The following packages will be installed automatically:

### Core Dependencies
- `openai>=1.0.0` - OpenAI API client
- `numpy>=1.24.0` - Numerical computing
- `datasets>=2.14.0` - Hugging Face datasets
- `tqdm>=4.65.0` - Progress bars
- `pyyaml>=6.0.1` - YAML configuration files

### API Clients
- `anthropic` - Anthropic API client
- `google-generativeai` - Google Gemini API client
- `tiktoken` - OpenAI tokenizer

### Text Processing
- `nltk>=3.8.1` - Natural language processing
- `html2text` - HTML to text conversion
- `beautifulsoup4` - HTML parsing
- `spacy` - Advanced NLP

### Utilities
- `tenacity>=8.2.3` - Retry mechanisms
- `wonderwords>=2.2.0` - Word generation
- `termcolor` - Colored terminal output

### Mathematical Computing
- `sympy` - Symbolic mathematics
- `networkx` - Graph algorithms
- `matplotlib` - Plotting
- `pydot` - Graph visualization


## Optional: Model Serving Setup

If you plan to use local models, you may want to install additional serving frameworks:

### vLLM (Recommended for high throughput)
```bash
pip install vllm
```

### SGLang (Alternative serving framework)
```bash
pip install sglang
```

### Transformers (For direct model loading)
```bash
pip install transformers torch
```

## API Keys Setup

GSM-Infinite supports multiple API providers. Set up your API keys as environment variables:

### OpenAI-Compatible APIs
```bash
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # or your custom endpoint
```

### Google Gemini
```bash
export GEMINI_API_KEY="your_gemini_api_key"
```

### Anthropic Claude
```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

## Configuration

After installation, you'll need to configure GSM-Infinite for your specific setup:

1. Navigate to the GSM-Infinite directory
2. Copy and edit the configuration file:
   ```bash
   cd gsm-infinite
   cp config.sh my_config.sh
   # Edit my_config.sh with your settings
   ```

3. See the [Usage Guide](USAGE.md) for detailed configuration instructions.

## Troubleshooting

### Common Issues

**ImportError: No module named 'gsm_infinite'**
- Make sure you've installed the package correctly
- If using source installation, ensure you're in the correct directory

**API Key Errors**
- Verify your API keys are set correctly
- Check that your API endpoints are accessible
- Ensure you have sufficient API credits/quota

**Permission Errors**
- Use `pip install --user` if you don't have system-wide permissions
- Consider using a virtual environment

**Dependency Conflicts**
- Create a fresh virtual environment:
  ```bash
  python -m venv gsm_infinite_env
  source gsm_infinite_env/bin/activate  # On Windows: gsm_infinite_env\Scripts\activate
  pip install gsm-infinite
  ```

### Getting Help

If you encounter issues not covered here:

1. Search existing [GitHub Issues](https://github.com/Infini-AI-Lab/gsm_infinite/issues)
2. Create a new issue with detailed error information

## Next Steps

After successful installation:

1. Read the [Usage Guide](USAGE.md) to learn how to use GSM-Infinite
<!-- 2. Explore the [Evaluation Guide](EVALUATION.md) to evaluate your models -->

