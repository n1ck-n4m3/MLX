# MLX Models Collection

A comprehensive collection of Large Language Models optimized for Apple Silicon using MLX framework.

## 🚀 Overview

This repository contains configuration files and documentation for running various state-of-the-art language models on Apple Silicon devices using the MLX framework. The models are optimized for efficient inference on M1/M2/M3 chips.

## 📋 Available Models

### Llama Models
- **llama-3-8b-instruct** - Meta's Llama 3 8B Instruct model
- **llama-3.1-8b** - Meta's Llama 3.1 8B base model  
- **llama-3.1-8b-instruct** - Meta's Llama 3.1 8B Instruct model

### Mistral Models
- **mistral7b-v0_1** - Mistral 7B base model v0.1
- **mistral7b-instruct-v0_3** - Mistral 7B Instruct v0.3
- **ministral8b-instruct-2410** - Ministral 8B Instruct (October 2024)

### Qwen Models
- **qwen2-7b-instruct** - Alibaba's Qwen2 7B Instruct model
- **qwen2.5-7b** - Alibaba's Qwen2.5 7B base model
- **qwen2.5-7b-instruct** - Alibaba's Qwen2.5 7B Instruct model

## 🔧 Model Configurations

Each model directory contains:

### Core Configuration Files
- `config.json` - Model architecture and parameters
- `generation_config.json` - Text generation settings (temperature, top_p, max_length)
- `tokenizer_config.json` - Tokenizer configuration
- `tokenizer.json` - Main tokenizer vocabulary and rules

### Special Files
- `chat_template.jinja` - Chat formatting template (Instruct models only)
- `special_tokens_map.json` - Special token definitions
- `added_tokens.json` - Additional tokens (Qwen models)
- `vocab.json` & `merges.txt` - BPE vocabulary (Qwen models)
- `tokenizer.model` - SentencePiece model (Mistral models)

## 📊 Model Specifications

| Model | Parameters | Context Length | Temperature | Top-P |
|-------|------------|----------------|-------------|-------|
| Llama 3/3.1 8B | 8B | 8,192 tokens | 0.6 | 0.9 |
| Mistral 7B | 7B | Variable | Default | Default |
| Qwen2/2.5 7B | 7B | Variable | 0.7 | 0.8 |
| Ministral 8B | 8B | Variable | Default | Default |

## 🛠️ Usage

### Prerequisites
```bash
# Install MLX
pip install mlx-lm

# Or install from source
git clone https://github.com/ml-explore/mlx-examples.git
cd mlx-examples/llms
pip install -e .
```

### Basic Usage
```python
from mlx_lm import load, generate

# Load a model
model, tokenizer = load("path/to/model")

# Generate text
response = generate(
    model, 
    tokenizer, 
    prompt="Hello, how are you?",
    temperature=0.7,
    max_tokens=100
)
```

### Custom Parameters
```python
# Adjust generation parameters
response = generate(
    model,
    tokenizer,
    prompt="Explain quantum computing",
    temperature=0.3,    # Lower for more focused responses
    top_p=0.9,         # Nucleus sampling
    max_tokens=500,    # Response length
    repetition_penalty=1.1
)
```

## 🎯 Model Selection Guide

### For Code Generation
- **Recommended**: Llama models with `temperature=0.1-0.3`
- **Alternative**: Qwen models for multilingual code

### For Creative Writing
- **Recommended**: Any Instruct model with `temperature=0.8-1.2`
- **Best**: Llama 3.1 or Qwen2.5 for coherence

### For Factual Q&A
- **Recommended**: Instruct models with `temperature=0.2-0.5`
- **Best**: Llama 3.1-8b-instruct for accuracy

### For Conversations
- **Recommended**: Any Instruct model with `temperature=0.6-0.8`
- **Best**: Models with chat templates

## ⚙️ Configuration Details

### Temperature Settings
- **0.1-0.3**: Conservative, accurate responses
- **0.6-0.8**: Balanced creativity and accuracy  
- **0.9-1.2**: Creative, diverse outputs

### Context Window Management
- **Llama 3/3.1**: 8,192 tokens (6,000-7,000 words)
- **Other models**: 4,096-8,192 tokens
- **Recommendation**: Monitor usage, summarize long conversations

## 📁 Directory Structure
```
MLX/
├── models/
│   ├── llama-3-8b-instruct/
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── chat_template.jinja
│   │   └── tokenizer files...
│   ├── qwen2-7b-instruct/
│   │   ├── config.json
│   │   ├── added_tokens.json
│   │   ├── vocab.json
│   │   └── tokenizer files...
│   └── ...
├── README.md
└── .gitignore
```

## 🚨 Important Notes

### Model Files Not Included
This repository contains **configuration files only**. The actual model weights (.safetensors files) are not included due to their large size (15-20GB per model).

### Getting Model Weights
To obtain the actual model files:

1. **Hugging Face Hub**:
   ```bash
   # Install huggingface-hub
   pip install huggingface-hub
   
   # Download models
   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./models/llama-3.1-8b-instruct
   ```

2. **MLX Community**:
   ```bash
   # Use MLX-optimized models
   huggingface-cli download mlx-community/Llama-3.1-8B-Instruct-4bit --local-dir ./models/llama-3.1-8b-instruct-4bit
   ```

### Hardware Requirements
- **Apple Silicon**: M1/M2/M3 Mac recommended
- **Memory**: 16GB+ RAM for 7-8B models
- **Storage**: 15-20GB per model

## 🔗 Useful Links

- [MLX Framework](https://github.com/ml-explore/mlx)
- [MLX Examples](https://github.com/ml-explore/mlx-examples)
- [MLX Community Models](https://huggingface.co/mlx-community)
- [Hugging Face Model Hub](https://huggingface.co/models)

## 📄 License

Model configurations follow their respective original licenses:
- Llama models: Custom Meta license
- Mistral models: Apache 2.0
- Qwen models: Tongyi Qianwen license

## 🤝 Contributing

Feel free to:
- Add new model configurations
- Improve documentation
- Share optimization tips
- Report issues

---

**Note**: This repository provides model configurations and documentation. Download actual model weights separately from official sources.
