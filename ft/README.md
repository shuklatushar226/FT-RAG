# Fine-tuning Pipeline

This directory contains the complete fine-tuning pipeline for customizing language models based on your ingested repositories.

## Overview

The fine-tuning pipeline uses **QLoRA (Quantized Low-Rank Adaptation)** with **Axolotl** to efficiently fine-tune language models while maintaining low memory usage. This approach allows you to create a specialized model that understands your codebase better than general-purpose models.

## Files

- **`axolotl.yaml`** - Main configuration for fine-tuning with Axolotl
- **`generate_training_data.py`** - Generates training data from ingested repositories
- **`build_sft_data.py`** - Helper class for building supervised fine-tuning datasets
- **`train_adapter.sh`** - Complete training script with dependency management
- **`sft.jsonl`** - Generated training data (JSONL format)
- **`sft_readable.json`** - Human-readable version of training data

## Quick Start

### 1. Generate Training Data

Extract training examples from your ingested repositories:

```bash
# Via API
curl -X POST http://localhost:8000/fine-tune/generate-data

# Or directly
cd ft
python generate_training_data.py
```

This will generate approximately 2600+ training examples based on the Hyperswitch repository.

### 2. Start Fine-tuning

```bash
# Via API (recommended for background processing)
curl -X POST http://localhost:8000/fine-tune/start

# Or directly (will run in foreground)
cd ft
./train_adapter.sh
```

### 3. Check Progress

```bash
# Get job status
curl http://localhost:8000/ingest/status/{job_id}

# Or check the training logs directly
tail -f ft/out/training.log
```

## Model Configuration

The default configuration uses **Qwen2-7B-Instruct** as the base model with the following optimizations:

- **4-bit quantization** for memory efficiency
- **LoRA** with rank 16 for parameter-efficient fine-tuning
- **2 epochs** with gradient accumulation
- **Cosine learning rate schedule**

### Key Settings (axolotl.yaml)

```yaml
base_model: Qwen/Qwen2-7B-Instruct
load_in_4bit: true
num_epochs: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
lora:
  r: 16
  alpha: 32
  dropout: 0.05
```

## Training Data Format

The training data includes several types of examples:

1. **Code Explanation** - Explain code functionality
2. **Code Completion** - Complete partial code snippets
3. **Documentation** - Generate documentation for code
4. **Hyperswitch-specific** - Payment processing and integration examples

Example training format:
```json
{
  "instruction": "Explain the following code from hyperswitch-client-core:",
  "input": "function createPayment(amount, currency) { ... }",
  "output": "This function creates a new payment request..."
}
```

## Hardware Requirements

### Minimum (CPU-only)
- 16GB RAM
- Training time: 8-12 hours

### Recommended (GPU)
- NVIDIA GPU with 8GB+ VRAM
- 16GB+ system RAM
- Training time: 2-4 hours

### Optimal (High-end GPU)
- NVIDIA GPU with 24GB+ VRAM (RTX 4090, A100)
- 32GB+ system RAM
- Training time: 30-60 minutes

## Using the Fine-tuned Model

Once training is complete, you can use the fine-tuned model via the API:

```bash
# Query with fine-tuned model
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"q": "How do I integrate Hyperswitch payments?", "model": "fine-tuned"}'

# Query with Gemini (default)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"q": "How do I integrate Hyperswitch payments?", "model": "gemini"}'
```

## Model Comparison

| Model | Pros | Cons |
|-------|------|------|
| **Gemini 2.0 Flash** | Fast, no local resources, large context | API costs, internet required, generic responses |
| **Fine-tuned Local** | Specialized knowledge, private, no API costs | Setup time, hardware requirements, smaller context |

## Advanced Configuration

### Customizing Training Data

Edit `generate_training_data.py` to:
- Add more training examples
- Change instruction formats
- Include domain-specific examples
- Filter by file types or repositories

### Adjusting Model Settings

Edit `axolotl.yaml` to:
- Change base model (`base_model`)
- Adjust training epochs (`num_epochs`)
- Modify LoRA rank (`lora.r`)
- Change batch size (`per_device_train_batch_size`)

### Different Base Models

Supported base models:
- `Qwen/Qwen2-7B-Instruct` (default)
- `microsoft/CodeGPT-small-py`
- `codellama/CodeLlama-7b-Instruct-hf`
- `WizardLM/WizardCoder-15B-V1.0`

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable `load_in_8bit` instead of 4-bit

### Slow Training
- Enable CUDA if available
- Increase batch size if you have more memory
- Use a smaller base model

### Poor Results
- Increase `num_epochs` (but watch for overfitting)
- Add more diverse training examples
- Adjust learning rate (`learning_rate`)

## API Integration

The fine-tuning pipeline integrates seamlessly with the main RAG system:

- **`/models/info`** - Check available models
- **`/fine-tune/generate-data`** - Generate training data
- **`/fine-tune/start`** - Start training process
- **`/query?model=fine-tuned`** - Use fine-tuned model

## Next Steps

1. **Experiment with prompts** - Test different question formats
2. **Add more repositories** - Ingest additional codebases for training
3. **Create specialized versions** - Fine-tune for specific programming languages
4. **Evaluate performance** - Compare responses between models
5. **Deploy in production** - Set up model serving infrastructure

## Resources

- [Axolotl Documentation](https://github.com/OpenAccess-AI-Collective/axolotl)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://github.com/huggingface/peft)