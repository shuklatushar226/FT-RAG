#!/bin/bash

# Fine-tuning script using Axolotl
# Make sure you have axolotl installed: pip install axolotl

echo "🚀 Starting fine-tuning with Axolotl..."

# Move to ft directory
cd "$(dirname "$0")"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: No virtual environment detected. Activating..."
    source ../venv/bin/activate
fi

# Check if axolotl is installed
if ! python -c "import axolotl" &> /dev/null; then
    echo "🔧 Axolotl not found. Installing..."
    pip install axolotl[flash-attn]
fi

# Check if required dependencies are installed
echo "📦 Checking dependencies..."
pip install accelerate transformers torch datasets

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 CUDA detected. Running with GPU acceleration."
    export ACCELERATE_USE_CUDA=1
else
    echo "💻 No CUDA detected. Running on CPU (this will be slow)."
    export ACCELERATE_USE_CUDA=0
fi

# Generate training data from ingested repositories
echo "📊 Generating training data from repositories..."
python generate_training_data.py

# Verify training data exists
if [ ! -f "sft.jsonl" ]; then
    echo "❌ Training data not found. Running basic data generation..."
    python build_sft_data.py
fi

# Initialize accelerate if not already done
echo "⚙️  Initializing accelerate configuration..."
accelerate config default

# Run the training
echo "🏋️  Starting training..."
accelerate launch -m axolotl.cli.train axolotl.yaml

echo "✅ Training completed!"
echo "📁 Model saved to ./out/"

# Optional: Convert to HuggingFace format
echo "🔄 Converting to HuggingFace format..."
python -m axolotl.cli.merge_lora axolotl.yaml --lora_model_dir="./out"

echo "🎉 Fine-tuning process complete!"
echo "📍 Fine-tuned model available at: ./out/"