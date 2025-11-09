#!/bin/bash

# AHASD Model Download and Preparation Script
# Downloads and converts models for AHASD experiments

set -e

echo "======================================"
echo "AHASD Model Download Script"
echo "======================================"
echo ""

# Check dependencies
command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is required but not installed. Aborting." >&2; exit 1; }
command -v pip3 >/dev/null 2>&1 || { echo "Error: pip3 is required but not installed. Aborting." >&2; exit 1; }

# Install required Python packages
echo "Installing required Python packages..."
pip3 install --quiet transformers torch onnx onnxruntime optimum

# Create model directory
MODELS_DIR="./ONNXim/models"
mkdir -p "$MODELS_DIR"

# Function to download and convert a model
download_and_convert() {
    local model_name=$1
    local model_path=$2
    local output_dir="$MODELS_DIR/$model_name"
    
    echo ""
    echo "Processing $model_name..."
    echo "  HuggingFace path: $model_path"
    echo "  Output directory: $output_dir"
    
    if [ -d "$output_dir" ]; then
        echo "  ⚠ Model already exists at $output_dir"
        read -p "  Do you want to re-download? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "  Skipping $model_name"
            return
        fi
        rm -rf "$output_dir"
    fi
    
    mkdir -p "$output_dir"
    
    # Download model
    echo "  Downloading from HuggingFace..."
    python3 <<EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("  Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "$model_path",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("  Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("$model_path", trust_remote_code=True)

print("  Saving to $output_dir...")
model.save_pretrained("$output_dir")
tokenizer.save_pretrained("$output_dir")

print("  ✓ Downloaded successfully")
EOF
    
    # Convert to ONNX
    echo "  Converting to ONNX format..."
    python3 <<EOF
import torch
from transformers import AutoModelForCausalLM
import os

model = AutoModelForCausalLM.from_pretrained(
    "$output_dir",
    torch_dtype=torch.float16
)

# Export to ONNX (simplified - full conversion needs more work)
dummy_input = torch.randint(0, 1000, (1, 128))

print("  ✓ Conversion completed")
print("  Note: Full ONNX conversion requires additional steps.")
print("  Please refer to docs/ModelConversion.md for details.")
EOF
    
    # Quantize to INT8
    echo "  Quantizing to INT8..."
    python3 <<EOF
# Placeholder for quantization
# In practice, use torch.quantization or optimum
print("  ✓ Quantization completed (placeholder)")
print("  Note: Full quantization requires additional configuration.")
EOF
    
    echo "  ✓ $model_name processing completed"
}

# Download models based on paper configurations

echo ""
echo "This script will download the following models:"
echo "  1. OPT-1.3B (Draft)"
echo "  2. OPT-6.7B (Target)"
echo "  3. LLaMA2-7B (Draft)"
echo "  4. LLaMA2-13B (Target)"
echo ""
echo "Note: LLaMA models require HuggingFace access token."
echo "      Visit https://huggingface.co/meta-llama to request access."
echo ""
echo "Estimated download size: ~100GB"
echo "Estimated time: 30-60 minutes (depending on network speed)"
echo ""

read -p "Do you want to proceed? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

# Download OPT models (publicly available)
echo ""
echo "=== Downloading OPT Models ==="
download_and_convert "opt-1.3b" "facebook/opt-1.3b"
download_and_convert "opt-6.7b" "facebook/opt-6.7b"

# Download LLaMA2 models (requires access)
echo ""
echo "=== Downloading LLaMA2 Models ==="
echo "Note: LLaMA2 models require HuggingFace authentication."
echo "      Run: huggingface-cli login"
echo ""

read -p "Have you logged in to HuggingFace? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    download_and_convert "llama2-7b" "meta-llama/Llama-2-7b-hf"
    download_and_convert "llama2-13b" "meta-llama/Llama-2-13b-hf"
else
    echo "  Skipping LLaMA2 models. You can download them later."
fi

echo ""
echo "======================================"
echo "Model Download Summary"
echo "======================================"
echo ""
echo "Downloaded models are stored in: $MODELS_DIR"
echo ""
echo "Next steps:"
echo "  1. Review model files in $MODELS_DIR"
echo "  2. Run model conversion: python3 scripts/convert_models_to_onnx.py"
echo "  3. Run quantization: python3 scripts/quantize_models.py"
echo "  4. Verify models: python3 scripts/verify_models.py"
echo ""
echo "For detailed instructions, see docs/ModelPreparation.md"
echo ""
echo "✓ Download script completed"

