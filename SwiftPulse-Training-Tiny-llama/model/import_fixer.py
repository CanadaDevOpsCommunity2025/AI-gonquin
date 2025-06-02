#!/usr/bin/env python3
"""
Fix import issues by patching the metadata system
Run this before your training script
"""

import sys
import os

# Set environment variables
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Patch importlib.metadata before any imports
import importlib.metadata

# Store original function
_original_distribution = importlib.metadata.distribution

def safe_distribution(name):
    """Patched distribution function that handles missing packages"""
    blocked_packages = {
        'auto-gptq', 'auto_gptq', 'optimum', 'bitsandbytes', 
        'peft', 'accelerate', 'deepspeed', 'xformers'
    }
    
    if name.lower() in blocked_packages:
        raise importlib.metadata.PackageNotFoundError(name)
    
    try:
        return _original_distribution(name)
    except importlib.metadata.PackageNotFoundError:
        # Re-raise the original error
        raise

# Apply the patch
importlib.metadata.distribution = safe_distribution

print("Import patches applied successfully!")
print("Environment variables set:")
for key, value in os.environ.items():
    if any(x in key.upper() for x in ['TRANSFORMERS', 'HF_', 'WANDB', 'MLFLOW']):
        print(f"  {key}={value}")

# Test imports
try:
    print("\nTesting imports...")
    from transformers import AutoTokenizer
    print("✓ AutoTokenizer imported successfully")
    
    from transformers import AutoModelForCausalLM  
    print("✓ AutoModelForCausalLM imported successfully")
    
    import torch
    print("✓ PyTorch imported successfully")
    
    print("\nAll imports successful! You can now run your training script.")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    print("\nTry installing missing dependencies:")
    print("pip install torch transformers")
