#!/usr/bin/env python3
"""
Quick test script to verify MPS compatibility and identify potential issues
"""

import torch
import sys
import os
sys.path.append('./rxnft_vae')

from config import get_device

def test_basic_operations():
    """Test basic PyTorch operations with MPS"""
    device = get_device()
    print(f"Testing with device: {device}")
    
    # Test basic tensor operations
    try:
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = x @ y  # Matrix multiplication
        print(f"✅ Basic tensor operations work: {z.shape}")
    except Exception as e:
        print(f"❌ Basic tensor operations failed: {e}")
        return False
        
    # Test embedding layer (common source of MPS issues)
    try:
        embedding = torch.nn.Embedding(100, 32).to(device)
        indices = torch.randint(0, 100, (5, 10)).to(device)
        output = embedding(indices)
        print(f"✅ Embedding layer works: {output.shape}")
    except Exception as e:
        print(f"❌ Embedding layer failed: {e}")
        return False
    
    # Test linear layers
    try:
        linear = torch.nn.Linear(32, 64).to(device)
        input_tensor = torch.randn(5, 32).to(device) 
        output = linear(input_tensor)
        print(f"✅ Linear layer works: {output.shape}")
    except Exception as e:
        print(f"❌ Linear layer failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = test_basic_operations()
    if success:
        print("\n🎉 MPS compatibility test passed!")
    else:
        print("\n💥 MPS compatibility test failed!")
        print("Consider using CPU fallback or CUDA if available")