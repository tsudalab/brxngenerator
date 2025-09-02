#!/usr/bin/env python3
"""
MPS compatibility fixes for the molecular VAE project.

This module provides patches and workarounds for known MPS issues.
"""

import torch
import functools

def mps_safe_embedding(original_embedding_fn):
    """
    Decorator to make embedding operations MPS-safe.
    
    The issue: "Placeholder storage has not been allocated on MPS device!"
    The fix: Ensure all tensors are properly allocated on the same device.
    """
    @functools.wraps(original_embedding_fn)
    def wrapper(input, weight, padding_idx=None, max_norm=None, 
                norm_type=2.0, scale_grad_by_freq=False, sparse=False):
        
        # If using MPS, ensure input is contiguous and properly allocated
        if hasattr(input, 'device') and input.device.type == 'mps':
            if not input.is_contiguous():
                input = input.contiguous()
            # Force synchronization to ensure proper allocation
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
        
        return original_embedding_fn(
            input, weight, padding_idx, max_norm, 
            norm_type, scale_grad_by_freq, sparse
        )
    return wrapper

def apply_mps_fixes():
    """Apply MPS compatibility fixes."""
    if torch.backends.mps.is_available():
        # Patch the embedding function
        torch.nn.functional.embedding = mps_safe_embedding(torch.nn.functional.embedding)
        print("✅ Applied MPS compatibility fixes")
        return True
    return False

def test_mps_embedding():
    """Test if MPS embedding works with our fixes."""
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return False
        
    try:
        device = torch.device("mps")
        
        # Test embedding layer (the main source of issues)
        embedding = torch.nn.Embedding(100, 32).to(device)
        
        # Create input tensor on MPS
        input_tensor = torch.randint(0, 100, (5, 10)).to(device)
        
        # Force memory allocation and synchronization
        torch.mps.empty_cache()
        torch.mps.synchronize()
        
        # Try the embedding operation
        output = embedding(input_tensor)
        
        # Force another sync to ensure completion
        torch.mps.synchronize()
        
        print(f"✅ MPS embedding test passed! Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ MPS embedding test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing MPS compatibility...")
    
    # Apply fixes
    apply_mps_fixes()
    
    # Test embedding
    if test_mps_embedding():
        print("\n🎉 MPS should work with this model!")
    else:
        print("\n💔 MPS still has issues. Use DISABLE_MPS=1 for CPU fallback.")