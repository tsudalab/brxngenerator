#!/usr/bin/env python3
"""
Device utilities for handling MPS-specific issues
"""

import torch
import os

def safe_to_device(tensor_or_model, device):
    """
    Safely move tensor or model to device, handling MPS-specific issues
    """
    if device.type == 'mps':
        # For MPS, ensure proper tensor types
        if hasattr(tensor_or_model, 'to'):
            # For models and tensors
            result = tensor_or_model.to(device)
            # Force synchronization for MPS
            if torch.is_tensor(result):
                torch.mps.synchronize()
            return result
        else:
            return tensor_or_model
    else:
        # For CUDA and CPU, use standard .to()
        if hasattr(tensor_or_model, 'to'):
            return tensor_or_model.to(device)
        else:
            return tensor_or_model

# Cache the device to avoid repeated detection
_cached_device = None

def get_compatible_device():
    """Get device with compatibility checks"""
    global _cached_device
    if _cached_device is not None:
        return _cached_device
        
    if torch.cuda.is_available():
        _cached_device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        print("🚀 MPS detected! Using Apple Silicon acceleration.")
        print("💡 Note: If you encounter issues, set DISABLE_MPS=1 to use CPU fallback")
        if os.environ.get('DISABLE_MPS', '0') == '1':
            print("DISABLE_MPS=1 detected, using CPU fallback...")
            _cached_device = torch.device("cpu")
        else:
            _cached_device = torch.device("mps")
    else:
        _cached_device = torch.device("cpu")
        
    return _cached_device