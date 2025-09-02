# [ECC] Minimal repetition code for binary latent space error correction
"""
Error-Correcting Code utilities for improving binary VAE latent representations.
Implements repetition codes with majority-vote decoding.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


class RepetitionECC:
    """
    Simple repetition code that repeats each information bit R times.
    Decoding uses majority vote to recover from up to ⌊(R-1)/2⌋ errors per group.
    
    Args:
        R (int): Repetition factor, must be >= 2
    """
    
    def __init__(self, R: int = 3):
        assert R >= 2, f"Repetition factor R must be >= 2, got {R}"
        self.R = R
    
    def encode(self, info_bits: torch.Tensor) -> torch.Tensor:
        """
        Encode information bits using repetition code.
        
        Args:
            info_bits: Binary tensor of shape (B, K) with values in {0, 1}
            
        Returns:
            code_bits: Binary tensor of shape (B, K*R) where each bit is repeated R times
        """
        # Repeat each bit R times along the last dimension
        return info_bits.repeat_interleave(self.R, dim=-1)
    
    def decode(self, code_bits: torch.Tensor) -> torch.Tensor:
        """
        Decode codeword using majority vote.
        
        Args:
            code_bits: Binary tensor of shape (B, N) where N = K*R
            
        Returns:
            info_bits: Binary tensor of shape (B, K) recovered via majority vote
        """
        batch_size = code_bits.size(0)
        N = code_bits.size(-1)
        
        assert N % self.R == 0, f"Code length {N} must be divisible by R={self.R}"
        K = N // self.R
        
        # Reshape into groups of R bits
        groups = code_bits.view(batch_size, K, self.R)
        
        # Majority vote: sum each group and threshold at ceil(R/2)
        votes = groups.sum(dim=-1)  # (B, K)
        threshold = (self.R + 1) // 2  # Ceiling division for tie-breaking to 1
        
        return (votes >= threshold).to(code_bits.dtype)
    
    def group_shape_ok(self, latent_size: int) -> bool:
        """Check if latent_size is compatible with this repetition code."""
        return latent_size % self.R == 0
    
    def get_info_size(self, code_size: int) -> int:
        """Get information size K from code size N."""
        assert code_size % self.R == 0, f"Code size {code_size} must be divisible by R={self.R}"
        return code_size // self.R
    
    def get_code_size(self, info_size: int) -> int:
        """Get code size N from information size K."""
        return info_size * self.R
    
    def to_groups(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor from (B, N) to (B, K, R) for group-wise operations."""
        batch_size = x.size(0)
        N = x.size(-1)
        K = self.get_info_size(N)
        return x.view(batch_size, K, self.R)
    
    def from_groups(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor from (B, K, R) to (B, N) format."""
        batch_size, K, R = x.shape
        assert R == self.R, f"Expected R={self.R}, got {R}"
        return x.view(batch_size, K * R)


def create_ecc_codec(ecc_type: str, **kwargs):
    """
    Factory function to create ECC codec.
    
    Args:
        ecc_type: Type of ECC ('none' or 'repetition')
        **kwargs: Additional arguments for the codec
        
    Returns:
        ECC codec instance or None if ecc_type is 'none'
    """
    if ecc_type == 'none':
        return None
    elif ecc_type == 'repetition':
        R = kwargs.get('R', 3)
        return RepetitionECC(R=R)
    else:
        raise ValueError(f"Unknown ECC type: {ecc_type}. Supported: 'none', 'repetition'")


# [ECC] Utility functions for ECC-aware latent handling
def sample_ecc_latent(info_size: int, batch_size: int, ecc_codec=None, device='cpu') -> torch.Tensor:
    """
    Sample binary latent vector with optional ECC encoding.
    
    Args:
        info_size: Size of information bits K
        batch_size: Number of samples
        ecc_codec: ECC codec instance (None for no ECC)
        device: Device to create tensor on
        
    Returns:
        Binary latent tensor of appropriate size (K if no ECC, N if ECC)
    """
    # Sample information bits
    info_bits = torch.bernoulli(torch.full((batch_size, info_size), 0.5, device=device))
    
    if ecc_codec is None:
        return info_bits
    else:
        # Encode with ECC
        return ecc_codec.encode(info_bits)


def extract_info_bits(latent: torch.Tensor, ecc_codec=None) -> torch.Tensor:
    """
    Extract information bits from latent vector.
    
    Args:
        latent: Binary latent tensor
        ecc_codec: ECC codec instance (None for no ECC)
        
    Returns:
        Information bits (same as input if no ECC, decoded if ECC)
    """
    if ecc_codec is None:
        return latent
    else:
        return ecc_codec.decode(latent)