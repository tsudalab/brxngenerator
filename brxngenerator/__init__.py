"""
brxngenerator - Binary Variational Autoencoder for Molecular Generation

A Python package for generating novel molecules using binary VAE with optional
error-correcting codes for improved quality.
"""

__version__ = "0.1.0"
__author__ = "brxngenerator team"

from .core.vae import bFTRXNVAE
from .core.ecc import RepetitionECC, create_ecc_codec

__all__ = [
    'bFTRXNVAE',
    'RepetitionECC', 
    'create_ecc_codec'
]