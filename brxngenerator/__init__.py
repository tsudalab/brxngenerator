"""
brxngenerator - Binary Variational Autoencoder for Molecular Generation

A consolidated Python package for generating novel molecules using binary VAE.
"""

__version__ = "0.1.0"
__author__ = "brxngenerator team"

from .core.vae import bFTRXNVAE
from .utils.core import get_device, safe_to_device
from .chemistry.chemistry_core import Evaluator, calculateScore
from .metrics.metrics import compute_molecular_metrics, LatentMetrics

__all__ = [
    'bFTRXNVAE',
    'get_device',
    'safe_to_device',
    'Evaluator',
    'calculateScore',
    'compute_molecular_metrics',
    'LatentMetrics'
]