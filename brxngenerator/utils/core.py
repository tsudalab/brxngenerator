# core.py - Consolidated core utilities: config and device management

import torch
import os

# === DEVICE UTILITIES ===

# Cache the device to avoid repeated detection
_cached_device = None

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

# === CONFIGURATION ===

# --- General Settings ---
RANDOM_SEED = 42

def get_device():
    """Get the best available device: MPS > CUDA > CPU"""
    return get_compatible_device()

DEVICE = get_device()

# --- Directory and Path Settings ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

DATA_FILENAME = os.path.join(DATA_DIR, "data.txt")
LOGP_VALUES_PATH = os.path.join(DATA_DIR, "logP_values.txt")
SA_SCORES_PATH = os.path.join(DATA_DIR, "SA_scores.txt")
CYCLE_SCORES_PATH = os.path.join(DATA_DIR, "cycle_scores.txt")

# --- B-VAE Model Architecture ---
HIDDEN_SIZE = 300
LATENT_SIZE = 100
DEPTH = 2
WEIGHTS_SAVE_PATH = os.path.join(WEIGHTS_DIR, "hidden_size_300_latent_size_100_depth_2_beta_1.0_lr_0.001/bvae_best_model_with.pt")

# --- Factorization Machine Surrogate Model ---
FACTOR_NUM = 8      # 'k' for Factorization Machine
PARAM_INIT = 0.03
PROP = "QED"        # Property being optimized, used for model naming

# --- Training & Optimization Settings ---
METRIC = "qed"          # "qed" or "logp"
MAX_EPOCH = 10000
BATCH_SIZE = 3000
LR = 0.001
DECAY_WEIGHT = 0.01
PATIENCE = 300
OPTIMIZE_NUM = 100 # Number of optimization iterations

# --- Solver Settings ---
CLIENT = "gurobi"

# 如果项目根目录存在 gurobi.lic，则默认使用它
LICENSE_FILE = os.path.join(BASE_DIR, "gurobi.lic")
if os.path.exists(LICENSE_FILE):
    os.environ.setdefault("GRB_LICENSE_FILE", LICENSE_FILE)

# 通过 gurobi.lic/GRB_LICENSE_FILE 管理授权，这里不再硬编码云端凭据
GUROBI_OPTIONS = {}