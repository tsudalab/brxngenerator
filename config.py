# config.py

import torch
import os

# --- General Settings ---
RANDOM_SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Directory and Path Settings ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
