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
WEIGHTS_SAVE_PATH = os.path.join(WEIGHTS_DIR, "hidden_size_300_latent_size_100_depth_2_beta_1.0_lr_0.001/bvae_iter-30-with.npy")

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
CLIENT = "Gurobi"
# Gurobi许可证信息，从代码中移到配置中
GUROBI_OPTIONS = {
    "LICENSEID": 2687913,
    "WLSACCESSID": "5cbfb8e1-0066-4b7f-ab40-579464946573",
    "WLSSECRET": "a5c475ea-ec91-4cd6-94e9-b73395e273d6",
    # 可以添加其他Gurobi参数，例如 'TimeLimit': 3600
}