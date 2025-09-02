# main.py

import argparse
import time
import os
import torch
import numpy as np
import random
import logging
from tqdm import tqdm
import sys

# 1. 从配置文件和工具文件中导入
import config
from ..core import binary_vae_utils

# 2. 导入所有必要的第三方和自定义库
# 确保 rxnft_vae 模块在 Python 路径中
from ..chemistry.reactions.reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates
from ..chemistry.fragments.fragment import FragmentVocab, FragmentTree
from ..core.vae import bFTRXNVAE
from ..chemistry.reactions.reaction_utils import read_multistep_rxns


def seed_all(seed):
    """Sets the random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Note: MPS doesn't need separate manual_seed - torch.manual_seed handles it

def main(seed_to_run, ecc_type='none', ecc_R=3):
    """Main execution function."""
    # --- 0. Setup ---
    config.RANDOM_SEED = seed_to_run
    seed_all(config.RANDOM_SEED)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- 1. Load Data and Pre-trained B-VAE Model ---
    logging.info("Loading data and pre-trained models...")
    routes, _ = read_multistep_rxns(config.DATA_FILENAME)
    rxn_trees = [ReactionTree(route) for route in routes]
    reactants = extract_starting_reactants(rxn_trees)
    templates, n_reacts = extract_templates(rxn_trees)
    reactantDic = StartingReactants(reactants)
    templateDic = Templates(templates, n_reacts)

    logging.info("Building fragment trees...")
    fgm_trees, valid_rxn_trees = [], []
    for tree in tqdm(rxn_trees, desc="Fragmenting molecules"):
        try:
            fgm_trees.append(FragmentTree(tree.molecule_nodes[0].smiles))
            valid_rxn_trees.append(tree)
        except Exception:
            continue
    
    data_pairs = list(zip(fgm_trees, valid_rxn_trees))
    cset = {node.smiles for fgm_tree in fgm_trees for node in fgm_tree.nodes}
    fragmentDic = FragmentVocab(list(cset))
    logging.info(f"Dictionaries created. Reactants: {reactantDic.size()}, Templates: {templateDic.size()}, Fragments: {fragmentDic.size()}")

    model = bFTRXNVAE(fragmentDic, reactantDic, templateDic, config.HIDDEN_SIZE, config.LATENT_SIZE, config.DEPTH, config.DEVICE).to(config.DEVICE)
    if not os.path.exists(config.WEIGHTS_SAVE_PATH):
        raise FileNotFoundError(f"Weight file not found at {config.WEIGHTS_SAVE_PATH}")
    checkpoint = torch.load(config.WEIGHTS_SAVE_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint)
    logging.info("Finished loading B-VAE model.")

    # --- 2. Prepare Initial Dataset for Surrogate Model ---
    logp_paths = {
        'logP': config.LOGP_VALUES_PATH,
        'SA': config.SA_SCORES_PATH,
        'cycle': config.CYCLE_SCORES_PATH
    }
    # [ECC] Use configurable ECC settings for dataset preparation
    print(f"[ECC] Using ECC type: {ecc_type}, R: {ecc_R}")
    X_train, y_train, X_test, y_test = binary_vae_utils.prepare_dataset(
        model=model, data_pairs=data_pairs, latent_size=config.LATENT_SIZE, metric=config.METRIC, logp_paths=logp_paths,
        ecc_type=ecc_type, ecc_R=ecc_R
    )
    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)
    
    # --- 3. Initialize Components based on Configuration ---
    logging.info("Initializing components for optimization...")
    # [ECC] Use actual feature size from X_train to handle ECC dimension reduction properly
    n_features = X_train.shape[1]
    print(f"[FM Surrogate] Using n_binary={n_features} (adapted from X_train shape)")
    fm_surrogate = binary_vae_utils.FactorizationMachineSurrogate(
        n_binary=n_features, k_factors=config.FACTOR_NUM, lr=config.LR,
        decay_weight=config.DECAY_WEIGHT, batch_size=config.BATCH_SIZE, max_epoch=config.MAX_EPOCH,
        patience=config.PATIENCE, param_init=config.PARAM_INIT, cache_dir=config.CACHE_DIR,
        prop=config.PROP, client=config.CLIENT, random_seed=config.RANDOM_SEED, device=config.DEVICE
    )
    gurobi_solver = binary_vae_utils.GurobiQuboSolver(options=config.GUROBI_OPTIONS)
    optimizer = binary_vae_utils.MoleculeOptimizer(
        bvae_model=model, surrogate_model=fm_surrogate, qubo_solver=gurobi_solver,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        optimize_num=config.OPTIMIZE_NUM, metric=config.METRIC, random_seed=config.RANDOM_SEED,
        results_dir=config.RESULTS_DIR, logp_paths=logp_paths, device=config.DEVICE
    )

    # --- 4. Start Optimization ---
    logging.info("Starting optimization process...")
    start_time = time.time()
    optimizer.optimize()
    end_time = time.time()
    
    logging.info(f"Total Running Time: {end_time - start_time:.2f} seconds")
    print(f"\nOptimization finished. Total time: {end_time - start_time:.2f} seconds")

# main.py (末尾部分)

if __name__ == "__main__":
    # [ECC] Add ECC options for A/B comparison during optimization
    parser = argparse.ArgumentParser(description="Run Molecule Optimization with a specific random seed.")
    parser.add_argument('--seed', type=int, required=True, help='The random seed to use for the experiment.')
    parser.add_argument('--ecc-type', type=str, choices=['none', 'repetition'], default='none',
                        help='ECC type for latent processing (default: none)')
    parser.add_argument('--ecc-R', type=int, default=3, 
                        help='Repetition factor for ECC (default: 3)')
    
    args = parser.parse_args()
    
    # [ECC] Pass ECC parameters to main function
    print("="*60)
    print(f"--- Starting experiment with RANDOM_SEED = {args.seed} ---")
    print(f"--- ECC type: {args.ecc_type}, R: {args.ecc_R} ---")
    print("="*60)
    main(seed_to_run=args.seed, ecc_type=args.ecc_type, ecc_R=args.ecc_R)
    print(f"\nExperiment with seed {args.seed} finished.")