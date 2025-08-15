#!/usr/bin/env python3
"""
A/B Comparison Script: Baseline vs ECC Binary VAE

Trains and evaluates both Baseline and ECC models using the same parameter set,
then compares real metrics including BER/WER, reconstruction loss, calibration,
and molecule quality metrics.

Usage:
    python ab_compare_ecc.py -n 1 --ecc-R 3 --train-subset 0 --eval-subset 2000 --gpu 0
"""

import os
import sys
import argparse
import time
import json
import pandas as pd
from datetime import datetime
import torch
import numpy as np
import random
from tqdm import tqdm

# [ECC] Import project modules
sys.path.append('./rxnft_vae')
from rxnft_vae.reaction_utils import read_multistep_rxns
from rxnft_vae.reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates
from rxnft_vae.fragment import FragmentVocab, FragmentTree
from rxnft_vae.vae import bFTRXNVAE
from rxnft_vae.mpn import MPN
from rxnft_vae.evaluate import Evaluator
from rxnft_vae.metrics_eval import MolecularMetrics, load_training_molecules
from rxnft_vae.latent_metrics import LatentMetrics
from metrics.molecule_metrics import canon_smi, to_mol, mean_sas, novelty, uniqueness


def seed_all(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_data(data_filename, train_subset=None, eval_subset=None):
    """
    Load and prepare data with optional subsetting.
    
    Returns:
        train_data_pairs: Training data pairs
        eval_data_pairs: Evaluation data pairs  
        vocabularies: (fragmentDic, reactantDic, templateDic)
    """
    print("Loading data...")
    routes, _ = read_multistep_rxns(data_filename)
    rxn_trees = [ReactionTree(route) for route in routes]
    
    # Apply training subset if requested
    if train_subset and train_subset > 0 and len(rxn_trees) > train_subset:
        print(f"Using training subset of {train_subset} reactions (out of {len(rxn_trees)})")
        rxn_trees = rxn_trees[:train_subset]
    
    # Build vocabularies from training data
    reactants = extract_starting_reactants(rxn_trees)
    templates, n_reacts = extract_templates(rxn_trees)
    reactantDic = StartingReactants(reactants)
    templateDic = Templates(templates, n_reacts)
    
    # Build fragment trees
    print("Building fragment trees...")
    fgm_trees, valid_rxn_trees = [], []
    for tree in tqdm(rxn_trees, desc="Fragmenting molecules", leave=False):
        try:
            fgm_trees.append(FragmentTree(tree.molecule_nodes[0].smiles))
            valid_rxn_trees.append(tree)
        except Exception:
            continue
    
    train_data_pairs = list(zip(fgm_trees, valid_rxn_trees))
    cset = {node.smiles for fgm_tree in fgm_trees for node in fgm_tree.nodes}
    fragmentDic = FragmentVocab(list(cset))
    
    # Prepare evaluation subset
    if eval_subset and eval_subset < len(train_data_pairs):
        eval_data_pairs = train_data_pairs[:eval_subset]
    else:
        eval_data_pairs = train_data_pairs
        
    print(f"Data prepared: {len(train_data_pairs)} training pairs, {len(eval_data_pairs)} eval pairs")
    print(f"Vocabularies - Fragments: {fragmentDic.size()}, Reactants: {reactantDic.size()}, Templates: {templateDic.size()}")
    
    return train_data_pairs, eval_data_pairs, (fragmentDic, reactantDic, templateDic)


def train_model(model, data_pairs, config, model_name, device):
    """
    Train a single model (baseline or ECC) with early stopping.
    
    Returns:
        best_model_path: Path to saved best model
        training_time: Time taken for training
    """
    print(f"\n=== Training {model_name} ===")
    start_time = time.time()
    
    # Training parameters
    batch_size = config['batch_size']
    epochs = config['epochs']
    patience = config.get('patience', 10)
    min_delta = config.get('min_delta', 0.0)
    
    # Train/val split (ensure at least 1 validation sample)
    val_size = max(1, min(500, len(data_pairs) // 10))
    val_pairs = data_pairs[:val_size]
    train_pairs = data_pairs[val_size:]
    
    print(f"Training: {len(train_pairs)}, Validation: {len(val_pairs)}")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0.0001)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Training loop with tqdm
    epoch_pbar = tqdm(range(epochs), desc=f"Training {model_name}", unit="epoch")
    for epoch in epoch_pbar:
        random.shuffle(train_pairs)
        
        # DataLoader setup
        from torch.utils.data import DataLoader
        num_workers = min(4, os.cpu_count() // 2) if device.type == 'cuda' else 0
        dataloader = DataLoader(train_pairs, batch_size=batch_size, shuffle=True,
                               collate_fn=lambda x: x, pin_memory=(device.type == 'cuda'),
                               num_workers=num_workers)
        
        # Training epoch
        model.train()
        epoch_loss = 0
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, unit="batch")
        
        for batch in batch_pbar:
            optimizer.zero_grad()
            
            # Schedule beta warmup (first 20 epochs)
            counter = epoch * len(dataloader) + len(batch_pbar)
            if epoch < 20:
                beta = min(1.0, counter / (20 * len(dataloader)))
            else:
                beta = config['beta']
            
            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    t_loss, pred_loss, stop_loss, template_loss, molecule_label_loss, pred_acc, stop_acc, template_acc, label_acc, kl_loss, molecule_distance_loss = model(batch, beta)
                scaler.scale(t_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                t_loss, pred_loss, stop_loss, template_loss, molecule_label_loss, pred_acc, stop_acc, template_acc, label_acc, kl_loss, molecule_distance_loss = model(batch, beta)
                t_loss.backward()
                optimizer.step()
            
            epoch_loss += t_loss.item()
            batch_pbar.set_postfix({'loss': f"{t_loss.item():.4f}", 'beta': f"{beta:.3f}"})
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_dataloader = DataLoader(val_pairs, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
            for batch in val_dataloader:
                t_loss, pred_loss, stop_loss, template_loss, molecule_label_loss, pred_acc, stop_acc, template_acc, label_acc, kl_loss, molecule_distance_loss = model(batch, config['beta'], epsilon_std=0.01)
                val_loss += (pred_loss + stop_loss + template_loss + molecule_label_loss).item()
            val_loss /= max(1, len(val_dataloader))
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        epoch_pbar.set_postfix({
            'val_loss': f"{val_loss:.4f}",
            'best': f"{best_val_loss:.4f}",
            'patience': f"{patience_counter}/{patience}"
        })
        
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Save best model
    save_dir = f"weights/compare_{model_name.lower()}"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = f"{save_dir}/best_model.pt"
    
    if best_model_state is not None:
        torch.save(best_model_state, best_model_path)
        print(f"Best model saved: {best_model_path}")
    else:
        print("Warning: No best model state to save")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s")
    
    return best_model_path, training_time


def compute_metrics(model, data_pairs, latent_size, ecc_type, ecc_R, model_name, n_samples=1000, reactions=False, 
                   latent_metrics_enabled=False, noise_epsilon=0.0, iwae_samples=0, device=None):
    """
    Compute standardized 5 metrics + optional latent metrics for a trained model.
    
    Returns:
        metrics: Dictionary of computed metrics
    """
    print(f"\n=== Evaluating {model_name} with 5 standardized metrics ===")
    
    # Build canonical training set from the actual training data_pairs used
    print(f"Extracting and canonicalizing training molecules from {len(data_pairs)} data pairs...")
    train_canon_set = set()
    
    for fgm_tree, rxn_tree in tqdm(data_pairs, desc="Canonicalizing training molecules"):
        # Extract molecule SMILES from the reaction tree
        mol_smiles = rxn_tree.molecule_nodes[0].smiles
        canonical = canon_smi(mol_smiles)
        if canonical is not None:
            train_canon_set.add(canonical)
    
    print(f"Training set: {len(train_canon_set)} canonical molecules")
    assert len(train_canon_set) > 0, "Canonical training set must be > 0"
    
    # Create a training_smiles list from canonical set for MolecularMetrics compatibility
    training_smiles = list(train_canon_set)
    metrics_evaluator = MolecularMetrics(training_smiles=training_smiles)
    
    # Initialize latent metrics if enabled
    latent_evaluator = None
    if latent_metrics_enabled:
        print(f"[Latent Metrics] Enabled with noise_epsilon={noise_epsilon}, iwae_samples={iwae_samples}")
        latent_evaluator = LatentMetrics(device or torch.device('cpu'))
    
    # Generate samples using existing Evaluator
    evaluator = Evaluator(latent_size, model, ecc_type=ecc_type, ecc_R=ecc_R)
    
    # Generate molecules/reactions for evaluation
    print(f"Generating {n_samples} samples...")
    generated_samples = []
    generated_reactions = []  # Track reactions separately
    max_attempts = n_samples * 3  # Allow some failures
    
    # Use tqdm progress bar for generation
    with tqdm(total=n_samples, desc="Generating samples") as pbar:
        attempts = 0
        while len(generated_samples) < n_samples and attempts < max_attempts:
            attempts += 1
            try:
                ft_latent = evaluator.generate_discrete_latent(latent_size, method="gumbel", temp=0.4)
                rxn_latent = evaluator.generate_discrete_latent(latent_size, method="gumbel", temp=0.4)
                product, reactions_str = evaluator.decode_from_prior(ft_latent, rxn_latent, n=5)
                
                if product:
                    # Always collect molecules for MOSES metrics
                    generated_samples.append(product)
                    # Collect reactions if available for reaction validity metric
                    if reactions_str:
                        generated_reactions.append(f"{reactions_str}>>{product}")
                    
                    # Update progress bar
                    pbar.update(1)
                        
            except Exception as e:
                continue
            
    if not generated_samples:
        print(f"[Warning] No valid samples generated for {model_name}")
        return {'model_name': model_name, 'error': 'No valid samples generated'}
    
    print(f"Generated {len(generated_samples)} molecules, {len(generated_reactions)} reactions")
    
    # Compute molecule metrics (MOSES standard: validity, QED, uniqueness, novelty, SAS)
    results = metrics_evaluator.evaluate_all_metrics(generated_samples, data_type="molecules")
    
    # Compute enhanced novelty and SAS using our improved implementations
    print("Computing enhanced novelty and SAS metrics...")
    gen_canonical = [canon_smi(smi) for smi in generated_samples]
    gen_mols = [to_mol(smi) for smi in generated_samples]
    
    # Override with enhanced implementations
    enhanced_novelty = novelty(gen_canonical, train_canon_set)
    enhanced_uniqueness = uniqueness(gen_canonical)
    enhanced_sas = mean_sas(gen_mols)
    
    if enhanced_novelty is not None:
        results['novelty'] = enhanced_novelty
    if enhanced_uniqueness is not None:
        results['uniqueness'] = enhanced_uniqueness  
    if enhanced_sas is not None:
        results['avg_sas'] = enhanced_sas
    
    # Add reaction validity if reactions were generated
    if generated_reactions:
        reaction_validity = metrics_evaluator.compute_valid_reaction_rate(generated_reactions)
        results.update(reaction_validity)
        print(f"[Reaction Metrics] {reaction_validity['valid_count']}/{reaction_validity['total_count']} reactions valid")
    else:
        # No reactions generated - mark as N/A
        results['valid_reaction_rate'] = 'N/A'
        results['valid_count_reactions'] = 'N/A'
        results['total_count_reactions'] = 'N/A'
        print("[Reaction Metrics] N/A - no reaction strings generated")
    
    # Add model identification
    results.update({
        'model_name': model_name,
        'ecc_type': ecc_type,
        'ecc_R': ecc_R,
        'latent_size': latent_size,
        'generated_samples': len(generated_samples)
    })
    
    # Compute latent metrics if enabled
    if latent_evaluator is not None:
        print(f"[Latent Metrics] Computing BER/WER/ECE/Entropy for {model_name}...")
        try:
            # Use a subset of data_pairs for latent evaluation (for speed)
            latent_eval_batch = data_pairs[:min(100, len(data_pairs))]
            
            latent_results = latent_evaluator.evaluate_all_latent_metrics(
                model=model,
                data_batch=latent_eval_batch,
                ecc_type=ecc_type,
                ecc_R=ecc_R,
                noise_epsilon=noise_epsilon,
                iwae_samples=iwae_samples
            )
            
            # Add latent metrics to results with prefix
            for key, value in latent_results.items():
                results[f"latent_{key}"] = value
                
            print(f"[Latent Metrics] ✅ BER={latent_results.get('ber_info', 0):.4f}, "
                  f"WER={latent_results.get('wer', 0):.4f}, "
                  f"ECE={latent_results.get('ece', 0):.4f}")
                  
        except Exception as e:
            print(f"[Latent Metrics] ❌ Error: {e}")
            results['latent_error'] = str(e)
    
    return results


def compare_and_save_results(baseline_metrics, ecc_metrics, args, baseline_time, ecc_time):
    """
    Compare metrics and save results to files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate improvements for the 5 standardized metrics + latent metrics
    improvements = {}
    key_metrics = ['valid_reaction_rate', 'valid_molecule_rate', 'avg_qed', 'uniqueness', 'novelty', 'avg_sas']
    
    # Add latent metrics if available
    latent_metric_keys = ['latent_ber_info', 'latent_wer', 'latent_ece', 'latent_entropy', 'latent_elbo']
    if any(key in baseline_metrics for key in latent_metric_keys):
        key_metrics.extend(latent_metric_keys)
    
    for key in key_metrics:
        if key in baseline_metrics and key in ecc_metrics:
            baseline_val = baseline_metrics[key]
            ecc_val = ecc_metrics[key]
            # Skip if either value is not numeric (e.g., 'N/A')
            if isinstance(baseline_val, (int, float)) and isinstance(ecc_val, (int, float)) and abs(baseline_val) > 1e-8:
                # Metrics where lower is better (negative improvement means improvement)
                lower_is_better = ['avg_sas', 'latent_ber_info', 'latent_wer', 'latent_ece', 'latent_entropy']
                if key in lower_is_better:
                    improvement = ((baseline_val - ecc_val) / baseline_val) * 100
                else:
                    # For other metrics, higher is better
                    improvement = ((ecc_val - baseline_val) / baseline_val) * 100
                improvements[f"{key}_improvement_%"] = improvement
    
    # Combine results
    comparison_results = {
        'experiment_config': {
            'parameter_set': args.n,
            'ecc_R': args.ecc_R,
            'train_subset': args.train_subset,
            'eval_subset': args.eval_subset,
            'timestamp': timestamp,
            'device': str(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')
        },
        'training_times': {
            'baseline_seconds': baseline_time,
            'ecc_seconds': ecc_time,
            'total_seconds': baseline_time + ecc_time
        },
        'baseline_metrics': baseline_metrics,
        'ecc_metrics': ecc_metrics,
        'improvements': improvements
    }
    
    # Save JSON results
    os.makedirs('results', exist_ok=True)
    json_path = f"results/compare_n{args.n}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Save CSV summary
    csv_data = []
    for model_name, metrics in [('baseline', baseline_metrics), ('ecc', ecc_metrics)]:
        row = {
            'model': model_name,
            'parameter_set': args.n,
            'ecc_R': args.ecc_R if model_name == 'ecc' else 'N/A',
            'timestamp': timestamp
        }
        row.update(metrics)
        csv_data.append(row)
    
    csv_path = f"results/compare_n{args.n}_{timestamp}.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    
    print(f"\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    
    # Print comparison table
    print("\n" + "="*80)
    print(f"{'COMPARISON RESULTS - 5 STANDARDIZED METRICS':^80}")
    print("="*80)
    print(f"{'METRIC':<25} {'BASELINE':<15} {'ECC':<15} {'IMPROVEMENT':<15}")
    print("-"*80)
    
    # Display the 5 standardized metrics + latent metrics if available
    display_metrics = [
        ('valid_reaction_rate', 'Valid Reaction Rate'),
        ('valid_molecule_rate', 'Valid Molecule Rate'),
        ('avg_qed', 'Average QED'),
        ('uniqueness', 'Uniqueness'),
        ('novelty', 'Novelty'),
        ('avg_sas', 'Average SAS')
    ]
    
    # Add latent metrics if computed
    latent_metrics = [
        ('latent_ber_info', 'BER (info bits)'),
        ('latent_wer', 'WER'),
        ('latent_ece', 'ECE'),
        ('latent_entropy', 'Entropy'),
        ('latent_elbo', 'ELBO')
    ]
    
    # Check if latent metrics are available
    has_latent_metrics = any(key in baseline_metrics for key, _ in latent_metrics)
    if has_latent_metrics:
        display_metrics.extend(latent_metrics)
    
    for key, display_name in display_metrics:
        baseline_val = baseline_metrics.get(key, 0)
        ecc_val = ecc_metrics.get(key, 0)
        improvement = improvements.get(f"{key}_improvement_%", 0)
        
        # Format values, handling 'N/A' strings
        baseline_str = f"{baseline_val:>15.4f}" if isinstance(baseline_val, (int, float)) else f"{str(baseline_val):>15}"
        ecc_str = f"{ecc_val:>15.4f}" if isinstance(ecc_val, (int, float)) else f"{str(ecc_val):>15}"
        improvement_str = f"{improvement:>14.1f}%" if isinstance(improvement, (int, float)) else f"{'N/A':>15}"
        
        print(f"{display_name:<25} {baseline_str} {ecc_str} {improvement_str}")
    
    print("-"*80)
    print(f"{'Training time (s)':<25} {baseline_time:<15.1f} {ecc_time:<15.1f} {'-':<15}")
    print("="*80)
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="A/B comparison: Baseline vs ECC Binary VAE")
    parser.add_argument('-n', type=int, default=1, help='Parameter set index (0-7)')
    parser.add_argument('--ecc-R', type=int, default=3, help='ECC repetition factor (default: 3)')
    parser.add_argument('--train-subset', type=int, default=0, help='Training subset size (0=full data)')
    parser.add_argument('--eval-subset', type=int, default=2000, help='Evaluation subset size')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index (ignored if no CUDA)')
    parser.add_argument('--fcd-samples', type=int, default=0, help='FCD sample count (0=skip FCD)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Latent metrics flags
    parser.add_argument('--latent-metrics', type=str, default='false', choices=['true', 'false'], 
                       help='Enable latent-space metrics (BER/WER/ECE/Entropy)')
    parser.add_argument('--noise-epsilon', type=float, default=0.0, 
                       help='Channel noise flip rate for ECC robustness test (0.0-0.1)')
    parser.add_argument('--iwae-samples', type=int, default=0, 
                       help='IWAE importance samples (0=skip, 64=recommended)')
    
    args = parser.parse_args()
    
    # Setup
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parameter sets (same as trainvae.py)
    params = [
        (100, 100, 2), (200, 100, 2), (200, 100, 3), (200, 100, 5),
        (200, 200, 2), (200, 300, 2), (300, 100, 2), (500, 300, 5),
    ]
    
    if args.n >= len(params):
        raise ValueError(f"Parameter set {args.n} not found. Available: 0-{len(params)-1}")
    
    hidden_size, latent_size, depth = params[args.n]
    print(f"Using parameter set {args.n}: hidden={hidden_size}, latent={latent_size}, depth={depth}")
    
    # ECC validation
    if latent_size % args.ecc_R != 0:
        raise ValueError(f"ECC repetition requires latent_size % ecc_R == 0. Got {latent_size} % {args.ecc_R} != 0")
    
    # Prepare data
    train_data_pairs, eval_data_pairs, (fragmentDic, reactantDic, templateDic) = prepare_data(
        "./data/data.txt",
        train_subset=args.train_subset if args.train_subset > 0 else None,
        eval_subset=args.eval_subset
    )
    
    # Training configuration
    config = {
        'batch_size': 1000,
        'lr': 0.001,
        'beta': 1.0,
        'epochs': 50,  # Reasonable for comparison
        'patience': args.patience,
        'min_delta': 0.0
    }
    
    print(f"\nExperiment Configuration:")
    print(f"  Parameter set: {args.n} -> {params[args.n]}")
    print(f"  ECC: R={args.ecc_R}")
    print(f"  Training data: {len(train_data_pairs)} pairs")
    print(f"  Evaluation data: {len(eval_data_pairs)} pairs")
    print(f"  Device: {device}")
    print(f"  Latent metrics: {args.latent_metrics}")
    if args.latent_metrics == 'true':
        print(f"    Noise epsilon: {args.noise_epsilon}")
        print(f"    IWAE samples: {args.iwae_samples}")
    
    # === BASELINE TRAINING ===
    # [ECC] Baseline model without ECC
    baseline_model = bFTRXNVAE(fragmentDic, reactantDic, templateDic, hidden_size, latent_size, depth, device=device,
                               ecc_type='none', ecc_R=args.ecc_R).to(device)
    baseline_path, baseline_time = train_model(baseline_model, train_data_pairs, config, "Baseline", device)
    
    # Load best baseline model for evaluation
    baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
    baseline_metrics = compute_metrics(baseline_model, eval_data_pairs, latent_size, 'none', args.ecc_R, "Baseline", 
                                     args.eval_subset, latent_metrics_enabled=(args.latent_metrics == 'true'), 
                                     noise_epsilon=args.noise_epsilon, iwae_samples=args.iwae_samples, device=device)
    
    # === ECC TRAINING ===
    # [ECC] ECC model with repetition code
    ecc_model = bFTRXNVAE(fragmentDic, reactantDic, templateDic, hidden_size, latent_size, depth, device=device,
                          ecc_type='repetition', ecc_R=args.ecc_R).to(device)
    ecc_path, ecc_time = train_model(ecc_model, train_data_pairs, config, "ECC", device)
    
    # Load best ECC model for evaluation  
    ecc_model.load_state_dict(torch.load(ecc_path, map_location=device))
    ecc_metrics = compute_metrics(ecc_model, eval_data_pairs, latent_size, 'repetition', args.ecc_R, "ECC", 
                                args.eval_subset, latent_metrics_enabled=(args.latent_metrics == 'true'), 
                                noise_epsilon=args.noise_epsilon, iwae_samples=args.iwae_samples, device=device)
    
    # === COMPARISON ===
    results = compare_and_save_results(baseline_metrics, ecc_metrics, args, baseline_time, ecc_time)
    
    print(f"\nA/B comparison completed successfully!")
    print(f"Total experiment time: {(baseline_time + ecc_time):.1f} seconds")


if __name__ == "__main__":
    main()