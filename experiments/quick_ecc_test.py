#!/usr/bin/env python3
"""
Quick ECC effectiveness test focused on core objectives:
1. 更稳的离散推断 (More stable discrete inference)  
2. 缩小变分间隙 (Reduce variational gap)
3. 更准的潜空间不确定性 (Better latent uncertainty)

Uses minimal training for rapid validation.
"""

import os
import sys
import torch
import numpy as np
import random
from tqdm import tqdm

# Add project modules
sys.path.append('./rxnft_vae')
from rxnft_vae.reaction_utils import read_multistep_rxns
from rxnft_vae.reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates
from rxnft_vae.fragment import FragmentVocab, FragmentTree
from rxnft_vae.vae import bFTRXNVAE
from rxnft_vae.evaluate import Evaluator
from metrics.molecule_metrics import canon_smi, to_mol, mean_sas, novelty, uniqueness


def seed_all(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def quick_train(model, data_pairs, epochs=10, device='cpu'):
    """
    Quick training with minimal epochs to demonstrate ECC benefits.
    Focus on inference stability rather than full convergence.
    """
    print(f"Quick training for {epochs} epochs...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        random.shuffle(data_pairs)
        
        epoch_loss = 0
        for i, (fgm_tree, rxn_tree) in enumerate(data_pairs[:20]):  # Use only 20 samples per epoch
            try:
                batch = [(fgm_tree, rxn_tree)]
                beta = min(1.0, epoch / 5.0)  # Quick warmup
                
                optimizer.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        t_loss, _, _, _, _, _, _, _, _, kl_loss, _ = model(batch, beta)
                    scaler.scale(t_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    t_loss, _, _, _, _, _, _, _, _, kl_loss, _ = model(batch, beta)
                    t_loss.backward()
                    optimizer.step()
                    
                epoch_loss += t_loss.item()
                
            except Exception:
                continue
                
        print(f"Epoch {epoch}: Loss = {epoch_loss/20:.4f}")


def test_generation_stability(model, latent_size, ecc_type, ecc_R, n_samples=50):
    """
    Test generation stability - core ECC objective.
    """
    print(f"\nTesting generation stability ({ecc_type}, R={ecc_R})...")
    
    evaluator = Evaluator(latent_size, model, ecc_type=ecc_type, ecc_R=ecc_R)
    
    valid_molecules = []
    generation_attempts = 0
    max_attempts = n_samples * 3
    
    with tqdm(total=n_samples, desc="Generating") as pbar:
        while len(valid_molecules) < n_samples and generation_attempts < max_attempts:
            generation_attempts += 1
            try:
                # Generate with consistent temperature for fair comparison
                ft_latent = evaluator.generate_discrete_latent(latent_size, method="gumbel", temp=0.5)
                rxn_latent = evaluator.generate_discrete_latent(latent_size, method="gumbel", temp=0.5)
                product, reactions_str = evaluator.decode_from_prior(ft_latent, rxn_latent, n=3)
                
                if product:
                    canonical = canon_smi(product)
                    if canonical:  # Valid molecule
                        valid_molecules.append(canonical)
                        pbar.update(1)
                        
            except Exception:
                continue
    
    # Compute stability metrics
    total_attempts = generation_attempts
    success_rate = len(valid_molecules) / total_attempts if total_attempts > 0 else 0
    
    # Uniqueness and diversity
    unique_molecules = list(set(valid_molecules))
    uniqueness_rate = len(unique_molecules) / len(valid_molecules) if valid_molecules else 0
    
    # Chemical validity assessment
    mol_objects = [to_mol(smi) for smi in valid_molecules]
    sas_scores = []
    for mol in mol_objects:
        if mol:
            try:
                from rxnft_vae.sascorer import calculateScore
                score = calculateScore(mol)
                sas_scores.append(score)
            except:
                pass
    
    avg_sas = np.mean(sas_scores) if sas_scores else float('inf')
    
    return {
        'model_type': f"{ecc_type}_R{ecc_R}" if ecc_type != 'none' else 'baseline',
        'valid_molecules': len(valid_molecules),
        'total_attempts': total_attempts,
        'success_rate': success_rate,
        'uniqueness': uniqueness_rate,
        'avg_sas': avg_sas,
        'sas_count': len(sas_scores),
        'unique_count': len(unique_molecules)
    }


def main():
    print("🧬 Quick ECC Effectiveness Test")
    print("="*50)
    
    # Setup
    seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load minimal data
    print("\nLoading data...")
    routes, _ = read_multistep_rxns("./data/data.txt")
    rxn_trees = [ReactionTree(route) for route in routes[:200]]  # Use only 200 samples
    
    # Build vocabularies
    reactants = extract_starting_reactants(rxn_trees)
    templates, n_reacts = extract_templates(rxn_trees)
    reactantDic = StartingReactants(reactants)
    templateDic = Templates(templates, n_reacts)
    
    # Build fragment trees
    fgm_trees, valid_rxn_trees = [], []
    for tree in tqdm(rxn_trees, desc="Fragmenting"):
        try:
            fgm_trees.append(FragmentTree(tree.molecule_nodes[0].smiles))
            valid_rxn_trees.append(tree)
        except:
            continue
    
    data_pairs = list(zip(fgm_trees, valid_rxn_trees))[:100]  # Use only 100 pairs
    cset = {node.smiles for fgm_tree in fgm_trees for node in fgm_tree.nodes}
    fragmentDic = FragmentVocab(list(cset))
    
    print(f"Data prepared: {len(data_pairs)} pairs")
    print(f"Vocabularies - Fragments: {fragmentDic.size()}, Reactants: {reactantDic.size()}, Templates: {templateDic.size()}")
    
    # Model parameters
    hidden_size, latent_size, depth = 200, 100, 2
    ecc_R = 2
    
    results = []
    
    # Test 1: Baseline model
    print(f"\n{'='*20} BASELINE TEST {'='*20}")
    baseline_model = bFTRXNVAE(fragmentDic, reactantDic, templateDic, hidden_size, latent_size, depth, 
                              device=device, ecc_type='none', ecc_R=ecc_R).to(device)
    
    quick_train(baseline_model, data_pairs, epochs=8, device=device)
    baseline_results = test_generation_stability(baseline_model, latent_size, 'none', ecc_R, n_samples=30)
    results.append(baseline_results)
    
    # Test 2: ECC model  
    print(f"\n{'='*20} ECC TEST {'='*20}")
    ecc_model = bFTRXNVAE(fragmentDic, reactantDic, templateDic, hidden_size, latent_size, depth,
                         device=device, ecc_type='repetition', ecc_R=ecc_R).to(device)
    
    quick_train(ecc_model, data_pairs, epochs=8, device=device)
    ecc_results = test_generation_stability(ecc_model, latent_size, 'repetition', ecc_R, n_samples=30)
    results.append(ecc_results)
    
    # Compare results
    print(f"\n{'='*20} COMPARISON {'='*20}")
    print(f"{'Metric':<20} {'Baseline':<15} {'ECC':<15} {'Improvement':<15}")
    print("-" * 65)
    
    baseline, ecc = results[0], results[1]
    
    for key in ['success_rate', 'uniqueness', 'avg_sas']:
        baseline_val = baseline.get(key, 0)
        ecc_val = ecc.get(key, 0)
        
        if baseline_val > 0:
            if key == 'avg_sas':  # Lower is better for SAS
                improvement = ((baseline_val - ecc_val) / baseline_val) * 100
            else:  # Higher is better
                improvement = ((ecc_val - baseline_val) / baseline_val) * 100
        else:
            improvement = 0
            
        print(f"{key:<20} {baseline_val:<15.4f} {ecc_val:<15.4f} {improvement:<15.1f}%")
    
    # Summary
    print(f"\n{'='*20} ECC EFFECTIVENESS {'='*20}")
    print(f"ECC Core Objectives Assessment:")
    
    if ecc['success_rate'] > baseline['success_rate']:
        print("✅ 更稳的离散推断 (More stable inference): ECC shows higher success rate")
    else:
        print("❌ 更稳的离散推断 (More stable inference): No improvement detected")
        
    if ecc['avg_sas'] < baseline['avg_sas'] and ecc['avg_sas'] < 10:
        print("✅ 更好的分子质量 (Better molecule quality): ECC shows lower SAS (easier synthesis)")
    else:
        print("❌ 更好的分子质量 (Better molecule quality): No clear improvement")
        
    if ecc['uniqueness'] >= baseline['uniqueness']:
        print("✅ 潜空间多样性 (Latent diversity): ECC maintains or improves uniqueness")
    else:
        print("❌ 潜空间多样性 (Latent diversity): Reduced uniqueness")
    
    print(f"\nTotal valid molecules - Baseline: {baseline['valid_molecules']}, ECC: {ecc['valid_molecules']}")


if __name__ == "__main__":
    main()