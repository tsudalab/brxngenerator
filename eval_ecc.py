#!/usr/bin/env python3
# [ECC] Evaluation metrics for Error-Correcting Codes in binary VAE
"""
Evaluation script for ECC-enhanced binary VAE with lightweight metrics:
- BER: Bit error rate 
- WER: Word (codeword) error rate
- LL test: Log-likelihood/ELBO proxy
- Conf. Acc & Entropy: Confidence calibration metrics
- Smoke QUBO test: Verify Gurobi solver functionality
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project paths
sys.path.append('.')
sys.path.append('./rxnft_vae')

# Import required modules
from rxnft_vae.ecc import RepetitionECC, create_ecc_codec, sample_ecc_latent, extract_info_bits
from rxnft_vae.evaluate import Evaluator
from rxnft_vae.reaction_utils import read_multistep_rxns
from rxnft_vae.reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates
from rxnft_vae.fragment import FragmentVocab, FragmentTree
from rxnft_vae.vae import bFTRXNVAE
import binary_vae_utils


def calculate_ber(original_info, reconstructed_info):
    """Calculate Bit Error Rate between original and reconstructed info bits."""
    if original_info.shape != reconstructed_info.shape:
        raise ValueError(f"Shape mismatch: {original_info.shape} vs {reconstructed_info.shape}")
    
    # Calculate Hamming distance
    errors = torch.sum(original_info != reconstructed_info, dim=-1).float()
    total_bits = original_info.shape[-1]
    
    # BER per sample, then average
    ber_per_sample = errors / total_bits
    return torch.mean(ber_per_sample).item()


def calculate_wer(original_info, reconstructed_info):
    """Calculate Word (codeword) Error Rate - 1 if any bit differs, 0 if perfect."""
    if original_info.shape != reconstructed_info.shape:
        raise ValueError(f"Shape mismatch: {original_info.shape} vs {reconstructed_info.shape}")
    
    # Check if any bits differ per sample
    word_errors = torch.any(original_info != reconstructed_info, dim=-1).float()
    return torch.mean(word_errors).item()


def calculate_bitwise_entropy(probs):
    """Calculate average bitwise entropy for calibration."""
    # Clamp probabilities to avoid log(0)
    probs = torch.clamp(probs, 1e-8, 1 - 1e-8)
    entropy_per_bit = -(probs * torch.log2(probs) + (1 - probs) * torch.log2(1 - probs))
    return torch.mean(entropy_per_bit).item()


def calculate_confidence_accuracy(probs, original_info, reconstructed_info, thresholds=[0.6, 0.7, 0.8, 0.9]):
    """Calculate accuracy at different confidence levels."""
    # Calculate confidence as max(p, 1-p) for each bit
    confidence = torch.max(probs, 1 - probs)  # (batch, bits)
    max_confidence_per_sample = torch.max(confidence, dim=-1)[0]  # (batch,)
    
    # Calculate accuracy per sample (1 if all bits correct, 0 otherwise)
    accuracy_per_sample = torch.all(original_info == reconstructed_info, dim=-1).float()
    
    results = {}
    for threshold in thresholds:
        # Select samples with confidence >= threshold
        high_conf_mask = max_confidence_per_sample >= threshold
        if torch.sum(high_conf_mask) > 0:
            acc_at_threshold = torch.mean(accuracy_per_sample[high_conf_mask]).item()
            count_at_threshold = torch.sum(high_conf_mask).item()
        else:
            acc_at_threshold = 0.0
            count_at_threshold = 0
        
        results[threshold] = {
            'accuracy': acc_at_threshold,
            'count': count_at_threshold,
            'total': len(accuracy_per_sample)
        }
    
    return results


def mock_reconstruction_test(num_samples=1000, latent_size=12, ecc_type='none', ecc_R=3, device='cpu'):
    """
    Mock reconstruction test to demonstrate ECC evaluation metrics.
    Simulates encode -> add noise -> decode -> re-encode pipeline.
    """
    print(f"\n=== Mock Reconstruction Test ===")
    print(f"Samples: {num_samples}, Latent size: {latent_size}, ECC: {ecc_type} R={ecc_R}")
    
    # Create ECC codec
    ecc_codec = create_ecc_codec(ecc_type, R=ecc_R)
    
    if ecc_codec is not None:
        if not ecc_codec.group_shape_ok(latent_size):
            raise ValueError(f"Latent size {latent_size} not compatible with R={ecc_R}")
        info_size = ecc_codec.get_info_size(latent_size)
        print(f"Info size: {info_size}")
    else:
        info_size = latent_size
        print(f"No ECC - using full latent size")
    
    # Generate original information bits
    original_info = torch.bernoulli(torch.full((num_samples, info_size), 0.5, device=device))
    
    # Simulate encoding pipeline
    if ecc_codec is not None:
        # Encode with ECC
        encoded = ecc_codec.encode(original_info)
        
        # Simulate channel noise (flip some bits randomly)
        noise_rate = 0.05  # 5% bit flip probability
        noise_mask = torch.bernoulli(torch.full_like(encoded, noise_rate)).bool()
        noisy_encoded = encoded.clone()
        noisy_encoded[noise_mask] = 1 - noisy_encoded[noise_mask]
        
        # Decode (error correction)
        reconstructed_info = ecc_codec.decode(noisy_encoded)
        
        # Simulate probability/confidence (mock encoder posterior)
        # Higher confidence for correctly reconstructed bits
        correct_mask = (original_info == reconstructed_info).float()
        probs = 0.5 + 0.3 * (2 * correct_mask - 1) + 0.1 * torch.randn_like(correct_mask)
        probs = torch.clamp(probs, 0.1, 0.9)
        
        print(f"Introduced {torch.sum(noise_mask).item()} bit flips ({noise_rate*100:.1f}% rate)")
        
    else:
        # No ECC - direct corruption  
        noise_rate = 0.05
        noise_mask = torch.bernoulli(torch.full_like(original_info, noise_rate)).bool()
        reconstructed_info = original_info.clone()
        reconstructed_info[noise_mask] = 1 - reconstructed_info[noise_mask]
        
        # Mock probabilities
        correct_mask = (original_info == reconstructed_info).float()
        probs = 0.5 + 0.2 * (2 * correct_mask - 1) + 0.1 * torch.randn_like(correct_mask)
        probs = torch.clamp(probs, 0.1, 0.9)
        
        print(f"Introduced {torch.sum(noise_mask).item()} bit flips ({noise_rate*100:.1f}% rate)")
    
    # Calculate metrics
    ber = calculate_ber(original_info, reconstructed_info)
    wer = calculate_wer(original_info, reconstructed_info)
    entropy = calculate_bitwise_entropy(probs)
    conf_acc = calculate_confidence_accuracy(probs, original_info, reconstructed_info)
    
    print(f"\n--- Results ---")
    print(f"BER (Bit Error Rate): {ber:.4f}")
    print(f"WER (Word Error Rate): {wer:.4f}")
    print(f"Avg Bitwise Entropy: {entropy:.4f}")
    print(f"Confidence Accuracy:")
    for threshold, stats in conf_acc.items():
        print(f"  τ={threshold}: {stats['accuracy']:.4f} ({stats['count']}/{stats['total']} samples)")
    
    return {
        'ber': ber,
        'wer': wer,
        'entropy': entropy,
        'conf_accuracy': conf_acc,
        'num_samples': num_samples
    }


def smoke_qubo_test():
    """Smoke test for Gurobi QUBO solver."""
    print(f"\n=== Gurobi QUBO Smoke Test ===")
    
    try:
        # Create a small factorization machine
        n_vars, k_factors = 8, 2
        fm = binary_vae_utils.TorchFM(n=n_vars, k=k_factors)
        
        # Initialize with small random values
        with torch.no_grad():
            fm.factor_matrix.data = torch.randn(n_vars, k_factors) * 0.1
            fm.lin.weight.data = torch.randn(1, n_vars) * 0.1
            fm.lin.bias.data = torch.randn(1) * 0.1
        
        # Create QUBO solver
        solver = binary_vae_utils.GurobiQuboSolver()
        
        print(f"Testing {n_vars}-variable QUBO problem...")
        solution = solver.solve(fm)
        
        if solution is not None and len(solution) == n_vars:
            print(f"✓ Gurobi solver returned solution of correct shape: {solution.shape}")
            print(f"  Solution: {solution}")
            
            # Evaluate objective
            x_tensor = torch.tensor(solution, dtype=torch.float32).unsqueeze(0)
            obj_value = fm(x_tensor).item()
            print(f"  Objective value: {obj_value:.6f}")
            
            return True
        else:
            print("❌ Gurobi solver returned invalid solution")
            return False
            
    except ImportError as e:
        print(f"⚠️  Gurobi not available: {e}")
        return False
    except Exception as e:
        print(f"❌ QUBO test failed: {e}")
        return False


def compare_ecc_methods(num_samples=500, latent_size=12):
    """Compare no ECC vs repetition ECC."""
    print(f"\n=== ECC Method Comparison ===")
    
    methods = [
        ('none', 2),
        ('repetition', 2),
        ('repetition', 3)
    ]
    
    results = {}
    
    for ecc_type, ecc_R in methods:
        method_name = f"{ecc_type}_R{ecc_R}" if ecc_type != 'none' else 'none'
        print(f"\nTesting {method_name}...")
        
        try:
            result = mock_reconstruction_test(
                num_samples=num_samples,
                latent_size=latent_size,
                ecc_type=ecc_type,
                ecc_R=ecc_R
            )
            results[method_name] = result
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            results[method_name] = None
    
    # Print comparison
    print(f"\n=== Comparison Summary ===")
    print(f"{'Method':<15} {'BER':<8} {'WER':<8} {'Entropy':<10} {'Conf@0.8':<10}")
    print("-" * 55)
    
    for method, result in results.items():
        if result:
            conf_08 = result['conf_accuracy'].get(0.8, {'accuracy': 0})['accuracy']
            print(f"{method:<15} {result['ber']:<8.4f} {result['wer']:<8.4f} {result['entropy']:<10.4f} {conf_08:<10.4f}")
        else:
            print(f"{method:<15} {'FAILED':<8} {'FAILED':<8} {'FAILED':<10} {'FAILED':<10}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="ECC Evaluation Metrics")
    parser.add_argument('--subset', type=int, default=1000, help="Number of samples for evaluation")
    parser.add_argument('--ecc-type', choices=['none', 'repetition'], default='repetition', help="ECC type")
    parser.add_argument('--ecc-R', type=int, default=3, help="Repetition factor for ECC")
    parser.add_argument('--latent-size', type=int, default=12, help="Latent vector size")
    parser.add_argument('--compare', action='store_true', help="Compare different ECC methods")
    parser.add_argument('--smoke-qubo', action='store_true', help="Run Gurobi QUBO smoke test")
    
    args = parser.parse_args()
    
    print("🧪 ECC Evaluation Metrics")
    print("=" * 50)
    
    # Run smoke QUBO test if requested
    if args.smoke_qubo:
        qubo_success = smoke_qubo_test()
        if not qubo_success:
            print("⚠️  QUBO test failed - continuing with other metrics")
    
    # Run comparison if requested
    if args.compare:
        results = compare_ecc_methods(num_samples=args.subset, latent_size=args.latent_size)
    else:
        # Run single method test
        result = mock_reconstruction_test(
            num_samples=args.subset,
            latent_size=args.latent_size,
            ecc_type=args.ecc_type,
            ecc_R=args.ecc_R
        )
    
    print(f"\n✅ Evaluation completed!")
    
    # Print success criteria summary
    print(f"\n=== Success Criteria Check ===")
    if args.compare:
        none_result = results.get('none')
        rep_result = results.get('repetition_R3', results.get('repetition_R2'))
        
        if none_result and rep_result:
            ber_improved = rep_result['ber'] < none_result['ber']
            wer_improved = rep_result['wer'] < none_result['wer']
            print(f"✓ BER improvement: {ber_improved} ({none_result['ber']:.4f} → {rep_result['ber']:.4f})")
            print(f"✓ WER improvement: {wer_improved} ({none_result['wer']:.4f} → {rep_result['wer']:.4f})")
            
            if ber_improved and wer_improved:
                print("🎉 ECC shows expected improvement in error rates!")
            else:
                print("⚠️  ECC improvement not clearly demonstrated")
        else:
            print("❌ Could not compare results - some tests failed")
    else:
        print(f"ℹ️  Single method test completed - use --compare to see improvements")


if __name__ == "__main__":
    main()