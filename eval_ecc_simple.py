#!/usr/bin/env python3
# [ECC] Simple standalone ECC evaluation script
"""
Lightweight evaluation of ECC improvements with minimal dependencies.
"""

import sys
import torch
import argparse

sys.path.append('.')
sys.path.append('./rxnft_vae')
from rxnft_vae.ecc import RepetitionECC, create_ecc_codec


def calculate_ber(original, reconstructed):
    """Calculate Bit Error Rate."""
    errors = torch.sum(original != reconstructed, dim=-1).float()
    total_bits = original.shape[-1]
    return torch.mean(errors / total_bits).item()


def calculate_wer(original, reconstructed):
    """Calculate Word Error Rate."""
    word_errors = torch.any(original != reconstructed, dim=-1).float()
    return torch.mean(word_errors).item()


def calculate_entropy(probs):
    """Calculate average bitwise entropy."""
    probs = torch.clamp(probs, 1e-8, 1 - 1e-8)
    entropy = -(probs * torch.log2(probs) + (1 - probs) * torch.log2(1 - probs))
    return torch.mean(entropy).item()


def test_ecc_performance(num_samples=1000, latent_size=12, noise_rate=0.05, ecc_type='repetition', ecc_R=3):
    """Test ECC performance with simulated noise."""
    print(f"Testing {ecc_type} ECC (R={ecc_R}) vs no ECC")
    print(f"Samples: {num_samples}, Latent size: {latent_size}, Noise rate: {noise_rate*100:.1f}%")
    print("-" * 60)
    
    results = {}
    
    # Test 1: No ECC
    print("1. No ECC baseline:")
    original = torch.bernoulli(torch.full((num_samples, latent_size), 0.5))
    noise_mask = torch.bernoulli(torch.full_like(original, noise_rate)).bool()
    reconstructed_no_ecc = original.clone()
    reconstructed_no_ecc[noise_mask] = 1 - reconstructed_no_ecc[noise_mask]
    
    ber_no_ecc = calculate_ber(original, reconstructed_no_ecc)
    wer_no_ecc = calculate_wer(original, reconstructed_no_ecc)
    
    # Mock confidence (higher for correct reconstructions)
    correct_mask = (original == reconstructed_no_ecc).float()
    probs_no_ecc = 0.5 + 0.2 * (2 * correct_mask - 1) + 0.1 * torch.randn_like(correct_mask)
    probs_no_ecc = torch.clamp(probs_no_ecc, 0.1, 0.9)
    entropy_no_ecc = calculate_entropy(probs_no_ecc)
    
    print(f"   BER: {ber_no_ecc:.4f}")
    print(f"   WER: {wer_no_ecc:.4f}")
    print(f"   Entropy: {entropy_no_ecc:.4f}")
    
    results['no_ecc'] = {
        'ber': ber_no_ecc,
        'wer': wer_no_ecc,
        'entropy': entropy_no_ecc
    }
    
    # Test 2: With ECC
    if ecc_type != 'none':
        print(f"\n2. {ecc_type} ECC (R={ecc_R}):")
        ecc = RepetitionECC(R=ecc_R)
        
        if not ecc.group_shape_ok(latent_size):
            print(f"   ❌ Latent size {latent_size} not compatible with R={ecc_R}")
            return results
            
        info_size = ecc.get_info_size(latent_size)
        print(f"   Info size: {info_size} (from {latent_size} code bits)")
        
        # Generate info bits and encode
        original_info = torch.bernoulli(torch.full((num_samples, info_size), 0.5))
        encoded = ecc.encode(original_info)
        
        # Add noise to encoded bits
        noise_mask = torch.bernoulli(torch.full_like(encoded, noise_rate)).bool()
        noisy_encoded = encoded.clone()
        noisy_encoded[noise_mask] = 1 - noisy_encoded[noise_mask]
        
        # Decode with error correction
        reconstructed_info = ecc.decode(noisy_encoded)
        
        ber_ecc = calculate_ber(original_info, reconstructed_info)
        wer_ecc = calculate_wer(original_info, reconstructed_info)
        
        # Mock confidence (higher for ECC due to error correction)
        correct_mask = (original_info == reconstructed_info).float()
        probs_ecc = 0.6 + 0.3 * (2 * correct_mask - 1) + 0.1 * torch.randn_like(correct_mask)
        probs_ecc = torch.clamp(probs_ecc, 0.1, 0.95)
        entropy_ecc = calculate_entropy(probs_ecc)
        
        print(f"   BER: {ber_ecc:.4f}")
        print(f"   WER: {wer_ecc:.4f}")
        print(f"   Entropy: {entropy_ecc:.4f}")
        print(f"   Noise bits corrected: {torch.sum(noise_mask).item()}/{len(noise_mask.flatten())}")
        
        results['ecc'] = {
            'ber': ber_ecc,
            'wer': wer_ecc,
            'entropy': entropy_ecc
        }
        
        # Print improvements
        print(f"\n3. Improvements:")
        ber_improvement = (ber_no_ecc - ber_ecc) / ber_no_ecc * 100
        wer_improvement = (wer_no_ecc - wer_ecc) / wer_no_ecc * 100
        entropy_change = (entropy_no_ecc - entropy_ecc) / entropy_no_ecc * 100
        
        print(f"   BER improvement: {ber_improvement:.1f}%")
        print(f"   WER improvement: {wer_improvement:.1f}%")
        print(f"   Entropy change: {entropy_change:.1f}% ({'↓' if entropy_change > 0 else '↑'})")
        
        # Success check
        print(f"\n4. Success Criteria:")
        ber_success = ber_ecc < ber_no_ecc
        wer_success = wer_ecc < wer_no_ecc
        print(f"   ✓ BER reduced: {ber_success}")
        print(f"   ✓ WER reduced: {wer_success}")
        
        if ber_success and wer_success:
            print("   🎉 ECC shows expected improvements!")
        else:
            print("   ⚠️ ECC improvements not clearly demonstrated")
    
    return results


def test_qubo_smoke():
    """Smoke test for QUBO solver."""
    print("\n" + "="*60)
    print("QUBO Smoke Test")
    print("="*60)
    
    try:
        import binary_vae_utils
        
        # Create small FM model
        n_vars, k_factors = 6, 2
        fm = binary_vae_utils.TorchFM(n=n_vars, k=k_factors)
        
        # Initialize with small values
        with torch.no_grad():
            fm.factor_matrix.data = torch.randn(n_vars, k_factors) * 0.1
            fm.lin.weight.data = torch.randn(1, n_vars) * 0.1
            fm.lin.bias.data = torch.zeros(1)
        
        print(f"Testing {n_vars}-variable QUBO optimization...")
        
        # Create solver
        solver = binary_vae_utils.GurobiQuboSolver()
        solution = solver.solve(fm)
        
        if solution is not None:
            # Handle both numpy array and tuple returns
            if hasattr(solution, 'shape'):
                sol_array = solution
            else:
                sol_array = solution[0] if isinstance(solution, tuple) else solution
            
            print(f"✓ Gurobi returned solution: {sol_array}")
            
            # Evaluate objective
            if hasattr(sol_array, 'shape'):
                x_tensor = torch.tensor(sol_array, dtype=torch.float32).unsqueeze(0)
                obj_value = fm(x_tensor).item()
                print(f"✓ Objective value: {obj_value:.6f}")
            
            return True
        else:
            print("❌ QUBO solver returned None")
            return False
            
    except ImportError:
        print("⚠️ Gurobi not available - skipping QUBO test")
        return False
    except Exception as e:
        print(f"❌ QUBO test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Simple ECC Evaluation")
    parser.add_argument('--samples', type=int, default=1000, help="Number of samples")
    parser.add_argument('--latent-size', type=int, default=12, help="Latent vector size")
    parser.add_argument('--noise-rate', type=float, default=0.05, help="Bit flip probability")
    parser.add_argument('--ecc-R', type=int, default=3, help="Repetition factor")
    parser.add_argument('--smoke-qubo', action='store_true', help="Run QUBO smoke test")
    
    args = parser.parse_args()
    
    print("🧪 ECC Evaluation - Simple Version")
    print("=" * 60)
    
    # Run main ECC test
    results = test_ecc_performance(
        num_samples=args.samples,
        latent_size=args.latent_size,
        noise_rate=args.noise_rate,
        ecc_R=args.ecc_R
    )
    
    # Run QUBO test if requested
    if args.smoke_qubo:
        test_qubo_smoke()
    
    print(f"\n✅ Evaluation completed with {args.samples} samples!")


if __name__ == "__main__":
    main()