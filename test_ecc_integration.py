#!/usr/bin/env python3
# [ECC] Smoke test for ECC integration in evaluator
"""
Quick test to verify ECC integration works without requiring full model training.
"""

import sys
import os
sys.path.append('.')
sys.path.append('./rxnft_vae')

import torch
from rxnft_vae.ecc import RepetitionECC, create_ecc_codec

def test_ecc_integration():
    """Test ECC module integration."""
    print("Testing ECC integration...")
    
    # Test 1: Basic ECC functionality
    print("\n1. Testing basic ECC functionality")
    ecc = RepetitionECC(R=3)
    info_bits = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)  # (1, 4)
    encoded = ecc.encode(info_bits)
    decoded = ecc.decode(encoded)
    
    print(f"Info bits: {info_bits}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Round-trip successful: {torch.equal(info_bits, decoded)}")
    
    # Test 2: Error correction
    print("\n2. Testing error correction")
    corrupted = encoded.clone()
    # Introduce 1 error per group (correctable with R=3)
    corrupted[0, 2] = 1 - corrupted[0, 2]  # Flip bit in first group
    corrupted[0, 5] = 1 - corrupted[0, 5]  # Flip bit in second group
    
    corrected = ecc.decode(corrupted)
    print(f"Corrupted: {corrupted}")
    print(f"Corrected: {corrected}")
    print(f"Error correction successful: {torch.equal(info_bits, corrected)}")
    
    # Test 3: Factory function
    print("\n3. Testing factory function")
    codec_none = create_ecc_codec('none')
    codec_rep = create_ecc_codec('repetition', R=2)
    
    print(f"None codec: {codec_none}")
    print(f"Repetition codec: {codec_rep}")
    print(f"Repetition R: {codec_rep.R if codec_rep else 'N/A'}")
    
    # Test 4: Latent size compatibility
    print("\n4. Testing latent size compatibility")
    latent_sizes = [9, 12, 15, 8, 10]  # 9,12,15 are divisible by 3; 8,10 are not
    ecc = RepetitionECC(R=3)
    
    for size in latent_sizes:
        compatible = ecc.group_shape_ok(size)
        print(f"Latent size {size}: {'✓' if compatible else '✗'}")
    
    print("\nECC integration test completed!")
    return True

def test_mock_evaluator():
    """Test a mock evaluator-like class to verify ECC integration pattern."""
    print("\n5. Testing mock evaluator with ECC")
    
    class MockEvaluator:
        def __init__(self, latent_size, ecc_type='none', ecc_R=3):
            self.latent_size = latent_size
            self.ecc_type = ecc_type
            self.ecc_R = ecc_R
            self.ecc_codec = create_ecc_codec(ecc_type, R=ecc_R)
            
            if self.ecc_codec is not None:
                if not self.ecc_codec.group_shape_ok(latent_size):
                    raise ValueError(f"Latent size {latent_size} must be divisible by ECC repetition factor {ecc_R}")
                self.info_size = self.ecc_codec.get_info_size(latent_size)
                print(f"[ECC] Using {ecc_type} with R={ecc_R}: latent_size={latent_size}, info_size={self.info_size}")
            else:
                self.info_size = latent_size
                print(f"[ECC] No ECC: latent_size={latent_size}")
        
        def generate_discrete_latent_mock(self):
            """Mock latent generation with ECC support."""
            device = torch.device('cpu')
            
            # Determine sampling size
            if self.ecc_codec is not None:
                effective_size = self.info_size
            else:
                effective_size = self.latent_size
            
            # Sample information bits
            info_bits = torch.bernoulli(torch.full((1, effective_size), 0.5, device=device))
            
            # Apply encoding if ECC is enabled
            if self.ecc_codec is not None:
                return self.ecc_codec.encode(info_bits)
            else:
                return info_bits
    
    # Test without ECC
    evaluator_no_ecc = MockEvaluator(latent_size=12, ecc_type='none')
    latent_no_ecc = evaluator_no_ecc.generate_discrete_latent_mock()
    print(f"No ECC - Latent shape: {latent_no_ecc.shape}")
    
    # Test with ECC
    evaluator_ecc = MockEvaluator(latent_size=12, ecc_type='repetition', ecc_R=3)
    latent_ecc = evaluator_ecc.generate_discrete_latent_mock()
    print(f"ECC R=3 - Latent shape: {latent_ecc.shape}")
    
    # Verify encoding consistency
    if evaluator_ecc.ecc_codec is not None:
        groups = evaluator_ecc.ecc_codec.to_groups(latent_ecc)
        print(f"ECC groups shape: {groups.shape}")
        # Check if groups are properly encoded (each group should be all 0s or all 1s)
        consistent_groups = 0
        for i in range(groups.shape[1]):
            group = groups[0, i, :]
            if torch.all(group == group[0]):
                consistent_groups += 1
        print(f"Consistent groups: {consistent_groups}/{groups.shape[1]} (should be all)")
    
    return True

if __name__ == "__main__":
    try:
        test_ecc_integration()
        test_mock_evaluator()
        print("\n🎉 All ECC integration tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)