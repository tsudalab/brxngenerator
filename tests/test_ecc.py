# [ECC] Unit tests for Error-Correcting Code module
"""
Unit tests for ECC functionality in rxnft_vae/ecc.py
"""

import torch
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rxnft_vae.ecc import RepetitionECC, create_ecc_codec, sample_ecc_latent, extract_info_bits


class TestRepetitionECC:
    """Test RepetitionECC class functionality."""
    
    def test_init_valid_R(self):
        """Test initialization with valid R values."""
        ecc = RepetitionECC(R=2)
        assert ecc.R == 2
        
        ecc = RepetitionECC(R=5)
        assert ecc.R == 5
    
    def test_init_invalid_R(self):
        """Test initialization with invalid R values."""
        with pytest.raises(AssertionError):
            RepetitionECC(R=1)
        
        with pytest.raises(AssertionError):
            RepetitionECC(R=0)
    
    def test_encode_basic(self):
        """Test basic encoding functionality."""
        ecc = RepetitionECC(R=3)
        
        # Test single sample
        info_bits = torch.tensor([[1, 0, 1]], dtype=torch.float32)  # (1, 3)
        encoded = ecc.encode(info_bits)
        
        expected = torch.tensor([[1, 1, 1, 0, 0, 0, 1, 1, 1]], dtype=torch.float32)  # (1, 9)
        assert torch.equal(encoded, expected)
        assert encoded.shape == (1, 9)
    
    def test_encode_batch(self):
        """Test encoding with batch dimension."""
        ecc = RepetitionECC(R=2)
        
        # Test batch
        info_bits = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)  # (2, 2)
        encoded = ecc.encode(info_bits)
        
        expected = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.float32)  # (2, 4)
        assert torch.equal(encoded, expected)
        assert encoded.shape == (2, 4)
    
    def test_decode_no_errors(self):
        """Test decoding with no errors."""
        ecc = RepetitionECC(R=3)
        
        # Perfect codeword
        code_bits = torch.tensor([[1, 1, 1, 0, 0, 0, 1, 1, 1]], dtype=torch.float32)  # (1, 9)
        decoded = ecc.decode(code_bits)
        
        expected = torch.tensor([[1, 0, 1]], dtype=torch.float32)  # (1, 3)
        assert torch.equal(decoded, expected)
    
    def test_decode_with_correctable_errors(self):
        """Test decoding with correctable errors (≤⌊(R-1)/2⌋ per group)."""
        ecc = RepetitionECC(R=3)  # Can correct 1 error per group
        
        # Original: [1, 0, 1] -> [1,1,1, 0,0,0, 1,1,1]
        # With 1 error per group: [1,1,0, 0,0,1, 1,0,1]
        code_bits = torch.tensor([[1, 1, 0, 0, 0, 1, 1, 0, 1]], dtype=torch.float32)
        decoded = ecc.decode(code_bits)
        
        expected = torch.tensor([[1, 0, 1]], dtype=torch.float32)
        assert torch.equal(decoded, expected)
    
    def test_decode_R2_correctable(self):
        """Test R=2 can't correct any errors but handles ties."""
        ecc = RepetitionECC(R=2)
        
        # No errors: [1,1, 0,0] -> [1, 0]
        code_bits = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32)
        decoded = ecc.decode(code_bits)
        expected = torch.tensor([[1, 0]], dtype=torch.float32)
        assert torch.equal(decoded, expected)
        
        # With errors (ties go to 1): [1,0, 0,1] -> [1, 1] (ties broken to 1)
        code_bits = torch.tensor([[1, 0, 0, 1]], dtype=torch.float32)
        decoded = ecc.decode(code_bits)
        expected = torch.tensor([[1, 1]], dtype=torch.float32)  # Both groups sum to 1, threshold is 1
        assert torch.equal(decoded, expected)
    
    def test_encode_decode_roundtrip(self):
        """Test encode->decode roundtrip preserves information."""
        ecc = RepetitionECC(R=3)
        
        # Random info bits
        torch.manual_seed(42)
        info_bits = torch.bernoulli(torch.full((5, 8), 0.5))  # (5, 8)
        
        # Roundtrip
        encoded = ecc.encode(info_bits)
        decoded = ecc.decode(encoded)
        
        assert torch.equal(decoded, info_bits)
        assert encoded.shape == (5, 24)  # 8 * 3
        assert decoded.shape == (5, 8)
    
    def test_group_shape_ok(self):
        """Test group shape validation."""
        ecc = RepetitionECC(R=3)
        
        assert ecc.group_shape_ok(9)   # 9 % 3 == 0
        assert ecc.group_shape_ok(12)  # 12 % 3 == 0
        assert not ecc.group_shape_ok(10)  # 10 % 3 != 0
        assert not ecc.group_shape_ok(8)   # 8 % 3 != 0
    
    def test_size_conversions(self):
        """Test size conversion utilities."""
        ecc = RepetitionECC(R=3)
        
        assert ecc.get_info_size(9) == 3
        assert ecc.get_info_size(12) == 4
        assert ecc.get_code_size(3) == 9
        assert ecc.get_code_size(4) == 12
        
        with pytest.raises(AssertionError):
            ecc.get_info_size(10)  # Not divisible by 3
    
    def test_to_from_groups(self):
        """Test group reshaping utilities."""
        ecc = RepetitionECC(R=3)
        
        x = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.float32)  # (1, 6)
        groups = ecc.to_groups(x)
        assert groups.shape == (1, 2, 3)  # (B, K, R)
        
        reconstructed = ecc.from_groups(groups)
        assert torch.equal(reconstructed, x)
        assert reconstructed.shape == (1, 6)


class TestECCFactory:
    """Test ECC factory functions."""
    
    def test_create_ecc_codec_none(self):
        """Test creating no ECC codec."""
        codec = create_ecc_codec('none')
        assert codec is None
    
    def test_create_ecc_codec_repetition(self):
        """Test creating repetition ECC codec."""
        codec = create_ecc_codec('repetition', R=3)
        assert isinstance(codec, RepetitionECC)
        assert codec.R == 3
        
        # Test default R
        codec_default = create_ecc_codec('repetition')
        assert codec_default.R == 3
    
    def test_create_ecc_codec_invalid(self):
        """Test creating codec with invalid type."""
        with pytest.raises(ValueError):
            create_ecc_codec('invalid_type')


class TestECCUtilities:
    """Test ECC utility functions."""
    
    def test_sample_ecc_latent_no_ecc(self):
        """Test sampling without ECC."""
        torch.manual_seed(42)
        latent = sample_ecc_latent(info_size=4, batch_size=2, ecc_codec=None)
        
        assert latent.shape == (2, 4)
        assert latent.dtype == torch.float32
        assert torch.all((latent == 0) | (latent == 1))  # Binary values
    
    def test_sample_ecc_latent_with_ecc(self):
        """Test sampling with ECC."""
        torch.manual_seed(42)
        ecc = RepetitionECC(R=3)
        latent = sample_ecc_latent(info_size=4, batch_size=2, ecc_codec=ecc)
        
        assert latent.shape == (2, 12)  # 4 * 3
        assert latent.dtype == torch.float32
        assert torch.all((latent == 0) | (latent == 1))
        
        # Verify it's properly encoded (groups should be consistent)
        groups = ecc.to_groups(latent)
        for b in range(2):
            for k in range(4):
                group = groups[b, k, :]
                # Each group should be all 0s or all 1s (no noise added yet)
                assert torch.all(group == group[0])
    
    def test_extract_info_bits_no_ecc(self):
        """Test extracting info bits without ECC."""
        latent = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
        info_bits = extract_info_bits(latent, ecc_codec=None)
        
        assert torch.equal(info_bits, latent)
    
    def test_extract_info_bits_with_ecc(self):
        """Test extracting info bits with ECC."""
        ecc = RepetitionECC(R=3)
        
        # Encoded latent: [1,1,1, 0,0,0] represents info [1, 0]
        latent = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.float32)
        info_bits = extract_info_bits(latent, ecc_codec=ecc)
        
        expected = torch.tensor([[1, 0]], dtype=torch.float32)
        assert torch.equal(info_bits, expected)


class TestErrorCorrection:
    """Test error correction capabilities."""
    
    def test_single_bit_error_correction_R3(self):
        """Test single bit error correction with R=3."""
        ecc = RepetitionECC(R=3)
        
        # Original info: [1, 0]
        # Encoded: [1,1,1, 0,0,0]
        # With errors: [1,1,0, 0,1,0] (1 error per group)
        corrupted = torch.tensor([[1, 1, 0, 0, 1, 0]], dtype=torch.float32)
        corrected = ecc.decode(corrupted)
        
        expected = torch.tensor([[1, 0]], dtype=torch.float32)
        assert torch.equal(corrected, expected)
    
    def test_multiple_samples_error_correction(self):
        """Test error correction on multiple samples."""
        ecc = RepetitionECC(R=3)
        
        # Batch of corrupted codewords
        # Original: [[1,0], [0,1]] -> [[1,1,1,0,0,0], [0,0,0,1,1,1]]
        # Corrupted: [[1,0,1,0,1,0], [0,1,0,1,0,1]] (1 error per group)
        corrupted = torch.tensor([
            [1, 0, 1, 0, 1, 0],  # Should decode to [1, 0]
            [0, 1, 0, 1, 0, 1]   # Should decode to [0, 1]
        ], dtype=torch.float32)
        
        corrected = ecc.decode(corrupted)
        expected = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
        assert torch.equal(corrected, expected)


if __name__ == "__main__":
    # Run basic smoke test
    print("Running ECC smoke tests...")
    
    # Test basic functionality
    ecc = RepetitionECC(R=3)
    print(f"Created RepetitionECC with R={ecc.R}")
    
    # Test encode/decode
    info = torch.tensor([[1, 0, 1]], dtype=torch.float32)
    encoded = ecc.encode(info)
    decoded = ecc.decode(encoded)
    print(f"Original: {info}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Roundtrip successful: {torch.equal(info, decoded)}")
    
    # Test error correction
    # Introduce 1 error per group
    corrupted = encoded.clone()
    corrupted[0, 2] = 1 - corrupted[0, 2]  # Flip bit in first group
    corrupted[0, 5] = 1 - corrupted[0, 5]  # Flip bit in second group
    corrupted[0, 7] = 1 - corrupted[0, 7]  # Flip bit in third group
    
    corrected = ecc.decode(corrupted)
    print(f"Corrupted: {corrupted}")
    print(f"Corrected: {corrected}")
    print(f"Error correction successful: {torch.equal(info, corrected)}")
    
    print("ECC smoke tests completed successfully!")