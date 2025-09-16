#!/usr/bin/env python3
"""
Standalone test script for the Factorization Machine (FM) component.
This script isolates and tests the FM training pipeline from latent vectors to property prediction.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Add parent directory to path to import from brxngenerator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brxngenerator.core.binary_vae_utils import TorchFM, FactorizationMachineSurrogate, MolData

class FMTester:
    def __init__(self, n_features=100, n_samples=1000, k_factors=10, device='cpu'):
        self.n_features = n_features
        self.n_samples = n_samples  
        self.k_factors = k_factors
        self.device = device
        
    def generate_synthetic_data(self, noise_level=0.1, random_seed=42):
        """Generate synthetic binary features and continuous targets for testing"""
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Generate binary features (0 or 1)
        X = np.random.binomial(1, 0.3, size=(self.n_samples, self.n_features)).astype(np.float32)
        
        # Generate ground truth FM parameters
        V_true = np.random.normal(0, 0.1, size=(self.n_features, self.k_factors))
        w_true = np.random.normal(0, 0.1, size=self.n_features)
        b_true = np.random.normal(0, 0.1)
        
        # Compute true FM output: bias + linear + interaction terms
        linear_term = X @ w_true
        interaction_term = 0.5 * ((X @ V_true) ** 2 - X**2 @ V_true**2).sum(axis=1)
        y_true = b_true + linear_term + interaction_term
        
        # Add noise
        y = y_true + noise_level * np.random.normal(0, 1, size=y_true.shape)
        y = y.reshape(-1, 1)
        
        print(f"Generated synthetic data:")
        print(f"  X shape: {X.shape}, y shape: {y.shape}")
        print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")
        
        return X, y, (V_true, w_true, b_true)
    
    def test_torch_fm_forward(self):
        """Test TorchFM forward pass"""
        print("\n=== Testing TorchFM Forward Pass ===")
        
        model = TorchFM(n=self.n_features, k=self.k_factors)
        X_sample = torch.randn(5, self.n_features)
        
        with torch.no_grad():
            output = model(X_sample)
        
        print(f"Input shape: {X_sample.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output values: {output.flatten()}")
        
        assert output.shape == (5, 1), f"Expected shape (5, 1), got {output.shape}"
        print("✓ TorchFM forward pass test passed")
        
    def test_torch_fm_qubo_conversion(self):
        """Test QUBO matrix generation from TorchFM"""
        print("\n=== Testing TorchFM QUBO Conversion ===")
        
        model = TorchFM(n=self.n_features, k=self.k_factors)
        
        # Initialize with known values for testing
        with torch.no_grad():
            model.factor_matrix.fill_(0.1)
            model.lin.weight.fill_(0.2)
            model.lin.bias.fill_(0.3)
        
        Q, f_E = model.to_qubo()
        
        print(f"QUBO matrix Q shape: {Q.shape}")
        print(f"Bias term f_E: {f_E}")
        print(f"Q diagonal (first 5 elements): {np.diag(Q)[:5]}")
        
        assert Q.shape == (self.n_features + 1, self.n_features + 1), f"Expected Q shape ({self.n_features + 1}, {self.n_features + 1}), got {Q.shape}"
        print("✓ QUBO conversion test passed")
        
    def test_fm_training_pipeline(self, plot_results=True):
        """Test complete FM training pipeline with synthetic data"""
        print("\n=== Testing FM Training Pipeline ===")
        
        # Generate synthetic data
        X, y, ground_truth = self.generate_synthetic_data()
        
        # Split data
        split_idx = int(0.8 * self.n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Initialize FactorizationMachineSurrogate
        fm_surrogate = FactorizationMachineSurrogate(
            n_binary=self.n_features,
            k_factors=self.k_factors,
            lr=0.001,
            decay_weight=0.001,
            batch_size=64,
            max_epoch=500,
            patience=50,
            param_init=0.1,
            cache_dir="./fm_testing",
            prop="test",
            client="synthetic",
            random_seed=42,
            device=self.device
        )
        
        # Train the model
        print("Training FM surrogate model...")
        fm_surrogate.train(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
        
        # Evaluate the trained model
        model = fm_surrogate.model
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train_tensor).cpu().numpy()
            y_pred_test = model(X_test_tensor).cpu().numpy()
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\nTraining Results:")
        print(f"  Train MSE: {train_mse:.6f}, R²: {train_r2:.6f}")
        print(f"  Test MSE: {test_mse:.6f}, R²: {test_r2:.6f}")
        
        # Plot results if requested
        if plot_results:
            self.plot_training_results(y_train, y_pred_train, y_test, y_pred_test)
        
        # Verify that the model learned something meaningful
        assert test_r2 > 0.5, f"Model performance too low: R² = {test_r2:.3f}"
        print("✓ FM training pipeline test passed")
        
        return fm_surrogate
    
    def plot_training_results(self, y_train_true, y_train_pred, y_test_true, y_test_pred):
        """Plot training results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training set
        ax1.scatter(y_train_true, y_train_pred, alpha=0.6)
        ax1.plot([y_train_true.min(), y_train_true.max()], 
                 [y_train_true.min(), y_train_true.max()], 'r--', lw=2)
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Training Set')
        ax1.grid(True)
        
        # Test set
        ax2.scatter(y_test_true, y_test_pred, alpha=0.6, color='orange')
        ax2.plot([y_test_true.min(), y_test_true.max()], 
                 [y_test_true.min(), y_test_true.max()], 'r--', lw=2)
        ax2.set_xlabel('True Values')
        ax2.set_ylabel('Predicted Values')
        ax2.set_title('Test Set')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('./fm_testing/fm_training_results.png', dpi=150)
        print("✓ Training results plot saved to ./fm_testing/fm_training_results.png")
    
    def test_binary_optimization_scenario(self):
        """Test FM in a binary optimization scenario similar to molecular optimization"""
        print("\n=== Testing Binary Optimization Scenario ===")
        
        # Generate data that mimics molecular latent vectors -> property scores
        X, y, _ = self.generate_synthetic_data(noise_level=0.05)
        
        # Make y more like QED/logP scores (bounded, positive)
        y = 1.0 / (1.0 + np.exp(-y))  # Sigmoid to [0,1] range
        
        # Split data
        split_idx = int(0.9 * self.n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train FM
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        fm_surrogate = FactorizationMachineSurrogate(
            n_binary=self.n_features, k_factors=self.k_factors, lr=0.01, 
            decay_weight=0.001, batch_size=128, max_epoch=300, patience=30,
            param_init=0.1, cache_dir="./fm_testing", prop="qed", 
            client="test", random_seed=123, device=self.device
        )
        
        fm_surrogate.train(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
        
        # Test QUBO generation for optimization
        Q, f_E = fm_surrogate.model.to_qubo()
        print(f"QUBO matrix generated: shape {Q.shape}, bias {f_E[0]:.4f}")
        
        # Simulate finding optimal binary vector
        model = fm_surrogate.model
        model.eval()
        
        # Try random binary vectors and find best predicted score
        n_candidates = 1000
        candidates = np.random.binomial(1, 0.3, size=(n_candidates, self.n_features))
        candidates_tensor = torch.FloatTensor(candidates)
        
        with torch.no_grad():
            predicted_scores = model(candidates_tensor).cpu().numpy()
        
        best_idx = np.argmax(predicted_scores)
        best_vector = candidates[best_idx]
        best_score = predicted_scores[best_idx, 0]
        
        print(f"Best predicted score: {best_score:.4f}")
        print(f"Best vector (first 10 elements): {best_vector[:10]}")
        
        # Verify that optimization found a reasonable solution
        mean_score = np.mean(predicted_scores)
        assert best_score > mean_score, f"Optimization failed: best {best_score:.3f} not better than mean {mean_score:.3f}"
        
        print("✓ Binary optimization scenario test passed")
        
    def run_all_tests(self):
        """Run all FM tests"""
        print("="*60)
        print("FM TESTING SUITE")
        print("="*60)
        
        try:
            self.test_torch_fm_forward()
            self.test_torch_fm_qubo_conversion()
            fm_surrogate = self.test_fm_training_pipeline()
            self.test_binary_optimization_scenario()
            
            print("\n" + "="*60)
            print("✓ ALL TESTS PASSED!")
            print("="*60)
            
            return fm_surrogate
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {str(e)}")
            raise

if __name__ == "__main__":
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tester
    tester = FMTester(
        n_features=100,  # Similar to latent_size//2 in actual usage
        n_samples=2000,  # Reasonable dataset size
        k_factors=20,    # Factor matrix dimension
        device=device
    )
    
    # Run tests
    trained_fm = tester.run_all_tests()