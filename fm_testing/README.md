# Factorization Machine (FM) Testing Directory

This directory contains standalone testing utilities for the Factorization Machine component of the brxngenerator project.

## Overview

The Factorization Machine (FM) is a key component in the molecular optimization pipeline that learns a surrogate model to predict molecular properties from binary latent vectors. The FM training pipeline consists of:

1. **TorchFM Model**: Neural FM implementation with factor matrix and linear terms
2. **FactorizationMachineSurrogate**: Training wrapper with early stopping and validation
3. **QUBO Conversion**: Converts trained FM to Quadratic Unconstrained Binary Optimization problem

## Key Files

- `test_fm_standalone.py`: Comprehensive test suite for FM components
- `README.md`: This documentation file

## FM Architecture

From the analysis of `brxngenerator/core/binary_vae_utils.py`:

### TorchFM Model (lines 22-44)
```python
class TorchFM(nn.Module):
    def __init__(self, n=None, k=None):
        self.factor_matrix = nn.Parameter(torch.randn(n, k), requires_grad=True)
        self.lin = nn.Linear(n, 1)

    def forward(self, x):
        # Interaction terms: 0.5 * ((xV)² - x²V²)
        out_1 = torch.matmul(x, self.factor_matrix).pow(2).sum(1, keepdim=True)
        out_2 = torch.matmul(x.pow(2), self.factor_matrix.pow(2)).sum(1, keepdim=True)
        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.lin(x)
        return out_inter + out_lin
```

**Key Components:**
- `factor_matrix`: (n × k) matrix for capturing feature interactions
- `lin`: Linear layer for first-order feature effects
- **Forward pass**: Computes linear + interaction terms efficiently

### FactorizationMachineSurrogate (lines 94-157)
**Training Pipeline:**
- Data loading with `MolData` dataset class
- Adam optimizer with weight decay
- Early stopping based on validation loss
- Model checkpointing in cache directory

**Key Parameters:**
- `n_binary`: Input feature dimension (latent_size // 2 in practice)
- `k_factors`: Factor matrix dimension (controls interaction complexity)
- `lr`, `decay_weight`: Learning rate and regularization
- `patience`: Early stopping patience

### QUBO Conversion (lines 36-44)
Converts trained FM to optimization problem:
```
Q[i,j] = V[i,:] @ V[j,:]  # Interaction matrix
Q[i,i] = W[i]             # Linear coefficients on diagonal
```

## Usage in Main Pipeline

From `mainstream.py` analysis:

### 1. Dataset Preparation (lines 81-84)
```python
X_train, y_train, X_test, y_test = binary_vae_utils.prepare_dataset(
    model=model, data_pairs=data_pairs, latent_size=config.LATENT_SIZE, 
    metric=config.METRIC, logp_paths=logp_paths, ecc_type=ecc_type, ecc_R=ecc_R
)
```

- Encodes molecular structures to latent vectors using B-VAE
- Extracts first half of latent vector (latent_size // 2)
- Computes property scores (QED, logP) for supervision
- Handles ECC decoding if enabled (reduces dimension further)

### 2. FM Training (lines 93-98)
```python
fm_surrogate = binary_vae_utils.FactorizationMachineSurrogate(
    n_binary=n_features, k_factors=config.FACTOR_NUM, lr=config.LR,
    # ... other parameters
)
```

### 3. Optimization Loop (lines 172-180)
```python
def optimize(self):
    for i in range(self.optimize_num):
        self.surrogate.train(self.X_train, self.y_train, self.X_test, self.y_test)
        solutions, _ = self.solver.solve(self.surrogate.model)
        self._update(solutions)  # Add new molecules to dataset
```

**Key Insight**: The FM is retrained in each iteration with augmented data from newly discovered molecules.

## Running Tests

```bash
# Navigate to project root
cd /Users/guanjiezou/CliecyPro/brxngenerator-2

# Run FM standalone tests
python fm_testing/test_fm_standalone.py
```

## Test Coverage

The test suite covers:

1. **Forward Pass Testing**: Verifies TorchFM computation
2. **QUBO Conversion**: Tests optimization matrix generation  
3. **Training Pipeline**: End-to-end training with synthetic data
4. **Binary Optimization**: Simulates molecular optimization scenario
5. **Performance Validation**: Ensures R² > 0.5 on test data

## Expected Output

```
=== Testing TorchFM Forward Pass ===
✓ TorchFM forward pass test passed

=== Testing TorchFM QUBO Conversion ===
✓ QUBO conversion test passed

=== Testing FM Training Pipeline ===
Training FM surrogate model...
Model -- Epoch 0 error on validation set: 0.8234, r2 on validation set: 0.2145
...
Model -- Early stopping at epoch 167. Best epoch was 117 with error 0.0234.
✓ FM training pipeline test passed

=== Testing Binary Optimization Scenario ===
✓ Binary optimization scenario test passed

============================================================
✓ ALL TESTS PASSED!
============================================================
```

## Key Insights

1. **Dimensionality**: FM operates on `latent_size // 2` features (50 for latent_size=100)
2. **ECC Integration**: With ECC enabled, dimension reduces further (e.g., 50→25 with R=2)
3. **Iterative Learning**: FM retrains with augmented datasets in optimization loop
4. **Property Prediction**: Maps binary latent vectors to molecular property scores
5. **QUBO Optimization**: Converts trained model to binary optimization problem

This testing framework allows isolated development and debugging of the FM component without requiring the full molecular pipeline.