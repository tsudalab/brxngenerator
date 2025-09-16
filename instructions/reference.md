# Command Reference 📖

Complete reference for all commands and parameters.

---

## 🏋️ Training Commands

### trainvae.py - Main Training Script

**Basic Usage:**
```bash
python trainvae.py -n PARAM_SET [OPTIONS]
```

**Core Parameters:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-n` | int | 0 | Parameter set index (0-7) |
| `--ecc-type` | str | `"none"` | ECC type: `none` or `repetition` |
| `--ecc-R` | int | 3 | ECC repetition factor (2 or 3) |
| `--subset` | int | None | Limit dataset size for testing |
| `--patience` | int | 10 | Early stopping patience |
| `--min-delta` | float | 0.0 | Minimum improvement threshold |
| `--seed` | int | None | Random seed for reproducibility |

**Examples:**
```bash
# Quick test
python trainvae.py -n 0 --subset 500 --patience 3

# Standard training
python trainvae.py -n 1

# ECC training  
python trainvae.py -n 1 --ecc-type repetition --ecc-R 2

# Reproducible training
python trainvae.py -n 1 --seed 42 --patience 15
```

**Parameter Set Reference:**
| Set | Hidden | Latent | Depth | ECC R=2 | ECC R=3 | Use Case |
|-----|--------|--------|-------|---------|---------|----------|
| 0 | 100 | 100 | 2 | ✅ | ❌ | Quick testing |
| 1 | 200 | 100 | 2 | ✅ | ❌ | **Recommended** |
| 2 | 100 | 200 | 2 | ✅ | ❌ | More latent capacity |
| 3 | 200 | 200 | 2 | ✅ | ❌ | Balanced large |
| 4 | 200 | 200 | 3 | ✅ | ❌ | Deeper network |
| 5 | 200 | 300 | 2 | ✅ | ✅ | **ECC R=3 compatible** |
| 6 | 300 | 200 | 3 | ✅ | ❌ | Wide + deep |
| 7 | 500 | 300 | 5 | ✅ | ✅ | Largest |

---

## 📊 Evaluation Commands

### ab_compare_ecc.py - A/B Model Comparison

**Basic Usage:**
```bash
python ab_compare_ecc.py -n PARAM_SET --ecc-R R [OPTIONS]
```

**Core Parameters:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-n` | int | 1 | Parameter set index |
| `--ecc-R` | int | Required | ECC repetition factor |
| `--train-subset` | int | 0 | Training molecules for comparison (0=all) |
| `--eval-subset` | int | 2000 | Molecules to generate and evaluate |
| `--latent-metrics` | bool | false | Enable BER/WER/calibration metrics |
| `--iwae-samples` | int | 64 | Importance weighted bounds samples |
| `--noise-epsilon` | float | None | Test noise robustness |
| `--seed` | int | None | Random seed |
| `--gpu` | int | None | GPU device ID |

**Examples:**
```bash
# Basic A/B comparison
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 2000

# Full comparison with latent metrics
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 5000 \
  --latent-metrics true --iwae-samples 64

# Noise robustness test
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 2000 \
  --noise-epsilon 0.05 --latent-metrics true

# Multi-seed evaluation
for seed in 42 43 44; do
  python ab_compare_ecc.py -n 1 --ecc-R 2 --seed $seed --eval-subset 1000
done
```

**Output Files:**
- `results/compare_n{N}_{timestamp}.json` - Detailed results
- `results/compare_n{N}_{timestamp}.csv` - Summary table
- Console output with side-by-side comparison

---

### sample.py - Molecule Generation

**Basic Usage:**
```bash
python sample.py -n PARAM_SET [OPTIONS]
```

**Core Parameters:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-n` | int | 1 | Parameter set index |
| `--w_save_path` | str | Auto | Model weight file path |
| `--subset` | int | None | Number of molecules to generate |
| `--ecc-type` | str | `"none"` | ECC type for generation |
| `--ecc-R` | int | 3 | ECC repetition factor |
| `--temperature` | float | 1.0 | Sampling temperature |
| `--seed` | int | None | Random seed |

**Examples:**
```bash
# Basic sampling
python sample.py -n 1 --subset 1000

# Sample with specific model
python sample.py -n 1 --w_save_path weights/best_model.pt --subset 500

# ECC sampling
python sample.py -n 1 --ecc-type repetition --ecc-R 2 --subset 1000

# Low-temperature sampling (more conservative)
python sample.py -n 1 --temperature 0.8 --subset 500
```

---

## 🎯 Optimization Commands

### mainstream.py - Property Optimization

**Basic Usage:**
```bash
python mainstream.py [OPTIONS]
```

**Core Parameters:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--seed` | int | 1 | Random seed |
| `--ecc-type` | str | `"none"` | ECC type |
| `--ecc-R` | int | 3 | ECC repetition factor |
| `--properties` | str | `"qed,logp,sas"` | Properties to optimize |
| `--weights` | str | `"0.4,0.3,0.3"` | Property weights |
| `--n-molecules` | int | 1000 | Number of molecules to optimize |
| `--n-iterations` | int | 100 | Optimization iterations |

**Examples:**
```bash
# Basic optimization
python mainstream.py --seed 1

# ECC optimization
python mainstream.py --seed 1 --ecc-type repetition --ecc-R 2

# Custom property weights
python mainstream.py --properties qed,logp,sas --weights 0.5,0.3,0.2 --seed 42

# Large-scale optimization
python mainstream.py --n-molecules 5000 --n-iterations 500 --seed 1
```

---

## 🔧 Utility Commands

### Device Detection
```bash
# Check current device
python -c "from config import get_device; print(f'Device: {get_device()}')"

# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check MPS availability (Apple Silicon)
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### MPS Testing
```bash
# Test MPS functionality
python mps_test.py

# Expected output: 🚀 MPS detected! Using Apple Silicon acceleration.
```

### Data Validation
```bash
# Check data loading
python -c "
from rxnft_vae.reaction_utils import read_multistep_rxns
data = read_multistep_rxns('data/data.txt')
print(f'✅ Loaded {len(data)} reactions')
"

# Verify data preprocessing
python -c "
from binary_vae_utils import prepare_dataset
dataset = prepare_dataset('data/data.txt', subset=100)
print(f'✅ Preprocessed {len(dataset)} samples')
"
```

---

## 🌐 Environment Variables

### Device Control
| Variable | Effect | Example |
|----------|--------|---------|
| `DISABLE_MPS` | Force disable MPS | `DISABLE_MPS=1 python trainvae.py -n 1` |
| `CUDA_VISIBLE_DEVICES` | Select GPU device | `CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 1` |
| `DISABLE_AMP` | Disable mixed precision | `DISABLE_AMP=1 python trainvae.py -n 1` |

### Debugging
| Variable | Effect | Example |
|----------|--------|---------|
| `PYTHONHASHSEED` | Fixed random seed | `PYTHONHASHSEED=42 python trainvae.py -n 1` |
| `CUDA_LAUNCH_BLOCKING` | Synchronous CUDA ops | `CUDA_LAUNCH_BLOCKING=1 python trainvae.py -n 1` |

### Examples:
```bash
# Force CPU training
DISABLE_MPS=1 CUDA_VISIBLE_DEVICES="" python trainvae.py -n 1

# Use specific GPU with debugging
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python trainvae.py -n 1

# Reproducible training
PYTHONHASHSEED=42 python trainvae.py -n 1 --seed 42
```

---

## 📊 Metrics Reference

### Molecular Quality Metrics (5 Core)

| Metric | Range | Higher/Lower | Good Score | Description |
|--------|-------|---------------|------------|-------------|
| **Valid Rate** | 0.0-1.0 | Higher ↑ | > 0.8 | Chemically valid molecules |
| **QED** | 0.0-1.0 | Higher ↑ | > 0.67 | Drug-likeness score |
| **Uniqueness** | 0.0-1.0 | Higher ↑ | > 0.85 | Diversity of molecules |
| **Novelty** | 0.0-1.0 | Higher ↑ | > 0.75 | New vs training molecules |
| **SAS** | 1.0-10.0 | Lower ↓ | < 4.0 | Synthetic accessibility |

### Latent Quality Metrics (4 Advanced)

| Metric | Range | Higher/Lower | ECC Effect | Description |
|--------|-------|---------------|------------|-------------|
| **BER** | 0.0-1.0 | Lower ↓ | 80-90% reduction | Bit error rate |
| **WER** | 0.0-1.0 | Lower ↓ | 90-95% reduction | Word error rate |
| **ECE** | 0.0-1.0 | Lower ↓ | Better calibration | Calibration error |
| **Entropy** | 0.0-1.0 | Context | Often lower | Bit-wise uncertainty |

---

## 🗂 File Paths & Naming

### Model Weight Files
```
weights/
├── hidden_size_{H}_latent_size_{L}_depth_{D}_beta_1.0_lr_0.001/
│   ├── bvae_best_model_with.pt          # ← Use this for sampling
│   ├── loss_record_with.txt             # Training history
│   └── config.json                      # Model configuration
```

### Result Files
```
results/
├── compare_n{N}_{YYYYMMDD}_{HHMMSS}.json    # A/B detailed results
├── compare_n{N}_{YYYYMMDD}_{HHMMSS}.csv     # A/B summary table
├── sample_n{N}_molecules.json               # Generated molecules
└── sample_n{N}_metrics.json                 # Quality metrics
```

### Data Files
```
data/
├── data.txt                    # Main training data
├── synthetic_routes.txt        # Reaction routes
├── logP_values.txt            # Property values
├── SA_scores.txt              # Accessibility scores
└── cycle_scores.txt           # Cycle scores
```

---

## ⚙️ Configuration Options

### Parameter Set Details
Each parameter set defines the model architecture:
- **Hidden Size**: Encoder/decoder hidden layer dimensions
- **Latent Size**: Binary latent vector length  
- **Depth**: Number of layers in encoder/decoder
- **Beta**: KL divergence weight (fixed at 1.0)
- **Learning Rate**: Adam optimizer rate (fixed at 0.001)

### ECC Compatibility Rules
- **R=2**: Latent size must be even (divisible by 2)
- **R=3**: Latent size must be divisible by 3
- **Info bits**: K = Latent_size / R (information capacity)
- **Codewords**: N = Latent_size (total bits including redundancy)

### Training Defaults
- **Batch Size**: Auto-determined based on parameter set and device
- **Early Stopping**: Patience=10, min_delta=0.0
- **Mixed Precision**: Auto-enabled on GPU/MPS
- **Progress Bars**: Always enabled with tqdm
- **Model Saving**: Only best validation model saved

---

## 🚀 Quick Command Recipes

### Development Workflow
```bash
# 1. Quick test (2-3 minutes)
python trainvae.py -n 0 --subset 500 --patience 3

# 2. Standard training (15-30 minutes)
python trainvae.py -n 1

# 3. ECC training (20-35 minutes)
python trainvae.py -n 1 --ecc-type repetition --ecc-R 2

# 4. A/B comparison (10-15 minutes)
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 2000

# 5. Generate samples (2-5 minutes)
python sample.py -n 1 --subset 1000

# 6. Property optimization (5-10 minutes)
python mainstream.py --seed 1 --ecc-type repetition --ecc-R 2
```

### Research Workflow
```bash
# Multi-seed A/B comparison for statistical significance
for seed in 42 43 44 45 46; do
  python ab_compare_ecc.py -n 1 --ecc-R 2 --seed $seed --eval-subset 2000
done

# ECC parameter sweep
for R in 2 3; do
  for n in 1 5; do
    if python -c "params={1:100,5:300}; exit(0 if params[$n] % $R == 0 else 1)"; then
      python trainvae.py -n $n --ecc-type repetition --ecc-R $R --subset 2000
      python ab_compare_ecc.py -n $n --ecc-R $R --eval-subset 1000
    fi
  done
done

# Comprehensive evaluation
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 5000 \
  --latent-metrics true --iwae-samples 128 --noise-epsilon 0.05
```

### Production Workflow
```bash
# Full training with validation
python trainvae.py -n 1 --ecc-type repetition --ecc-R 2 --seed 42

# Validation with large evaluation set
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 10000 --seed 42

# Large-scale generation
python sample.py -n 1 --ecc-type repetition --ecc-R 2 --subset 50000

# Property optimization at scale
python mainstream.py --n-molecules 10000 --n-iterations 1000 \
  --ecc-type repetition --ecc-R 2 --seed 42
```

---

## 📋 Command Validation

### Before Training
```bash
# Check system readiness
python -c "
import torch
from config import get_device
print(f'Device: {get_device()}')
print(f'Data exists: {open(\"data/data.txt\").readline().strip()[:50]}...')
print('✅ Ready for training')
"
```

### Before Evaluation
```bash
# Check model exists
ls weights/*/bvae_best_model_with.pt

# Verify parameter compatibility
python -c "
n, R = 1, 2  # Your parameters
latent_sizes = {0:100, 1:100, 2:200, 3:200, 4:200, 5:300, 6:200, 7:300}
compatible = latent_sizes[n] % R == 0
print(f'✅ Compatible' if compatible else f'❌ Incompatible: {latent_sizes[n]} % {R} = {latent_sizes[n] % R}')
"
```

### Command Syntax Validation
```bash
# Test command syntax without execution
python trainvae.py --help
python ab_compare_ecc.py --help
python sample.py --help
python mainstream.py --help
```

---

**Need help with specific issues?** → [Troubleshooting](troubleshooting.md)

**Want to understand the project?** → [Getting Started](getting-started.md)