# Training Guide 🏋️

Comprehensive guide to training binary VAE models with and without error correction.

---

## 🎯 Training Overview

### Two Model Types
1. **Baseline VAE**: Standard binary variational autoencoder
2. **ECC VAE**: With error-correcting codes for improved quality

### Key Features (Auto-Enabled)
- **Early Stopping**: Automatically stops when no improvement (patience=10)
- **Mixed Precision**: 2-3x faster training on GPU/MPS
- **Progress Monitoring**: Real-time loss tracking with tqdm
- **Smart Saving**: Only saves best models based on validation loss

---

## 🚀 Basic Training Commands

### Baseline Model
```bash
# Quick test (2-3 minutes)
python trainvae.py -n 0 --subset 500 --patience 3

# Standard training (15-30 minutes)
python trainvae.py -n 1

# Large model (45-60 minutes)
python trainvae.py -n 4
```

### ECC Model (Error Correction)
```bash
# ECC with R=2 (20-35 minutes)
python trainvae.py -n 1 --ecc-type repetition --ecc-R 2

# ECC with R=3 - stronger correction (25-40 minutes)
python trainvae.py -n 5 --ecc-type repetition --ecc-R 3
```

---

## 📊 Parameter Sets Reference

| Set | Command | Hidden | Latent | Depth | ECC R=2 | ECC R=3 | Use Case |
|-----|---------|--------|--------|-------|---------|---------|----------|
| 0 | `-n 0` | 100 | 100 | 2 | ✅ | ❌ | Quick testing |
| 1 | `-n 1` | 200 | 100 | 2 | ✅ | ❌ | **Recommended** |
| 2 | `-n 2` | 100 | 200 | 2 | ✅ | ❌ | More latent capacity |
| 3 | `-n 3` | 200 | 200 | 2 | ✅ | ❌ | Balanced large |
| 4 | `-n 4` | 200 | 200 | 3 | ✅ | ❌ | Deeper network |
| 5 | `-n 5` | 200 | 300 | 2 | ✅ | ✅ | **ECC R=3 compatible** |
| 6 | `-n 6` | 300 | 200 | 3 | ✅ | ❌ | Wide + deep |
| 7 | `-n 7` | 500 | 300 | 5 | ✅ | ✅ | Largest |

**Recommendations:**
- **Start with `-n 1`**: Best balance of quality and training time
- **For ECC R=3**: Use `-n 5` (latent=300, divisible by 3)
- **For quick tests**: Use `-n 0` with `--subset`

---

## 🛠 Training Options

### Core Parameters
```bash
# Parameter set and ECC
python trainvae.py -n 1 --ecc-type repetition --ecc-R 2

# Data subset for faster training/testing
python trainvae.py -n 1 --subset 2000

# Early stopping configuration
python trainvae.py -n 1 --patience 5 --min-delta 0.01

# Reproducibility
python trainvae.py -n 1 --seed 42
```

### Device Control
```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 1

# Force CPU (if GPU issues)
DISABLE_MPS=1 python trainvae.py -n 1

# Check device detection
python -c "from config import get_device; print(f'Device: {get_device()}')"
```

---

## 📈 Understanding Training Output

### Progress Bars
```
Epoch 1: 100%|████| 1000/1000 [02:15<00:00, 7.41batch/s, loss=2.34, kl=0.67, beta=1.0, val_loss=2.12, patience=0/10]
```

**Key metrics:**
- **loss**: Total training loss (lower = better)
- **kl**: KL divergence term (measures latent regularity)
- **beta**: KL weight (starts at 0, increases to 1)
- **val_loss**: Validation loss (determines early stopping)
- **patience**: Steps since improvement (stops at 10)

### Early Stopping
```
✅ Early stopping triggered!
Validation loss improved from 1.987 → 1.984
Best epoch: 23, Patience exceeded: 10
Model saved: weights/hidden_size_200_latent_size_100_depth_2_beta_1.0_lr_0.001/bvae_best_model_with.pt
```

---

## ⚡ Performance Optimization

### Hardware-Specific Settings

#### Apple Silicon (M1/M2/M3/M4)
```bash
# Native MPS acceleration (2-5x faster than CPU)
python trainvae.py -n 1 --ecc-type repetition --ecc-R 2

# If MPS issues, force CPU
DISABLE_MPS=1 python trainvae.py -n 1
```

#### NVIDIA GPU
```bash
# Mixed precision training (automatic)
CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 1

# Multiple GPUs (use first GPU)
CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 1
```

### Memory Optimization
```bash
# Reduce memory usage with smaller batches or models
python trainvae.py -n 0 --subset 1000  # Smaller model + data
```

### Speed vs Quality Trade-offs

| Priority | Settings | Training Time | Quality |
|----------|----------|---------------|---------|
| **Speed** | `-n 0 --subset 1000 --patience 3` | 2-3 min | Basic |
| **Balanced** | `-n 1 --subset 5000` | 10-15 min | Good |
| **Quality** | `-n 1 --ecc-type repetition --ecc-R 2` | 20-35 min | High |
| **Maximum** | `-n 5 --ecc-type repetition --ecc-R 3` | 45-60 min | Best |

---

## 🔍 ECC Training Deep Dive

### What Error Correction Does
- **Adds redundancy**: Encodes info bits K into codewords N (N > K)
- **Corrects errors**: Majority voting fixes up to ⌊(R-1)/2⌋ errors per group
- **Improves quality**: 80-90% reduction in generation errors

### ECC Parameter Compatibility
```bash
# R=2 compatible (latent size must be even)
python trainvae.py -n 0 --ecc-R 2  # ✅ 100%2=0
python trainvae.py -n 1 --ecc-R 2  # ✅ 100%2=0
python trainvae.py -n 4 --ecc-R 2  # ✅ 200%2=0

# R=3 compatible (latent size must be divisible by 3)
python trainvae.py -n 5 --ecc-R 3  # ✅ 300%3=0
python trainvae.py -n 7 --ecc-R 3  # ✅ 300%3=0

# INVALID combinations
python trainvae.py -n 0 --ecc-R 3  # ❌ 100%3≠0 (validation error)
```

### ECC Training Process
1. **Info bits**: Model learns to use K information bits
2. **Encoding**: Info bits encoded to N codewords during generation
3. **Error correction**: Majority vote decoding fixes transmission errors
4. **Consistency loss**: Additional regularization term (weight: 0.01)

---

## 📁 Output Files

### Model Files
```
weights/hidden_size_200_latent_size_100_depth_2_beta_1.0_lr_0.001/
├── bvae_best_model_with.pt          # Best model (use this for sampling)
├── loss_record_with.txt             # Training loss history
├── config.json                      # Model configuration
└── training_log.txt                 # Detailed training log
```

### Training Logs
```
# View training progress
tail -f weights/.../training_log.txt

# Plot loss curves
python -c "
import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('weights/.../loss_record_with.txt')
plt.plot(data[:, 0], label='Loss')
plt.plot(data[:, 1], label='Val Loss')
plt.legend()
plt.show()
"
```

---

## 🔧 Advanced Training Configuration

### Custom Training Parameters
```bash
# Adjust KL weight scheduling
python trainvae.py -n 1 --beta-max 1.0 --beta-warmup-epochs 10

# Modify learning rate
python trainvae.py -n 1 --lr 0.0005

# Change batch size (if memory allows)
python trainvae.py -n 1 --batch-size 2000
```

### Debugging Training
```bash
# Verbose output
python trainvae.py -n 1 --verbose

# Save every epoch (not recommended for full training)
python trainvae.py -n 1 --save-every-epoch --patience 3

# Monitor GPU usage (separate terminal)
# For NVIDIA
nvidia-smi -l 1
# For Apple Silicon  
sudo powermetrics --samplers gpu_power -n 1
```

---

## 🚨 Training Troubleshooting

### Common Issues

#### Loss Not Decreasing
```bash
# Try different learning rate
python trainvae.py -n 1 --lr 0.0001

# Use more data
python trainvae.py -n 1 --subset 0  # Use full dataset

# Different parameter set
python trainvae.py -n 2  # Try different architecture
```

#### Out of Memory
```bash
# Reduce batch size or use smaller model
python trainvae.py -n 0 --subset 1000

# Force CPU
DISABLE_MPS=1 python trainvae.py -n 1
```

#### Training Too Slow
```bash
# Check device usage
python -c "from config import get_device; print(get_device())"

# Use subset for testing
python trainvae.py -n 1 --subset 2000 --patience 3
```

#### ECC Compatibility Error
```bash
# Check parameter compatibility
python -c "
params = [(100,100,2), (200,100,2), (100,200,2), (200,200,2), (200,200,3), (200,300,2)]
for i, (h,l,d) in enumerate(params):
    r2_ok = l % 2 == 0
    r3_ok = l % 3 == 0  
    print(f'Set {i}: R=2 {\"✅\" if r2_ok else \"❌\"}, R=3 {\"✅\" if r3_ok else \"❌\"}')
"
```

---

## 📊 Training Validation

### Check Model Quality
```bash
# Quick validation after training
python sample.py -n 1 --subset 100 --w_save_path weights/your_model.pt

# Full A/B comparison
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 1000
```

### Expected Training Times

| Setup | Parameter Set | ECC | Subset | Time | Quality |
|-------|---------------|-----|---------|------|---------|
| CPU | -n 0 | No | 500 | 2-3 min | Test |
| CPU | -n 1 | No | Full | 60-90 min | Good |
| GPU/MPS | -n 1 | No | Full | 15-30 min | Good |
| GPU/MPS | -n 1 | R=2 | Full | 20-35 min | High |
| GPU/MPS | -n 5 | R=3 | Full | 25-40 min | Best |

---

## 🎯 Training Best Practices

1. **Start small**: Always test with `-n 0 --subset 500` first
2. **Use early stopping**: Default settings work well (patience=10)
3. **Monitor progress**: Watch validation loss, not just training loss
4. **Save good models**: Keep models that generalize well
5. **A/B test**: Always compare baseline vs ECC with `ab_compare_ecc.py`
6. **Reproducibility**: Use fixed seeds for consistent results
7. **Resource monitoring**: Watch memory/GPU usage during training

---

**Ready to evaluate your models?** → [Evaluation Guide](evaluation.md)

**Need to troubleshoot?** → [Troubleshooting](troubleshooting.md)