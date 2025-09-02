# Getting Started 🚀

Complete setup guide for brxngenerator molecular generation.

---

## 📦 Installation

### 1. Check Python Version
```bash
python --version  # Should be 3.12+
```

### 2. Install Dependencies
```bash
# Option 1: Using uv (recommended - fastest)
uv sync

# Option 2: Using pip
pip install -r requirements.txt

# Option 3: Manual install
pip install torch torchvision torchaudio rdkit gurobi-optimods numpy matplotlib tqdm scikit-learn jupyter pyyaml pylint
```

### 3. Verify Installation
```bash
python -c "import torch, rdkit; print('✅ Core dependencies OK')"
```

---

## 🖥 Device Setup

### GPU (NVIDIA)
```bash
# Check GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If False, install GPU PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Apple Silicon (M1/M2/M3/M4)
```bash
# Check MPS availability  
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Test MPS acceleration
python mps_test.py
```
**Expected output:** `🚀 MPS detected! Using Apple Silicon acceleration.`

### CPU Only
No special setup needed - everything works on CPU (just slower).

---

## 🧪 First Test Run

### 5-Minute Quick Test
```bash
# Train a tiny model to verify everything works
python trainvae.py -n 0 --subset 500 --patience 3
```

**Expected output:**
```
🚀 Using device: cuda/mps/cpu
Epoch 1: 100%|████| loss=2.34, val_loss=2.12, patience=0/3
Epoch 2: 100%|████| loss=2.01, val_loss=1.98, patience=0/3
...
✅ Early stopping: val_loss improved from 1.98 → 1.95
Model saved: weights/...bvae_best_model_with.pt
```

**If successful:** You're ready to go! 🎉

**If errors:** Check [troubleshooting.md](troubleshooting.md)

---

## 🎯 Your First Real Model

### Basic Training (15-30 minutes)
```bash
# Train baseline model
python trainvae.py -n 1
```

### With Error Correction (20-35 minutes)
```bash  
# Train improved model with ECC
python trainvae.py -n 1 --ecc-type repetition --ecc-R 2
```

**Monitor progress:**
- Progress bars show real-time loss values
- Training auto-stops when no improvement
- Models saved to `weights/` directory

---

## 📊 Verify Quality

### A/B Comparison Test
```bash
# Compare baseline vs ECC (10-15 minutes)
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 2000
```

**Expected results:**
- **BER/WER reduction**: ECC should show 80-90% lower error rates
- **Molecule quality**: Higher Valid%, QED, Uniqueness, Novelty
- **Output files**: `results/compare_n1_timestamp.json/.csv`

### Generate Sample Molecules
```bash
# Create 1000 new molecules
python sample.py -n 1 --subset 1000
```

---

## 🔧 Configuration Files

### Parameter Sets (Choose One)
```bash
# Quick testing
python trainvae.py -n 0    # (100,100,2) - smallest

# General use (recommended)  
python trainvae.py -n 1    # (200,100,2) - balanced

# ECC compatible
python trainvae.py -n 5    # (200,300,2) - supports R=3
```

### ECC Options
```bash
# No error correction (baseline)
--ecc-type none

# Simple error correction (most stable)
--ecc-type repetition --ecc-R 2

# Stronger error correction (requires -n 5)
--ecc-type repetition --ecc-R 3
```

---

## 🗂 Understanding Output Files

### Model Files (weights/ directory)
```
weights/
├── hidden_size_200_latent_size_100_depth_2_beta_1.0_lr_0.001/
│   ├── bvae_best_model_with.pt          # Best model
│   ├── loss_record_with.txt             # Training history
│   └── config.json                      # Model configuration
```

### Results Files
```
results/
├── compare_n1_20240902_143055.json      # Detailed A/B results  
├── compare_n1_20240902_143055.csv       # Summary table
└── sample_molecules_n1.json             # Generated molecules
```

---

## 🎛 Important Environment Variables

```bash
# Force CPU (if GPU/MPS problems)
DISABLE_MPS=1 python trainvae.py -n 1

# Control CUDA device
CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 1

# Set random seed for reproducibility  
export PYTHONHASHSEED=42
```

---

## 💡 Pro Tips

### Speed Optimization
- **Use GPU/MPS**: 2-5x faster than CPU
- **Increase batch size**: If you have memory (check training output)
- **Use subset for testing**: `--subset 1000` for quick experiments

### Memory Management
```bash
# If out of memory, reduce batch size or use smaller parameter set
python trainvae.py -n 0 --subset 1000  # Smaller model + data
```

### Reproducibility  
```bash
# For reproducible results, always use same seed
python trainvae.py -n 1 --seed 42
python ab_compare_ecc.py -n 1 --ecc-R 2 --seed 42
```

---

## 🧪 Advanced Testing

### Full System Test
```bash
# Complete workflow test (30-45 minutes)
python trainvae.py -n 1 --subset 2000 --patience 5      # Train
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 1000  # Evaluate  
python sample.py -n 1 --subset 500                      # Generate
python mainstream.py --seed 1                           # Optimize
```

### Performance Benchmarking
```bash
# Time baseline training
time python trainvae.py -n 1 --subset 1000 --patience 2

# Time ECC training
time python trainvae.py -n 1 --ecc-type repetition --ecc-R 2 --subset 1000 --patience 2
```

---

## ❗ Common First-Time Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'rdkit'` | Run `pip install rdkit` |
| `CUDA out of memory` | Use `--subset 1000` or smaller `-n` parameter |
| `MPS not available` | Update macOS or use `DISABLE_MPS=1` |
| Training seems stuck | Normal - first epoch is slowest |
| No improvement after many epochs | Use different parameter set or check data |

---

## ✅ Success Checklist

- [ ] Dependencies installed without errors
- [ ] Device detection working (CUDA/MPS/CPU)
- [ ] Quick test completes successfully  
- [ ] First real model trains to completion
- [ ] A/B comparison shows ECC improvements
- [ ] Generated molecules look reasonable

**All checked?** You're ready for [Training Guide →](training.md)

**Having issues?** Check [Troubleshooting →](troubleshooting.md)