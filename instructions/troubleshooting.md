# Troubleshooting Guide 🚨

Quick solutions for common issues and problems.

---

## 🆘 Quick Fixes

### "It's not working!"
1. **Check the error message** - It usually tells you exactly what's wrong
2. **Try the quick test**: `python trainvae.py -n 0 --subset 500 --patience 3`  
3. **Check device detection**: `python -c "from config import get_device; print(get_device())"`
4. **Use CPU as fallback**: `DISABLE_MPS=1 python trainvae.py -n 1`

### "Training is too slow!"
1. **Check your device**: Should see `🚀 MPS detected!` or `CUDA available: True`
2. **Use smaller subset**: `--subset 1000` for testing
3. **Reduce patience**: `--patience 3` for quicker experiments
4. **Monitor device usage**: `nvidia-smi` (GPU) or Activity Monitor (Mac)

---

## 🐛 Installation Issues

### Missing Dependencies
```bash
# Error: ModuleNotFoundError: No module named 'rdkit'
pip install rdkit

# Error: No module named 'torch'  
pip install torch torchvision torchaudio

# Error: No module named 'gurobi_optimods'
pip install gurobi-optimods

# Install everything at once
pip install torch rdkit gurobi-optimods numpy matplotlib tqdm scikit-learn jupyter pyyaml pylint
```

### Python Version Issues
```bash
# Check Python version (need 3.12+)
python --version

# If too old, install newer Python
# macOS: brew install python@3.12
# Linux: apt install python3.12
# Windows: Download from python.org
```

### Gurobi License Issues
```bash
# Check if license file exists
ls gurobi.lic  # Should be in project root

# Test Gurobi solver
python -c "
try:
    from gurobi_optimods import solve_qubo
    import numpy as np
    Q = np.eye(3)
    result = solve_qubo(Q)
    print('✅ Gurobi working')
except Exception as e:
    print('❌ Gurobi issue:', e)
"
```

---

## 💻 Device & Hardware Issues

### Apple Silicon (MPS) Issues

#### MPS Not Detected
```bash
# Check MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# If False, check:
# 1. Are you on M1/M2/M3/M4 Mac?
# 2. Is macOS 12.3+ installed?
# 3. Is PyTorch recent (2.0+)?

# Update PyTorch if needed
pip install --upgrade torch torchvision torchaudio
```

#### MPS Errors During Training
```bash
# Common MPS error: "Placeholder storage not allocated"
# Already fixed in mps_fix.py, but if still happening:

# Force CPU instead
DISABLE_MPS=1 python trainvae.py -n 1

# Or test MPS functionality
python mps_test.py
```

#### MPS Performance Issues
```bash
# Monitor MPS usage
sudo powermetrics --samplers gpu_power -n 1

# Check memory pressure
# Activity Monitor → GPU History

# Reduce memory usage
python trainvae.py -n 0 --subset 1000  # Smaller model + data
```

### NVIDIA GPU Issues

#### CUDA Not Available
```bash
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check NVIDIA driver
nvidia-smi
```

#### GPU Out of Memory
```bash
# Error: "CUDA out of memory"

# Solutions:
# 1. Use smaller model
python trainvae.py -n 0 --subset 1000

# 2. Reduce batch size (edit config.py or use subset)
python trainvae.py -n 1 --subset 2000

# 3. Use CPU instead
CUDA_VISIBLE_DEVICES="" python trainvae.py -n 1

# 4. Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### Multiple GPUs Issues
```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 1

# Check GPU usage
nvidia-smi -l 1

# If multi-GPU issues, use single GPU
export CUDA_VISIBLE_DEVICES=0
```

---

## 🧠 Training Issues

### Training Not Starting
```bash
# Error: Dataset loading issues
# Check data file exists
ls data/data.txt

# Test data loading
python -c "
from rxnft_vae.reaction_utils import read_multistep_rxns
data = read_multistep_rxns('data/data.txt')
print(f'✅ Loaded {len(data)} reactions')
"
```

### Loss Not Decreasing
```bash
# Check training progress
tail -f weights/.../loss_record_with.txt

# Try different learning rate
python trainvae.py -n 1 --lr 0.0001

# Use more data
python trainvae.py -n 1 --subset 0  # Full dataset

# Try different parameter set
python trainvae.py -n 2
```

### Early Stopping Too Early
```bash
# Increase patience
python trainvae.py -n 1 --patience 20

# Reduce min-delta
python trainvae.py -n 1 --min-delta 0.0

# Check if validation loss actually improving
cat weights/.../loss_record_with.txt | awk '{print $2}' | tail -20
```

### Training Crashes
```bash
# Enable debugging
python -u trainvae.py -n 1 --verbose 2>&1 | tee training.log

# Check memory usage
free -h  # Linux
vm_stat  # macOS

# Use smaller configuration
python trainvae.py -n 0 --subset 500 --patience 3
```

---

## ⚙️ ECC-Specific Issues

### ECC Parameter Compatibility
```bash
# Error: "Latent size not divisible by R"

# Check compatibility
python -c "
params = {0: 100, 1: 100, 2: 200, 3: 200, 4: 200, 5: 300, 6: 200, 7: 300}
for n, latent in params.items():
    r2 = '✅' if latent % 2 == 0 else '❌'
    r3 = '✅' if latent % 3 == 0 else '❌'
    print(f'Set {n}: R=2 {r2}, R=3 {r3}')
"

# Use compatible combinations:
python trainvae.py -n 1 --ecc-R 2  # ✅
python trainvae.py -n 5 --ecc-R 3  # ✅  
python trainvae.py -n 1 --ecc-R 3  # ❌ Won't work
```

### ECC Not Improving Quality
```bash
# Verify ECC is actually enabled
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 100 --verbose

# Check BER/WER reduction (should be 80%+)
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 1000

# Try stronger ECC
python trainvae.py -n 5 --ecc-type repetition --ecc-R 3
```

### ECC Training Slower Than Expected
```bash
# ECC adds ~10-20% training overhead, not 2x
# If much slower, check:

# 1. Device acceleration enabled?
python -c "from config import get_device; print(get_device())"

# 2. Mixed precision working?  
# Should see AMP-related messages in training

# 3. Using appropriate parameter set?
python trainvae.py -n 1 --ecc-R 2  # Good balance
```

---

## 🔬 Evaluation Issues

### A/B Comparison Failures
```bash
# Error in ab_compare_ecc.py

# 1. Check both models can be loaded
python -c "
import torch
try:
    baseline = torch.load('weights/baseline_model.pt', map_location='cpu')
    print('✅ Baseline model OK')
except Exception as e:
    print('❌ Baseline model issue:', e)
"

# 2. Use smaller evaluation set
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 100

# 3. Check parameter compatibility  
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 500 --verbose
```

### Poor Generation Quality
```bash
# Low valid rates (< 0.5)

# 1. Check model training convergence
cat weights/.../loss_record_with.txt | tail -10

# 2. Try different sampling parameters
python sample.py -n 1 --temperature 0.8 --subset 500

# 3. Use different decoding
python sample.py -n 1 --decoding-strategy greedy --subset 500

# 4. Check if model overfitted
python sample.py -n 1 --subset 100 --eval-training-reconstruction
```

### Inconsistent Results
```bash
# Results vary between runs

# 1. Fix random seeds
python ab_compare_ecc.py -n 1 --ecc-R 2 --seed 42 --eval-subset 1000

# 2. Use larger evaluation set
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 5000

# 3. Multiple runs for confidence
for seed in 42 43 44; do
    python ab_compare_ecc.py -n 1 --ecc-R 2 --seed $seed --eval-subset 1000
done
```

---

## 🗂 File & Data Issues

### Missing Weight Files
```bash
# Error: Model weight file not found

# 1. Check if training completed
ls -la weights/

# 2. Find the correct path
find weights/ -name "*.pt" | head -5

# 3. Use correct path in evaluation
python sample.py -n 1 --w_save_path weights/actual_path/bvae_best_model_with.pt
```

### Data Loading Issues
```bash
# Error: Cannot load data.txt

# 1. Check file exists and has content
ls -la data/data.txt
head -5 data/data.txt

# 2. Test data format
python -c "
from rxnft_vae.reaction_utils import read_multistep_rxns
try:
    data = read_multistep_rxns('data/data.txt')[:5]
    print(f'✅ Sample data: {data}')
except Exception as e:
    print(f'❌ Data loading error: {e}')
"

# 3. Re-download or regenerate data if corrupted
```

### Permission Issues
```bash
# Error: Permission denied writing to weights/

# 1. Check write permissions
ls -la weights/

# 2. Fix permissions
chmod -R u+w weights/

# 3. Create directory if missing
mkdir -p weights/
```

---

## ⚡ Performance Issues

### Training Very Slow
```bash
# Expected times:
# CPU: 60-90 min for full training
# GPU/MPS: 15-30 min for full training
# Subset (1000): 2-5 min

# If much slower:

# 1. Check device
python -c "from config import get_device; print(get_device())"

# 2. Monitor resource usage
top  # CPU usage
nvidia-smi  # GPU usage (NVIDIA)
# Activity Monitor → GPU (Mac)

# 3. Use profiling
python -m cProfile -o profile.out trainvae.py -n 0 --subset 500 --patience 2
```

### Memory Issues
```bash
# Error: Out of memory

# 1. Monitor memory usage
free -h  # Linux
vm_stat  # macOS

# 2. Reduce memory usage
python trainvae.py -n 0 --subset 1000  # Smaller everything

# 3. Close other applications
# 4. Use swap file (Linux) or increase virtual memory
```

### Disk Space Issues
```bash
# Error: No space left on device

# 1. Check space
df -h

# 2. Clean up old models
rm weights/*/intermediate_models_*.pt

# 3. Use subset training to reduce output
python trainvae.py -n 1 --subset 2000 --no-save-intermediate
```

---

## 🔍 Debugging Commands

### Comprehensive System Check
```bash
# Run this to check everything
python -c "
import sys
print('Python version:', sys.version)

import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)

try:
    import rdkit
    print('RDKit version:', rdkit.__version__)
except:
    print('RDKit: Not installed')

try:
    from gurobi_optimods import solve_qubo
    print('Gurobi: Available')
except:
    print('Gurobi: Not available')

from config import get_device
print('Device detected:', get_device())

import os
print('Data file exists:', os.path.exists('data/data.txt'))
print('Weights dir exists:', os.path.exists('weights/'))
"
```

### Minimal Working Example
```bash
# This should always work (2-3 minutes)
python trainvae.py -n 0 --subset 100 --patience 2

# If this fails, there's a fundamental issue
```

### Clean Reset
```bash
# Start fresh if everything is broken

# 1. Remove all model weights
rm -rf weights/*/

# 2. Clear Python cache
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete

# 3. Reinstall dependencies
pip uninstall torch rdkit gurobi-optimods -y
pip install torch torchvision torchaudio rdkit gurobi-optimods numpy matplotlib tqdm scikit-learn

# 4. Test minimal setup
python trainvae.py -n 0 --subset 100 --patience 2
```

---

## 📞 Getting Help

### Before Asking for Help

1. **Run the system check** (see debugging commands above)
2. **Try the minimal working example**
3. **Check the error message carefully** - it usually tells you what's wrong
4. **Try CPU fallback**: `DISABLE_MPS=1 python trainvae.py -n 1`
5. **Use smaller test**: `--subset 500 --patience 3`

### What to Include When Reporting Issues

```bash
# Run this and include output:
echo "=== System Info ==="
python -c "
import sys, torch, platform
print('OS:', platform.system(), platform.release())
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if hasattr(torch.backends, 'mps'):
    print('MPS available:', torch.backends.mps.is_available())
"

echo "=== Device Detection ==="
python -c "from config import get_device; print('Device:', get_device())"

echo "=== Data Check ==="
ls -la data/data.txt
python -c "
try:
    from rxnft_vae.reaction_utils import read_multistep_rxns
    data = read_multistep_rxns('data/data.txt')
    print('Data loaded:', len(data), 'reactions')
except Exception as e:
    print('Data loading error:', e)
"

echo "=== Error Command ==="
# Include the exact command that failed and the full error message
```

---

## ✅ Common Solutions Summary

| Problem | Quick Fix | Full Solution |
|---------|-----------|---------------|
| **ModuleNotFoundError** | `pip install rdkit` | Install all dependencies |
| **CUDA/MPS not available** | `DISABLE_MPS=1 ...` | Install proper PyTorch version |
| **Out of memory** | `--subset 1000` | Use smaller model or more RAM |
| **Training too slow** | Check device detection | Use GPU/MPS acceleration |
| **ECC compatibility** | Use `-n 1 --ecc-R 2` | Check parameter set table |
| **Low generation quality** | Check model convergence | Longer training, better data |
| **Files not found** | Check paths with `ls` | Ensure training completed |
| **Permission denied** | `chmod -R u+w weights/` | Fix directory permissions |

**Still stuck?** Try the minimal working example: `python trainvae.py -n 0 --subset 100 --patience 2`

---

**Need complete command reference?** → [Command Reference](reference.md)

**Want to understand the project better?** → [Getting Started](getting-started.md)