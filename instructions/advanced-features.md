# Advanced Features ⚡

Comprehensive guide to advanced functionality, optimization, and customization.

---

## 🎯 Molecular Property Optimization

### Basic Property Optimization
```bash
# Optimize for drug-likeness and synthetic accessibility
python mainstream.py --seed 1

# With ECC optimization
python mainstream.py --seed 1 --ecc-type repetition --ecc-R 2
```

### Advanced Optimization
```bash
# Multi-property optimization
python mainstream.py --properties qed,logp,sas --weights 0.4,0.3,0.3 --seed 42

# Custom optimization targets
python mainstream.py --qed-target 0.8 --logp-range "[-1,3]" --sas-max 4.0

# Large-scale optimization
python mainstream.py --n-molecules 10000 --n-iterations 500 --seed 1
```

### Gurobi QUBO Solver
```bash
# Ensure Gurobi license is available
ls gurobi.lic  # Should exist in project root

# Test QUBO solver
python -c "
from gurobi_optimods import solve_qubo
import numpy as np
Q = np.random.randn(10, 10)
Q = Q @ Q.T  # Make positive semi-definite
result = solve_qubo(Q)
print('✅ Gurobi QUBO solver working')
"
```

---

## 🖥 Device Configuration & Acceleration

### Apple Silicon (MPS) Setup
```bash
# Check MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Test MPS functionality
python mps_test.py
```

**Expected output:**
```
🍎 Apple Silicon detected!
🚀 MPS detected! Using Apple Silicon acceleration.
Testing basic tensor operations... ✅
Testing neural network operations... ✅
All MPS tests passed!
```

### MPS Optimization
```bash
# Native Apple Silicon training (2-5x speedup)
python trainvae.py -n 1 --ecc-type repetition --ecc-R 2

# Monitor MPS usage (separate terminal)
sudo powermetrics --samplers gpu_power -n 1

# Force CPU if MPS issues
DISABLE_MPS=1 python trainvae.py -n 1
```

### NVIDIA GPU Setup
```bash
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"

# Multi-GPU training (use specific GPU)
CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 1
CUDA_VISIBLE_DEVICES=1 python trainvae.py -n 2

# Monitor GPU usage
nvidia-smi -l 1  # Update every second
```

### Mixed Precision Training
Automatically enabled on GPU/MPS for 2-3x speedup:
```bash
# Mixed precision automatically used when available
python trainvae.py -n 1  # Uses AMP if GPU/MPS detected

# Force disable mixed precision (if issues)
DISABLE_AMP=1 python trainvae.py -n 1
```

---

## ⚙️ Advanced Training Configuration

### Custom Parameter Sets
```python
# Create custom parameter configuration
# Edit config.py or create custom_config.py

CUSTOM_PARAMS = {
    'hidden_size': 256,
    'latent_size': 128,  # Must be divisible by ECC R
    'depth': 3,
    'beta': 1.0,
    'lr': 0.0008,
    'batch_size': 1500
}

# Use custom parameters
python trainvae.py --config custom_config.py
```

### Advanced Early Stopping
```bash
# Fine-tune early stopping
python trainvae.py -n 1 --patience 15 --min-delta 0.005 --restore-best-weights

# Validation-based stopping with custom metrics
python trainvae.py -n 1 --validation-metric reconstruction_loss --patience 8
```

### Learning Rate Scheduling
```bash
# Exponential decay
python trainvae.py -n 1 --lr-scheduler exp --lr-decay 0.95

# Step scheduling  
python trainvae.py -n 1 --lr-scheduler step --lr-step-size 50 --lr-gamma 0.8

# Cosine annealing
python trainvae.py -n 1 --lr-scheduler cosine --lr-max-epochs 100
```

---

## 🧪 Research & Experimental Features

### Latent Space Analysis
```bash
# Deep latent metrics with IWAE bounds
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 5000 \
  --latent-metrics true --iwae-samples 128

# Noise robustness testing
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 2000 \
  --noise-epsilon 0.01 --noise-epsilon 0.05 --noise-epsilon 0.1

# Calibration analysis
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 3000 \
  --calibration-bins 20 --confidence-thresholds "0.8,0.9,0.95"
```

### Advanced ECC Configurations
```bash
# Test different ECC strengths
for R in 2 3 5; do
  if python -c "exit(0 if 300 % $R == 0 else 1)"; then
    python trainvae.py -n 5 --ecc-type repetition --ecc-R $R --subset 1000
  fi
done

# ECC parameter sweep
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 1000 --ecc-sweep-R "2,3"
```

### Custom Evaluation Metrics
```python
# Add custom molecular property evaluator
# Create custom_metrics.py

def custom_lipinski_violations(mol):
    """Count Lipinski rule violations"""
    violations = 0
    if mol.GetDescriptor('MolWt') > 500: violations += 1
    if mol.GetDescriptor('MolLogP') > 5: violations += 1
    if mol.GetDescriptor('NumHDonors') > 5: violations += 1
    if mol.GetDescriptor('NumHAcceptors') > 10: violations += 1
    return violations

# Use in evaluation
python sample.py -n 1 --custom-metrics custom_metrics.py --subset 1000
```

---

## 📊 Advanced Analysis & Visualization

### Training Analysis
```python
# Analyze training curves
import matplotlib.pyplot as plt
import numpy as np

# Load training history
data = np.loadtxt('weights/.../loss_record_with.txt')
epochs = range(len(data))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].plot(epochs, data[:, 0], label='Training Loss')
axes[0,0].plot(epochs, data[:, 1], label='Validation Loss')  
axes[0,0].set_title('Loss Curves')
axes[0,0].legend()

axes[0,1].plot(epochs, data[:, 2], label='KL Divergence')
axes[0,1].set_title('KL Divergence')

axes[1,0].plot(epochs, data[:, 3], label='Beta Schedule')
axes[1,0].set_title('Beta Annealing')

axes[1,1].plot(epochs, data[:, 4], label='Learning Rate')
axes[1,1].set_title('Learning Rate Schedule')

plt.tight_layout()
plt.show()
```

### Latent Space Visualization
```python
# Visualize latent representations
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Generate latent codes
python -c "
from rxnft_vae.evaluate import Evaluator
evaluator = Evaluator(...)
latents, molecules = evaluator.sample_latent_molecules(1000)
np.save('latent_codes.npy', latents)
"

# Visualize with t-SNE
latents = np.load('latent_codes.npy')
tsne = TSNE(n_components=2, random_state=42)
latent_2d = tsne.fit_transform(latents)

plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6)
plt.title('Latent Space Visualization')
plt.show()
```

---

## 🔧 Performance Optimization

### Memory Optimization
```bash
# Reduce memory usage
python trainvae.py -n 0 --batch-size 500 --subset 2000

# Gradient accumulation for large effective batch sizes
python trainvae.py -n 1 --batch-size 800 --accumulation-steps 4  # Effective: 3200

# Memory-efficient evaluation
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-batch-size 100 --eval-subset 1000
```

### Distributed Training (Experimental)
```bash
# Multi-GPU data parallel (if multiple GPUs)
python -m torch.distributed.launch --nproc_per_node=2 trainvae.py -n 1 --distributed

# Model parallel for very large models
python trainvae.py -n 7 --model-parallel --split-layers "encoder,decoder"
```

### Profiling & Debugging
```bash
# Profile training performance
python -m cProfile -o training_profile.prof trainvae.py -n 1 --subset 1000 --patience 2

# Analyze profile
python -c "
import pstats
p = pstats.Stats('training_profile.prof')
p.sort_stats('cumulative').print_stats(20)
"

# Memory profiling
python -m memory_profiler trainvae.py -n 0 --subset 500 --patience 2
```

---

## 🔬 Custom Extensions

### Custom Loss Functions
```python
# Create custom_losses.py
import torch
import torch.nn as nn

class WeightedReconstructionLoss(nn.Module):
    def __init__(self, atom_weights=None):
        super().__init__()
        self.atom_weights = atom_weights or {'C': 1.0, 'N': 1.5, 'O': 1.2}
    
    def forward(self, pred, target, molecule_info):
        # Implement custom weighting based on atom types
        base_loss = nn.CrossEntropyLoss()(pred, target)
        # Apply custom weighting logic
        return weighted_loss

# Use custom loss
python trainvae.py -n 1 --custom-loss custom_losses.WeightedReconstructionLoss
```

### Custom Evaluation Pipeline
```python
# Create custom_eval.py
def custom_evaluation_pipeline(model, data_loader, device):
    """Custom evaluation with domain-specific metrics"""
    results = {}
    
    # Generate molecules
    molecules = []
    for batch in data_loader:
        generated = model.sample(batch_size=100)
        molecules.extend(generated)
    
    # Compute custom metrics
    results['bioactivity_score'] = compute_bioactivity(molecules)
    results['synthetic_feasibility'] = compute_feasibility(molecules)
    results['patent_novelty'] = check_patent_space(molecules)
    
    return results

# Use custom evaluation
python ab_compare_ecc.py -n 1 --ecc-R 2 --custom-eval custom_eval.py
```

---

## 🌐 Integration & Export

### Export Models
```bash
# Export to ONNX format
python -c "
import torch
from rxnft_vae.vae import bFTRXNVAE
model = bFTRXNVAE.load('weights/best_model.pt')
dummy_input = torch.randn(1, 100)
torch.onnx.export(model, dummy_input, 'model.onnx')
"

# Export to TorchScript
python -c "
import torch
from rxnft_vae.vae import bFTRXNVAE
model = bFTRXNVAE.load('weights/best_model.pt')
scripted = torch.jit.script(model)
scripted.save('model_scripted.pt')
"
```

### REST API Server
```python
# Create api_server.py
from flask import Flask, request, jsonify
from rxnft_vae.evaluate import Evaluator

app = Flask(__name__)
evaluator = Evaluator.load('weights/best_model.pt')

@app.route('/generate', methods=['POST'])
def generate_molecules():
    data = request.json
    n_molecules = data.get('n_molecules', 100)
    
    molecules = evaluator.sample_molecules(n_molecules)
    metrics = evaluator.evaluate_molecules(molecules)
    
    return jsonify({
        'molecules': molecules,
        'metrics': metrics
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# Run API server
python api_server.py
```

### Database Integration
```python
# Save results to database
import sqlite3
import json

def save_results_to_db(results, db_path='results.db'):
    conn = sqlite3.connect(db_path)
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            config TEXT,
            metrics TEXT
        )
    ''')
    
    conn.execute('''
        INSERT INTO experiments (timestamp, config, metrics)
        VALUES (?, ?, ?)
    ''', (
        results['timestamp'],
        json.dumps(results['config']),
        json.dumps(results['metrics'])
    ))
    
    conn.commit()
    conn.close()

# Use in A/B comparison
python ab_compare_ecc.py -n 1 --ecc-R 2 --save-to-db results.db
```

---

## 🚨 Advanced Troubleshooting

### Debug Model Loading Issues
```bash
# Check model file integrity
python -c "
import torch
try:
    model_data = torch.load('weights/your_model.pt', map_location='cpu')
    print('✅ Model file loads successfully')
    print('Keys:', list(model_data.keys()))
except Exception as e:
    print('❌ Model loading failed:', e)
"

# Convert between devices
python -c "
import torch
model = torch.load('weights/your_model.pt', map_location='cpu')
torch.save(model, 'weights/your_model_cpu.pt')
"
```

### Performance Debugging
```bash
# Check bottlenecks
python -m torch.profiler trainvae.py -n 0 --subset 200 --patience 1

# Memory leak detection  
python -m objgraph trainvae.py -n 0 --subset 200 --patience 2

# CUDA debugging (if GPU issues)
CUDA_LAUNCH_BLOCKING=1 python trainvae.py -n 1 --subset 500
```

### Data Pipeline Debugging
```bash
# Verify data loading
python -c "
from rxnft_vae.reaction_utils import read_multistep_rxns
data = read_multistep_rxns('data/data.txt')
print(f'✅ Loaded {len(data)} reactions')
print('Sample:', data[0] if data else 'No data')
"

# Check data preprocessing
python -c "
from binary_vae_utils import prepare_dataset
dataset = prepare_dataset('data/data.txt', subset=100)
print(f'✅ Preprocessed {len(dataset)} samples')
"
```

---

## 📋 Advanced Usage Checklist

### Research Setup
- [ ] Multi-seed evaluation for statistical significance
- [ ] Hyperparameter sweeps with different configurations
- [ ] Ablation studies (ECC R values, architectures)
- [ ] Noise robustness testing
- [ ] Custom metrics implementation
- [ ] Results database integration

### Production Setup  
- [ ] Model export to deployment format (ONNX/TorchScript)
- [ ] REST API server implementation
- [ ] Batch generation pipelines
- [ ] Monitoring and logging
- [ ] Error handling and recovery
- [ ] Performance optimization

### Development Setup
- [ ] Custom loss functions and metrics
- [ ] Extended evaluation pipelines  
- [ ] Visualization and analysis tools
- [ ] Automated testing and validation
- [ ] Documentation and reproducibility
- [ ] Version control and model tracking

---

**Need help with specific issues?** → [Troubleshooting](troubleshooting.md)

**Want complete command reference?** → [Command Reference](reference.md)