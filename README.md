# brxngenerator Instructions & User Guide

**Binary Variational Autoencoder for Molecular Synthesis Route Generation**

A consolidated Python package for generating novel molecules and synthesis routes using binary VAE with molecular optimization capabilities.

## 🚀 Overview

brxngenerator combines cutting-edge machine learning with computational chemistry to:

- **Generate novel molecular structures** using binary latent representations
- **Optimize molecular properties** (QED, logP, synthetic accessibility)
- **Design synthesis routes** with template-based reaction planning
- **Evaluate generation quality** with comprehensive metrics

### Key Features

- ✨ **Binary VAE Architecture**: Discrete latent space for improved molecular generation
- 🧪 **Chemistry Integration**: RDKit-based molecular validation and property computation
- 🎯 **Property Optimization**: Gurobi-based QUBO optimization for molecular properties
- 📊 **Comprehensive Metrics**: MOSES-compatible evaluation with novelty, uniqueness, and SA scoring
- 🖥️ **Multi-Device Support**: CUDA, MPS (Apple Silicon), and CPU compatibility
- ⚡ **Optimized Training**: Mixed precision, early stopping, and progress tracking

## 📦 Installation

### Prerequisites

```bash
# Python 3.8+ required
conda create -n brxngenerator python=3.8
conda activate brxngenerator
```

### Core Dependencies

```bash
# Essential packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install rdkit numpy scipy scikit-learn tqdm

# Optimization (optional but recommended)
pip install gurobi-optimods

# Visualization (optional)
pip install matplotlib seaborn pandas
```

### Project Setup

```bash
git clone <repository-url>
cd brxngenerator
pip install -e .
```

### Gurobi License (for optimization)

Place your `gurobi.lic` file in the project root for molecular property optimization features.

## 🏗️ Project Structure

```
brxngenerator/
├── brxngenerator/              # Core package
│   ├── chemistry/              # Chemical utilities
│   │   ├── chemistry_core.py   # Consolidated chemistry utilities
│   │   ├── fragments/          # Fragment processing
│   │   └── reactions/          # Reaction handling
│   ├── core/                   # VAE implementation
│   │   ├── vae.py             # Binary VAE models
│   │   └── binary_vae_utils.py # Training utilities
│   ├── models/                 # Neural architectures
│   │   └── models.py          # Encoders, decoders, networks
│   ├── metrics/                # Evaluation metrics
│   │   └── metrics.py         # Molecular and latent metrics
│   ├── utils/                  # Core utilities
│   │   └── core.py            # Config and device management
│   └── optimization/           # Property optimization
├── data/                       # Training data
├── weights/                    # Model checkpoints
├── trainvae.py                # Training script
├── sample.py                  # Sampling script
├── mainstream.py              # Optimization pipeline
└── README.md                  # Project overview
```

## 🚀 Quick Start Guide

### Step 1: Prepare Your Data

Place your molecular reaction data in the `data/` directory:
```bash
# Expected format: data/data.txt
# Each line should contain reaction SMILES or molecular data
```

### Step 2: Train a Binary VAE Model

```bash
# Basic training with recommended settings
CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 1

# For different model sizes:
python trainvae.py -n 0  # Small model (100,100,2)
python trainvae.py -n 1  # Recommended (200,100,2)
python trainvae.py -n 4  # Larger latent (200,200,2)
python trainvae.py -n 7  # Largest model (500,300,5)
```

**Parameter Sets Available:**
- Set 0: `(100,100,2)` - Small/fast training
- Set 1: `(200,100,2)` - **Recommended balance**
- Set 4: `(200,200,2)` - Larger latent space
- Set 5: `(200,300,2)` - Large latent space
- Set 7: `(500,300,5)` - Largest model

### Step 3: Generate New Molecules

```bash
# Generate molecules using your trained model
CUDA_VISIBLE_DEVICES=0 python sample.py -n 1 \
    --w_save_path weights/bvae_best_model_with.pt \
    --subset 500
```

### Step 4: Optimize Molecular Properties

```bash
# Run property optimization pipeline
python mainstream.py --seed 1
```

## 📊 Understanding the Metrics

The project provides **5 standardized evaluation metrics**:

### 1. Validity (0.0 - 1.0, higher better)
- Fraction of chemically valid generated molecules
- Uses RDKit sanitization and validation

### 2. Uniqueness (0.0 - 1.0, higher better)
- Fraction of unique molecules among valid ones
- Based on canonical SMILES deduplication

### 3. Novelty (0.0 - 1.0, higher better)
- Fraction of molecules not in training set
- Measures true generative capability vs. memorization

### 4. Average QED (0.0 - 1.0, higher better)
- Quantitative Estimate of Drug-likeness
- Values >0.67 considered drug-like

### 5. Average SA Score (1.0 - 10.0, lower better)
- Synthetic Accessibility Score
- 1-3: easy to synthesize, 6+: difficult

## 🔧 Configuration Guide

### Training Configuration

Key training parameters you can modify:

```python
# In trainvae.py or via command line
batch_size = 1000       # Larger batches work well with GPU
patience = 10           # Early stopping patience
learning_rate = 0.001   # Learning rate
beta = 1.0             # KL divergence weight
```

### Device Configuration

The system automatically detects the best device:

```bash
# Force CPU usage
export DISABLE_MPS=1

# Specify GPU
export CUDA_VISIBLE_DEVICES=0

# Check device detection
python -c "from brxngenerator import get_device; print(get_device())"
```

### Memory Optimization

For limited GPU memory:
```bash
# Use smaller model
python trainvae.py -n 0

# Reduce dataset size
python trainvae.py -n 1 --subset 1000

# Reduce batch size (edit in script)
```

## 🧪 API Usage Examples

### Basic Model Usage

```python
from brxngenerator import bFTRXNVAE, get_device
import torch

# Setup
device = get_device()

# Load vocabularies (implement based on your data)
# fragment_vocab, reactant_vocab, template_vocab = load_vocabularies()

# Initialize model
model = bFTRXNVAE(
    fragment_vocab=fragment_vocab,
    reactant_vocab=reactant_vocab,
    template_vocab=template_vocab,
    hidden_size=200,
    latent_size=100,
    depth=2,
    device=device
).to(device)

# Load trained weights
checkpoint = torch.load('weights/bvae_best_model_with.pt', map_location=device)
model.load_state_dict(checkpoint)
```

### Molecule Generation

```python
from brxngenerator import Evaluator

# Initialize evaluator
evaluator = Evaluator(latent_size=100, model=model)

# Generate molecules
ft_latent = evaluator.generate_discrete_latent(50)  # Half latent size
rxn_latent = evaluator.generate_discrete_latent(50)

for product, reaction in evaluator.decode_from_prior(ft_latent, rxn_latent, n=10):
    if product:
        print(f"Generated molecule: {product}")
```

### Evaluation and Metrics

```python
from brxngenerator import compute_molecular_metrics

# Evaluate generated molecules
generated_smiles = ['CCO', 'CC(=O)O', 'c1ccccc1', ...]
training_smiles = ['CCO', 'CC(C)O', ...]  # Your training set

metrics = compute_molecular_metrics(
    generated_smiles=generated_smiles,
    training_smiles=training_smiles
)

print(f"Validity: {metrics['validity']:.3f}")
print(f"Uniqueness: {metrics['uniqueness']:.3f}")
print(f"Novelty: {metrics['novelty']:.3f}")
print(f"Average QED: {metrics['avg_qed']:.3f}")
print(f"Average SA: {metrics['avg_sa']:.3f}")
```

### Custom Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Setup training
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()  # For mixed precision

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        with autocast():  # Mixed precision
            loss, *other_losses = model(batch, beta=1.0, temp=0.4)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
```

## 🛠️ Command Line Reference

### Training Commands

```bash
# Basic training
python trainvae.py -n 1

# With options
python trainvae.py -n 1 --subset 5000 --patience 15

# Quick test (small dataset, early stopping)
python trainvae.py -n 0 --subset 500 --patience 3
```

### Sampling Commands

```bash
# Basic sampling
python sample.py -n 1 --w_save_path weights/bvae_best_model_with.pt

# Limited sampling for testing
python sample.py -n 1 --w_save_path weights/bvae_best_model_with.pt --subset 100
```

### Optimization Commands

```bash
# Property optimization
python mainstream.py --seed 42

# Multiple runs with different seeds
for seed in {1..5}; do
    python mainstream.py --seed $seed
done
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```bash
# Solution 1: Use smaller model
python trainvae.py -n 0

# Solution 2: Reduce data size
python trainvae.py -n 1 --subset 1000

# Solution 3: Use CPU
export CUDA_VISIBLE_DEVICES=""
```

#### 2. MPS Issues (Apple Silicon)
```bash
# Disable MPS acceleration
export DISABLE_MPS=1
python trainvae.py -n 1
```

#### 3. Import Errors
```bash
# Ensure proper installation
pip install -e .

# Check Python path
python -c "import brxngenerator; print('Installation OK')"
```

#### 4. RDKit Issues
```bash
# Reinstall RDKit
conda install -c conda-forge rdkit

# Verify installation
python -c "from rdkit import Chem; print('RDKit OK')"
```

#### 5. Gurobi License Issues
```bash
# Check license file
ls gurobi.lic

# Set environment variable
export GRB_LICENSE_FILE=/path/to/gurobi.lic

# Test Gurobi
python -c "import gurobipy; print('Gurobi OK')"
```

### Performance Optimization Tips

1. **Use GPU**: 10-50x speedup over CPU
2. **Larger Batches**: Better GPU utilization (1000-3000)
3. **Mixed Precision**: Automatic on GPU, faster training
4. **Early Stopping**: Prevents overfitting, saves time
5. **Parameter Sets**: Start with Set 1, scale up as needed

### Debugging Training

```python
# Monitor training progress
# Check generated_reactions.txt for sample outputs
# Watch loss curves in terminal output
# Monitor GPU usage: nvidia-smi -l 1

# Validation checks
assert model.latent_size == 100  # Check model configuration
assert len(data_pairs) > 0       # Ensure data loaded
assert torch.cuda.is_available() # GPU availability
```

## 📈 Best Practices

### 1. Data Preparation
- Ensure reaction data is clean and properly formatted
- Use subset for initial testing: `--subset 1000`
- Validate data loading before full training

### 2. Model Selection
- Start with parameter set 1 (balanced performance)
- Use set 0 for quick prototyping
- Scale to larger sets (4, 7) for production

### 3. Training Strategy
- Always use early stopping (patience=10)
- Monitor validation loss trends
- Save multiple checkpoints for comparison

### 4. Evaluation Protocol
- Generate at least 1000 molecules for reliable metrics
- Include training set for novelty computation
- Compare multiple model configurations

### 5. Production Deployment
- Use GPU for best performance
- Set up proper logging and monitoring
- Implement proper error handling

## 📚 References and Further Reading

- **MOSES Benchmark**: [Polykovskiy et al., Frontiers in Pharmacology (2020)](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.565644/full)
- **QED Drug-likeness**: [Bickerton et al., Nature Chemistry (2012)](https://pubmed.ncbi.nlm.nih.gov/22270643/)
- **Synthetic Accessibility**: [Ertl & Schuffenhauer, J. Cheminformatics (2009)](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8)
- **RDKit Documentation**: [https://www.rdkit.org/docs/](https://www.rdkit.org/docs/)
- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)

## 🤝 Support and Contributing

### Getting Help
1. Check this guide for common solutions
2. Review error messages carefully
3. Test with smaller datasets first
4. Check device compatibility

### Contributing Guidelines
- Maintain the consolidated architecture
- Update imports when adding new modules
- Include proper documentation
- Test on multiple devices when possible

### Architecture Notes
This project uses a **consolidated architecture** for maintainability:
- `chemistry/chemistry_core.py` - All chemistry utilities
- `models/models.py` - All neural network components
- `metrics/metrics.py` - All evaluation metrics
- `utils/core.py` - Configuration and device management

When extending the project, follow this consolidation pattern.

---

**Happy molecular generation! 🧬✨**

For additional support, please refer to the consolidated module documentation within each file.