# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**brxngenerator** is a binary Variational Autoencoder (B-VAE) implementation for molecular synthesis route generation and optimization. The project combines:

- **Binary VAE**: Neural architecture with binary latent space for molecular fragment encoding
- **Molecular Optimization**: Gurobi-based QUBO optimization for molecular properties (QED, logP)

## Core Architecture

### Key Components
1. **B-VAE Model** (`brxngenerator/core/vae.py`): `bFTRXNVAE` with binary latent vectors
2. **Optimization Pipeline** (`mainstream.py`): Factorization Machine + Gurobi QUBO solver
3. **Training Interface** (`trainvae.py`): Modern training with early stopping

## Full-Data Training (Server, 1×GPU, AMP, Early Stop ON by Default)

### Standard Training
```bash
# B-VAE training (auto early stopping, tqdm, AMP)
CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 1    # Parameter set 1: hidden=200, latent=100, depth=2

# Available parameter sets (0-7):
# Set 0: (100,100,2)
# Set 1: (200,100,2) - recommended
# Set 4: (200,200,2)
# Set 5: (200,300,2)
# Set 7: (500,300,5) - largest
```

### Features (Auto-Enabled)
- **Early Stopping**: Default ON (patience=10), saves only best model `bvae_best_model_with.pt`
- **Mixed Precision**: Automatic AMP on GPU with GradScaler for faster training
- **Progress Bars**: Real-time tqdm showing loss, KL, beta, validation loss, patience
- **GPU Optimization**: Pin memory, CUDNN benchmarking, optimal num_workers

## Standardized Evaluation Metrics

The project implements **5 standardized metrics** for molecular generation evaluation, ensuring consistency with MOSES benchmarking and research reproducibility:

### Core Metrics

1. **Valid Reaction Rate** - Fraction of generated reactions that pass RDKit parsing and validation
   - **Purpose**: Measures chemical reaction validity and parser compatibility
   - **Computation**: Uses `rdChemReactions.ReactionFromSmarts/SMILES` with sanitization checks
   - **Range**: [0.0, 1.0], higher is better

2. **Average QED** - Drug-likeness score based on molecular descriptors
   - **Purpose**: Quantitative Estimate of Drug-likeness for pharmaceutical relevance
   - **Reference**: [Bickerton et al., Nature Chemistry (2012)](https://pubmed.ncbi.nlm.nih.gov/22270643/)
   - **Computation**: RDKit `QED.qed()` with Lipinski's rule considerations
   - **Range**: [0.0, 1.0], higher is better (0.67+ considered drug-like)

3. **Uniqueness** - Fraction of unique molecules among valid generated molecules
   - **Purpose**: Measures generative diversity and mode collapse detection
   - **Computation**: Canonical SMILES deduplication via RDKit `MolToSmiles`
   - **Range**: [0.0, 1.0], higher is better

4. **Novelty** - Fraction of unique valid molecules not present in training set
   - **Purpose**: Measures true generative capability vs. memorization
   - **Computation**: Set difference between generated and training canonical SMILES
   - **Range**: [0.0, 1.0], higher is better

5. **Average SAS** - Synthetic Accessibility Score for synthesis difficulty
   - **Purpose**: Estimates synthetic feasibility of generated molecules
   - **Reference**: [Ertl & Schuffenhauer, J. Cheminformatics (2009)](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8)
   - **Computation**: RDKit `sascorer.calculateScore()` based on reaction frequency
   - **Range**: [1.0, 10.0], lower is better (1-3: easy, 6+: difficult)

### Research Validation

Our implementation follows established benchmarking standards:
- **MOSES Metrics**: [Polykovskiy et al., Frontiers in Pharmacology (2020)](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.565644/full)
- **RDKit Standards**: All computations use RDKit 2023+ with proper sanitization and error handling

## Sampling & Property Optimization

```bash
# Sampling (subset allowed to control runtime)
CUDA_VISIBLE_DEVICES=0 python sample.py -n 1 --w_save_path weights/best_model.pt --subset 500

# Property optimization
python mainstream.py --seed 1
```

## GPU & Installs (Server CUDA 12.4)

### GPU Detection and Setup
```bash
# Check GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Monitor GPU usage during training
nvidia-smi -l 1  # Update every second
```

### PyTorch Installation for CUDA 12.4
If GPU is detected but PyTorch lacks CUDA libraries, use the **official installer** for the latest stable release. Often **cu121** wheels are provided by PyTorch and run on newer drivers:

```bash
# Install PyTorch with CUDA support (example for cu121)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Reference Links**:
- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/)
- [PyTorch Forums Discussion](https://discuss.pytorch.org/t/pytorch-2-3-with-cuda-12-4-wont-download-gpu-version/202757)

### Performance Notes
- **Batch Size**: Larger batches (1000-3000) work well with GPU memory
- **Parameter Sets**: Start with set 1 (200,100,2) for good balance

## Minimal Copy-Pasteable Workflow

### 1) Single GPU Training
```bash
# Training (auto early stopping, tqdm progress bars)
CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 1
```

### 2) Sampling and Optimization
```bash
# Sampling with trained models
CUDA_VISIBLE_DEVICES=0 python sample.py -n 1 --w_save_path weights/best_model.pt --subset 500

# Property optimization
python mainstream.py --seed 1
```

### 3) Developer Testing
```bash
# Quick smoke test (subset data, early patience)
python trainvae.py -n 0 --subset 500 --patience 3
```

## Recent Optimizations ✨

### Enhanced Pipeline Performance
- **Fixed Weight Extension**: Consistent `.pt` format across training, sampling, and evaluation
- **Validated Training Flow**: Comprehensive testing with subset validation

## Key Features

### Auto-Enabled Optimizations
- **Early Stopping**: Default ON, saves only best model, prevents overtraining
- **Mixed Precision (AMP)**: GPU acceleration with GradScaler for faster training
- **Progress Tracking**: Real-time tqdm with loss/validation metrics and patience
- **Smart Defaults**: Optimal DataLoader, CUDNN benchmarking, device detection

### Evidence-Based Implementation
- **MOSES Metrics**: Standard molecule benchmarking ([Frontiers in Pharmacology](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.565644/full))
- **PyTorch Best Practices**: Model saving, mixed precision, validation patterns
- **Real Data Validation**: All metrics computed from actual training/generation

### Dependencies & Setup
- **Core**: `torch`, `rdkit`, `gurobi-optimods`, `numpy`, `tqdm`
- **Optional**: `pandas`, `matplotlib` (for visualization in evaluation)
- **License**: Place `gurobi.lic` in project root for optimization features

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.