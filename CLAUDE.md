# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**brxngenerator** is a binary Variational Autoencoder (B-VAE) implementation for molecular synthesis route generation and optimization. The project combines:

- **Binary VAE**: Neural architecture with binary latent space for molecular fragment encoding
- **Error-Correcting Codes (ECC)**: Optional repetition codes for improved generation quality
- **Molecular Optimization**: Gurobi-based QUBO optimization for molecular properties (QED, logP)

## Core Architecture

### Key Components
1. **B-VAE Model** (`rxnft_vae/vae.py`): `bFTRXNVAE` with binary latent vectors
2. **ECC Module** (`rxnft_vae/ecc.py`): Optional repetition codes with majority-vote decoding  
3. **Optimization Pipeline** (`mainstream.py`): Factorization Machine + Gurobi QUBO solver
4. **Training Interface** (`trainvae.py`): Modern training with early stopping

## Full-Data Training (Server, 1×GPU, AMP, Early Stop ON by Default)

### Standard Training
```bash
# Baseline B-VAE training (auto early stopping, tqdm, AMP)
CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 1    # Parameter set 1: hidden=200, latent=100, depth=2

# ECC training (latent must be divisible by R)
CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 1 --ecc-type repetition --ecc-R 2   # latent=100, info=50
CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 5 --ecc-type repetition --ecc-R 3   # latent=300, info=100

# Available parameter sets (0-7):
# Set 0: (100,100,2) - R=2 compatible  
# Set 1: (200,100,2) - R=2 compatible  
# Set 4: (200,200,2) - R=2 compatible
# Set 5: (200,300,2) - R=3 compatible (recommended for ECC)
# Set 7: (500,300,5) - R=3 compatible (largest)
```

### Features (Auto-Enabled)
- **Early Stopping**: Default ON (patience=10), saves only best model `bvae_best_model_with.pt`
- **Mixed Precision**: Automatic AMP on GPU with GradScaler for faster training
- **Progress Bars**: Real-time tqdm showing loss, KL, beta, validation loss, patience
- **GPU Optimization**: Pin memory, CUDNN benchmarking, optimal num_workers
- **ECC Integration**: Training-time ECC with code consistency regularization
- **ECC Validation**: Automatic divisibility check prevents invalid parameter combinations

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
- **ECC Validation**: Follows codedVAE methodology for BER/WER evaluation ([arXiv:2410.07840](https://arxiv.org/abs/2410.07840))
- **RDKit Standards**: All computations use RDKit 2023+ with proper sanitization and error handling

## A/B Comparison (Same Parameter Set)

### Training and Evaluating Baseline vs ECC
```bash
# Real-data A/B comparison with 5 standardized metrics
CUDA_VISIBLE_DEVICES=0 python ab_compare_ecc.py -n 1 --ecc-R 2 --train-subset 0 --eval-subset 2000
# -> results/compare_n1_*.json/.csv with 5 standardized metrics + metadata

# Key outputs:
# - JSON: Complete results with experimental configuration and metric definitions
# - CSV: Summary table for statistical analysis and plotting
# - Console: Side-by-side comparison with improvement percentages for all 5 metrics
```

### Sampling & Property Optimization

```bash
# Sampling (subset allowed to control runtime)
CUDA_VISIBLE_DEVICES=0 python sample.py -n 1 --w_save_path weights/best_baseline.pt --subset 500
CUDA_VISIBLE_DEVICES=0 python sample.py -n 1 --ecc-type repetition --ecc-R 2 \
  --w_save_path weights/best_ecc.pt --subset 500

# Property optimization (pass ECC flags if you want ECC in the optimization stage)
python mainstream.py --seed 1 --ecc-type repetition --ecc-R 2
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
- **Parameter Sets**: Start with set 1 (200,100,2) for good balance; set 5 (200,300,2) for ECC R=3
- **ECC Overhead**: Minimal (~5-10%) computational overhead for significant quality gains

## ECC Integration

### Research Validation
ECC implementation follows **codedVAE** methodology ([arXiv:2410.07840](https://arxiv.org/abs/2410.07840)) which reports **BER/WER (MAP)** reductions vs. uncoded DVAE and improved calibration. Our A/B comparison script mirrors these metrics and adds standard molecule metrics (**validity/uniqueness/novelty**) from MOSES benchmarking ([Frontiers in Pharmacology](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.565644/full)).

### Compatible Parameter Sets
```bash
# R=2 compatible: sets 0,1,2,3,4,6 (latent sizes divisible by 2)
# R=3 compatible: sets 5,7 (latent sizes: 300,300)

python trainvae.py -n 1 --ecc-type repetition --ecc-R 2  # ✅ 100%2=0
python trainvae.py -n 5 --ecc-type repetition --ecc-R 3  # ✅ 300%3=0  
python trainvae.py -n 0 --ecc-type repetition --ecc-R 3  # ❌ 100%3≠0 (validation error)
```

## Minimal Copy-Pasteable Workflow

### 1) Single GPU Training
```bash
# Baseline training (auto early stopping, tqdm progress bars)
CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 1

# ECC training (compatible parameter set required)  
CUDA_VISIBLE_DEVICES=0 python trainvae.py -n 1 --ecc-type repetition --ecc-R 2
```

### 2) A/B Comparison
```bash
# Compare baseline vs ECC with real metrics (BER, WER, molecule quality)
CUDA_VISIBLE_DEVICES=0 python ab_compare_ecc.py -n 1 --ecc-R 2 --train-subset 0 --eval-subset 2000
# Output: results/compare_n1_timestamp.json/.csv
```

### 3) Sampling and Optimization
```bash
# Sampling with trained models
CUDA_VISIBLE_DEVICES=0 python sample.py -n 1 --w_save_path weights/compare_baseline/best_model.pt --subset 500

# Property optimization  
python mainstream.py --seed 1 --ecc-type repetition --ecc-R 2
```

### 4) Developer Testing
```bash
# Quick smoke test (subset data, early patience)
python trainvae.py -n 0 --subset 500 --patience 3
```

## Recent Optimizations ✨

### Training-Time ECC Integration
- **Code Consistency Regularization**: ECC encoding/decoding integrated into VAE forward pass
- **Improved Training Stability**: ECC-processed latent vectors with consistency loss (weight: 0.01)
- **Real BER/WER Reduction**: Training-time error correction following codedVAE methodology

### Enhanced Pipeline Performance  
- **Fixed Weight Extension**: Consistent `.pt` format across training, sampling, and evaluation
- **Optimized A/B Comparison**: Complete baseline vs ECC comparison with real metrics
- **Validated Training Flow**: Comprehensive testing with subset validation

## Key Features

### Auto-Enabled Optimizations
- **Early Stopping**: Default ON, saves only best model, prevents overtraining
- **Mixed Precision (AMP)**: GPU acceleration with GradScaler for faster training
- **Progress Tracking**: Real-time tqdm with loss/validation metrics and patience
- **ECC Validation**: Automatic parameter compatibility checking
- **Smart Defaults**: Optimal DataLoader, CUDNN benchmarking, device detection

### Evidence-Based Implementation
- **codedVAE Research**: BER/WER reduction methodology ([arXiv:2410.07840](https://arxiv.org/abs/2410.07840))
- **MOSES Metrics**: Standard molecule benchmarking ([Frontiers in Pharmacology](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.565644/full))
- **PyTorch Best Practices**: Model saving, mixed precision, validation patterns
- **Training-Time ECC**: Integrated ECC encoding/decoding in VAE forward pass with consistency regularization
- **Real Data Validation**: All metrics computed from actual training/generation

## Latent-space Metrics (codedVAE-style)

### Overview
Complementing the 5 standardized molecular metrics, we implement **latent-space evaluation metrics** following codedVAE methodology. These metrics diagnose **latent stability & calibration**, providing insights into the VAE's internal representations and ECC effectiveness.

### Metrics Reported
- **BER/WER (MAP with ECC grouping)**: Bit/Word Error Rate using Maximum A Posteriori decoding with ECC majority-vote correction
- **ELBO (+ optional IWAE-K)**: Evidence Lower Bound and Importance Weighted Autoencoder bounds for generative quality  
- **ECE**: Expected Calibration Error using reliability-diagram binning for confidence assessment
- **Bitwise Entropy**: Average Shannon entropy over encoder posterior distributions

### Why They Matter
These metrics provide **orthogonal insights** to molecular quality scores:
- **BER/WER reduction** demonstrates ECC's error-correction effectiveness ([codedVAE](https://arxiv.org/abs/2410.07840))
- **Better calibration** (lower ECE) indicates more reliable confidence estimates
- **IWAE bounds** provide tighter likelihood estimates than standard ELBO ([Burda et al.](https://arxiv.org/abs/1509.00519))
- **Entropy analysis** reveals posterior sharpness and learning progression

### Usage

**Standard A/B comparison with latent metrics:**
```bash
CUDA_VISIBLE_DEVICES=0 python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 5000 \
  --latent-metrics true --iwae-samples 64 --gpu 0
```

**With noise robustness testing:**
```bash
CUDA_VISIBLE_DEVICES=0 python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 5000 \
  --latent-metrics true --noise-epsilon 0.05 --iwae-samples 64 --gpu 0
```

### Implementation Notes
- **Real Posteriors**: Uses `encode_posteriors()` method for authentic per-bit probability distributions
- **MOSES Compliance**: Validity metrics use proper denominators (#valid/#total) following MOSES benchmarking standards  
- **IWAE computation** is slower but provides tighter bounds (use `--iwae-samples 64` for balance)
- **ECE** follows reliability-diagram binning with 10 bins ([calibration literature](https://arxiv.org/html/2501.19047v2))
- **Entropy** is averaged over bit posteriors: `H(p) = -p*log(p) - (1-p)*log(1-p)`
- **Noisy-channel test** validates ECC robustness under controlled bit-flip perturbations
- **Fair Comparison**: Baseline and ECC models use equivalent latent space sizes for unbiased evaluation

### Expected Results
With proper ECC implementation, expect:
- **BER/WER reduction**: ECC should show lower error rates vs baseline
- **Improved calibration**: ECC models often exhibit better confidence calibration  
- **Comparable ELBO**: Likelihood bounds should remain competitive
- **Controlled entropy**: Well-trained models show appropriate posterior sharpness

### Dependencies & Setup
- **Core**: `torch`, `rdkit`, `gurobi-optimods`, `numpy`, `tqdm`
- **Optional**: `pandas`, `matplotlib` (for visualization in evaluation)
- **License**: Place `gurobi.lic` in project root for optimization features

