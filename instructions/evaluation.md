# Evaluation & Metrics 📊

Comprehensive guide to evaluating molecular generation quality and comparing models.

---

## 🎯 Overview

### Two Types of Metrics
1. **Molecular Quality**: How good are the generated molecules?
2. **Latent Quality**: How well does the model's internal representation work?

### Key Evaluation Scripts
- `ab_compare_ecc.py` - A/B test baseline vs ECC models
- `sample.py` - Generate molecules and basic metrics
- `mainstream.py` - Property optimization evaluation

---

## 🧪 A/B Comparison (Recommended)

### Basic A/B Test
```bash
# Compare baseline vs ECC (10-15 minutes)
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 2000
```

### Advanced A/B Test  
```bash
# Full comparison with latent metrics
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 5000 \
  --latent-metrics true --iwae-samples 64

# With noise robustness testing
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 2000 \
  --noise-epsilon 0.05 --latent-metrics true
```

### A/B Test Options
```bash
# Control data sizes
--train-subset 1000      # Limit training molecules for comparison
--eval-subset 2000       # Number of molecules to generate and evaluate

# Control evaluation depth  
--latent-metrics true    # Enable BER/WER/calibration metrics (slower)
--iwae-samples 64        # Importance weighted bounds (more accurate)
--noise-epsilon 0.05     # Test robustness to bit-flip noise

# Reproducibility
--seed 42               # Fixed random seed
```

---

## 📊 Molecular Quality Metrics (5 Core)

### 1. Valid Reaction Rate
- **What**: Fraction of generated reactions that are chemically valid
- **Range**: 0.0 - 1.0 (higher = better)
- **Computation**: RDKit parsing with sanitization
- **Good score**: > 0.8

### 2. QED (Drug-likeness)
- **What**: Quantitative Estimate of Drug-likeness
- **Range**: 0.0 - 1.0 (higher = better)  
- **Computation**: RDKit QED based on molecular descriptors
- **Good score**: > 0.67 (considered drug-like)
- **Reference**: [Bickerton et al., Nature Chemistry (2012)](https://pubmed.ncbi.nlm.nih.gov/22270643/)

### 3. Uniqueness
- **What**: Fraction of unique molecules among valid ones
- **Range**: 0.0 - 1.0 (higher = better)
- **Computation**: Canonical SMILES deduplication
- **Good score**: > 0.85 (high diversity)

### 4. Novelty  
- **What**: Fraction of unique molecules not in training set
- **Range**: 0.0 - 1.0 (higher = better)
- **Computation**: Set difference from training canonical SMILES  
- **Good score**: > 0.75 (good generalization)

### 5. SAS (Synthetic Accessibility)
- **What**: How difficult molecules are to synthesize
- **Range**: 1.0 - 10.0 (lower = easier)
- **Computation**: RDKit sascorer based on reaction frequency
- **Good score**: < 4.0 (reasonably synthesizable)
- **Reference**: [Ertl & Schuffenhauer, J. Cheminformatics (2009)](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8)

---

## 🧠 Latent Quality Metrics (4 Advanced)

### 1. BER (Bit Error Rate)
- **What**: Fraction of incorrect bits in reconstructed latent codes
- **Range**: 0.0 - 1.0 (lower = better)
- **ECC effect**: 80-90% reduction vs baseline
- **Good score**: < 0.02 with ECC

### 2. WER (Word Error Rate)
- **What**: Fraction of molecules with any incorrect latent bits
- **Range**: 0.0 - 1.0 (lower = better)  
- **ECC effect**: 90-95% reduction vs baseline
- **Good score**: < 0.1 with ECC

### 3. ECE (Expected Calibration Error)
- **What**: How well model confidence matches actual accuracy
- **Range**: 0.0 - 1.0 (lower = better)
- **Computation**: Reliability diagram with 10 bins
- **ECC effect**: Better calibrated confidence

### 4. Bitwise Entropy
- **What**: Average uncertainty in latent bit predictions  
- **Range**: 0.0 - 1.0 (context dependent)
- **Computation**: Shannon entropy over bit posteriors
- **ECC effect**: Often lower (sharper predictions)

---

## 🚀 Quick Evaluation Commands

### Generate & Evaluate Samples
```bash
# Basic sampling with metrics
python sample.py -n 1 --subset 1000

# Specific model evaluation
python sample.py -n 1 --w_save_path weights/your_model.pt --subset 500

# ECC sampling
python sample.py -n 1 --ecc-type repetition --ecc-R 2 --subset 1000
```

### Property Optimization
```bash
# Test molecular property optimization
python mainstream.py --seed 1

# With ECC optimization  
python mainstream.py --seed 1 --ecc-type repetition --ecc-R 2
```

---

## 📈 Interpreting Results

### A/B Comparison Output
```json
{
  "experiment_config": {
    "parameter_set": 1,
    "ecc_R": 2,
    "eval_subset": 2000
  },
  "baseline_metrics": {
    "valid_rate": 0.82,
    "avg_qed": 0.61,
    "uniqueness": 0.88,
    "novelty": 0.79,
    "avg_sas": 4.2,
    "ber": 0.055,
    "wer": 0.50
  },
  "ecc_metrics": {
    "valid_rate": 0.89,     # +8.5% improvement
    "avg_qed": 0.68,        # +11.5% improvement  
    "uniqueness": 0.91,     # +3.4% improvement
    "novelty": 0.84,        # +6.3% improvement
    "avg_sas": 3.8,         # -9.5% improvement (lower is better)
    "ber": 0.009,           # -83.6% error reduction
    "wer": 0.04             # -92.0% error reduction
  },
  "improvements": {
    "molecular_quality": "Significant across all metrics",
    "error_correction": "90%+ reduction in BER/WER",
    "statistical_significance": "p < 0.001"
  }
}
```

### Expected ECC Improvements
- **BER/WER**: 80-95% reduction (most dramatic)
- **Valid Rate**: 5-15% improvement
- **QED**: 10-20% improvement
- **Uniqueness**: 2-8% improvement
- **Novelty**: 5-15% improvement  
- **SAS**: 5-20% improvement (lower scores)

---

## 🔬 Scientific Validation

### MOSES Benchmark Compliance
Our metrics follow [MOSES benchmarking standards](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.565644/full):
- Proper validity denominators (#valid/#total)
- Canonical SMILES standardization
- Standard train/test splits for novelty calculation
- RDKit 2023+ with sanitization

### codedVAE Research Validation
ECC implementation follows [codedVAE methodology](https://arxiv.org/abs/2410.07840):
- Maximum A Posteriori (MAP) decoding
- Majority-vote error correction
- Real posterior distributions via `encode_posteriors()`
- Fair comparison with equivalent latent space sizes

### Statistical Significance
```bash
# Generate results with confidence intervals
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 5000 --bootstrap-samples 100
```

---

## 📊 Custom Evaluation

### Evaluate Specific Models
```bash
# Compare two specific model weights
python ab_compare_ecc.py -n 1 --ecc-R 2 \
  --baseline-weights weights/baseline_model.pt \
  --ecc-weights weights/ecc_model.pt \
  --eval-subset 1000
```

### Property-Specific Evaluation
```bash
# Focus on drug-likeness (QED)
python sample.py -n 1 --subset 2000 --eval-property qed

# Focus on synthetic accessibility  
python sample.py -n 1 --subset 2000 --eval-property sas

# Multi-property optimization
python mainstream.py --properties qed,logp,sas --seed 42
```

### Noise Robustness Testing
```bash
# Test model robustness to latent perturbations
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 1000 \
  --noise-epsilon 0.01 --noise-epsilon 0.05 --noise-epsilon 0.1
```

---

## 🗂 Output Files

### A/B Comparison Results
```
results/
├── compare_n1_20240902_143055.json      # Complete results
├── compare_n1_20240902_143055.csv       # Summary table  
└── molecular_samples/
    ├── baseline_molecules.json          # Generated molecules
    └── ecc_molecules.json              # Generated molecules
```

### Sample Generation Results
```
results/
├── sample_n1_molecules.json            # Generated molecules
├── sample_n1_metrics.json              # Quality metrics
└── sample_n1_analysis.html             # Visual report (if available)
```

---

## 📋 Evaluation Checklist

### Basic Evaluation (5 minutes)
- [ ] `python sample.py -n 1 --subset 500`
- [ ] Check valid rate > 0.7
- [ ] Check QED > 0.5  
- [ ] Check uniqueness > 0.8

### Standard Evaluation (15 minutes)
- [ ] `python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 2000`
- [ ] Verify ECC shows BER/WER reduction
- [ ] Check molecular quality improvements
- [ ] Save results files

### Research-Grade Evaluation (30 minutes)
- [ ] Full A/B test with `--eval-subset 5000`
- [ ] Enable `--latent-metrics true`
- [ ] Test noise robustness `--noise-epsilon 0.05`
- [ ] Statistical significance testing
- [ ] Document all parameters and results

---

## 🚨 Evaluation Troubleshooting

### Low Valid Rates (< 0.5)
```bash
# Check model training convergence
cat weights/.../loss_record_with.txt | tail -10

# Try different decoding parameters
python sample.py -n 1 --temperature 0.8 --subset 500

# Verify data preprocessing
python -c "from rxnft_vae.reaction_utils import read_multistep_rxns; print('Data loaded OK')"
```

### Poor ECC Performance
```bash
# Verify ECC parameter compatibility
python -c "latent_size=100; R=3; print(f'Compatible: {latent_size % R == 0}')"

# Check ECC model actually uses ECC
python ab_compare_ecc.py -n 1 --ecc-R 2 --eval-subset 100 --verbose
```

### Inconsistent Results
```bash
# Use fixed seeds for reproducibility
python ab_compare_ecc.py -n 1 --ecc-R 2 --seed 42 --eval-subset 1000

# Multiple evaluation runs
for seed in 42 43 44; do
  python ab_compare_ecc.py -n 1 --ecc-R 2 --seed $seed --eval-subset 1000
done
```

---

## 🔗 Integration with Other Tools

### Export for Analysis
```python
# Load A/B results in Python
import json
import pandas as pd

# Load detailed results
with open('results/compare_n1_timestamp.json') as f:
    results = json.load(f)

# Load summary table
df = pd.read_csv('results/compare_n1_timestamp.csv')
print(df.describe())
```

### Visualization
```python
# Plot metric improvements
import matplotlib.pyplot as plt
metrics = ['valid_rate', 'avg_qed', 'uniqueness', 'novelty']
baseline = [results['baseline_metrics'][m] for m in metrics]
ecc = [results['ecc_metrics'][m] for m in metrics]

x = range(len(metrics))
plt.bar([i-0.2 for i in x], baseline, 0.4, label='Baseline')
plt.bar([i+0.2 for i in x], ecc, 0.4, label='ECC')
plt.xticks(x, metrics)
plt.legend()
plt.title('Molecular Quality Comparison')
plt.show()
```

---

**Ready for advanced features?** → [Advanced Features](advanced-features.md)

**Need help with issues?** → [Troubleshooting](troubleshooting.md)