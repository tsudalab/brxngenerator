# metrics.py - Consolidated metrics for molecular and latent space evaluation

"""
Comprehensive metrics module combining molecular and latent-space evaluation.

Molecular Metrics:
- Enhanced molecular metrics with canonicalization, SAS, novelty, and uniqueness
- MOSES-compatible metrics for standardized evaluation
- SAS scoring for synthetic accessibility

Latent Metrics:
- BER/WER evaluation for binary representations
- ELBO/IWAE log-likelihood bounds
- Calibration metrics and entropy analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
import sys
import os

# Optional dependencies
try:
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDKIT_AVAILABLE = True
except ImportError:
    print("[Warning] RDKit not available")
    RDKIT_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Import SAS scorer from chemistry module
try:
    from ..chemistry.chemistry_core import calculateScore
    SAS_AVAILABLE = True
    print("✅ SAS scorer loaded successfully")
except Exception as e:
    print(f"[Warning] SAS scorer not available: {e}")
    SAS_AVAILABLE = False

# === MOLECULAR METRICS ===

def to_mol(smi: str):
    """Convert SMILES to RDKit molecule with error handling."""
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        # Basic sanitization
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None

def to_canonical_smiles(smi: str) -> Optional[str]:
    """Convert SMILES to canonical form using RDKit standardization."""
    if not RDKIT_AVAILABLE:
        return smi

    mol = to_mol(smi)
    if mol is None:
        return None

    try:
        # Enhanced canonicalization with standardization
        standardizer = rdMolStandardize.Standardizer()
        mol = standardizer.standardize(mol)

        # Remove stereochemistry for consistent canonicalization
        Chem.RemoveStereochemistry(mol)

        # Generate canonical SMILES
        canonical_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return canonical_smi
    except:
        # Fallback to basic canonicalization
        try:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        except:
            return None

def compute_validity(smiles_list: List[str]) -> float:
    """Compute fraction of valid SMILES strings."""
    if not smiles_list:
        return 0.0

    valid_count = 0
    for smi in smiles_list:
        if to_mol(smi) is not None:
            valid_count += 1

    return valid_count / len(smiles_list)

def compute_uniqueness(smiles_list: List[str], valid_only: bool = True) -> float:
    """Compute fraction of unique molecules (MOSES-compatible)."""
    if not smiles_list:
        return 0.0

    if valid_only:
        # Filter to valid molecules first
        valid_smiles = [smi for smi in smiles_list if to_mol(smi) is not None]
        if not valid_smiles:
            return 0.0
        smiles_to_process = valid_smiles
    else:
        smiles_to_process = smiles_list

    # Canonicalize
    canonical_smiles = []
    for smi in smiles_to_process:
        canonical = to_canonical_smiles(smi)
        if canonical is not None:
            canonical_smiles.append(canonical)

    if not canonical_smiles:
        return 0.0

    unique_smiles = set(canonical_smiles)
    return len(unique_smiles) / len(canonical_smiles)

def compute_novelty(generated_smiles: List[str], training_smiles: List[str], valid_only: bool = True) -> float:
    """Compute fraction of generated molecules not in training set (MOSES-compatible)."""
    if not generated_smiles:
        return 0.0

    # Canonicalize training set
    training_canonical = set()
    for smi in training_smiles:
        canonical = to_canonical_smiles(smi)
        if canonical is not None:
            training_canonical.add(canonical)

    if valid_only:
        # Filter to valid molecules first
        valid_generated = [smi for smi in generated_smiles if to_mol(smi) is not None]
        if not valid_generated:
            return 0.0
        smiles_to_process = valid_generated
    else:
        smiles_to_process = generated_smiles

    # Check novelty
    novel_count = 0
    total_count = 0

    for smi in smiles_to_process:
        canonical = to_canonical_smiles(smi)
        if canonical is not None:
            total_count += 1
            if canonical not in training_canonical:
                novel_count += 1

    if total_count == 0:
        return 0.0

    return novel_count / total_count

def compute_qed_scores(smiles_list: List[str]) -> List[float]:
    """Compute QED scores for a list of SMILES."""
    if not RDKIT_AVAILABLE:
        return [0.0] * len(smiles_list)

    qed_scores = []
    for smi in smiles_list:
        mol = to_mol(smi)
        if mol is not None:
            try:
                qed_score = QED.qed(mol)
                qed_scores.append(qed_score)
            except:
                qed_scores.append(0.0)
        else:
            qed_scores.append(0.0)

    return qed_scores

def compute_sa_scores(smiles_list: List[str]) -> List[float]:
    """Compute Synthetic Accessibility scores."""
    if not SAS_AVAILABLE:
        return [5.0] * len(smiles_list)  # Neutral SA score

    sa_scores = []
    for smi in smiles_list:
        mol = to_mol(smi)
        if mol is not None:
            try:
                sa_score = calculateScore(mol)
                sa_scores.append(sa_score)
            except:
                sa_scores.append(5.0)  # Neutral SA score
        else:
            sa_scores.append(10.0)  # High SA score for invalid molecules

    return sa_scores

def compute_molecular_metrics(generated_smiles: List[str],
                            training_smiles: List[str] = None) -> Dict[str, float]:
    """
    Compute comprehensive molecular metrics.

    Args:
        generated_smiles: List of generated SMILES strings
        training_smiles: List of training SMILES for novelty computation

    Returns:
        Dictionary with metrics: validity, uniqueness, novelty, avg_qed, avg_sa
    """
    metrics = {}

    # Basic validity
    metrics['validity'] = compute_validity(generated_smiles)

    # Uniqueness (among valid molecules)
    metrics['uniqueness'] = compute_uniqueness(generated_smiles, valid_only=True)

    # Novelty (if training set provided)
    if training_smiles is not None:
        metrics['novelty'] = compute_novelty(generated_smiles, training_smiles, valid_only=True)

    # QED scores
    qed_scores = compute_qed_scores(generated_smiles)
    valid_qed = [score for score in qed_scores if score > 0]
    metrics['avg_qed'] = np.mean(valid_qed) if valid_qed else 0.0

    # SA scores
    sa_scores = compute_sa_scores(generated_smiles)
    valid_sa = [score for score in sa_scores if score < 10.0]
    metrics['avg_sa'] = np.mean(valid_sa) if valid_sa else 10.0

    return metrics

# === LATENT METRICS ===

class LatentMetrics:
    """
    Latent-space evaluation metrics for binary VAE.

    Computes BER/WER via MAP decoding, likelihood bounds, and calibration metrics.
    """

    def __init__(self, device: torch.device = None):
        """Initialize latent metrics evaluator."""
        self.device = device or torch.device('cpu')

    def compute_ber_wer(self, posteriors: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        """
        Compute Bit Error Rate and Word Error Rate via MAP decoding.

        Args:
            posteriors: Posterior probabilities p(z_i=1|x) of shape [B, L]
            targets: Ground truth binary codes of shape [B, L]

        Returns:
            Tuple of (BER, WER)
        """
        # MAP decoding: threshold at 0.5
        predictions = (posteriors > 0.5).float()

        # Bit Error Rate
        bit_errors = (predictions != targets).float()
        ber = bit_errors.mean().item()

        # Word Error Rate
        word_errors = (bit_errors.sum(dim=1) > 0).float()
        wer = word_errors.mean().item()

        return ber, wer

    def compute_elbo(self, posteriors: torch.Tensor,
                    targets: torch.Tensor,
                    prior_prob: float = 0.5) -> float:
        """
        Compute Evidence Lower Bound (ELBO).

        Args:
            posteriors: Posterior probabilities p(z_i=1|x) of shape [B, L]
            targets: Ground truth binary codes of shape [B, L]
            prior_prob: Prior probability p(z_i=1)

        Returns:
            ELBO value
        """
        # Reconstruction term: log p(z|x)
        log_posterior = targets * torch.log(posteriors + 1e-8) + \
                       (1 - targets) * torch.log(1 - posteriors + 1e-8)
        recon_term = log_posterior.sum(dim=1)

        # KL divergence term: KL[q(z|x) || p(z)]
        prior_logits = torch.logit(torch.tensor(prior_prob))
        posterior_logits = torch.logit(posteriors)
        kl_term = F.kl_div(F.log_softmax(posterior_logits, dim=-1),
                          F.softmax(prior_logits.expand_as(posterior_logits), dim=-1),
                          reduction='none').sum(dim=1)

        elbo = (recon_term - kl_term).mean()
        return elbo.item()

    def compute_iwae(self, posteriors: torch.Tensor,
                    targets: torch.Tensor,
                    k: int = 10,
                    prior_prob: float = 0.5) -> float:
        """
        Compute Importance Weighted Autoencoder (IWAE) bound.

        Args:
            posteriors: Posterior probabilities p(z_i=1|x) of shape [B, L]
            targets: Ground truth binary codes of shape [B, L]
            k: Number of importance samples
            prior_prob: Prior probability p(z_i=1)

        Returns:
            IWAE bound
        """
        B, L = posteriors.shape

        # Sample k importance samples for each datapoint
        # Expand posteriors: [B, L] -> [B, k, L]
        posteriors_expanded = posteriors.unsqueeze(1).expand(B, k, L)

        # Sample from posterior: [B, k, L]
        samples = torch.bernoulli(posteriors_expanded)

        # Compute log weights: log p(x,z) - log q(z|x)
        # log p(z) (prior term)
        log_prior = samples * np.log(prior_prob) + (1 - samples) * np.log(1 - prior_prob)
        log_prior = log_prior.sum(dim=2)  # [B, k]

        # log q(z|x) (posterior term)
        log_posterior = samples * torch.log(posteriors_expanded + 1e-8) + \
                       (1 - samples) * torch.log(1 - posteriors_expanded + 1e-8)
        log_posterior = log_posterior.sum(dim=2)  # [B, k]

        # log p(x|z) - simplified as reconstruction probability
        log_likelihood = (samples == targets.unsqueeze(1)).float().sum(dim=2)  # [B, k]

        # Log weights
        log_weights = log_likelihood + log_prior - log_posterior

        # IWAE bound: log mean_k exp(log_weights)
        iwae_bound = torch.logsumexp(log_weights, dim=1) - np.log(k)
        return iwae_bound.mean().item()

    def compute_calibration_error(self, posteriors: torch.Tensor,
                                 targets: torch.Tensor,
                                 n_bins: int = 10) -> Tuple[float, List]:
        """
        Compute Expected Calibration Error (ECE) using reliability diagrams.

        Args:
            posteriors: Posterior probabilities p(z_i=1|x) of shape [B, L]
            targets: Ground truth binary codes of shape [B, L]
            n_bins: Number of bins for reliability diagram

        Returns:
            Tuple of (ECE, bin_stats)
        """
        # Flatten for per-bit analysis
        posteriors_flat = posteriors.view(-1)
        targets_flat = targets.view(-1)

        # Create bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=self.device)
        bin_stats = []

        total_ece = 0.0
        total_samples = len(posteriors_flat)

        for i in range(n_bins):
            # Find samples in this bin
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            if i == n_bins - 1:  # Last bin includes upper boundary
                in_bin = (posteriors_flat >= bin_lower) & (posteriors_flat <= bin_upper)
            else:
                in_bin = (posteriors_flat >= bin_lower) & (posteriors_flat < bin_upper)

            if in_bin.sum() > 0:
                bin_confidence = posteriors_flat[in_bin].mean()
                bin_accuracy = targets_flat[in_bin].float().mean()
                bin_count = in_bin.sum().item()

                # Contribution to ECE
                bin_weight = bin_count / total_samples
                ece_contribution = bin_weight * abs(bin_confidence - bin_accuracy)
                total_ece += ece_contribution.item()

                bin_stats.append({
                    'bin_lower': bin_lower.item(),
                    'bin_upper': bin_upper.item(),
                    'confidence': bin_confidence.item(),
                    'accuracy': bin_accuracy.item(),
                    'count': bin_count,
                    'weight': bin_weight
                })
            else:
                bin_stats.append({
                    'bin_lower': bin_lower.item(),
                    'bin_upper': bin_upper.item(),
                    'confidence': 0.0,
                    'accuracy': 0.0,
                    'count': 0,
                    'weight': 0.0
                })

        return total_ece, bin_stats

    def compute_entropy(self, posteriors: torch.Tensor) -> float:
        """
        Compute average bitwise entropy of posterior distributions.

        Args:
            posteriors: Posterior probabilities p(z_i=1|x) of shape [B, L]

        Returns:
            Average entropy
        """
        # Bitwise entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
        entropy = -(posteriors * torch.log(posteriors + 1e-8) +
                   (1 - posteriors) * torch.log(1 - posteriors + 1e-8))

        return entropy.mean().item()

    def compute_latent_metrics(self, posteriors: torch.Tensor,
                              targets: torch.Tensor,
                              iwae_samples: int = 10) -> Dict[str, float]:
        """
        Compute comprehensive latent-space metrics.

        Args:
            posteriors: Posterior probabilities p(z_i=1|x) of shape [B, L]
            targets: Ground truth binary codes of shape [B, L]
            iwae_samples: Number of samples for IWAE computation

        Returns:
            Dictionary with metrics: BER, WER, ELBO, IWAE, ECE, entropy
        """
        metrics = {}

        # Error rates
        ber, wer = self.compute_ber_wer(posteriors, targets)
        metrics['BER'] = ber
        metrics['WER'] = wer

        # Likelihood bounds
        metrics['ELBO'] = self.compute_elbo(posteriors, targets)
        metrics['IWAE'] = self.compute_iwae(posteriors, targets, k=iwae_samples)

        # Calibration
        ece, _ = self.compute_calibration_error(posteriors, targets)
        metrics['ECE'] = ece

        # Entropy
        metrics['entropy'] = self.compute_entropy(posteriors)

        return metrics

# === EVALUATION WRAPPER ===

def evaluate_model_comprehensive(model, data_loader,
                                training_smiles: List[str] = None,
                                device: torch.device = None) -> Dict[str, float]:
    """
    Comprehensive model evaluation combining molecular and latent metrics.

    Args:
        model: Trained model with encode/decode capabilities
        data_loader: DataLoader for evaluation data
        training_smiles: Training SMILES for novelty computation
        device: Device for computation

    Returns:
        Dictionary with all computed metrics
    """
    device = device or torch.device('cpu')
    model.eval()

    # Collect generated molecules and latent representations
    generated_smiles = []
    all_posteriors = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            # Generate molecules (implement based on your model interface)
            # This is a placeholder - adapt to your specific model
            try:
                # Example generation - adapt to your model
                latent_sample = torch.bernoulli(torch.full((1, 100), 0.5)).to(device)
                # generated_molecules = model.decode(latent_sample)
                # generated_smiles.extend(generated_molecules)

                # Example latent evaluation - adapt to your model
                # posteriors = model.encode_posteriors(batch)
                # targets = batch['binary_codes']  # Adapt to your batch format
                # all_posteriors.append(posteriors)
                # all_targets.append(targets)
                pass
            except Exception as e:
                print(f"Warning: Could not evaluate batch: {e}")
                continue

    metrics = {}

    # Molecular metrics
    if generated_smiles:
        mol_metrics = compute_molecular_metrics(generated_smiles, training_smiles)
        metrics.update(mol_metrics)

    # Latent metrics
    if all_posteriors and all_targets:
        posteriors = torch.cat(all_posteriors, dim=0)
        targets = torch.cat(all_targets, dim=0)

        latent_evaluator = LatentMetrics(device=device)
        latent_metrics = latent_evaluator.compute_latent_metrics(posteriors, targets)
        metrics.update(latent_metrics)

    return metrics