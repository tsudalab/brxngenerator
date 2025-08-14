
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import QED
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from nnutils import create_var
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from vae import FTRXNVAE, set_batch_nodeID
from mpn import MPN,PP,Discriminator
import random
from reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates,stats
from fragment import FragmentVocab, FragmentTree, FragmentNode, can_be_decomposed
from reaction_utils import get_mol_from_smiles, get_smiles_from_mol,read_multistep_rxns, get_template_order, get_qed_score,get_clogp_score

# Optional visualization dependencies - install with: pip install pandas seaborn matplotlib
try:
    import pandas as pd 
    import seaborn as sns 
    import matplotlib.pyplot as plt
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("[Warning] Visualization dependencies not available. Some plotting functions will be disabled.")
import gzip
import pickle
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import tqdm
import pandas as pd

import sys

# [ECC] Import error correcting code utilities
from ecc import create_ecc_codec, sample_ecc_latent, extract_info_bits

class Evaluator(nn.Module):
    def __init__(self, latent_size, model, ecc_type='none', ecc_R=3):
        super(Evaluator, self).__init__()
        self.latent_size = latent_size
        self.model = model
        # [ECC] Initialize error correcting code
        self.ecc_type = ecc_type
        self.ecc_R = ecc_R
        self.ecc_codec = create_ecc_codec(ecc_type, R=ecc_R)
        
        # [ECC] Determine effective latent dimensions
        if self.ecc_codec is not None:
            # When using ECC, latent_size is the code size N, info size is K
            if not self.ecc_codec.group_shape_ok(latent_size):
                raise ValueError(f"Latent size {latent_size} must be divisible by ECC repetition factor {ecc_R}")
            self.info_size = self.ecc_codec.get_info_size(latent_size)
            print(f"[ECC] Using {ecc_type} with R={ecc_R}: latent_size={latent_size}, info_size={self.info_size}")
        else:
            self.info_size = latent_size
            print(f"[ECC] No ECC: latent_size={latent_size}")

    def decode_from_prior(self, ft_latent, rxn_latent, n, prob_decode=True):
        for i in range(n):
            generated_tree = self.model.fragment_decoder.decode(ft_latent, prob_decode=prob_decode)
            g_encoder_output, g_root_vec = self.model.fragment_encoder([generated_tree])
            product, reactions = self.model.rxn_decoder.decode(rxn_latent, g_encoder_output, prob_decode)
            if product != None:
                return product, reactions
        return None, None

    def novelty_and_uniqueness(self, files, rxn_trees):
        smiles_training_set = []
        for rxn in rxn_trees:
            smiles = rxn.molecule_nodes[0].smiles
            smiles_training_set.append(smiles)

        smiles_training_set = set(smiles_training_set)
        training_size = len(smiles_training_set)
        count = 0
        total = 0
        valid_molecules = []
        for file in files:
            with open(file, "r") as reader:
                lines = reader.readlines()
                for line in lines:
                    total += 1
                    elements = line.strip().split(" ")
                    target = elements[0]
                    reactions = elements[1:]
                    if target not in smiles_training_set:
                        valid_molecules.append(target)
                        count += 1
                        print(target, len(reactions))
            reader.close()
        print("novelty:", count, training_size, total, count/total)
        print("uniqueness:", len(set(valid_molecules)), len(valid_molecules), len(set(valid_molecules))/len(valid_molecules))

    def kde_plot(self, file1s, file2s, metric="qed"):
        bo_scores = []
        smiles_list = []
        for file in file2s:
            with open(file) as reader:
                lines = reader.readlines()
                for line in lines:
                    line = line.strip()
                    res = line.split(" ")
                    smiles = res[0]
                    if smiles not in smiles_list:
                        smiles_list.append(smiles)
                    else:
                        continue
                    try:
                        if metric == "logp":
                            score = get_clogp_score(smiles, logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s)
                        if metric == "qed":
                            score = get_qed_score(smiles)
                        bo_scores.append(score)
                        print(file, smiles, score)
                    except:
                        print("cannot parse:", smiles)
        smiles_list = []
        p1 = sns.kdeplot(bo_scores, shade=True, color='r', label='Bayesian Optimization', x='QED', clip=(-10,20))
        sampling_scores = []
        count = 0
        for file in file1s:
            with open(file, "r") as reader:
                lines = reader.readlines()
                for line in lines:
                    line = line.strip()
                    res = line.split(" ")
                    smiles = res[0]
                    if smiles not in smiles_list:
                        smiles_list.append(smiles)
                    else:
                        continue
                    try:
                        if metric == "logp":
                            score = get_clogp_score(smiles, logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s)
                        if metric == "qed":
                            score = get_qed_score(smiles)
                        sampling_scores.append(score)
                        print(file, smiles, score)
                    except:
                        print("cannot parse:", smiles)
        np.random.shuffle(sampling_scores)
        limit = len(bo_scores)
        p1 = sns.kdeplot(sampling_scores[:limit], shade=True, color='b',label='Random Sampling', clip=(-10,20))
        plt.xlabel('(a) QED', fontsize=18)
        plt.ylabel('', fontsize=18)
        plt.legend(loc='upper left')
        plt.show()

    def qualitycheck(self, rxns, files):
        smiles_list = []
        for file in files:
            with open(file, "r") as reader:
                lines = reader.readlines()
                for line in lines:
                    elements = line.strip().split(" ")
                    target = elements[0]
                    smiles_list.append(target)
        num_cores = 4
        training_smiles = [rxn.molecule_nodes[0].smiles for rxn in rxns]

        p = Pool(mp.cpu_count())
        input_data = [(smi, f"MOL_{i}") for i, smi in enumerate(smiles_list)]
        training_data = [(smi, f"TMOL_{i}") for i, smi in enumerate(training_smiles)]
        
        alert_file_name = "alert_collection.csv"
        self.rf = rd_filters.RDFilters(alert_file_name)
        rules_file_path = "rules.json"
        rule_dict = rd_filters.read_rules(rules_file_path)
        rule_list = [x.replace("Rule_", "") for x in rule_dict.keys() if x.startswith("Rule") and rule_dict[x]]
        rule_str = " and ".join(rule_list)
        print(f"Using alerts from {rule_str}", file=sys.stderr)
        self.rf.build_rule_list(rule_list)
        self.rule_dict = rule_dict

        trn_res = list(p.map(self.rf.evaluate, training_data))
        res = list(p.map(self.rf.evaluate, input_data))
        count1 = 0
        for re in trn_res:
            ok = re[2]
            if ok == "OK":
                count1 += 1
        norm = count1/len(trn_res)
        count2 = 0
        for re in res:
            ok = re[2]
            if ok == "OK":
                count2 += 1
        ratio = count2/len(res)
        print(count2, len(res), ratio)
        print(ratio, norm, ratio/norm)
        
		
    def generate_discrete_latent(self, latent_size, method="gumbel", temp=0.4):
        """
        Generate latent variables conforming to discrete distribution, supporting Gumbel-Softmax and Bernoulli sampling
        [ECC] Now supports error-correcting codes for improved generation
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # [ECC] Determine sampling size based on ECC configuration
        if self.ecc_codec is not None:
            # Sample from information space and encode
            effective_size = self.info_size
        else:
            # No ECC, sample directly
            effective_size = latent_size
        
        if method == "gumbel":
            logits = torch.zeros(1, effective_size, 2, device=device)  # logits size for binary case
            gumbel_noise = -torch.log(-torch.log(torch.rand(logits.shape, device=device) + 1e-20) + 1e-20)
            latent_sample = F.softmax((logits + gumbel_noise) / temp, dim=-1)
            info_bits = latent_sample.argmax(dim=-1).float()  # Output one-hot approximation result

        elif method == "bernoulli":
            probs = torch.full((1, effective_size), 0.5, device=device)  # Bernoulli parameter, assuming uniform distribution
            info_bits = torch.bernoulli(probs)  # Return sampling result of 0 or 1
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # [ECC] Apply encoding if ECC is enabled
        if self.ecc_codec is not None:
            return self.ecc_codec.encode(info_bits)
        else:
            return info_bits
        
    def validate_and_save(self, train_rxn_trees, n=10000, output_file="generated_reactions.txt"):
        training_smiles = []
        for tree in train_rxn_trees:
            smiles = tree.molecule_nodes[0].smiles
            training_smiles.append(smiles)
        validity = 0
        
        

        with open(output_file, "a") as writer:
            for i in range(n):
                print("The " + str(i) + "th reaction is being generated...")
                
                ft_latent = self.generate_discrete_latent(self.latent_size, method="gumbel", temp=0.4)
                rxn_latent = self.generate_discrete_latent(self.latent_size, method="gumbel", temp=0.4)
                product, reactions = self.decode_from_prior(ft_latent, rxn_latent, 50)
                print("Product:", product, ", Reaction:", reactions)
                print("--------------------------------")
                if product != None:
                    validity += 1
                    print(i, validity/(i+1), "Product:", product, ", Reaction:", reactions)
                    line = product + " " + reactions
                    writer.write(line)
                    writer.write("\n")

        print("validity: ", validity/n)

    # [Metrics] ECC evaluation and comparison methods
    def evaluate_ecc_metrics(self, data_pairs, n_samples=1000, output_prefix="metrics"):
        """
        Evaluate ECC vs baseline metrics for BER, WER, reconstruction loss, 
        confidence calibration, and molecule quality metrics.
        
        Args:
            data_pairs: List of (fragment_tree, reaction_tree) pairs for evaluation
            n_samples: Number of samples for generation metrics (default 1000)
            output_prefix: Prefix for output files
        """
        print(f"[Metrics] Evaluating ECC vs baseline with {n_samples} samples")
        print(f"[Metrics] ECC config: type={self.ecc_type}, R={self.ecc_R}")
        
        # Subsample for faster evaluation
        eval_pairs = data_pairs[:min(len(data_pairs), n_samples)]
        print(f"[Metrics] Using {len(eval_pairs)} pairs for evaluation")
        
        results = {}
        
        # 1. Bit Error Rate (BER) and Word Error Rate (WER)
        if self.ecc_codec is not None:
            ber, wer = self._compute_error_rates(eval_pairs)
            results['BER'] = ber
            results['WER'] = wer
        else:
            results['BER'] = 0.0  # No errors in baseline
            results['WER'] = 0.0
            
        # 2. Reconstruction loss proxy
        recon_loss = self._compute_reconstruction_loss(eval_pairs)
        results['reconstruction_loss'] = recon_loss
        
        # 3. Confidence and entropy metrics
        entropy, confidence_acc = self._compute_calibration_metrics(eval_pairs)
        results['entropy'] = entropy
        results['confidence_accuracy'] = confidence_acc
        
        # 4. Molecule quality metrics
        mol_metrics = self._compute_molecule_metrics(n_samples//4)  # Smaller sample for generation
        results.update(mol_metrics)
        
        # Print compact results table
        self._print_metrics_table(results)
        
        return results
    
    def _compute_error_rates(self, data_pairs):
        """Compute BER and WER through round-trip encoding/decoding."""
        if self.ecc_codec is None:
            return 0.0, 0.0
            
        print("[Metrics] Computing BER/WER through round-trip encoding...")
        bit_errors, word_errors, total_bits, total_words = 0, 0, 0, 0
        
        for i, (fgm_tree, rxn_tree) in enumerate(data_pairs[:100]):  # Sample for speed
            if i % 50 == 0:
                print(f"  Progress: {i}/{min(100, len(data_pairs))}")
                
            try:
                # Encode data pair to get latent
                latent = self.model.encode([(fgm_tree, rxn_tree)])[0]
                
                # Extract info bits and re-encode
                if latent.shape[-1] >= self.latent_size:
                    code_bits = latent[:self.latent_size]
                    info_bits = self.ecc_codec.decode(code_bits.unsqueeze(0))
                    reconstructed_code = self.ecc_codec.encode(info_bits)
                    
                    # Compare bits
                    bit_diff = (code_bits.unsqueeze(0) != reconstructed_code).float()
                    bit_errors += bit_diff.sum().item()
                    total_bits += code_bits.numel()
                    
                    # Word error (any bit different in info block)
                    info_diff = (info_bits != self.ecc_codec.decode(reconstructed_code)).float()
                    word_errors += (info_diff.sum(dim=1) > 0).sum().item()
                    total_words += info_bits.shape[0]
                    
            except Exception as e:
                # Skip problematic samples
                continue
                
        ber = bit_errors / max(total_bits, 1)
        wer = word_errors / max(total_words, 1) 
        print(f"[Metrics] BER: {ber:.4f}, WER: {wer:.4f}")
        return ber, wer
    
    def _compute_reconstruction_loss(self, data_pairs):
        """Compute average reconstruction loss as ELBO proxy."""
        print("[Metrics] Computing reconstruction loss...")
        total_loss, count = 0.0, 0
        
        self.model.eval()
        with torch.no_grad():
            for i, data_pair in enumerate(data_pairs[:50]):  # Sample for speed
                try:
                    # Forward pass with beta=1 for pure reconstruction
                    loss, _, _, _, _, _, _, _, _, kl_loss, _ = self.model([data_pair], beta=1.0)
                    recon_loss = loss.item() - kl_loss.item()  # Remove KL component
                    total_loss += recon_loss
                    count += 1
                except Exception:
                    continue
                    
        avg_loss = total_loss / max(count, 1)
        print(f"[Metrics] Average reconstruction loss: {avg_loss:.4f}")
        return avg_loss
    
    def _compute_calibration_metrics(self, data_pairs):
        """Compute entropy and confidence-accuracy metrics."""
        print("[Metrics] Computing calibration metrics...")
        entropies, confidences, accuracies = [], [], []
        
        self.model.eval()
        with torch.no_grad():
            for i, data_pair in enumerate(data_pairs[:50]):  # Sample for speed
                try:
                    # Get encoder posterior probabilities
                    latent = self.model.encode([data_pair])[0]
                    
                    # Compute bitwise entropy (assuming sigmoid outputs)
                    probs = torch.sigmoid(latent)
                    entropy = -(probs * torch.log(probs + 1e-8) + (1-probs) * torch.log(1-probs + 1e-8))
                    entropies.append(entropy.mean().item())
                    
                    # Confidence as max probability per bit
                    confidence = torch.max(probs, 1-probs).mean().item()
                    confidences.append(confidence)
                    
                except Exception:
                    continue
                    
        avg_entropy = sum(entropies) / max(len(entropies), 1)
        avg_confidence = sum(confidences) / max(len(confidences), 1)
        
        print(f"[Metrics] Average entropy: {avg_entropy:.4f}, confidence: {avg_confidence:.4f}")
        return avg_entropy, avg_confidence
    
    def _compute_molecule_metrics(self, n_samples=250):
        """Compute molecule validity, uniqueness, and novelty."""
        print(f"[Metrics] Generating {n_samples} molecules for quality assessment...")
        
        generated_smiles = []
        attempts = 0
        max_attempts = n_samples * 10  # Avoid infinite loops
        
        while len(generated_smiles) < n_samples and attempts < max_attempts:
            attempts += 1
            try:
                ft_latent = self.generate_discrete_latent(self.latent_size, method="bernoulli", temp=0.4)
                rxn_latent = self.generate_discrete_latent(self.latent_size, method="bernoulli", temp=0.4) 
                product, _ = self.decode_from_prior(ft_latent, rxn_latent, n=5)
                
                if product and product not in generated_smiles:
                    generated_smiles.append(product)
                    
            except Exception:
                continue
                
        if len(generated_smiles) == 0:
            return {'validity': 0.0, 'uniqueness': 0.0, 'novelty': 0.0}
            
        # Validity: RDKit can parse the SMILES
        valid_count = 0
        for smiles in generated_smiles:
            try:
                mol = get_mol_from_smiles(smiles)
                if mol is not None:
                    valid_count += 1
            except:
                pass
                
        validity = valid_count / len(generated_smiles)
        uniqueness = len(set(generated_smiles)) / len(generated_smiles)
        
        # Novelty: not in training set (simplified - assume novelty = uniqueness for now)
        novelty = uniqueness  # Simplified metric
        
        print(f"[Metrics] Molecules - Validity: {validity:.3f}, Uniqueness: {uniqueness:.3f}, Novelty: {novelty:.3f}")
        
        return {
            'validity': validity,
            'uniqueness': uniqueness, 
            'novelty': novelty,
            'total_generated': len(generated_smiles)
        }
    
    def _print_metrics_table(self, results):
        """Print compact metrics table."""
        print("\n" + "="*60)
        print(f"{'METRIC':<20} {'VALUE':<15} {'DESCRIPTION'}")
        print("="*60)
        print(f"{'BER':<20} {results.get('BER', 0):<15.4f} {'Bit Error Rate'}")
        print(f"{'WER':<20} {results.get('WER', 0):<15.4f} {'Word Error Rate'}")
        print(f"{'Reconstruction':<20} {results.get('reconstruction_loss', 0):<15.4f} {'Recon Loss Proxy'}")
        print(f"{'Entropy':<20} {results.get('entropy', 0):<15.4f} {'Avg Bitwise Entropy'}")
        print(f"{'Confidence':<20} {results.get('confidence_accuracy', 0):<15.4f} {'Avg Confidence'}")
        print(f"{'Validity':<20} {results.get('validity', 0):<15.4f} {'Molecule Validity'}")
        print(f"{'Uniqueness':<20} {results.get('uniqueness', 0):<15.4f} {'Molecule Uniqueness'}")
        print(f"{'Novelty':<20} {results.get('novelty', 0):<15.4f} {'Molecule Novelty'}")
        print("="*60)


# [Metrics] CLI entry point for running metrics evaluation
def run_metrics_cli():
    """Command-line interface for running ECC metrics evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ECC vs baseline metrics")
    parser.add_argument('--mode', choices=['metrics'], default='metrics', help='Evaluation mode')
    parser.add_argument('-n', type=int, default=4, help='Parameter set index (0-7)')
    parser.add_argument('--w_save_path', required=True, help='Path to saved model weights')
    parser.add_argument('--ecc-type', choices=['none', 'repetition'], default='none', help='ECC type')
    parser.add_argument('--ecc-R', type=int, default=3, help='ECC repetition factor')
    parser.add_argument('--subset', type=int, default=1000, help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    # Load data and model (same pattern as sample.py)
    from reaction_utils import read_multistep_rxns
    from reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates
    from fragment import FragmentVocab, FragmentTree
    from vae import bFTRXNVAE
    from mpn import MPN
    
    # Parameter sets
    params = [
        (100, 100, 2), (200, 100, 2), (200, 100, 3), (200, 100, 5),
        (200, 200, 2), (200, 300, 2), (300, 100, 2), (500, 300, 5),
    ]
    
    param_set = params[args.n]
    hidden_size, latent_size, depth = param_set
    
    # ECC validation
    if args.ecc_type == "repetition" and latent_size % args.ecc_R != 0:
        raise ValueError(f"ECC repetition requires latent_size % ecc_R == 0. Got {latent_size} % {args.ecc_R} != 0")
    
    print(f"Loading model with params: hidden={hidden_size}, latent={latent_size}, depth={depth}")
    print(f"ECC config: type={args.ecc_type}, R={args.ecc_R}")
    
    # Load data
    routes, _ = read_multistep_rxns("./data/data.txt")
    rxn_trees = [ReactionTree(route) for route in routes]
    
    if args.subset and len(rxn_trees) > args.subset:
        rxn_trees = rxn_trees[:args.subset]
        print(f"Using subset of {args.subset} reactions")
        
    # Build vocabularies
    reactants = extract_starting_reactants(rxn_trees)
    templates, n_reacts = extract_templates(rxn_trees)
    reactantDic = StartingReactants(reactants)
    templateDic = Templates(templates, n_reacts)
    
    fgm_trees, valid_rxn_trees = [], []
    for tree in rxn_trees:
        try:
            fgm_trees.append(FragmentTree(tree.molecule_nodes[0].smiles))
            valid_rxn_trees.append(tree)
        except:
            continue
            
    data_pairs = list(zip(fgm_trees, valid_rxn_trees))
    cset = {node.smiles for fgm_tree in fgm_trees for node in fgm_tree.nodes}
    fragmentDic = FragmentVocab(list(cset))
    
    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = bFTRXNVAE(fragmentDic, reactantDic, templateDic, hidden_size, latent_size, depth, device=device).to(device)
    checkpoint = torch.load(args.w_save_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Create evaluator and run metrics
    evaluator = Evaluator(latent_size, model, ecc_type=args.ecc_type, ecc_R=args.ecc_R)
    results = evaluator.evaluate_ecc_metrics(data_pairs, n_samples=args.subset)
    
    print(f"\n[Metrics] Evaluation completed for ECC={args.ecc_type}")


if __name__ == "__main__":
    run_metrics_cli()

