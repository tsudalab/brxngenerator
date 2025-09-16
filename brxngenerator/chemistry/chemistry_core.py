# chemistry_core.py - Consolidated chemistry utilities

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors, QED
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict, deque
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
import pickle
import gzip
import math
import os.path as op
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from ...models.networks.nnutils import create_var
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ...core.vae import FTRXNVAE, set_batch_nodeID
from ...models.networks.mpn import MPN, PP, Discriminator
import random
from ..reactions.reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates, stats
from ..fragments.fragment import FragmentVocab, FragmentTree, FragmentNode, can_be_decomposed
from ..reactions.reaction_utils import get_mol_from_smiles, get_smiles_from_mol, read_multistep_rxns, get_template_order, get_qed_score, get_clogp_score

# Optional visualization dependencies
try:
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("[Warning] Visualization dependencies not available. Some plotting functions will be disabled.")

import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import tqdm
import sys

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

# === CHEMICAL UTILITIES ===

MST_MAX_WEIGHT = 100
MAX_NCAND = 2000

def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap:
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

# === SA SCORER ===

_fscores = None

def readFragmentScores(name='fpscores'):
    global _fscores
    if name == "fpscores":
        # Look for fpscores.pkl.gz in the project root directory
        project_root = op.dirname(op.dirname(op.dirname(op.dirname(__file__))))
        name = op.join(project_root, name)
    data = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" sascore into scale between 1 and 10
    min_val = -4.0
    max_val = 2.5
    sascore = 11. - (sascore - min_val + 1) / (max_val - min_val) * 9.
    # smooth the boundaries
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore < 1.:
        sascore = 1.

    return sascore

# === EVALUATOR CLASS ===

class Evaluator(nn.Module):
    def __init__(self, latent_size, model):
        super(Evaluator, self).__init__()
        self.latent_size = latent_size
        self.model = model

    def decode_from_prior(self, ft_latent, rxn_latent, n, prob_decode=True):
        for i in range(n):
            generated_tree = self.model.fragment_decoder.decode(ft_latent, prob_decode=prob_decode)
            g_encoder_output, g_root_vec = self.model.fragment_encoder([generated_tree])

            generated_rxn_tree = self.model.rxn_decoder.decode(rxn_latent, generated_tree, g_encoder_output, prob_decode=prob_decode)
            if generated_rxn_tree and len(generated_rxn_tree.molecule_nodes) > 0:
                yield generated_rxn_tree.molecule_nodes[0].smiles, generated_rxn_tree
            else:
                yield None, None

    def generate_discrete_latent(self, latent_size, method="gumbel", temp=0.4):
        """
        Generate latent variables conforming to discrete distribution, supporting Gumbel-Softmax and Bernoulli sampling
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if method == "gumbel":
            logits = torch.zeros(1, latent_size, 2, device=device)  # logits size for binary case
            gumbel_noise = -torch.log(-torch.log(torch.rand(logits.shape, device=device) + 1e-20) + 1e-20)
            latent_sample = F.softmax((logits + gumbel_noise) / temp, dim=-1)
            info_bits = latent_sample.argmax(dim=-1).float()  # Output one-hot approximation result

        elif method == "bernoulli":
            probs = torch.full((1, latent_size), 0.5, device=device)  # Bernoulli parameter, assuming uniform distribution
            info_bits = torch.bernoulli(probs)  # Return sampling result of 0 or 1
        else:
            raise ValueError(f"Unknown method: {method}")

        return info_bits

    def validate_and_save(self, train_rxn_trees, n=10000, output_file="generated_reactions.txt"):
        training_smiles = []
        for tree in train_rxn_trees:
            smiles = tree.molecule_nodes[0].smiles
            training_smiles.append(smiles)

        training_smiles = set(training_smiles)

        valid_molecules = []
        unique_molecules = set()
        novel_molecules = []
        count2 = 0
        validity = 0

        with open(output_file, "w") as writer:
            writer.write("")

        with open(output_file, "a") as writer:
            for i in range(n):
                ft_latent = self.generate_discrete_latent(self.latent_size // 2)
                rxn_latent = self.generate_discrete_latent(self.latent_size // 2)

                for product, reaction_tree in self.decode_from_prior(ft_latent, rxn_latent, 1, prob_decode=False):
                    if product is None:
                        continue

                    count2 += 1
                    mol = Chem.MolFromSmiles(product)
                    if mol is not None:
                        valid_molecules.append(mol)
                        unique_molecules.add(product)

                        if product not in training_smiles:
                            novel_molecules.append(product)

                        reactions = ""
                        if reaction_tree and hasattr(reaction_tree, 'templates'):
                            reactions = " ".join([str(t) for t in reaction_tree.templates])

                        validity += 1
                        print(i, validity/(i+1), "Product:", product, ", Reaction:", reactions)
                        line = product + " " + reactions
                        writer.write(line)
                        writer.write("\n")

        print("validity: ", validity/n)

def run_metrics_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model metrics")
    parser.add_argument('--mode', choices=['metrics'], default='metrics', help='Evaluation mode')
    parser.add_argument('-n', type=int, default=4, help='Parameter set index (0-7)')
    parser.add_argument('--w_save_path', required=True, help='Path to saved model weights')
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

    print(f"Loading model with params: hidden={hidden_size}, latent={latent_size}, depth={depth}")

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

    fragmentDic = FragmentVocab(fgm_trees)

    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = bFTRXNVAE(fragmentDic, reactantDic, templateDic, hidden_size, latent_size, depth, device=device).to(device)
    checkpoint = torch.load(args.w_save_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Create evaluator and run metrics
    evaluator = Evaluator(latent_size, model)
    print("Evaluator created successfully")

if __name__ == "__main__":
    run_metrics_cli()