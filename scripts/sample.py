import os
import sys
sys.path.append('./rxnft_vae')
from rxnft_vae.evaluate import Evaluator
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import argparse
from collections import deque

from rxnft_vae.reaction_utils import get_mol_from_smiles, get_smiles_from_mol,read_multistep_rxns, get_template_order, get_qed_score,get_clogp_score
from rxnft_vae.reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates,stats
from rxnft_vae.fragment import FragmentVocab, FragmentTree, FragmentNode, can_be_decomposed
from rxnft_vae.vae import FTRXNVAE, set_batch_nodeID, bFTRXNVAE
from rxnft_vae.mpn import MPN,PP,Discriminator
import rxnft_vae.sascorer as sascorer
import random

# TaskID =os.environ["TaskID"]
TaskID = "1"
def schedule(counter, M):
	x = counter/(2*M)
	if x > M:
		return 1.0
	else:
		return 1.0 * x/M

# [CLI] Unified parameter set approach consistent with trainvae.py
parser = argparse.ArgumentParser(description="Sample from binary VAE with optional ECC")
parser.add_argument("-n", type=int, dest="params_num", default=4, help="Parameter set index (0-7)")
parser.add_argument("--w_save_path", dest="w_save_path", required=True, help="Path to saved model weights")
parser.add_argument("--ecc-type", dest="ecc_type", default="none", choices=["none", "repetition"], help="ECC type: none or repetition")
parser.add_argument("--ecc-R", dest="ecc_R", type=int, default=3, help="Repetition factor for ECC")
parser.add_argument("--subset", dest="subset", type=int, default=None, help="Limit dataset size for testing")

args = parser.parse_args()

# [CLI] Parameter sets (same as trainvae.py)
params = [
    (100, 100, 2),  # Set 0
    (200, 100, 2),  # Set 1
    (200, 100, 3),  # Set 2
    (200, 100, 5),  # Set 3
    (200, 200, 2),  # Set 4 - recommended
    (200, 300, 2),  # Set 5 - ECC R=3 compatible
    (300, 100, 2),  # Set 6
    (500, 300, 5),  # Set 7 - largest
]

param_set = params[args.params_num]
hidden_size = param_set[0]
latent_size = param_set[1]  
depth = param_set[2]

# [ECC] Validate ECC parameters
if args.ecc_type == "repetition" and latent_size % args.ecc_R != 0:
    raise ValueError(f"ECC repetition requires latent_size % ecc_R == 0. Got {latent_size} % {args.ecc_R} != 0")

# [CLI] Fixed parameters
batch_size = 32
beta = 1.0
lr = 0.001
epochs = 100
save_path = "weights"
vocab_path = None  # Not used in current implementation

w_save_path = args.w_save_path
data_filename = "./data/data.txt"  # [CLI] Fixed data path
# [ECC] Parse ECC options
ecc_type = args.ecc_type
ecc_R = args.ecc_R
subset_size = args.subset

config_args={}
config_args['beta'] = beta
config_args['lr'] = lr
config_args['batch_size'] = batch_size
config_args['datasetname'] = data_filename
config_args['epochs'] = epochs
config_args['save_path'] = save_path


print("hidden size:", hidden_size, "latent_size:", latent_size, "batch size:", batch_size, "depth:", depth)
print("beta:", beta, "lr:", lr)
print("ECC type:", ecc_type, "ECC R:", ecc_R)  # [ECC] Show ECC settings

# [ECC] CLI validation completed successfully

print("loading data.....")
routes, scores = read_multistep_rxns(data_filename)


rxn_trees = [ReactionTree(route) for route in routes]
molecules = [rxn_tree.molecule_nodes[0].smiles for rxn_tree in rxn_trees]

# [ECC] Apply subset filtering if requested
if subset_size is not None and len(rxn_trees) > subset_size:
    print(f"Using subset of {subset_size} reactions (out of {len(rxn_trees)})")
    rxn_trees = rxn_trees[:subset_size]
    molecules = molecules[:subset_size]
reactants = extract_starting_reactants(rxn_trees)
templates, n_reacts = extract_templates(rxn_trees)
reactantDic = StartingReactants(reactants)
templateDic = Templates(templates, n_reacts)

print("size of reactant dic:", reactantDic.size())
print("size of template dic:", templateDic.size())


n_pairs = len(routes)
ind_list = [i for i in range(n_pairs)]

fgm_trees = []
valid_id = []
for i in ind_list:
    try:
        fgm_trees.append(FragmentTree(rxn_trees[i].molecule_nodes[0].smiles))
        valid_id.append(i)
    except Exception as e:
        # print(e)
        continue
rxn_trees = [rxn_trees[i] for i in valid_id]

print("size of fgm_trees:", len(fgm_trees))
print("size of rxn_trees:", len(rxn_trees))

data_pairs=[]
for fgm_tree, rxn_tree in zip(fgm_trees, rxn_trees):
	data_pairs.append((fgm_tree, rxn_tree))
cset=set()
for fgm_tree in fgm_trees:
	for node in fgm_tree.nodes:
		cset.add(node.smiles)
cset = list(cset)
# if vocab_path is None:
# else:
	# fragmentDic = FragmentVocab(cset, filename =vocab_path)
fragmentDic = FragmentVocab(cset)

print("size of fragment dic:", fragmentDic.size())
# Use the improved device detection from config
from config import get_device
device = get_device()
print(f"Using device: {device}")


mpn = MPN(hidden_size, depth)
# [ECC] Pass ECC parameters to model for consistent behavior
model = bFTRXNVAE(fragmentDic, reactantDic, templateDic, hidden_size, latent_size, depth, 
                  fragment_embedding=None, reactant_embedding=None, template_embedding=None,
                  device=device, ecc_type=ecc_type, ecc_R=ecc_R).to(device)
checkpoint = torch.load(w_save_path, map_location=device)
model.load_state_dict(checkpoint)
print("loaded model....")
evaluator = Evaluator(latent_size, model, ecc_type=ecc_type, ecc_R=ecc_R)
# Ensure the output file is empty
with open("generated_reactions.txt", "w") as writer:
    writer.write("")
    
# [ECC] Test with small sample count
n_samples = 10 if subset_size else 100
evaluator.validate_and_save(rxn_trees, n=n_samples, output_file="generated_reactions.txt")



