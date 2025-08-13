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
from optparse import OptionParser
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

parser = OptionParser()
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=50)
parser.add_option("-d", "--depth", dest="depth", default=2)
parser.add_option("-b", "--batch", dest="batch_size", default = 32)
parser.add_option("-s", "--save_dir", dest="save_path", default="weights")
parser.add_option("--w_save_path", dest="w_save_path")
parser.add_option("-t", "--data_path", dest="data_path")
parser.add_option("-v", "--vocab_path", dest="vocab_path")
parser.add_option("-q", "--lr", dest="lr", default = 0.001)
parser.add_option("-z", "--beta", dest="beta", default = 1.0)
parser.add_option("-e", "--epochs", dest="epochs", default = 100)
# [ECC] Error-correcting code options
parser.add_option("--ecc-type", dest="ecc_type", default="none", help="ECC type: none or repetition")
parser.add_option("--ecc-R", dest="ecc_R", type="int", default=3, help="Repetition factor for ECC")
parser.add_option("--subset", dest="subset", type="int", default=None, help="Limit dataset size for testing")

opts, _ = parser.parse_args()

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
beta = float(opts.beta)
lr = float(opts.lr)
vocab_path = opts.vocab_path
data_filename = opts.data_path
epochs = int(opts.epochs)
save_path = opts.save_path
w_save_path = opts.w_save_path
# [ECC] Parse ECC options
ecc_type = opts.ecc_type
ecc_R = opts.ecc_R
subset_size = opts.subset

args={}
args['beta'] = beta
args['lr'] = lr
args['batch_size'] = batch_size
args['datasetname'] = data_filename
args['epochs'] = epochs
args['save_path'] = save_path


print("hidden size:", hidden_size, "latent_size:", latent_size, "batch size:", batch_size, "depth:", depth)
print("beta:", beta, "lr:", lr)
print("loading data.....")
data_filename = opts.data_path
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


mpn = MPN(hidden_size, depth)
model = bFTRXNVAE(fragmentDic, reactantDic, templateDic, hidden_size, latent_size, depth, fragment_embedding=None, reactant_embedding=None, template_embedding=None,device=device).to(device)
checkpoint = torch.load(w_save_path, map_location=device)
model.load_state_dict(checkpoint)
print("loaded model....")
evaluator = Evaluator(latent_size, model, ecc_type=ecc_type, ecc_R=ecc_R)
        # Ensure the output file is empty
with open("generated_reactions.txt", "w") as writer:
    writer.write("")
    
evaluator.validate_and_save(rxn_trees, output_file="generated_reactions.txt")



