
# python import
import time
import random
import numpy as np
import logging
import random
import sys
sys.path.append('./rxnft_vae')

# rxnft_vae imports
from rxnft_vae.reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates
from rxnft_vae.fragment import FragmentVocab, FragmentTree
from rxnft_vae.vae import bFTRXNVAE
from rxnft_vae.mpn import MPN
from rxnft_vae.reaction_utils import read_multistep_rxns, get_qed_score,get_clogp_score

# torch
import torch

# tqdm
from tqdm import tqdm

# my binary vae utils
import binary_vae_utils




def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

hidden_size = 300

latent_size = 100

depth = 2

data_filename = "./data/data.txt"

w_save_path = "./weights/hidden_size_300_latent_size_100_depth_2_beta_1.0_lr_0.001/bvae_iter-30-with.npy"

metric = "qed"

seed = binary_vae_utils.RANDOM_SEED

device = binary_vae_utils.DEVICE


print("hidden size:", hidden_size, "latent_size:", latent_size, "depth:", depth)
print("loading data.....")
routes, scores = read_multistep_rxns(data_filename)
rxn_trees = [ReactionTree(route) for route in routes]
molecules = [rxn_tree.molecule_nodes[0].smiles for rxn_tree in rxn_trees]
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
for i in tqdm(ind_list):
    try:
        fgm_trees.append(FragmentTree(rxn_trees[i].molecule_nodes[0].smiles))
        valid_id.append(i)
    except Exception as e:
        # print(e)
        continue
rxn_trees = [rxn_trees[i] for i in valid_id]

print("size of fgm_trees:", len(fgm_trees))
print("size of rxn_trees:", len(rxn_trees))
data_pairs = []
for fgm_tree, rxn_tree in zip(fgm_trees, rxn_trees):
    data_pairs.append((fgm_tree, rxn_tree))
cset = set()
for fgm_tree in fgm_trees:
    for node in fgm_tree.nodes:
        cset.add(node.smiles)
cset = list(cset)
fragmentDic = FragmentVocab(cset)

print("size of fragment dic:", fragmentDic.size())


mpn = MPN(hidden_size, depth)
model = bFTRXNVAE(fragmentDic, reactantDic, templateDic, hidden_size, latent_size, depth, device,
                    fragment_embedding=None, reactant_embedding=None, template_embedding=None).to(device)
checkpoint = torch.load(w_save_path, map_location=device)
model.load_state_dict(checkpoint)
print("finished loading model...")


seed_all(seed)


X_train, y_train, X_test, y_test = binary_vae_utils.prepare_dataset(model=model, data_pairs=data_pairs,latent_size=latent_size)

X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

FM_surrogate = binary_vae_utils.FactorizationMachineSurrogate(n_binary=latent_size//2,k_factors=binary_vae_utils.FACTOR_NUM,random_seed=seed)

options = {
    "LICENSEID": 2687913,
    "WLSACCESSID": "5cbfb8e1-0066-4b7f-ab40-579464946573",
    "WLSSECRET": "a5c475ea-ec91-4cd6-94e9-b73395e273d6"
}

gurobi_solver = binary_vae_utils.GurobiQuboSolver(options)


import binary_vae_utils
optimizer = binary_vae_utils.MoleculeOptimizer(bvae_model=model,surrogate_model=FM_surrogate,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,qubo_solver=gurobi_solver)

start_time = time.time()

optimizer.optimize()

logging.info("Running Time: %f" % (time.time() - start_time))