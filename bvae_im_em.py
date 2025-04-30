import sys, os
sys.path.append('./rxnft_vae')

import rdkit
from rdkit.Chem import QED, Descriptors, rdmolops
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
import math, random, sys
from optparse import OptionParser
import pickle as pickle
import yaml
from rxnft_vae.reaction_utils import get_mol_from_smiles, get_smiles_from_mol, read_multistep_rxns, get_template_order, \
    get_qed_score, get_clogp_score
from rxnft_vae.reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, \
    extract_templates, stats
from rxnft_vae.fragment import FragmentVocab, FragmentTree, FragmentNode, can_be_decomposed
from rxnft_vae.vae import FTRXNVAE, set_batch_nodeID, bFTRXNVAE
from rxnft_vae.mpn import MPN, PP, Discriminator
import random
import rxnft_vae.sascorer as sascorer
from sklearn.model_selection import train_test_split
from amplify import BinaryMatrix, BinaryPoly, gen_symbols, sum_poly
from amplify import decode_solution, Solver
from amplify.client import FixstarsClient
from amplify.client.ocean import DWaveSamplerClient
import logging
import time

UPDATE_ITER = 1

metric = "qed"

class TorchFM(nn.Module):

    def __init__(self, n=None, k=None):
        super().__init__()
        self.V = nn.Parameter(torch.randn(n, k), requires_grad=True)
        self.lin = nn.Linear(n, 1)

    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)  # S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)  # S_2

        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin

        return out


class MolData(Dataset):

    def __init__(self, binary, targets):
        self.binary = binary
        self.targets = targets

    def __len__(self):
        return len(self.binary)

    def __getitem__(self, index):
        return self.binary[index], self.targets[index]


class RandomBinaryData(Dataset):

    def __init__(self, binary):
        self.binary = binary

    def __len__(self):
        return len(self.binary)

    def __getitem__(self, index):
        return self.binary[index]


class bVAE_IM(object):
    def __init__(self, bvae_model=None, smiles=None, targets=None, seed=0, n_sample=1):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bvae_model = bvae_model.to(self.device)
        self.train_smiles = smiles
        self.train_targets = targets
        
        self.random_seed = seed
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        if self.random_seed is not None:
            seed_all(self.random_seed)

        self.n_sample = n_sample  # configs['opt']['n_sample']
        self.sleep_count = 0

    def decode_many_times(self, latent):

        prob_decode = True
        binary_size = self.bvae_model.binary_size

        product_list = []
        for i in range(10):
            if len(product_list) > 5:
                break
            latent_new = latent
            # latent_new = torch.cat([latent, torch.randint(0, 2, (latent.shape[0], latent.shape[1] * 2 - latent.shape[1]))], dim=1).to(self.device)
            # print("latent_new shape", latent_new.shape)
            binary = F.one_hot(latent_new.long(), num_classes=2).float().to(self.device)
            binary = binary.view(1, -1)
            ft_mean = binary[:, :binary_size * 2]
            rxn_mean = binary[:, binary_size * 2:]
            # print("ft_mean shape", ft_mean.shape)
            # print("rxn_mean shape", rxn_mean.shape)
            generated_tree = self.bvae_model.fragment_decoder.decode(ft_mean, prob_decode)
            # print("generated_tree shape", generated_tree.shape)
            g_encoder_output, g_root_vec = self.bvae_model.fragment_encoder([generated_tree])
            # print("g_encoder_output shape", g_encoder_output.shape)
            # print("g_root_vec shape", g_root_vec.shape)
            product, reactions = self.bvae_model.rxn_decoder.decode(rxn_mean, g_encoder_output, prob_decode)
            if product != None:
                product_list.append([product, reactions])
        if len(product_list) == 0:
            return None
        else:
            return product_list

    def optimize(self, X_train, y_train, X_test, y_test, configs):
        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)
        self.X_test = X_test.to(self.device)
        self.y_test = y_test.to(self.device)
        
        
        n_opt = 100  # configs['opt']['num_end']
        self.train_binary = torch.vstack((X_train, X_test))
        self.n_binary = self.train_binary.shape[1]

        self.valid_smiles = []
        self.new_features = []
        self.full_rxn_strs = []


        self.end_cond = configs['opt']['end_cond']
        if self.end_cond not in [0, 1, 2]:
            raise ValueError("end_cond should be 0, 1 or 2.")
        if self.end_cond == 2:
            n_opt = 100  # n_opt is patience in this condition. When patience exceeds 100, exhaustion searching ends.

        self.results_smiles = []
        self.results_binary = []
        self.results_scores = []


        client = FixstarsClient()
        client.token = configs['opt']['client_token']
        client.parameters.timeout = 1000

        solver = Solver(client)

        self.iteration = 0

        while self.iteration < n_opt:
            qubo = self._build_qubo(configs)

            solution, energy = self._solve_qubo(qubo=qubo,
                                                qubo_solver=solver)

            self._update(solution=solution,
                         energy=energy)

        result_save_dir = configs['opt']['output']
        if not os.path.exists(result_save_dir):
            os.mkdir(result_save_dir)

        with open((os.path.join(result_save_dir, "%s_smiles.pkl" % configs['opt']['prop'])), "wb") as f:
            pickle.dump(self.results_smiles, f)
        with open((os.path.join(result_save_dir, "%s_scores.pkl" % configs['opt']['prop'])), "wb") as f:
            pickle.dump(self.results_scores, f)

        logging.info("Sleeped for %d minutes..." % self.sleep_count)





    def _build_qubo(self, configs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = TorchFM(self.n_binary, configs['opt']['factor_num']).to(device)
        
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)  # bias
            else:
                nn.init.uniform_(param, -configs['opt']['param_init'], configs['opt']['param_init'])  # weights

        print('========shape: ', self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape)
        dataset_train = MolData(self.X_train, self.y_train)
        dataloader_train = DataLoader(dataset=dataset_train,
                                      batch_size=configs['opt']['batch_size'],
                                      shuffle=True)
        dataset_valid = MolData(self.X_test, self.y_test)
        dataloader_valid = DataLoader(dataset=dataset_valid,
                                      batch_size=configs['opt']['batch_size'],
                                      shuffle=False)

        print('lr: ', configs['opt']['lr'])
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=configs['opt']['lr'],
                                     weight_decay=configs['opt']['decay_weight'])
        criterion = nn.MSELoss()

        lowest_error = float('inf')
        best_epoch = 0

        for epoch in range(configs['opt']['maxepoch']):
            model.train()
            for batch_x, batch_y in dataloader_train:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # Ensure data is on the GPU
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_hat_test = []
                for batch_x, _ in dataloader_valid:
                    batch_x = batch_x.to(device)  # Ensure data is on the GPU
                    valid = model(batch_x)
                    y_hat_test.append(valid)
                y_hat_test = torch.cat(y_hat_test)

                epoch_error = criterion(self.y_test.to(device), y_hat_test)  # Ensure target is on the GPU
                r2_test = r2_score(self.y_test.cpu().numpy(), y_hat_test.cpu().numpy())
                epoch_error = epoch_error.detach().cpu().numpy()
                if epoch % 100 == 0:
                    print("Model -- Epoch %d error on validation set: %.4f, r2 on validation set: %.4f" % (epoch, epoch_error, r2_test))

                if epoch_error < lowest_error:
                    torch.save(model.state_dict(),
                               os.path.join(configs['opt']['cache'],
                                            "fm_model-%s-%s-dim%d-seed%d-end%d" % (
                                                configs['opt']['prop'],
                                                configs['opt']['client'],
                                                self.n_binary,
                                                self.random_seed,
                                                self.end_cond)))
                    lowest_error = epoch_error
                    best_epoch = epoch

                if epoch > best_epoch + configs['opt']['patience']:
                    print("Model -- Epoch %d has lowest error!" % (best_epoch))
                    break

        y_hat_test = y_hat_test.unsqueeze(1).detach().cpu().numpy()
        y_test = self.y_test.detach().cpu().numpy()
        print(y_hat_test.shape, y_test.shape)
        model.load_state_dict(torch.load(
            os.path.join(configs['opt']['cache'],
                         "fm_model-%s-%s-dim%d-seed%d-end%d" % (
                             configs['opt']['prop'],
                             configs['opt']['client'],
                             self.n_binary,
                             self.random_seed,
                             self.end_cond)))
        )

        for p in model.parameters():
            if tuple(p.shape) == (self.n_binary, configs['opt']['factor_num']):
                Vi_f = p.to("cpu").detach().numpy()
            elif tuple(p.shape) == (1, self.n_binary):
                Wi = p.to("cpu").detach().numpy()
            elif tuple(p.shape) == (1,):
                W0 = p.to("cpu").detach().numpy()

        q = gen_symbols(BinaryPoly, self.n_binary)
        f_E = sum_poly(configs['opt']['factor_num'], lambda f: (
                    (sum_poly(self.n_binary, lambda i: Vi_f[i][f] * q[i])) ** 2 - sum_poly(self.n_binary,
                                                                                           lambda i: Vi_f[i][f] ** 2 *
                                                                                                     q[i] ** 2))) / 2 \
              + sum_poly(self.n_binary, lambda i: Wi[0][i] * q[i]) \
              + W0[0]
        qubo = (q, f_E)

        return qubo

    def _solve_qubo(self,
                    qubo,
                    qubo_solver):

        if isinstance(qubo, tuple):
            q, qubo = qubo

        solved = False
        while not solved:
            try:
                result = qubo_solver.solve(qubo)
                solved = True
            except RuntimeError as e:  # retry after 60s if connection to the solver fails..
                time.sleep(60)
                self.sleep_count += 1

        sols = []
        sol_E = []
        for sol in result:  # Iterate over multiple solutions
            if isinstance(qubo, BinaryMatrix):
                solution = [sol.values[i] for i in range(self.n_binary)]
            elif isinstance(qubo, BinaryPoly):
                solution = decode_solution(q, sol.values)
            else:
                raise ValueError("qubo type unknown!")
            sols.append(solution)
            sol_E.append(sol.energy)
        return np.array(sols), np.array(sol_E).astype(np.float32)
    def _update(self,
            solution,
            energy):

        if self.end_cond == 0:
            self.iteration += 1

        binary_new = torch.from_numpy(solution).to(torch.float)
        print('========binary_new shape')
        print(binary_new.shape)
        print("shape of binary_new", binary_new.shape)
        res = self.decode_many_times(binary_new)
        print('========res')
        print(res)

        if res is None:
            print('========res is None')
            return

        preLength = 0
        new_smiles = []
        new_rxn_strs = []

        for re in res:
            smiles = re[0]
            if len(re[1].split(" ")) > 0 and smiles not in self.valid_smiles:
                preLength += 1
                self.valid_smiles.append(smiles)
                self.new_features.append(latent)
                self.full_rxn_strs.append(re[1])
                new_smiles.append(smiles)
                new_rxn_strs.append(re[1])

        if preLength == 0:
            print("No new valid molecules generated.")
            return

        print("Number of new molecules:", preLength)

        scores = []
        b_valid_smiles = []
        b_full_rxn_strs = []
        b_scores = []

        for i in range(preLength):
            mol = rdkit.Chem.MolFromSmiles(new_smiles[i])
            if mol is None:
                continue
            if metric == "logp":
                print('========computing logp of molecule{}'.format(i))
                logP_values = np.loadtxt('./data/logP_values.txt')
                SA_scores = np.loadtxt('./data/SA_scores.txt')
                cycle_scores = np.loadtxt('./data/cycle_scores.txt')

                logp_m = np.mean(logP_values)
                logp_s = np.std(logP_values)

                sascore_m = np.mean(SA_scores)
                sascore_s = np.std(SA_scores)

                cycle_m = np.mean(cycle_scores)
                cycle_s = np.std(cycle_scores)
                smiles = new_smiles[i]
                score = get_clogp_score(smiles, logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s)
                scores.append(-score)

            elif metric == "qed":
                print('========computing qed of molecule{}'.format(i))
                score = QED.qed(mol)
                scores.append(-score)
            else:
                raise ValueError("Unsupported metric: {}".format(metric))
            b_valid_smiles.append(new_smiles[i])
            b_full_rxn_strs.append(new_rxn_strs[i])

        if len(scores) >= 1:
            b_scores = scores.copy()
            avg_score = np.mean(scores)
            training_score = [avg_score]
        else:
            print("No valid scores calculated.")
            return

        if len(binary_new) > 0:
            print('========Updating training set')
            print('X_train shape before update:', self.X_train.shape)
            print('y_train shape before update:', self.y_train.shape)
            self.X_train = np.concatenate([self.X_train, binary_new], 0)
            self.y_train = np.concatenate([self.y_train, np.array(training_score)[:, None]], 0)
            self.y_train = self.y_train.astype(np.float32)
            
            print('X_train shape after update:', self.X_train.shape)
            print('y_train shape after update:', self.y_train.shape)

        TaskID = os.environ.get("TaskID", "default_task")

        if metric == "logp":
            filename = "./Results/" + TaskID + "_logp.txt"
        elif metric == "qed":
            filename = "./Results/" + TaskID + "_qed.txt"

        print("Writing to file:", filename)
        with open(filename, "a") as writer:
            for i in range(len(b_valid_smiles)):
                line = " ".join([b_valid_smiles[i], b_full_rxn_strs[i], str(b_scores[i])])
                writer.write(line + "\n")

        assert self.X_train.shape[0] == self.y_train.shape[0]

        return

def main(X_train, y_train, X_test, y_test, smiles, targets, model, parameters, configs, metric, seed):
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    optimizer = bVAE_IM(smiles=smiles, targets=targets, bvae_model=model, seed=seed)

    start_time = time.time()

    optimizer.optimize(X_train, y_train, X_test, y_test, configs)

    logging.info("Running Time: %f" % (time.time() - start_time))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    

hidden_size = 300
latent_size = 100
depth = 2
data_filename = "/home/gzou/fitcheck/newnnn/brxngenerator-master/data/data.txt"
w_save_path = "/home/gzou/fitcheck/newnnn/brxngenerator-master/weights/hidden_size_300_latent_size_100_depth_2_beta_1.0_lr_0.001/bvae_iter-30-with.npy"
metric = "qed"
seed = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

print("number of samples:", len(data_pairs))
data_pairs = data_pairs
latent_list = []
score_list = []
print("num of samples:", len(rxn_trees))
latent_list = []
score_list = []
print('========start to compute all scores')
if metric == "qed":
    for i, data_pair in enumerate(data_pairs):
        latent = model.encode([data_pair])
        latent_list.append(latent[0])
        rxn_tree = data_pair[1]
        smiles = rxn_tree.molecule_nodes[0].smiles
        score_list.append(get_qed_score(smiles))
if metric == "logp":
    logP_values = np.loadtxt('./data/logP_values.txt')
    SA_scores = np.loadtxt('./data/SA_scores.txt')
    cycle_scores = np.loadtxt('./data/cycle_scores.txt')

    logp_m = np.mean(logP_values)
    logp_s = np.std(logP_values)

    sascore_m = np.mean(SA_scores)
    sascore_s = np.std(SA_scores)

    cycle_m = np.mean(cycle_scores)
    cycle_s = np.std(cycle_scores)
    for i, data_pair in enumerate(data_pairs):
        latent = model.encode([data_pair])
        latent_list.append(latent[0])
        rxn_tree = data_pair[1]
        smiles = rxn_tree.molecule_nodes[0].smiles
        score_list.append(get_clogp_score(smiles, logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s))
latents = torch.stack(latent_list, dim=0)
scores = np.array(score_list)
scores = scores.reshape((-1, 1))
# move to cpu first
latents = latents.detach().cpu().numpy()
n = latents.shape[0]
print('===================', n)
permutation = np.random.choice(n, n, replace=False)
X_train = latents[permutation, :][0: int(np.round(0.9 * n)), :]
X_test = latents[permutation, :][int(np.round(0.9 * n)):, :]
y_train = -scores[permutation][0: int(np.round(0.9 * n))]
y_test = -scores[permutation][int(np.round(0.9 * n)):]
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
if metric == "logp":
    parameters = [logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s]
else:
    parameters = []

with open('config/config.yaml', 'r') as f:
    configs = yaml.safe_load(f)



main(X_train, y_train, X_test, y_test, molecules, -scores, model, parameters, configs, metric, seed)
