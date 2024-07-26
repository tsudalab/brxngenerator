import sys, os
sys.path.append('./rxnft_vae')

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import QED, Descriptors, rdmolops

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import math, random, sys
from optparse import OptionParser
from collections import deque
import pickle as pickle
import yaml
import networkx as nx

from reaction_utils import get_mol_from_smiles, get_smiles_from_mol,read_multistep_rxns, get_template_order, get_qed_score,get_clogp_score
from reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates,stats
from fragment import FragmentVocab, FragmentTree, FragmentNode, can_be_decomposed
from vae import FTRXNVAE, set_batch_nodeID, bFTRXNVAE
from mpn import MPN,PP,Discriminator
import random
import sascorer
  
from rdkit import Chem
from sklearn.model_selection import train_test_split

from amplify import BinaryMatrix, BinaryPoly, gen_symbols, sum_poly
from amplify import decode_solution, Solver
from amplify.client import FixstarsClient
from amplify.client.ocean import DWaveSamplerClient

from sklearn.linear_model import Ridge, Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
import logging
import time

UPDATE_ITER = 1

class TorchFM(nn.Module):
    
    def __init__(self, n=None, k=None):
        # n: size of binary features
        # k: size of latent features
        super().__init__()
        self.V = nn.Parameter(torch.randn(n, k), requires_grad=True)
        self.lin = nn.Linear(n, 1)

    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        out = out.squeeze(dim=1)

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

        self.bvae_model = bvae_model
        self.train_smiles = smiles
        self.train_targets = targets

        self.random_seed = seed
        if self.random_seed is not None:
            seed_all(self.random_seed)
        
        self.n_sample = n_sample # configs['opt']['n_sample']
        # self._initialize()
        self.sleep_count = 0
        
    def decode_many_times(self, latent):
        binary = F.one_hot(latent.long(),num_classes=2).float()
        binary = binary.view(1, -1)
        prob_decode = True
        binary_size = self.bvae_model.binary_size
        # ft_mean = latent[:, :latent_size]
        # rxn_mean = latent[:, latent_size:]
        ft_mean = binary[:, :binary_size*2]
        rxn_mean = binary[:, binary_size*2:]
        product_list=[]
        for i in range(50):
            generated_tree = self.bvae_model.fragment_decoder.decode(ft_mean, prob_decode)
            g_encoder_output, g_root_vec = self.bvae_model.fragment_encoder([generated_tree])
            product, reactions = self.bvae_model.rxn_decoder.decode(rxn_mean, g_encoder_output, prob_decode)
            if product != None:
                product_list.append([product, reactions])
            # break
        if len(product_list) == 0:
            return None
        else:
            return product_list


    def optimize(self, X_train, y_train, X_test, y_test, configs):
        
        n_opt = 100 # configs['opt']['num_end']
        self.train_binary = torch.vstack((X_train, X_test))
        self.n_binary = self.train_binary.shape[1]
        
        self.valid_smiles =[]
        self.new_features =[]
        self.full_rxn_strs=[]
        self.X_train = X_train
        self.y_train = y_train
        
        self.end_cond = configs['opt']['end_cond']
        if self.end_cond not in [0, 1, 2]:
            raise ValueError("end_cond should be 0, 1 or 2.")
        if self.end_cond == 2:
            n_opt = 100 # n_opt is patience in this condition. When patience exceeds 100, exhaustion searching ends.

        self.results_smiles = []
        self.results_binary = []
        self.results_scores = []

        # # config Ising machine

        # client = configs['opt']['client']
        client = FixstarsClient()
        client.token = configs['opt']['client_token']
        client.parameters.timeout = 1000
        # elif client == "dwave":
        #     client = DWaveSamplerClient()
        #     client.token = configs['opt']['client_token']
        #     client.solver = configs['opt']['dwave_sys']
        #     client.parameters.num_reads = 1000
        #     client.parameters.max_answers = 1
        # else:
        #     raise ValueError("Wrong client!")

        solver = Solver(client)

        self.iteration = 0

        while self.iteration < n_opt:

            # train factorization machine
            qubo = self._build_qubo(X_train, X_test, y_train, y_test, configs)

            solution, energy = self._solve_qubo(qubo = qubo,
                                    qubo_solver = solver)

            # merge new data into dataset
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
        

    # def _initialize(self):
    #     self.train_smiles = self.train_smiles.tolist()
    #     self.train_targets = self.train_targets.astype('float')
    #     self.train_mols = [Chem.MolFromSmiles(s) for s in self.train_smiles]

    #     self.train_binary = self._encode_to_binary(self.train_smiles)

    #     if self.opt_target == 'max':
    #         self.train_targets = [-self.get_score(m) for m in self.train_mols]
    #     elif self.opt_target == 'min':
    #         self.train_targets = [self.get_score(m) for m in self.train_mols]
    #     self.train_targets = np.repeat(self.train_targets, self.n_sample).tolist()
    #     # plus --> minimization; minus --> maximization

    # def _encode_to_binary(self, smiles, batch_size = 64):
    #     encoded = []
    #     print("encoding molecules to binary sequences...")
    #     for i in tqdm(range(int(np.ceil(len(smiles) / batch_size)))):
    #         smiles_batch = smiles[i*batch_size: (i+1)*batch_size]
    #         if self.n_sample == 1:
    #             encoded_batch = self.bvae_model.encode_from_smiles(smiles_batch)
    #         else:
    #             encoded_batch = self.bvae_model.encode_from_smiles(smiles_batch, self.n_sample)
    #         encoded.append(encoded_batch)
    #     train_binary = torch.vstack(encoded)
    #     train_binary = train_binary.to('cpu').numpy()
    #     return train_binary

    def _build_qubo(self, X_train, X_valid, y_train, y_valid, configs):
        
        model = TorchFM(self.n_binary, configs['opt']['factor_num'])# .to(self.device)
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)     # bias
            else:
                nn.init.uniform_(param, -configs['opt']['param_init'], configs['opt']['param_init'])   # weights

        # X_train, X_valid, y_train, y_valid = train_test_split(self.train_binary,
        #                                                 self.train_targets,
        #                                                 test_size=0.1,
        #                                                 random_state=self.iteration)

        # X_train = torch.from_numpy(X_train).to(torch.float).to(self.device)
        # X_valid = torch.from_numpy(X_valid).to(torch.float).to(self.device)
        # y_train = torch.tensor(y_train).to(torch.float).to(self.device)
        # y_valid = torch.tensor(y_valid).to(torch.float).to(self.device)

        print('========shape: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        dataset_train = MolData(X_train, y_train)
        dataloader_train = DataLoader(dataset=dataset_train,
                                    batch_size=configs['opt']['batch_size'],
                                    shuffle=True)
        dataset_valid = MolData(X_valid, y_valid)
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
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_hat_valid = []
                for batch_x, _ in dataloader_valid:
                    valid = model(batch_x)
                    y_hat_valid.append(valid)
                y_hat_valid = torch.cat(y_hat_valid)

                epoch_error = criterion(y_valid, y_hat_valid)
                epoch_error = epoch_error.detach().cpu().numpy()
                if epoch % 100 == 0:
                    print("Model -- Epoch %d error on validation set: %.4f" % (epoch, epoch_error))
                
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

                if epoch > best_epoch+configs['opt']['patience']:
                    print("Model -- Epoch %d has lowest error!" % (best_epoch))
                    break
        
        y_hat_valid = y_hat_valid.unsqueeze(1).detach().cpu().numpy()
        y_valid = y_valid.detach().cpu().numpy()
        print(y_hat_valid.shape, y_valid.shape)
        # print(np.corrcoef(y_hat_valid, y_valid))
        # reload best epoch
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
                Vi_f= p.to("cpu").detach().numpy()
            elif tuple(p.shape) == (1, self.n_binary):
                Wi = p.to("cpu").detach().numpy()
            elif tuple(p.shape) == (1, ):
                W0 = p.to("cpu").detach().numpy()

        # build the QUBO graph
        q = gen_symbols(BinaryPoly, self.n_binary)
        f_E = sum_poly(configs['opt']['factor_num'], lambda f: ((sum_poly(self.n_binary, lambda i: Vi_f[i][f] * q[i]))**2 - sum_poly(self.n_binary, lambda i: Vi_f[i][f]**2 * q[i]**2)))/2 \
            + sum_poly(self.n_binary, lambda i: Wi[0][i]*q[i]) \
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
            except RuntimeError as e: # retry after 60s if connection to the solver fails..
                time.sleep(60)
                self.sleep_count += 1

        sols = []
        sol_E = []
        for sol in result:  # Iterate over multiple solutions
            # solution = [sol.values[i] for i in range(self.n_binary)]
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

        # 0 --> certain number of iterations;
        # 1 --> certain number of new molecule;
        # 2 --> exhaustion
        if self.end_cond == 0:
            self.iteration += 1

        binary_new = torch.from_numpy(solution).to(torch.float)# .to(self.device)
        # smiles_new = self.bvae_model.decode_from_binary(binary_new)   # 1 x 1
        # mol_new = Chem.MolFromSmiles(smiles_new)
        print('========', binary_new.shape)
        res= self.decode_many_times(binary_new)
        if res is not None:
            smiles_list = [re[0] for re in res]
            n_reactions = [len(re[1].split(" ")) for re in res]
            #print(n_reactions)
            for re in res:
                smiles = re[0]
                if len(re[1].split(" ")) > 0 and smiles not in self.valid_smiles:
                    #print(smiles, re[1].split(" "))
                    self.valid_smiles.append(smiles)
                    self.new_features.append(latent)
                    self.full_rxn_strs.append(re[1])
        
        #new_features = np.vstack(new_features)
        print('========cal new score')
        scores =[]
        b_valid_smiles=[]
        b_full_rxn_strs=[]
        b_scores=[]
        # b_new_features=[]
        for i in range(len(self.valid_smiles)):
            if metric =="logp":
                mol = rdkit.Chem.MolFromSmiles(self.valid_smiles[i])
                if mol is None:
                    continue
                current_log_P_value = Descriptors.MolLogP(mol)
                current_SA_score = -sascorer.calculateScore(mol)
                cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
                if len(cycle_list) == 0:
                    cycle_length = 0
                else:
                    cycle_length = max([ len(j) for j in cycle_list ])
                if cycle_length <= 6:
                    cycle_length = 0
                else:
                    cycle_length = cycle_length - 6
                current_cycle_score = -cycle_length
                current_SA_score_normalized = (current_SA_score - sascore_m) / sascore_s
                current_log_P_value_normalized = (current_log_P_value - logp_m) / logp_s
                current_cycle_score_normalized = (current_cycle_score - cycle_m) / cycle_s
                score = current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized
                scores.append(-score)
                b_valid_smiles.append(self.valid_smiles[i])
                b_full_rxn_strs.append(self.full_rxn_strs[i])
                # b_new_features.append(new_features[i])
            if metric=="qed":
                mol = rdkit.Chem.MolFromSmiles(self.valid_smiles[i])
                if mol!=None:
                    score = QED.qed(mol)
                    scores.append(-score)
                    b_valid_smiles.append(self.valid_smiles[i])
                    b_full_rxn_strs.append(self.full_rxn_strs[i])
                    # b_new_features.append(new_features[i])
        # new_features = np.vstack(b_new_features)
        if len(scores) > 0:
            self.X_train = np.concatenate([ self.X_train, binary_new ], 0)
            self.y_train = np.concatenate([ self.y_train, np.array(scores)[ :, None ] ], 0)

        # for i in range(len(b_valid_smiles)):
        #     line = " ".join([b_valid_smiles[i], b_full_rxn_strs[i], str(scores[i])])
        #     writer.write(line + "\n")

        # # skip invalid smiles
        # if mol_new is None:
        #     return

        # if smiles_new in self.train_smiles:
        #     if self.end_cond == 2:
        #         self.iteration += 1
        #     return

        # # fm_pred = fm_model(binary_new)
        # # fm_pred = fm_pred.detach().cpu().numpy()

        # # assert np.round(fm_pred, 3) == np.round(energy[0], 3)    # ensure correctness of qubo
        # if self.opt_target == 'max':
        #     target_new = -self.get_score(mol_new)
        # else:
        #     target_new = self.get_score(mol_new)
        # print("energy: %.3f; target: %.3f" % (energy[0], target_new))
        # self.train_smiles.append(smiles_new)
        # # self.train_binary = torch.vstack((self.train_binary, binary_new))
        # binary_new = binary_new.to('cpu').numpy()
        # self.train_binary = np.vstack((self.train_binary, binary_new))
        # print(self.train_binary.shape)
        # self.train_targets.append(target_new)


        # # if new molecule is generated:
        # if self.end_cond == 1:
        #     self.iteration += 1
        # if self.end_cond == 2:
        #     self.iteration = 0 # if new molecule is generated, reset to 0

        # self.results_smiles.append(smiles_new)
        # self.results_binary.extend(solution)
        # self.results_scores.append(-target_new)

        # print(self.X_train.shape, self.y_train.shape, np.array(scores)[ :, None ].shape)
        # logging.info("Iteration %d: QUBO energy -- %.4f, actual energy -- %.4f, smiles -- %s" % (self.iteration, energy[0]))
        
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


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
    parser.add_option("-l", "--latent", dest="latent_size", default=50)
    parser.add_option("-d", "--depth", dest="depth", default=2)
    parser.add_option("-s", "--save_dir", dest="save_path")
    parser.add_option("-t", "--data_path", dest="data_path")
    parser.add_option("-v", "--vocab_path", dest="vocab_path")
    parser.add_option("-m", "--metric", dest="metric")
    parser.add_option("-r", "--seed", dest="seed", default=1)
    opts, _ = parser.parse_args()

    # get parameters
    hidden_size = int(opts.hidden_size)
    latent_size = int(opts.latent_size)
    depth = int(opts.depth)
    vocab_path = opts.vocab_path
    data_filename = opts.data_path
    w_save_path = opts.save_path
    metric = opts.metric
    seed = int(opts.seed)


    # load model
    if torch.cuda.is_available():
        #device = torch.device("cuda:1")
        device = torch.device("cuda")
        torch.cuda.set_device(1)
    else:
        device = torch.device("cpu")


    print("hidden size:", hidden_size, "latent_size:", latent_size, "depth:", depth)
    print("loading data.....")
    data_filename = opts.data_path
    routes, scores = read_multistep_rxns(data_filename)
    # routes = routes[:10]
    # scores = scores[:10]
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
    fgm_trees = [FragmentTree(rxn_trees[i].molecule_nodes[0].smiles) for i in ind_list]
    rxn_trees = [rxn_trees[i] for i in ind_list]
    data_pairs=[]
    for fgm_tree, rxn_tree in zip(fgm_trees, rxn_trees):
        data_pairs.append((fgm_tree, rxn_tree))
    cset=set()
    for fgm_tree in fgm_trees:
        for node in fgm_tree.nodes:
            cset.add(node.smiles)
    cset = list(cset)
    if vocab_path is None:
        fragmentDic = FragmentVocab(cset)
    else:
        fragmentDic = FragmentVocab(cset, filename =vocab_path)

    print("size of fragment dic:", fragmentDic.size())



    # loading model

    mpn = MPN(hidden_size, depth)
    model = bFTRXNVAE(fragmentDic, reactantDic, templateDic, hidden_size, latent_size, depth, device, fragment_embedding=None, reactant_embedding=None, template_embedding=None)
    checkpoint = torch.load(w_save_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("finished loading model...")

    print("number of samples:", len(data_pairs))
    data_pairs = data_pairs
    latent_list=[]
    score_list=[]
    print("num of samples:", len(rxn_trees))
    latent_list =[]
    score_list=[]
    if metric =="qed":
        for i, data_pair in enumerate(data_pairs):
            latent = model.encode([data_pair])
            #print(i, latent.size(), latent)
            latent_list.append(latent[0])
            rxn_tree = data_pair[1]
            smiles = rxn_tree.molecule_nodes[0].smiles
            score_list.append(get_qed_score(smiles))
            print(i, len(score_list))
    if metric =="logp":
        logP_values = np.loadtxt('logP_values.txt')
        SA_scores = np.loadtxt('SA_scores.txt')
        cycle_scores = np.loadtxt('cycle_scores.txt')

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
    scores = scores.reshape((-1,1))
    latents = latents.detach().numpy()
    n = latents.shape[0]
    print('===================', n)
    permutation = np.random.choice(n, n, replace = False)
    X_train = latents[ permutation, : ][ 0 : int(np.round(0.9 * n)), : ]
    X_test = latents[ permutation, : ][ int(np.round(0.9 * n)) :, : ]
    y_train = -scores[ permutation ][ 0 : int(np.round(0.9 * n)) ]
    y_test = -scores[ permutation ][ int(np.round(0.9 * n)) : ]
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    if metric == "logp":
        parameters = [logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s]
    else: 
        parameters =[]
    
    with open('config/config.yaml','r') as f:
        configs = yaml.safe_load(f)

    main(X_train, y_train, X_test, y_test, molecules, -scores, model, parameters, configs, metric, seed)