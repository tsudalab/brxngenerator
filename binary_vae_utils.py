# binary_vae_utils.py

import sys
from abc import ABC, abstractmethod
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import rdkit
from rdkit.Chem import QED, MolFromSmiles
import gurobipy as gp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import r2_score
sys.path.append('./rxnft_vae')


sys.path.append(os.path.join(os.path.dirname(__file__), 'rxnft_vae'))
from rxnft_vae.reaction_utils import get_qed_score, get_clogp_score
# [ECC] Import ECC utilities for latent processing
from rxnft_vae.ecc import create_ecc_codec, extract_info_bits

class TorchFM(nn.Module):
    def __init__(self, n=None, k=None):
        super().__init__()
        self.factor_matrix = nn.Parameter(torch.randn(n, k), requires_grad=True)
        self.lin = nn.Linear(n, 1)

    def forward(self, x):
        out_1 = torch.matmul(x, self.factor_matrix).pow(2).sum(1, keepdim=True)
        out_2 = torch.matmul(x.pow(2), self.factor_matrix.pow(2)).sum(1, keepdim=True)
        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        return out
    
    def to_qubo(self):
        n, k = self.factor_matrix.shape
        V = self.factor_matrix.detach().numpy()
        W = self.lin.weight.detach().numpy().flatten()
        f_E = self.lin.bias.detach().numpy().flatten()
        q = np.zeros((n + 1, n + 1))
        q[:-1, :-1] = V @ V.T
        np.fill_diagonal(q[:-1, :-1], W)
        return (q, f_E)


class BaseQuboSolver(ABC):
    @abstractmethod
    def solve(self, model: TorchFM):
        pass


class GurobiQuboSolver(BaseQuboSolver):
    def __init__(self, options=None):
        env_params = options or {}
        # Prefer local license via gurobi.lic or GRB_LICENSE_FILE; fall back to provided params
        # Auto-wire gurobi.lic in project root if present and no GRB_LICENSE_FILE set
        project_root = os.path.dirname(os.path.abspath(__file__))
        license_path = os.path.join(project_root, "gurobi.lic")
        if not os.environ.get("GRB_LICENSE_FILE") and os.path.exists(license_path):
            os.environ["GRB_LICENSE_FILE"] = license_path

        cloud_params = {k: v for k, v in env_params.items() if k in ["WLSACCESSID", "WLSSECRET", "LICENSEID"]}
        if cloud_params:
            print("Using Gurobi with provided parameters (Cloud/Env).")
            self.gurobi_env = gp.Env(params=cloud_params)
        else:
            print("Using Gurobi with local license or defaults.")
            self.gurobi_env = gp.Env()

    def solve(self, model: TorchFM):
        V = model.factor_matrix.detach().cpu().numpy()
        W = model.lin.weight.detach().cpu().numpy().flatten()
        Q_dense = V @ V.T
        np.fill_diagonal(Q_dense, W)
        
        m = gp.Model("qubo_fm", env=self.gurobi_env)
        m.setParam('OutputFlag', 0)
        n_vars = Q_dense.shape[0]
        x = m.addMVar(shape=n_vars, vtype=gp.GRB.BINARY, name="x")
        
        m.setObjective(x @ Q_dense @ x, gp.GRB.MAXIMIZE)
        m.optimize()

        if m.status == gp.GRB.OPTIMAL:
            solutions = x.X.reshape(1, -1)
            energies = np.array([m.objVal])
            return solutions, energies
        else:
            print("Gurobi did not find an optimal solution. Status code:", m.status)
            return np.array([]), np.array([])


class FactorizationMachineSurrogate:
    def __init__(self, n_binary, k_factors, lr, decay_weight, batch_size, max_epoch, patience, param_init, cache_dir, prop, client, random_seed, device):
        self.device = device
        self.n_binary = n_binary
        self.k_factors = k_factors
        self.lr = lr
        self.decay_weight = decay_weight
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.patience = patience
        self.param_init = param_init
        self.cache_dir = cache_dir
        self.prop = prop
        self.client = client
        self.random_seed = random_seed
        self.model = TorchFM(self.n_binary, self.k_factors).to(self.device)

    def train(self, X_train, y_train, X_test, y_test):
        model = self.model
        for param in model.parameters():
            if param.dim() == 1: nn.init.constant_(param, 0)
            else: nn.init.uniform_(param, -self.param_init, self.param_init)

        dataset_train = MolData(X_train, y_train)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True)
        dataset_valid = MolData(X_test, y_test)
        dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.decay_weight)
        criterion = nn.MSELoss()
        lowest_error, best_epoch = float('inf'), 0

        for epoch in range(self.max_epoch):
            model.train()
            for batch_x, batch_y in dataloader_train:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_hat_test = torch.cat([model(batch_x.to(self.device)) for batch_x, _ in dataloader_valid])
                epoch_error = criterion(y_test.to(self.device), y_hat_test).item()
                r2_test = r2_score(y_test.cpu().numpy(), y_hat_test.cpu().numpy())
                if epoch % 100 == 0:
                    print(f"Model -- Epoch {epoch} error on validation set: {epoch_error:.4f}, r2 on validation set: {r2_test:.4f}")

                if epoch_error < lowest_error:
                    model_path = os.path.join(self.cache_dir, f"fm_model-{self.prop}-{self.client}-dim{self.n_binary}-seed{self.random_seed}")
                    torch.save(model.state_dict(), model_path)
                    lowest_error = epoch_error
                    best_epoch = epoch

                if epoch > best_epoch + self.patience:
                    print(f"Model -- Early stopping at epoch {epoch}. Best epoch was {best_epoch} with error {lowest_error:.4f}.")
                    break

        best_model_path = os.path.join(self.cache_dir, f"fm_model-{self.prop}-{self.client}-dim{self.n_binary}-seed{self.random_seed}")
        self.model.load_state_dict(torch.load(best_model_path))
        print("Best torchfm model loaded.")


class MoleculeOptimizer:
    def __init__(self, bvae_model, surrogate_model, qubo_solver, X_train, y_train, X_test, y_test, optimize_num, metric, random_seed, results_dir, logp_paths, device):
        self.device = device
        self.bvae_model = bvae_model.to(self.device)
        self.surrogate = surrogate_model
        self.solver = qubo_solver
        self.X_train, self.y_train = X_train.to(self.device), y_train.to(self.device)
        self.X_test, self.y_test = X_test.to(self.device), y_test.to(self.device)
        self.optimize_num, self.metric, self.random_seed, self.results_dir = optimize_num, metric, random_seed, results_dir
        self.logp_paths = logp_paths
        self.valid_smiles, self.full_rxn_strs, self.new_features = [], [], []

    def optimize(self):
        for i in range(self.optimize_num):
            self.iteration = i
            print(f"--- Starting Optimization Iteration {self.iteration + 1}/{self.optimize_num} ---")
            self.surrogate.train(self.X_train, self.y_train, self.X_test, self.y_test)
            solutions, _ = self.solver.solve(self.surrogate.model)
            if solutions.size == 0:
                print("Solver failed to find a solution. Skipping update.")
                continue
            self._update(solutions)

    def _update(self, solution):
        print('========Updating dataset with new solutions=========')
        binary_new = torch.from_numpy(solution).to(torch.float).to(self.device)
        print("Shape of new binary vector:", binary_new.shape)
        
        res = self.decode_many_times(binary_new[0]) # Assuming single solution
        if not res:
            print('========Decoding failed to produce valid molecules.=========')
            return

        new_smiles, new_rxn_strs = [], []
        for smiles, rxn_str in res:
            if smiles and smiles not in self.valid_smiles:
                self.valid_smiles.append(smiles)
                new_smiles.append(smiles)
                new_rxn_strs.append(rxn_str)
        
        if not new_smiles:
            print("No new valid molecules generated.")
            return

        print(f"Generated {len(new_smiles)} new unique and valid molecules.")
        self.new_features.append(binary_new) # Storing the feature vector
        
        scores = self.calculate_scores(new_smiles)
        if not scores:
            print("No valid scores could be calculated.")
            return

        # Use average score of newly generated molecules for the new data point
        avg_score = torch.tensor([[np.mean(scores)]], dtype=torch.float32, device=self.device)
        
        self.X_train = torch.cat([self.X_train, binary_new], dim=0)
        self.y_train = torch.cat([self.y_train, -avg_score], dim=0) # Use negative score for maximization
        print('========Training set updated. New size:', self.X_train.shape[0])

        self.write_results(new_smiles, new_rxn_strs, scores)
    
    def calculate_scores(self, smiles_list):
        scores = []
        if self.metric == "logp":
            logP_values, SA_scores, cycle_scores = [np.loadtxt(p) for p in self.logp_paths.values()]
            logp_m, logp_s = np.mean(logP_values), np.std(logP_values)
            sascore_m, sascore_s = np.mean(SA_scores), np.std(SA_scores)
            cycle_m, cycle_s = np.mean(cycle_scores), np.std(cycle_scores)
            
        for smiles in smiles_list:
            mol = MolFromSmiles(smiles)
            if mol is None: continue
            if self.metric == "qed":
                scores.append(QED.qed(mol))
            elif self.metric == "logp":
                score = get_clogp_score(smiles, logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s)
                scores.append(score)
        return scores

    def write_results(self, smiles, rxns, scores):
        filename = os.path.join(self.results_dir, f"{self.random_seed}_{self.metric}.txt")
        print("Writing results to:", filename)
        with open(filename, "a") as writer:
            for i in range(len(smiles)):
                line = f"{smiles[i]} {rxns[i]} {-scores[i]}" # Store negative score
                writer.write(line + "\n")

    def decode_many_times(self, latent_half):
        prob_decode = True
        binary_size = self.bvae_model.binary_size
        product_list = []
        
        latent_half = latent_half.long()
        
        for _ in tqdm(range(5000), desc="Decoding Attempts"):
            if len(product_list) > 5: break
            
            random_half = torch.randint(0, 2, (1, binary_size), device=self.device).long()
            latent_full = torch.cat([latent_half.unsqueeze(0), random_half], dim=1)

            binary = F.one_hot(latent_full, num_classes=2).float().view(1, -1)
            ft_mean, rxn_mean = binary[:, :binary_size*2], binary[:, binary_size*2:]
            
            generated_tree = self.bvae_model.fragment_decoder.decode(ft_mean, prob_decode)
            g_encoder_output, _ = self.bvae_model.fragment_encoder([generated_tree])
            product, reactions = self.bvae_model.rxn_decoder.decode(rxn_mean, g_encoder_output, prob_decode)

            if product: product_list.append((product, reactions))

        return product_list if product_list else None


class MolData(Dataset):
    def __init__(self, binary, targets):
        self.binary, self.targets = binary, targets
    def __len__(self): return len(self.binary)
    def __getitem__(self, index): return self.binary[index], self.targets[index]


# [ECC] Helper function for ECC-aware latent processing
def extract_latent_info_bits(latent_tensor, ecc_codec=None):
    """
    Extract information bits from latent tensor with optional ECC decoding.
    
    Args:
        latent_tensor: Tensor of shape (batch_size, latent_size) 
        ecc_codec: ECC codec instance (None for no ECC)
        
    Returns:
        Information bits tensor
    """
    if ecc_codec is None:
        return latent_tensor
    else:
        # Decode each sample in the batch
        return extract_info_bits(latent_tensor, ecc_codec)


def prepare_dataset(model, data_pairs, latent_size, metric, logp_paths, ecc_type='none', ecc_R=3):
    print("Preparing dataset. Number of samples:", len(data_pairs))
    latent_list, score_list = [], []
    
    # [ECC] Initialize ECC codec for latent processing
    ecc_codec = create_ecc_codec(ecc_type, R=ecc_R)
    if ecc_codec is not None:
        print(f"[ECC] Using {ecc_type} with R={ecc_R} for latent processing")

    if metric == "logp":
        logP_values, SA_scores, cycle_scores = [np.loadtxt(p) for p in logp_paths.values()]
        logp_m, logp_s = np.mean(logP_values), np.std(logP_values)
        sascore_m, sascore_s = np.mean(SA_scores), np.std(SA_scores)
        cycle_m, cycle_s = np.mean(cycle_scores), np.std(cycle_scores)

    for data_pair in tqdm(data_pairs, desc="Encoding data and getting scores"):
        latent = model.encode([data_pair])[0]
        latent_list.append(latent)
        smiles = data_pair[1].molecule_nodes[0].smiles
        if metric == "qed":
            score_list.append(get_qed_score(smiles))
        elif metric == "logp":
            score_list.append(get_clogp_score(smiles, logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s))

    latents = torch.stack(latent_list, dim=0)
    scores = np.array(score_list).reshape((-1, 1))
    
    # Use only the first half of the latent vector as per original logic
    half_latents = latents[:, : latent_size // 2]
    
    # [ECC] Apply ECC decoding if enabled to extract information bits
    if ecc_codec is not None:
        # Extract information bits from the first half
        half_latents = extract_latent_info_bits(half_latents, ecc_codec)
        print(f"[ECC] Extracted {half_latents.shape[1]} info bits from {latent_size // 2} code bits")
    
    latents = half_latents.detach().cpu().numpy()
    n = latents.shape[0]
    
    permutation = np.random.permutation(n)
    train_idx, test_idx = permutation[:int(0.9 * n)], permutation[int(0.9 * n):]
    
    X_train, X_test = latents[train_idx, :], latents[test_idx, :]
    y_train, y_test = -scores[train_idx], -scores[test_idx] # Maximize score = Minimize -score
    
    print(f"Dataset prepared. Shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")
    return X_train, y_train, X_test, y_test
