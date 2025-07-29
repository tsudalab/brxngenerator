import sys
from abc import ABC, abstractmethod
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import rdkit
from rdkit.Chem import QED
import gurobipy as gp # 假设你使用 Gurobi 作为求解器
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import r2_score
from rxnft_vae.reaction_utils import get_qed_score, get_clogp_score
from rxnft_vae.reaction_utils import get_clogp_score





sys.path.append('./rxnft_vae')

UPDATE_ITER = 1

METRIC = "qed"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FACTOR_NUM = 8

PARAM_INIT = 0.03

BATCH_SIZE = 3000

LR = 0.001     

DECAY_WEIGHT = 0.01

MAX_EPOCH = 10000

CACHE = "./cache"

PROP = "QED"

CLIENT = "Gurobi"

PATIENCE = 300

OPTIMIZE_NUM = 100
    
RANDOM_SEED = 42


class TorchFM(nn.Module):
    '''This is the Factorization Machine model implemented in PyTorch, which helps
    to compute the quotation with the binary'''
    def __init__(self, n=None, k=None):
        super().__init__()
        # factor_matrix is the factor matrix, which has shape (n, k)
        self.factor_matrix = nn.Parameter(torch.randn(n, k), requires_grad=True)
        # from n input to 1 output's linear layer
        # n is the number of binary features, k is the number of factors
        self.lin = nn.Linear(n, 1)

    def forward(self, x): 
        '''x is the input binary vector, which has shape (batch_size, n)
        # Σ_{i<j} <v_i, v_j> x_i x_j = 0.5 * ((Σ(v_i * x_i))^2 - Σ((v_i * x_i)^2))。
        out_1 is (Σ(v_i * x_i))^2
        out_2 is Σ((v_i * x_i)^2)
        we can regard out_inter as the interaction term of the factorization machine
        out_lin is the linear term of the factorization machine
        out_lin is the output of the linear layer
        x is the input binary vector, which has shape (batch_size, n)
        matmul is matrix multiplication
        '''
        out_1 = torch.matmul(x, self.factor_matrix).pow(2).sum(1, keepdim=True)
        out_2 = torch.matmul(x.pow(2), self.factor_matrix.pow(2)).sum(1, keepdim=True)
        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        return out
    
    def to_qubo(self):
        """
        将模型转换为 QUBO 问题的表示。

        Returns:
            一个包含 QUBO 矩阵和线性项的元组 (q, f_E)。
        """
        n, k = self.factor_matrix.shape
        V = self.factor_matrix.detach().numpy()
        W = self.lin.weight.detach().numpy().flatten()
        f_E = self.lin.bias.detach().numpy().flatten()

        # 创建 QUBO 矩阵
        q = np.zeros((n + 1, n + 1))
        q[:-1, :-1] = V @ V.T
        np.fill_diagonal(q[:-1, :-1], W)
        return (q, f_E)


class BaseQuboSolver(ABC):
    """QUBO 求解器的抽象基类"""
    @abstractmethod
    def solve(self, model: TorchFM):
        """
        接收一个 QUBO 问题并返回解。

        Args:
            qubo_problem: QUBO 问题的表示（可以是你的 (q, f_E) 元组）。

        Returns:
            一个包含解和能量的 NumPy 数组。
        """
        pass


class GurobiQuboSolver(BaseQuboSolver):
    """使用 Gurobi 求解 QUBO 问题的具体实现"""
    def __init__(self, options=None):
        if options is not None:
            print("Using Gurobi with provided options:", options)
        else:
            print("Using Gurobi with default options.")
            
        self.gurobi_env = gp.Env(params=options) if options else gp.Env()

    # 在 GurobiQuboSolver 类的 solve 方法中
    def solve(self, model: TorchFM):
        # 1. 从 TorchFM 模型中获取参数 V 和 W
        V = model.factor_matrix.detach().cpu().numpy()
        W = model.lin.weight.detach().cpu().numpy().flatten()

        # 2. 构建 QUBO 矩阵 Q (最大化问题)
        Q_dense = V @ V.T
        np.fill_diagonal(Q_dense, W)
        
        # 3. 创建 Gurobi 模型
        m = gp.Model("qubo_fm")
        m.setParam('OutputFlag', 0) # 关闭冗余输出
        n_vars = Q_dense.shape[0]
        x = m.addMVar(shape=n_vars, vtype=gp.GRB.BINARY, name="x")

        # 4. 设置目标函数 (我们想最大化 y = x'Qx, Gurobi 默认最小化)
        m.setObjective(x @ Q_dense @ x, gp.GRB.MAXIMIZE)
        
        # 5. 求解
        m.optimize()

        # 6. 返回解和目标值
        if m.status == gp.GRB.OPTIMAL:
            solutions = x.X.reshape(1, -1) # Gurobi 返回一个解
            energies = np.array([m.objVal])
            return solutions, energies
        else:
            # 如果没有找到最优解，返回空值
            return np.array([]), np.array([])
        
    
class FactorizationMachineSurrogate:
    
    def __init__(self, n_binary, k_factors, random_seed=42):
        self.device = DEVICE
        self.model = TorchFM(n_binary, k_factors).to(self.device)
        self.n_binary = n_binary
        self.k_factors = k_factors
        self.random_seed = random_seed


    def train(self, X_train, y_train, X_test, y_test):
            model = self.model
            
            for param in model.parameters():
                if param.dim() == 1:
                    nn.init.constant_(param, 0)  # bias
                else:
                    nn.init.uniform_(param, -PARAM_INIT, PARAM_INIT)  # weights

            print('========shape: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            dataset_train = MolData(X_train, y_train) 
            dataloader_train = DataLoader(dataset=dataset_train,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
            dataset_valid = MolData(X_test, y_test)
            dataloader_valid = DataLoader(dataset=dataset_valid,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

            print('lr: ', LR)
            optimizer = torch.optim.Adam(model.parameters(),
                                        lr=LR,
                                        weight_decay=DECAY_WEIGHT)
            criterion = nn.MSELoss()

            lowest_error = float('inf')
            best_epoch = 0

            for epoch in range(MAX_EPOCH):
                model.train()
                for batch_x, batch_y in dataloader_train:
                    batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)  # Ensure data is on the GPU
                    optimizer.zero_grad()
                    out = model(batch_x)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    y_hat_test = []
                    for batch_x, _ in dataloader_valid:
                        batch_x = batch_x.to(DEVICE)  # Ensure data is on the GPU
                        valid = model(batch_x)
                        y_hat_test.append(valid)
                    y_hat_test = torch.cat(y_hat_test)

                    epoch_error = criterion(y_test.to(DEVICE), y_hat_test)  # Ensure target is on the GPU
                    r2_test = r2_score(y_test.cpu().numpy(), y_hat_test.cpu().numpy())
                    epoch_error = epoch_error.detach().cpu().numpy()
                    if epoch % 100 == 0:
                        print("Model -- Epoch %d error on validation set: %.4f, r2 on validation set: %.4f" % (epoch, epoch_error, r2_test))

                    if epoch_error < lowest_error:
                        torch.save(model.state_dict(),
                                os.path.join(CACHE,
                                                "fm_model-%s-%s-dim%d-seed%d" % (
                                                    PROP,
                                                    CLIENT,
                                                    self.n_binary,
                                                    self.random_seed)))
                        lowest_error = epoch_error
                        best_epoch = epoch

                    if epoch > best_epoch + PATIENCE:
                        print("Model -- Epoch %d has lowest error!" % (best_epoch))
                        break

            y_hat_test = y_hat_test.unsqueeze(1).detach().cpu().numpy()
            y_test = y_test.detach().cpu().numpy()
            print(y_hat_test.shape, y_test.shape)
            
            # 加载最佳模型
            self.model.load_state_dict(torch.load(
                os.path.join(CACHE,
                             "fm_model-%s-%s-dim%d-seed%d" % (
                                 PROP,
                                 CLIENT,
                                 self.n_binary,
                                 self.random_seed)))
            )
            print("best torchfm model loaded.")

class MolData(Dataset):
    '''This class is used to create a dataset for the binary and targets.'''
    def __init__(self, binary, targets):
        self.binary = binary
        self.targets = targets

    def __len__(self):
        return len(self.binary)

    def __getitem__(self, index):
        return self.binary[index], self.targets[index]


class RandomBinaryData(Dataset):
    '''This class is used to create a dataset for the binary. It is used to generate random binary data.'''
    def __init__(self, binary):
        self.binary = binary

    def __len__(self):
        return len(self.binary)

    def __getitem__(self, index):
        return self.binary[index]

class MoleculeOptimizer:
    def __init__(self, bvae_model, X_train,y_train,X_test,y_test, surrogate_model, qubo_solver):
        """
        通过依赖注入的方式，接收各个组件
        """
        self.device = DEVICE
        
        # receive the trained binary VAE model 
        self.bvae_model = bvae_model.to(self.device)
        # receive the surrogate model(which is a factorization machine model)
        self.surrogate = surrogate_model
        # receive the QUBO solver
        self.solver = qubo_solver
        
        # 数据和状态
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.X_test = self.X_test.to(self.device)
        self.y_test = self.y_test.to(self.device)
        
        self.valid_smiles = []
        self.full_rxn_strs = []
        self.new_features = [] # 或者你可能想保存 binary_new
        self.random_seed = RANDOM_SEED

    def optimize(self):
        self.iteration = 0
        n_opt = OPTIMIZE_NUM

        while self.iteration < n_opt:
            print(f"--- Starting Iteration {self.iteration} ---")
            
            # 1. train surrogate model
            self.surrogate.train(self.X_train, self.y_train, self.X_test, self.y_test)
            
            solutions, energies = self.solver.solve(self.surrogate.model)

            
            # 4. update the dataset 
            self._update(solutions, energies)
            
            self.iteration += 1


    def _update(self,
            solution,
            energy):
        print('========Updating dataset with new solutions=========')
        print(solution)
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
                self.new_features.append(binary_new)
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
            if METRIC == "logp":
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

            elif METRIC == "qed":
                print('========computing qed of molecule{}'.format(i))
                score = QED.qed(mol)
                scores.append(-score)
            else:
                raise ValueError("Unsupported metric: {}".format(METRIC))
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
            # 1. 把新的 binary_new 放到同一个 device 上
            binary_new = binary_new.to(self.device)

            # 2. 把 training_score 转成 (N,1) 的 float Tensor，直接指定 device
            scores = torch.tensor(training_score, dtype=torch.float32, device=self.device).unsqueeze(1)

            # 3. 用 torch.cat 拼接
            self.X_train = torch.cat([self.X_train, binary_new], dim=0)
            self.y_train = torch.cat([self.y_train, scores],    dim=0)
            
            print('X_train shape after update:', self.X_train.shape)
            print('y_train shape after update:', self.y_train.shape)


        if METRIC == "logp":
            filename = "./Results/" + str(self.random_seed) + "_logp.txt"
        elif METRIC == "qed":
            filename = "./Results/" + str(self.random_seed) + "_qed.txt"

        print("Writing to file:", filename)
        with open(filename, "a") as writer:
            for i in range(len(b_valid_smiles)):
                line = " ".join([b_valid_smiles[i], b_full_rxn_strs[i], str(b_scores[i])])
                writer.write(line + "\n")

        assert self.X_train.shape[0] == self.y_train.shape[0]

        return

    
    def decode_many_times(self, latent):

        prob_decode = True
        binary_size = self.bvae_model.binary_size

        product_list = []
        for i in tqdm(range(5000)):
            if len(product_list) > 5:
                break
            # latent_new = latent
            # print("start decode many times")
            # print("latent shape", latent.shape)
            latent_new = torch.cat([latent, torch.randint(0, 2, (latent.shape[0], latent.shape[1] * 2 - latent.shape[1]))], dim=1).to(self.device)
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



def prepare_dataset(model=None, data_pairs=None,latent_size=None):
    '''This function is used for generate the metric we need for every molecule in the dataset.'''
    print("number of samples:", len(data_pairs))
    
    data_pairs = data_pairs
    latent_list = []
    score_list = []
    
    
    latent_list = []
    score_list = []
    
    print('========start to compute all scores')

    if METRIC == "qed":
        # tqdm
        for i, data_pair in tqdm(enumerate(data_pairs)):
            latent = model.encode([data_pair])
            latent_list.append(latent[0])
            rxn_tree = data_pair[1]
            smiles = rxn_tree.molecule_nodes[0].smiles
            score_list.append(get_qed_score(smiles))
    
    if METRIC == "logp":
        logP_values = np.loadtxt('./data/logP_values.txt')
        SA_scores = np.loadtxt('./data/SA_scores.txt')
        cycle_scores = np.loadtxt('./data/cycle_scores.txt')

        logp_m = np.mean(logP_values)
        logp_s = np.std(logP_values)

        sascore_m = np.mean(SA_scores)
        sascore_s = np.std(SA_scores)

        cycle_m = np.mean(cycle_scores)
        cycle_s = np.std(cycle_scores)
        for i, data_pair in tqdm(enumerate(data_pairs)):
            # only need the previous half of the latent vector
            latent = model.encode([data_pair])
            latent_list.append(latent[0])
            print("ForTraining, latent shape:", latent_list[-1].shape)
            rxn_tree = data_pair[1]
            smiles = rxn_tree.molecule_nodes[0].smiles
            score_list.append(get_clogp_score(smiles, logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s))
    
    latents = torch.stack(latent_list, dim=0)
    scores = np.array(score_list)
    scores = scores.reshape((-1, 1))

    latents = latents[:, : latent_size // 2]

    
    # move to cpu first
    latents = latents.detach().cpu().numpy()
    
    n = latents.shape[0]
    
    print('===================latent shape', latents.shape)
    
    permutation = np.random.choice(n, n, replace=False)
    X_train = latents[permutation, :][0: int(np.round(0.9 * n)), :]
    X_test = latents[permutation, :][int(np.round(0.9 * n)):, :]
    y_train = -scores[permutation][0: int(np.round(0.9 * n))]
    y_test = -scores[permutation][int(np.round(0.9 * n)):]
    
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    return X_train, y_train, X_test, y_test
    