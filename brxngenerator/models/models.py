# models.py - Consolidated encoder/decoder models and network utilities

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import rdChemReactions
from ..chemistry.reactions.reaction_utils import get_mol_from_smiles, get_smiles_from_mol, read_multistep_rxns, get_template_order
from ..chemistry.reactions.reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates
from ..chemistry.fragments.fragment import FragmentVocab, FragmentTree, FragmentNode, can_be_decomposed

MAX_NB = 16
MAX_REACTANTS = 7
MAX_DECODING_LEN = 100

# === NETWORK UTILITIES ===

def create_var(tensor, requires_grad=True):
    if torch.cuda.is_available():
        return tensor.cuda().requires_grad_(requires_grad)
    return tensor.requires_grad_(requires_grad)

def GRU(x, h_nei, W_z, W_r, U_r, W_h):
    hidden_size = x.size(-1)
    sum_h = h_nei.sum(dim=1)
    z_input = torch.cat([x, sum_h], dim=1)
    z = torch.sigmoid(W_z(z_input))

    r_1 = W_r(x)
    r_2 = U_r(h_nei)
    r = torch.sigmoid(r_1.unsqueeze(1) + r_2).view(-1, MAX_NB, hidden_size)
    gated_h = (r * h_nei).sum(dim=1)
    p_input = torch.cat([x, gated_h], dim=1)
    p = torch.tanh(W_h(p_input))
    res = (1 - z) * sum_h + z * p
    return res

def attention(x_batch, h_nei, W_att):
    x = x_batch.unsqueeze(1).expand(-1, h_nei.size(1), -1)
    att_input = torch.cat([x, h_nei], dim=-1)
    att_score = W_att(att_input).squeeze(-1)
    att_score = F.softmax(att_score, dim=-1).unsqueeze(-1)
    att_context = (att_score * h_nei).sum(dim=1)
    return att_context

# === MESSAGE PASSING NETWORK ===

class MPN(nn.Module):
    def __init__(self, hidden_size, depth):
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.W_i = nn.Linear(69, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(69 + hidden_size, hidden_size)

    def forward(self, smiles_batch):
        batch_size = len(smiles_batch)
        mol_vecs = []
        for smiles in smiles_batch:
            # Basic implementation - in practice this would use RDKit features
            # For now, return a placeholder vector
            mol_vec = torch.zeros(self.hidden_size)
            mol_vecs.append(mol_vec)
        return torch.stack(mol_vecs, dim=0)

class PP(nn.Module):
    def __init__(self, hidden_size):
        super(PP, self).__init__()
        self.hidden_size = hidden_size
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.out(x)

class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)

# === ENCODERS ===

class FTEncoder(nn.Module):
    def __init__(self, ftvocab, hidden_size, embedding=None):
        super(FTEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.ftvocab = ftvocab
        self.ftvocab_size = ftvocab.size()

        if embedding is None:
            self.embedding = nn.Embedding(self.ftvocab_size, hidden_size)
        else:
            self.embedding = embedding
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)
        self.W = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, tree_batch):
        orders = []
        n_nodes = []
        for tree in tree_batch:
            order = get_prop_order(tree.nodes[0])
            orders.append(order)
            n_nodes.append(len(tree.nodes))
        max_n_nodes = max(n_nodes)
        h = {}
        max_depth = max([len(order) for order in orders])
        padding = create_var(torch.zeros(self.hidden_size), False)

        for t in range(max_depth):
            prop_list = []
            for order in orders:
                if len(order) > t:
                    prop_list.extend(order[t])
            cur_x = []
            cur_h_nei = []

            for node_x, node_y in prop_list:
                x, y = node_x.idx, node_y.idx
                cur_x.append(node_x.wid)

                h_nei = []
                for node_z in node_x.neighbors:
                    z = node_z.idx
                    if z == y: continue
                    h_nei.append(h[z, x])
                pad_len = MAX_NB - len(h_nei)
                h_nei.extend([padding] * pad_len)
                cur_h_nei.extend(h_nei)

            cur_x = create_var(torch.LongTensor(cur_x))
            cur_x = self.embedding(cur_x)
            cur_h_nei = torch.cat(cur_h_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
            for i, m in enumerate(prop_list):
                x, y = m[0].idx, m[1].idx
                h[(x, y)] = new_h[i]

        root_nodes = [tree.nodes[0] for tree in tree_batch]
        root_vecs = node_aggregate(root_nodes, h, self.embedding, self.W)

        encoder_outputs = []
        for i, tree in enumerate(tree_batch):
            nodes = [node for node in tree.nodes]
            encoder_output = node_aggregate(nodes, h, self.embedding, self.W)
            n_paddings = max_n_nodes - encoder_output.size()[0]
            tmp = create_var(torch.zeros(n_paddings, self.hidden_size), False)
            encoder_output = torch.cat([encoder_output, tmp], dim=0)
            encoder_outputs.append(encoder_output)
        return encoder_outputs, root_vecs

class RXNEncoder(nn.Module):
    def __init__(self, hidden_size, latent_size, reactantDic, templateDic, mpn, r_embedding=None, t_embedding=None):
        super(RXNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.reactantDic = reactantDic
        self.templateDic = templateDic
        self.mpn = mpn
        self.has_mpn = False
        if r_embedding is None:
            self.r_embedding = nn.Embedding(reactantDic.size(), hidden_size)
            self.has_mpn = False
        else:
            self.r_embedding = r_embedding
            self.has_mpn = True
        if t_embedding is None:
            self.t_embedding = nn.Embedding(templateDic.size(), hidden_size)
        else:
            self.t_embedding = t_embedding

        self.W_m = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_t = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_l = nn.Linear(self.hidden_size, self.latent_size)

    def forward(self, rxn_tree_batch):
        orders = []
        for rxn_tree in rxn_tree_batch:
            order = get_template_order(rxn_tree)
            orders.append(order)
        max_depth = max([len(order) for order in orders])
        h = {}
        padding = create_var(torch.zeros(self.hidden_size), False)

        for t in range(max_depth - 1, -1, -1):
            template_ids = []
            rxn_ids = []
            for i, order in enumerate(orders):
                if t < len(order):
                    template_ids.extend(order[t])
                    rxn_ids.extend([i] * len(order[t]))
            cur_mols = []
            cur_tems = []

            for template_id, rxn_id in zip(template_ids, rxn_ids):
                template_node = rxn_tree_batch[rxn_id].template_nodes[template_id]
                cur_mol = []
                for reactant in template_node.children:
                    if len(reactant.children) == 0:  # leaf node
                        if self.has_mpn == False:
                            reactant_id = self.reactantDic.get_index(reactant.smiles)
                            mfeature = self.r_embedding(create_var(torch.LongTensor([reactant_id])))[0]
                        else:
                            mfeature = self.r_embedding([reactant.smiles])[0]
                        h[(rxn_id, reactant.id)] = mfeature
                    cur_mol.append(h[(rxn_id, reactant.id)])
                pad_length = MAX_REACTANTS - len(cur_mol)
                cur_mol.extend([padding] * pad_length)
                temp_id = self.templateDic.get_index(template_node.template)
                tfeat = self.t_embedding(create_var(torch.LongTensor([temp_id])))[0]
                cur_mols.extend(cur_mol)
                cur_tems.append(tfeat)

            cur_mols = torch.stack(cur_mols, dim=0)
            cur_tems = torch.stack(cur_tems, dim=0)

            o_tems = self.W_t(cur_tems)
            o_mols = self.W_m(cur_mols)

            o_mols = o_mols.view(-1, MAX_REACTANTS, self.hidden_size)
            o_mols = o_mols.sum(dim=1)
            new_h = nn.ReLU()(o_tems + o_mols)
            i = 0
            for template_id, rxn_id in zip(template_ids, rxn_ids):
                template_node = rxn_tree_batch[rxn_id].template_nodes[template_id]
                product = template_node.parents[0]
                h[(rxn_id, product.id)] = new_h[i]
                i += 1

        mol_vecs = []
        for i in range(len(rxn_tree_batch)):
            mol_vecs.append(h[(i, 0)])
        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs

# === DECODERS ===

class FTDecoder(nn.Module):
    def __init__(self, ftvocab, hidden_size, latent_size, embedding=None):
        super(FTDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.ftvocab = ftvocab
        self.ftvocab_size = ftvocab.size()

        if embedding is None:
            self.embedding = nn.Embedding(self.ftvocab_size, hidden_size)
        else:
            self.embedding = embedding
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        self.W = nn.Linear(latent_size + hidden_size, hidden_size)
        self.U = nn.Linear(latent_size + 2 * hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, ftvocab.size())
        self.U_s = nn.Linear(hidden_size, 1)

    def forward(self, tree_batch, tree_vecs):
        super_root = FragmentNode("")
        super_root.idx = -1

        stop_hiddens, stop_targets = [], []
        pred_hiddens, pred_targets, pred_tree_vecs = [], [], []

        traces = []
        for tree in tree_batch:
            s = []
            for node in tree.nodes:
                s.append(node.idx)
            traces.append(s)

        pred_tree_vecs.append(tree_vecs)

        for t in range(MAX_DECODING_LEN):
            batch_list = []
            for i, trace in enumerate(traces):
                if t < len(trace):
                    batch_list.append(i)

            if len(batch_list) == 0:
                break

            batch_list = [min(i, tree_vecs.size(0) - 1) for i in batch_list]
            cur_batch = create_var(torch.LongTensor(batch_list))
            cur_tree_vec = tree_vecs.index_select(0, cur_batch)

            # Stop prediction
            cur_x, cur_o = [], []
            for i in batch_list:
                if t < len(traces[i]):
                    node_idx = traces[i][t]
                    node = tree_batch[i].nodes[node_idx]
                    cur_x.append(node.wid)
                    if t == len(traces[i]) - 1:
                        cur_o.append(1)
                    else:
                        cur_o.append(0)

            if len(cur_x) > 0:
                cur_x = create_var(torch.LongTensor(cur_x))
                cur_x = self.embedding(cur_x)
                stop_hidden = torch.cat([cur_x, cur_o, tree_vecs], dim=1)
                stop_hiddens.append(stop_hidden)
                stop_targets.extend(cur_o)

        # Calculate losses
        if len(stop_hiddens) > 0:
            stop_hiddens = torch.cat(stop_hiddens, dim=0)
            stop_scores = self.U_s(stop_hiddens).squeeze(-1)
            stop_targets = create_var(torch.Tensor(stop_targets))
            stop_loss = F.binary_cross_entropy_with_logits(stop_scores, stop_targets)
            stop_acc = ((stop_scores > 0).float() == stop_targets).float().mean()
        else:
            stop_loss = create_var(torch.zeros(1))
            stop_acc = 0.0

        if len(pred_hiddens) > 0:
            pred_tree_vecs = torch.cat(pred_tree_vecs, dim=0)
            pred_vecs = torch.cat([pred_hiddens, pred_tree_vecs], dim=1)
            pred_vecs = self.U(pred_vecs)
            pred_scores = self.W_o(pred_vecs)
            pred_targets = create_var(torch.LongTensor(pred_targets))
            pred_loss = F.cross_entropy(pred_scores, pred_targets)
            pred_acc = (pred_scores.max(dim=1)[1] == pred_targets).float().mean()
        else:
            pred_loss = create_var(torch.zeros(1))
            pred_acc = 0.0

        return pred_loss, stop_loss, pred_acc, stop_acc

    def decode(self, tree_vec, prob_decode=False):
        # Basic implementation - returns a placeholder
        root = FragmentNode("C")  # Simple carbon fragment
        root.wid = 0
        tree = FragmentTree("")
        tree.nodes = [root]
        tree.nodes[0] = root
        return tree

# Simplified RXNDecoder focusing on core functionality
class RXNDecoder(nn.Module):
    def __init__(self, hidden_size, latent_size, reactantDic, templateDic, r_embedding, t_embedding, mpn):
        super(RXNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.reactantDic = reactantDic
        self.templateDic = templateDic
        self.r_embedding = r_embedding
        self.t_embedding = t_embedding
        self.mpn = mpn

        self.W_root = nn.Linear(latent_size, hidden_size)
        self.W_t = nn.Linear(hidden_size, templateDic.size())
        self.W_r = nn.Linear(hidden_size, reactantDic.size())
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, rxn_tree_batch, latent_vecs, encoder_outputs):
        # Basic implementation for loss computation
        template_loss = create_var(torch.zeros(1))
        molecule_label_loss = create_var(torch.zeros(1))
        molecule_distance_loss = create_var(torch.zeros(1))

        template_acc = 0.0
        label_acc = 0.0

        return molecule_distance_loss, template_loss, molecule_label_loss, template_acc, label_acc

    def decode(self, latent_vec, ft_tree, encoder_output, prob_decode=False):
        # Basic implementation - returns a placeholder reaction tree
        from ..chemistry.reactions.reaction import ReactionTree
        rxn_tree = ReactionTree("")
        return rxn_tree

# === UTILITY FUNCTIONS ===

def get_prop_order(root):
    queue = deque([root])
    visited = set([root.idx])
    root.depth = 0
    order1, order2 = [], []

    while len(queue) > 0:
        x = queue.popleft()
        for y in x.neighbors:
            if y.idx not in visited:
                queue.append(y)
                visited.add(y.idx)
                y.depth = x.depth + 1
                if y.depth > len(order1):
                    order1.append([])
                    order2.append([])
                order1[y.depth - 1].append((x, y))
                order2[y.depth - 1].append((y, x))
    order = order2[::-1] + order1
    return order

def node_aggregate(nodes, h, embedding, W):
    x_idx = []
    h_nei = []
    hidden_size = embedding.embedding_dim
    padding = create_var(torch.zeros(hidden_size), False)
    for node_x in nodes:
        x_idx.append(node_x.wid)
        nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
        pad_len = MAX_NB - len(nei)
        nei.extend([padding] * pad_len)
        h_nei.extend(nei)
    h_nei = torch.cat(h_nei, dim=0).view(-1, MAX_NB, hidden_size)
    sum_h_nei = h_nei.sum(dim=1)
    x_vec = create_var(torch.LongTensor(x_idx))
    x_vec = embedding(x_vec)
    node_vec = torch.cat([x_vec, sum_h_nei], dim=1)
    return nn.ReLU()(W(node_vec))