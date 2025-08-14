
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnutils import create_var, attention
from ftencoder import FTEncoder
from ftdecoder import FTDecoder
from rxndecoder import RXNDecoder, RXNDecoder1
from rxnencoder import RXNEncoder
from mpn import MPN,PP,Discriminator
# [ECC] Import ECC utilities for training-time integration
from ecc import create_ecc_codec


def set_batch_nodeID(ft_trees, ft_vocab):
	tot = 0
	for ft_tree in ft_trees:
		for node in ft_tree.nodes:
			node.idx = tot
			node.wid = ft_vocab.get_index(node.smiles)
			tot +=1
def log_Normal_diag(x, mean, log_var):
	log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
	return torch.mean(log_normal)


class FTRXNVAE(nn.Module):
	def __init__(self, fragment_vocab, reactant_vocab, template_vocab, hidden_size, latent_size, depth, fragment_embedding=None, reactant_embedding=None, template_embedding=None):
		super(FTRXNVAE, self).__init__()
		self.fragment_vocab = fragment_vocab
		self.reactant_vocab = reactant_vocab
		self.template_vocab = template_vocab
		self.depth = depth


		self.hidden_size = hidden_size
		self.latent_size = latent_size

		if fragment_embedding is None:
			self.fragment_embedding = nn.Embedding(self.fragment_vocab.size(), hidden_size)
		else:
			self.fragment_embedding = fragment_embedding

		if reactant_embedding is None:
			self.reactant_embedding = nn.Embedding(self.reactant_vocab.size(), hidden_size)
		else:
			self.reactant_embedding = reactant_embedding

		if template_embedding is None:
			self.template_embedding = nn.Embedding(self.template_vocab.size(), hidden_size)
		else:
			self.template_embedding = template_embedding
		self.mpn = MPN(hidden_size, 2)


		self.fragment_encoder = FTEncoder(self.fragment_vocab, self.hidden_size, self.fragment_embedding)
		self.fragment_decoder = FTDecoder(self.fragment_vocab, self.hidden_size, self.latent_size, self.fragment_embedding)

		self.rxn_decoder = RXNDecoder(self.hidden_size, self.latent_size, self.reactant_vocab, self.template_vocab, self.reactant_embedding, self.template_embedding, self.mpn)
		self.rxn_encoder = RXNEncoder(self.hidden_size, self.latent_size, self.reactant_vocab, self.template_vocab, self.mpn, self.template_embedding)

		self.combine_layer = nn.Linear(2 *hidden_size, hidden_size)

		self.FT_mean = nn.Linear(hidden_size, int(latent_size))
		self.FT_var = nn.Linear(hidden_size, int(latent_size))

		self.RXN_mean = nn.Linear(hidden_size, int(latent_size))
		self.RXN_var = nn.Linear(hidden_size, int(latent_size))

	def encode(self, ftrxn_tree_batch):
		batch_size = len(ftrxn_tree_batch)
		ft_trees = [ftrxn_tree[0] for ftrxn_tree in ftrxn_tree_batch]
		rxn_trees = [ftrxn_tree[1] for ftrxn_tree in ftrxn_tree_batch]
		set_batch_nodeID(ft_trees, self.fragment_vocab)
		encoder_outputs, root_vecs = self.fragment_encoder(ft_trees)
		root_vecs_rxn = self.rxn_encoder(rxn_trees)
		ft_mean = self.FT_mean(root_vecs)
		rxn_mean = self.RXN_mean(root_vecs_rxn)
		z_mean = torch.cat([ft_mean, rxn_mean], dim=1)
		return z_mean

	def forward(self, ftrxn_tree_batch, beta, a = 1.0, b = 1.0, epsilon_std=0.1):
		batch_size = len(ftrxn_tree_batch)
		ft_trees = [ftrxn_tree[0] for ftrxn_tree in ftrxn_tree_batch]
		rxn_trees = [ftrxn_tree[1] for ftrxn_tree in ftrxn_tree_batch]
		set_batch_nodeID(ft_trees, self.fragment_vocab)

		encoder_outputs, root_vecs = self.fragment_encoder(ft_trees)
		root_vecs_rxn = self.rxn_encoder(rxn_trees)
		ft_mean = self.FT_mean(root_vecs)
		ft_log_var = -torch.abs(self.FT_var(root_vecs))

		rxn_mean = self.RXN_mean(root_vecs_rxn)
		rxn_log_var = -torch.abs(self.RXN_var(root_vecs_rxn))

		z_mean = torch.cat([ft_mean, rxn_mean], dim=1)
		z_log_var = torch.cat([ft_log_var,rxn_log_var], dim=1)
		kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size

		
		epsilon = create_var(torch.randn(batch_size, int(self.latent_size)), False)*epsilon_std
		ft_vec = ft_mean + torch.exp(ft_log_var / 2) * epsilon

		epsilon = create_var(torch.randn(batch_size, int(self.latent_size)), False)*epsilon_std
		rxn_vec = rxn_mean + torch.exp(rxn_log_var / 2) * epsilon

		pred_loss, stop_loss, pred_acc, stop_acc = self.fragment_decoder(ft_trees, ft_vec)
		molecule_distance_loss, template_loss, molecule_label_loss, template_acc, label_acc = self.rxn_decoder(rxn_trees, rxn_vec, encoder_outputs)
		
		rxn_decoding_loss = template_loss  + molecule_label_loss# + molecule_distance_loss
		fragment_decoding_loss = pred_loss + stop_loss
		total_loss = fragment_decoding_loss+ rxn_decoding_loss + beta * (kl_loss) 

		return total_loss, pred_loss, stop_loss, template_loss, molecule_label_loss, pred_acc, stop_acc, template_acc, label_acc, kl_loss, molecule_distance_loss


class bFTRXNVAE(nn.Module):
	def __init__(self, fragment_vocab, reactant_vocab, template_vocab, hidden_size, latent_size, depth, device, fragment_embedding=None, reactant_embedding=None, template_embedding=None, ecc_type='none', ecc_R=3):
		super(bFTRXNVAE, self).__init__()
		self.fragment_vocab = fragment_vocab
		self.reactant_vocab = reactant_vocab
		self.template_vocab = template_vocab
		self.depth = depth
		self.n_class = 2
		self.binary_size = latent_size//2
		self.device = device
		
		# [ECC] Initialize error correcting code
		self.ecc_type = ecc_type
		self.ecc_R = ecc_R
		self.ecc_codec = create_ecc_codec(ecc_type, R=ecc_R)
		
		# [ECC] Determine effective binary size based on ECC configuration
		if self.ecc_codec is not None:
			if not self.ecc_codec.group_shape_ok(self.binary_size):
				raise ValueError(f"Binary size {self.binary_size} must be divisible by ECC repetition factor {ecc_R}")
			self.info_size = self.ecc_codec.get_info_size(self.binary_size)
			print(f"[ECC] VAE using {ecc_type} with R={ecc_R}: binary_size={self.binary_size}, info_size={self.info_size}")
		else:
			self.info_size = self.binary_size
			print(f"[ECC] VAE without ECC: binary_size={self.binary_size}")


		self.hidden_size = hidden_size
		self.latent_size = latent_size

		if fragment_embedding is None:
			self.fragment_embedding = nn.Embedding(self.fragment_vocab.size(), hidden_size)
		else:
			self.fragment_embedding = fragment_embedding

		if reactant_embedding is None:
			self.reactant_embedding = nn.Embedding(self.reactant_vocab.size(), hidden_size)
		else:
			self.reactant_embedding = reactant_embedding

		if template_embedding is None:
			self.template_embedding = nn.Embedding(self.template_vocab.size(), hidden_size)
		else:
			self.template_embedding = template_embedding
		self.mpn = MPN(hidden_size, 2)


		self.fragment_encoder = FTEncoder(self.fragment_vocab, self.hidden_size, self.fragment_embedding)
		self.fragment_decoder = FTDecoder(self.fragment_vocab, self.hidden_size, self.latent_size, self.fragment_embedding)

		self.rxn_decoder = RXNDecoder(self.hidden_size, self.latent_size, self.reactant_vocab, self.template_vocab, self.reactant_embedding, self.template_embedding, self.mpn)
		self.rxn_encoder = RXNEncoder(self.hidden_size, self.latent_size, self.reactant_vocab, self.template_vocab, self.mpn, self.template_embedding)

		self.combine_layer = nn.Linear(2 *hidden_size, hidden_size)

		self.FT_mean = nn.Linear(hidden_size, int(latent_size))
		self.FT_var = nn.Linear(hidden_size, int(latent_size))

		self.RXN_mean = nn.Linear(hidden_size, int(latent_size))
		self.RXN_var = nn.Linear(hidden_size, int(latent_size))

	def encode(self, ftrxn_tree_batch):
		batch_size = len(ftrxn_tree_batch)
		ft_trees = [ftrxn_tree[0] for ftrxn_tree in ftrxn_tree_batch]
		rxn_trees = [ftrxn_tree[1] for ftrxn_tree in ftrxn_tree_batch]
		set_batch_nodeID(ft_trees, self.fragment_vocab)
		encoder_outputs, root_vecs = self.fragment_encoder(ft_trees)
		root_vecs_rxn = self.rxn_encoder(rxn_trees)
		ft_mean = self.FT_mean(root_vecs)
		rxn_mean = self.RXN_mean(root_vecs_rxn)
		log_ft = ft_mean.view(-1, self.n_class)# .to(self.device)
		q_ft = F.softmax(log_ft, dim=-1).view(-1, self.binary_size*self.n_class)
		log_rxn = rxn_mean.view(-1, self.n_class)# .to(self.device)
		q_rxn = F.softmax(log_rxn, dim=-1).view(-1, self.binary_size*self.n_class)
  
		g_ft_vecs, _ = self.gumbel_softmax(q_ft, log_ft, 0.4)
		g_rxn_vecs, _ = self.gumbel_softmax(q_rxn, log_rxn, 0.4)
		g_ft_vecs = g_ft_vecs.view(-1, self.binary_size, self.n_class)
		g_rxn_vecs = g_rxn_vecs.view(-1, self.binary_size, self.n_class)
		z_binary = torch.cat([torch.argmax(g_ft_vecs, dim=-1), torch.argmax(g_rxn_vecs, dim=-1)], dim=-1)
		return z_binary

	# def gumbel_softmax(self, q, logits, temp):
	# 	G_sample = self.gumbel_sample(logits.shape)# .to(self.device)
	# 	y = F.softmax((logits + G_sample) / temp, dim=-1).view(-1, self.binary_size*self.n_class)
	# 	kl_loss = torch.sum(q * torch.log(q * self.n_class + 1e-20), dim=-1).mean()# .to(self.device)
	# 	return y, kl_loss
	def gumbel_softmax(self, q, logits, temp):
		# 生成 Gumbel 噪声并显式指定设备
		G_sample = self.gumbel_sample(logits.shape).to(logits.device)  # 关键修复！
		y = F.softmax((logits + G_sample) / temp, dim=-1).view(-1, self.binary_size*self.n_class)
		kl_loss = torch.sum(q * torch.log(q * self.n_class + 1e-20), dim=-1).mean()
		return y, kl_loss

	# def gumbel_sample(self, shape, eps=1e-20):
	# 	U = torch.rand(shape)
	# 	return -torch.log(-torch.log(U + eps) + eps)
 
	def gumbel_sample(self, shape):
		# 直接在目标设备上生成噪声，避免跨设备传输
		device = next(self.parameters()).device  # 获取模型当前设备
		u = torch.rand(shape).to(device)  # 在 GPU 上生成均匀分布
		gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)  # Gumbel 噪声
		return gumbel

	def forward(self, ftrxn_tree_batch, beta, a = 1.0, b = 1.0, epsilon_std=0.1, temp=0.8):
		batch_size = len(ftrxn_tree_batch)
		ft_trees = [ftrxn_tree[0] for ftrxn_tree in ftrxn_tree_batch]
		rxn_trees = [ftrxn_tree[1] for ftrxn_tree in ftrxn_tree_batch]
		set_batch_nodeID(ft_trees, self.fragment_vocab)

		encoder_outputs, root_vecs = self.fragment_encoder(ft_trees)
		root_vecs_rxn = self.rxn_encoder(rxn_trees)
		ft_mean = self.FT_mean(root_vecs)
		rxn_mean = self.RXN_mean(root_vecs_rxn)

		log_ft = ft_mean.view(-1, self.n_class)# .to(self.device)
		q_ft = F.softmax(log_ft, dim=-1).view(-1, self.binary_size*self.n_class)
		log_rxn = rxn_mean.view(-1, self.n_class)# .to(self.device)
		q_rxn = F.softmax(log_rxn, dim=-1).view(-1, self.binary_size*self.n_class)
  
		g_ft_vecs, ft_kl = self.gumbel_softmax(q_ft, log_ft, temp)
		g_rxn_vecs, rxn_kl = self.gumbel_softmax(q_rxn, log_rxn, temp)
  
		# [ECC] Apply ECC encoding/decoding with code consistency regularization
		ecc_consistency_loss = 0.0
		if self.ecc_codec is not None:
			# Process fragment vectors
			g_ft_binary = torch.argmax(g_ft_vecs.view(-1, self.binary_size, self.n_class), dim=-1).float()
			ft_info_bits = self.ecc_codec.decode(g_ft_binary)
			ft_encoded = self.ecc_codec.encode(ft_info_bits)
			
			# Code consistency regularization (minimize distance between original and reconstructed)
			ecc_consistency_loss += F.mse_loss(g_ft_binary, ft_encoded)
			
			# Process reaction vectors
			g_rxn_binary = torch.argmax(g_rxn_vecs.view(-1, self.binary_size, self.n_class), dim=-1).float()
			rxn_info_bits = self.ecc_codec.decode(g_rxn_binary)
			rxn_encoded = self.ecc_codec.encode(rxn_info_bits)
			
			# Code consistency regularization
			ecc_consistency_loss += F.mse_loss(g_rxn_binary, rxn_encoded)
			
			# Update vectors with ECC-processed versions for improved training stability
			g_ft_vecs = F.one_hot(ft_encoded.long(), num_classes=self.n_class).float().view(-1, self.binary_size*self.n_class)
			g_rxn_vecs = F.one_hot(rxn_encoded.long(), num_classes=self.n_class).float().view(-1, self.binary_size*self.n_class)
  
		kl_loss = ft_kl + rxn_kl

		pred_loss, stop_loss, pred_acc, stop_acc = self.fragment_decoder(ft_trees, g_ft_vecs)
		molecule_distance_loss, template_loss, molecule_label_loss, template_acc, label_acc = self.rxn_decoder(rxn_trees, g_rxn_vecs, encoder_outputs)
		
		rxn_decoding_loss = template_loss  + molecule_label_loss# + molecule_distance_loss
		fragment_decoding_loss = pred_loss + stop_loss
		
		# [ECC] Add ECC consistency loss with small weight to avoid overwhelming main objectives
		ecc_weight = 0.01 if self.ecc_codec is not None else 0.0
		total_loss = fragment_decoding_loss + rxn_decoding_loss + beta * (kl_loss) + ecc_weight * ecc_consistency_loss

		return total_loss, pred_loss, stop_loss, template_loss, molecule_label_loss, pred_acc, stop_acc, template_acc, label_acc, kl_loss, molecule_distance_loss









