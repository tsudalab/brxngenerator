
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.networks.nnutils import create_var, attention
from ..models.encoders.ftencoder import FTEncoder
from ..models.decoders.ftdecoder import FTDecoder
from ..models.decoders.rxndecoder import RXNDecoder, RXNDecoder1
from ..models.encoders.rxnencoder import RXNEncoder
from ..models.networks.mpn import MPN,PP,Discriminator
# [ECC] Import ECC utilities for training-time integration
from .ecc import create_ecc_codec


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
		# [FIX] For fairness: baseline uses full latent_size, only ECC uses half for each part (ft/rxn)
		if ecc_type == 'none':
			self.binary_size = latent_size  # Baseline: full latent size as binary
		else:
			self.binary_size = latent_size//2  # ECC: split between ft and rxn parts
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
		# [FIX] Keep original latent_size, adjust vector dimensions at runtime
		self.fragment_decoder = FTDecoder(self.fragment_vocab, self.hidden_size, self.latent_size, self.fragment_embedding)

		self.rxn_decoder = RXNDecoder(self.hidden_size, self.latent_size, self.reactant_vocab, self.template_vocab, self.reactant_embedding, self.template_embedding, self.mpn)
		self.rxn_encoder = RXNEncoder(self.hidden_size, self.latent_size, self.reactant_vocab, self.template_vocab, self.mpn, self.template_embedding)

		self.combine_layer = nn.Linear(2 *hidden_size, hidden_size)

		# [FIX] Keep linear layers at binary_size for decoder compatibility
		self.FT_mean = nn.Linear(hidden_size, int(self.binary_size))
		self.FT_var = nn.Linear(hidden_size, int(self.binary_size))

		self.RXN_mean = nn.Linear(hidden_size, int(self.binary_size))
		self.RXN_var = nn.Linear(hidden_size, int(self.binary_size))

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
		# [FIX] Reshape for proper softmax computation over class dimension
		batch_size = q.size(0)
		# Reshape to [batch_size * binary_size, n_class] for class-wise softmax
		logits_reshaped = logits.view(-1, self.n_class)
		q_reshaped = q.view(-1, self.n_class)
		
		# Generate Gumbel noise
		G_sample = self.gumbel_sample(logits_reshaped.shape).to(logits.device)
		# Apply Gumbel softmax over class dimension
		y = F.softmax((logits_reshaped + G_sample) / temp, dim=-1)
		
		# Reshape back to match q shape
		y = y.view(q.shape)
		kl_loss = torch.sum(q_reshaped * torch.log(q_reshaped * self.n_class + 1e-20), dim=-1).mean()
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
		# [FIX] For binary VAE, treat each output as a separate binary choice
		# Convert single logits to binary choice logits [value, 0] → [0, value] for Gumbel softmax
		ft_binary_logits = torch.stack([torch.zeros_like(ft_mean), ft_mean], dim=-1)  # [batch, binary_size, 2]
		rxn_binary_logits = torch.stack([torch.zeros_like(rxn_mean), rxn_mean], dim=-1)  # [batch, binary_size, 2]
		
		# Reshape for Gumbel softmax: [batch_size, binary_size*2]
		log_ft = ft_binary_logits.view(-1, self.n_class)  # [batch*binary_size, 2]
		q_ft = F.softmax(log_ft, dim=-1).view(batch_size, -1)  # [batch_size, binary_size*2]
		log_rxn = rxn_binary_logits.view(-1, self.n_class)  # [batch*binary_size, 2] 
		q_rxn = F.softmax(log_rxn, dim=-1).view(batch_size, -1)  # [batch_size, binary_size*2]
  
		# Reshape log tensors to match q shape for gumbel_softmax
		log_ft_reshaped = log_ft.view(batch_size, -1)  # [batch_size, binary_size*2]
		log_rxn_reshaped = log_rxn.view(batch_size, -1)  # [batch_size, binary_size*2]
		
		g_ft_vecs, ft_kl = self.gumbel_softmax(q_ft, log_ft_reshaped, temp)
		g_rxn_vecs, rxn_kl = self.gumbel_softmax(q_rxn, log_rxn_reshaped, temp)
  
		# [ECC] Apply ECC encoding/decoding with code consistency regularization
		ecc_consistency_loss = 0.0
		if self.ecc_codec is not None:
			# [FIX] Properly reshape vectors for ECC processing
			# g_ft_vecs shape should be [batch_size, binary_size*n_class]
			g_ft_reshaped = g_ft_vecs.view(batch_size, self.binary_size, self.n_class)
			g_rxn_reshaped = g_rxn_vecs.view(batch_size, self.binary_size, self.n_class)
			
			# Process fragment vectors
			g_ft_binary = torch.argmax(g_ft_reshaped, dim=-1).float()  # [batch_size, binary_size]
			ft_info_bits = self.ecc_codec.decode(g_ft_binary)
			ft_encoded = self.ecc_codec.encode(ft_info_bits)
			
			# Code consistency regularization (minimize distance between original and reconstructed)
			ecc_consistency_loss += F.mse_loss(g_ft_binary, ft_encoded)
			
			# Process reaction vectors
			g_rxn_binary = torch.argmax(g_rxn_reshaped, dim=-1).float()  # [batch_size, binary_size]
			rxn_info_bits = self.ecc_codec.decode(g_rxn_binary)
			rxn_encoded = self.ecc_codec.encode(rxn_info_bits)
			
			# Code consistency regularization
			ecc_consistency_loss += F.mse_loss(g_rxn_binary, rxn_encoded)
			
			# Update vectors with ECC-processed versions for improved training stability
			g_ft_vecs = F.one_hot(ft_encoded.long(), num_classes=self.n_class).float().view(batch_size, self.binary_size*self.n_class)
			g_rxn_vecs = F.one_hot(rxn_encoded.long(), num_classes=self.n_class).float().view(batch_size, self.binary_size*self.n_class)
  
		kl_loss = ft_kl + rxn_kl

		# [FIX] Extract binary representations for decoder input
		# Convert Gumbel vectors back to binary latent representations
		g_ft_binary = torch.argmax(g_ft_vecs.view(batch_size, self.binary_size, self.n_class), dim=-1).float()
		g_rxn_binary = torch.argmax(g_rxn_vecs.view(batch_size, self.binary_size, self.n_class), dim=-1).float()
		
		# For decoders, we need vectors of size latent_size
		# Pad to latent_size if needed (for fair comparison between baseline and ECC)
		if self.binary_size < self.latent_size:
			pad_size = self.latent_size - self.binary_size
			ft_decoder_vec = F.pad(g_ft_binary, (0, pad_size))
			rxn_decoder_vec = F.pad(g_rxn_binary, (0, pad_size))
		else:
			ft_decoder_vec = g_ft_binary[:, :self.latent_size]
			rxn_decoder_vec = g_rxn_binary[:, :self.latent_size]
		
		pred_loss, stop_loss, pred_acc, stop_acc = self.fragment_decoder(ft_trees, ft_decoder_vec)
		molecule_distance_loss, template_loss, molecule_label_loss, template_acc, label_acc = self.rxn_decoder(rxn_trees, rxn_decoder_vec, encoder_outputs)
		
		rxn_decoding_loss = template_loss  + molecule_label_loss# + molecule_distance_loss
		fragment_decoding_loss = pred_loss + stop_loss
		
		# [ECC] Add ECC consistency loss with small weight to avoid overwhelming main objectives
		ecc_weight = 0.01 if self.ecc_codec is not None else 0.0
		total_loss = fragment_decoding_loss + rxn_decoding_loss + beta * (kl_loss) + ecc_weight * ecc_consistency_loss

		return total_loss, pred_loss, stop_loss, template_loss, molecule_label_loss, pred_acc, stop_acc, template_acc, label_acc, kl_loss, molecule_distance_loss
	
	def encode_posteriors(self, ftrxn_tree_batch):
		"""
		Encode batch and return posterior probabilities for latent metrics.
		
		Args:
			ftrxn_tree_batch: Batch of (fragment_tree, reaction_tree) pairs
			
		Returns:
			posterior_probs: Tensor of shape [batch_size, binary_size] with p(z_i=1|x)
		"""
		self.eval()
		with torch.no_grad():
			batch_size = len(ftrxn_tree_batch)
			ft_trees = [ftrxn_tree[0] for ftrxn_tree in ftrxn_tree_batch]
			rxn_trees = [ftrxn_tree[1] for ftrxn_tree in ftrxn_tree_batch]
			set_batch_nodeID(ft_trees, self.fragment_vocab)

			# Encode to get posterior parameters
			encoder_outputs, root_vecs = self.fragment_encoder(ft_trees)
			root_vecs_rxn = self.rxn_encoder(rxn_trees)
			
			ft_mean = self.FT_mean(root_vecs)  # [batch, binary_size]
			rxn_mean = self.RXN_mean(root_vecs_rxn)  # [batch, binary_size]
			
			# Convert single logits to probabilities for binary choices
			ft_probs = torch.sigmoid(ft_mean)  # [batch, binary_size]
			rxn_probs = torch.sigmoid(rxn_mean)  # [batch, binary_size]
			
			# Concatenate both parts to get full posterior
			posterior_probs = torch.cat([ft_probs, rxn_probs], dim=1)  # shape: [batch, 2*binary_size]
			
			return posterior_probs