import os
import sys
sys.path.append('./rxnft_vae')

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

TaskID =os.environ["TaskID"]

def schedule(counter, M):
	x = counter/(2*M)
	if x > M:
		return 1.0
	else:
		return 1.0 * x/M

def train(data_pairs, model,args):
	n_pairs = len(data_pairs)
	ind_list = [i for i in range(n_pairs)]
	data_pairs = [data_pairs[i] for i in ind_list]
	lr = args['lr']
	batch_size = args['batch_size']
	beta = args['beta']
	val_pairs = data_pairs[:1000]
	train_pairs = data_pairs[1000:-1]
	print("trainng size:", len(train_pairs))
	print("valid size:", len(val_pairs))
	optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 0.0001)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.5)
	tr_rec_loss_list = []
	tr_kl_loss_list=[]
	beta_list=[]
	M = 100
	init_temp=1
	temp_anneal_rate=1e-4
	min_temp=0.4
	total_step = 0

	counter = 0
	temp = init_temp

	for epoch in range(args['epochs']):
		random.shuffle(train_pairs)
		dataloader = DataLoader(train_pairs, batch_size = batch_size, shuffle = True, collate_fn = lambda x:x)
		total_loss = 0
		total_pred_loss=0
		total_stop_loss =0
		total_kl_loss =0
		total_pred_acc =0
		total_stop_acc = 0
		total_template_loss = 0
		total_template_acc = 0
		total_molecule_distance_loss =0
		total_molecule_label_loss = 0
		total_label_acc =0
		for it, batch in enumerate(dataloader):
			if epoch < 20:
				beta = schedule(counter, M)
			else:
				beta = args['beta']
			counter +=1
			total_step += 1
			temp = max(min_temp, temp*np.exp(-temp_anneal_rate*total_step))
			model.zero_grad()
			t_loss, pred_loss, stop_loss, template_loss, molecule_label_loss, pred_acc, stop_acc, template_acc, label_acc, kl_loss, molecule_distance_loss = model(batch, beta, temp=temp)
			t_loss.backward()
			optimizer.step()
			print('loss: ', t_loss.item(), kl_loss.item())
			total_loss += t_loss
			total_pred_loss += pred_loss
			total_stop_loss += stop_loss
			total_kl_loss += kl_loss
			total_pred_acc += pred_acc
			total_stop_acc += stop_acc
			total_template_loss += template_loss
			total_template_acc += template_acc
			total_molecule_distance_loss += molecule_distance_loss
			total_molecule_label_loss += molecule_label_loss
			total_label_acc += label_acc

				
		print("*******************Epoch", epoch, "******************", counter, beta)
		print("Validation Loss")
		val_loss = validate(val_pairs, model, args)
		print("Train Loss")
		print("---> pred loss:", total_pred_loss.item()/len(dataloader), "pred acc:", total_pred_acc/len(dataloader))
		print("---> stop loss:", total_stop_loss.item()/len(dataloader), "stop acc:", total_stop_acc/len(dataloader))
		print("---> template loss:", total_template_loss.item()/len(dataloader), "tempalte acc:", total_template_acc.item()/len(dataloader))
		print("---> molecule label loss:", total_molecule_label_loss.item()/len(dataloader), "molecule acc:", total_label_acc.item()/len(dataloader))
		print("---> kl loss:", total_kl_loss.item()/len(dataloader))
		print("---> reconstruction loss:", total_loss.item()/len(dataloader)-beta * total_kl_loss.item()/len(dataloader))
		torch.save(model.state_dict(),"./weights/bvae_iter-{}-with{}.npy".format(epoch+1,TaskID))
		print("saving file:./weights/bvae_iter-{}-with{}.npy".format(epoch+1,TaskID))

def validate(data_pairs, model, args):
	beta = args['beta']
	batch_size = args['batch_size']
	dataloader = DataLoader(data_pairs, batch_size = batch_size, shuffle = True, collate_fn = lambda x:x)

	total_pred_acc =0
	total_stop_acc = 0
	total_template_loss = 0
	total_template_acc = 0
	total_molecule_distance_loss =0
	total_label_acc =0
	total_pred_loss=0
	total_stop_loss =0
	total_template_loss = 0
	total_molecule_label_loss = 0
	total_loss = 0

	with torch.no_grad():
		for it, batch in enumerate(dataloader):
			t_loss, pred_loss, stop_loss, template_loss, molecule_label_loss, pred_acc, stop_acc, template_acc, label_acc, kl_loss, molecule_distance_loss = model(batch, beta, epsilon_std=0.01)
			total_pred_acc += pred_acc
			total_stop_acc += stop_acc
			total_template_acc += template_acc
			total_label_acc += label_acc
			total_pred_loss += pred_loss
			total_stop_loss += stop_loss
			total_template_loss += template_loss
			total_molecule_label_loss += molecule_label_loss

	print("*** pred loss: ",total_pred_loss.item()/len(dataloader), "pred acc:", total_pred_acc/len(dataloader))
	print("*** stop loss: ",total_stop_loss.item()/len(dataloader), "stop acc:", total_stop_acc/len(dataloader))

	print("*** template loss: ",total_template_loss.item()/len(dataloader), "template acc:", total_template_acc/len(dataloader))
	print("*** label loss: ",total_molecule_label_loss.item()/len(dataloader), "label acc:", total_label_acc/len(dataloader))
	return t_loss - beta * kl_loss




parser = OptionParser()
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=50)
parser.add_option("-d", "--depth", dest="depth", default=2)
parser.add_option("-b", "--batch", dest="batch_size", default = 32)
parser.add_option("-s", "--save_dir", dest="save_path", default="weights")
parser.add_option("-t", "--data_path", dest="data_path")
parser.add_option("-v", "--vocab_path", dest="vocab_path")
parser.add_option("-q", "--lr", dest="lr", default = 0.001)
parser.add_option("-z", "--beta", dest="beta", default = 1.0)
parser.add_option("-e", "--epochs", dest="epochs", default = 100)

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mpn = MPN(hidden_size, depth)
model = bFTRXNVAE(fragmentDic, reactantDic, templateDic, hidden_size, latent_size, depth, device, fragment_embedding=None, reactant_embedding=None, template_embedding=None)
train(data_pairs, model,args)




