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

TaskID =""
# TaskID = "1"
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
	save_path = args['save_path']
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
		train_pred_loss = total_pred_loss.item()/len(dataloader)
		train_stop_loss = total_stop_loss.item()/len(dataloader)
		train_template_loss = total_template_loss.item()/len(dataloader)
		train_molecule_label_loss = total_molecule_label_loss.item()/len(dataloader)
		train_kl_loss = total_kl_loss.item()/len(dataloader)
		train_reconstruction_loss = total_loss.item()/len(dataloader) - beta * train_kl_loss

		print("---> pred loss:", train_pred_loss, "pred acc:", total_pred_acc/len(dataloader))
		print("---> stop loss:", train_stop_loss, "stop acc:", total_stop_acc/len(dataloader))
		print("---> template loss:", train_template_loss, "tempalte acc:", total_template_acc.item()/len(dataloader))
		print("---> molecule label loss:", train_molecule_label_loss, "molecule acc:", total_label_acc.item()/len(dataloader))
		print("---> kl loss:", train_kl_loss)
		print("---> reconstruction loss:", train_reconstruction_loss)

		# Record train and validation losses to a file
		try:
			with open(f"{save_path}/loss_record_with{TaskID}.txt", "a") as f:
				f.write(f"Epoch {epoch}, Counter {counter}, Beta {beta}\n")
				f.write(f"Validation Loss: {val_loss}\n") 
				f.write(f"Train Losses:\n")
				f.write(f"  pred loss: {train_pred_loss}, pred acc: {total_pred_acc/len(dataloader)}\n")
				f.write(f"  stop loss: {train_stop_loss}, stop acc: {total_stop_acc/len(dataloader)}\n")
				f.write(f"  template loss: {train_template_loss}, template acc: {total_template_acc.item()/len(dataloader)}\n")
				f.write(f"  molecule label loss: {train_molecule_label_loss}, molecule acc: {total_label_acc.item()/len(dataloader)}\n")
				f.write(f"  kl loss: {train_kl_loss}\n")
				f.write(f"  reconstruction loss: {train_reconstruction_loss}\n\n")
		except FileNotFoundError:
			os.makedirs(save_path, exist_ok=True)
			with open(f"{save_path}/loss_record_with{TaskID}.txt", "a") as f:
				f.write(f"Epoch {epoch}, Counter {counter}, Beta {beta}\n")
				f.write(f"Validation Loss: {val_loss}\n")
				f.write(f"Train Losses:\n")
				f.write(f"  pred loss: {train_pred_loss}, pred acc: {total_pred_acc/len(dataloader)}\n")
				f.write(f"  stop loss: {train_stop_loss}, stop acc: {total_stop_acc/len(dataloader)}\n")
				f.write(f"  template loss: {train_template_loss}, template acc: {total_template_acc.item()/len(dataloader)}\n")
				f.write(f"  molecule label loss: {train_molecule_label_loss}, molecule acc: {total_label_acc.item()/len(dataloader)}\n")
				f.write(f"  kl loss: {train_kl_loss}\n")
				f.write(f"  reconstruction loss: {train_reconstruction_loss}\n\n")

		torch.save(model.state_dict(),"{}/bvae_iter-{}-with{}.npy".format(save_path,epoch+1,TaskID))
		print("saving file:{}/bvae_iter-{}-with{}.npy".format(save_path,epoch+1,TaskID))

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



print("cuda is ", torch.cuda.is_available())


batch_size = 1000


vocab_path = "./weights/data.txt_fragmentvocab.txt"
data_filename = "./data/data.txt"
epochs = 100

args={}

args['batch_size'] = batch_size
args['datasetname'] = data_filename
args['epochs'] = epochs

print("loading data.....")

routes, scores = read_multistep_rxns(data_filename)

# debug 
# routes = routes[:3000]

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


params = [
    (100, 100, 2, 1.0, 0.001,"/hidden_size_100_latent_size_100_depth_2_beta_1.0_lr_0.001"),
    (200, 100, 2, 1.0, 0.00001,"/hidden_size_200_latent_size_100_depth_2_beta_1.0_lr_0.00001"),
    (200, 100, 3, 1.0, 0.001,"/hidden_size_200_latent_size_100_depth_3_beta_1.0_lr_0.001"),
    (200, 100, 5, 1.0, 0.001,"/hidden_size_200_latent_size_100_depth_5_beta_1.0_lr_0.001"),
    (200, 200, 2, 1.0, 0.001,"/hidden_size_200_latent_size_200_depth_2_beta_1.0_lr_0.001"),
    (200, 300, 2, 1.0, 0.001,"/hidden_size_200_latent_size_300_depth_2_beta_1.0_lr_0.001"),
    (300, 100, 2, 1.0, 0.001,"/hidden_size_300_latent_size_100_depth_2_beta_1.0_lr_0.001"),
    (500, 300, 5, 1.0, 0.001,"/hidden_size_500_latent_size_300_depth_5_beta_1.0_lr_0.001"),
]


parser = OptionParser()
parser.add_option("-n", dest="params_num", default=0)
opts, _ = parser.parse_args()

i = params[int(opts.params_num)]

hidden_size = i[0]
latent_size = i[1]
depth = i[2]
beta = i[3]
lr = i[4]
save_path = "weights" + i[5]

args['save_path'] = save_path
args['beta'] = beta
args['lr'] = lr
print("hidden size:", hidden_size, "latent_size:", latent_size, "batch size:", batch_size, "depth:", depth)
print("beta:", beta, "lr:", lr)
mpn = MPN(hidden_size, depth)
model = bFTRXNVAE(fragmentDic, reactantDic, templateDic, hidden_size, latent_size, depth, device=device, fragment_embedding=None, reactant_embedding=None, template_embedding=None).to(device)
print("size of data pairs:", len(data_pairs))
train(data_pairs, model,args)




