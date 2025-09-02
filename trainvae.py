import os
import sys

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
# [GPU] Mixed precision training
from torch.cuda.amp import autocast, GradScaler

import math, random, sys, argparse
from collections import deque
# [tqdm] Import for progress bars
from tqdm import tqdm

from brxngenerator.chemistry.reactions.reaction_utils import get_mol_from_smiles, get_smiles_from_mol,read_multistep_rxns, get_template_order, get_qed_score,get_clogp_score
from brxngenerator.chemistry.reactions.reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates,stats
from brxngenerator.chemistry.fragments.fragment import FragmentVocab, FragmentTree, FragmentNode, can_be_decomposed
from brxngenerator.core.vae import FTRXNVAE, set_batch_nodeID, bFTRXNVAE
from brxngenerator.models.networks.mpn import MPN,PP,Discriminator
import brxngenerator.chemistry.utils.sascorer as sascorer
import random

TaskID =""
# TaskID = "1"

# [ECC] Parse command line arguments early to fix NameError  
parser = argparse.ArgumentParser(description="Train binary VAE with optional ECC")
parser.add_argument("-n", type=int, dest="params_num", default=0, help="Parameter set index (0-7)")
parser.add_argument("--ecc-type", type=str, choices=["none", "repetition"], default="none", 
                    help="ECC type: none or repetition")
parser.add_argument("--ecc-R", type=int, default=3, help="Repetition factor for ECC")
parser.add_argument("--subset", type=int, default=None, help="Limit dataset size for testing")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum improvement threshold")
args = parser.parse_args()

def schedule(counter, M):
	x = counter/(2*M)
	if x > M:
		return 1.0
	else:
		return 1.0 * x/M

def train(data_pairs, model, config_args, train_args):
	n_pairs = len(data_pairs)
	ind_list = [i for i in range(n_pairs)]
	data_pairs = [data_pairs[i] for i in ind_list]
	lr = config_args['lr']
	batch_size = config_args['batch_size']
	beta = config_args['beta']
	save_path = config_args['save_path']
	device = config_args['device']  # [GPU] Get device from config
	
	# Proper train/val split that works with small datasets
	val_size = min(1000, len(data_pairs) // 10)  # Use 10% for validation, capped at 1000
	val_pairs = data_pairs[:val_size]
	train_pairs = data_pairs[val_size:]
	print("training size:", len(train_pairs))
	print("valid size:", len(val_pairs))
	
	# [GPU] Enable CUDNN benchmarking for stable input sizes
	if device.type == 'cuda':
		torch.backends.cudnn.benchmark = True
		print(f"[GPU] Training on {device} with CUDNN benchmark enabled")
	
	optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 0.0001)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.5)
	
	# [GPU] Initialize mixed precision scaler
	scaler = GradScaler() if device.type == 'cuda' else None
	
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
	
	# Early stopping variables
	best_val_loss = float('inf')
	best_model_state = None
	patience_counter = 0
	patience = train_args.patience
	min_delta = train_args.min_delta

	# [tqdm] Epoch progress bar
	epoch_pbar = tqdm(range(config_args['epochs']), desc="Training Epochs", unit="epoch")
	for epoch in epoch_pbar:
		random.shuffle(train_pairs)
		# [GPU] Optimize DataLoader with pin_memory and num_workers
		num_workers = min(4, os.cpu_count() // 2) if device.type == 'cuda' else 0
		dataloader = DataLoader(train_pairs, batch_size = batch_size, shuffle = True, 
		                       collate_fn = lambda x:x, pin_memory=(device.type == 'cuda'),
		                       num_workers=num_workers)
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
		
		# [tqdm] Batch progress bar  
		batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, unit="batch")
		for it, batch in enumerate(batch_pbar):
			if epoch < 20:
				beta = schedule(counter, M)
			else:
				beta = config_args['beta']  # [Fix] Use config_args instead of undefined args
			counter +=1
			total_step += 1
			temp = max(min_temp, temp*np.exp(-temp_anneal_rate*total_step))
			
			optimizer.zero_grad()
			
			# [GPU] Mixed precision forward pass
			if scaler is not None:
				with autocast():
					t_loss, pred_loss, stop_loss, template_loss, molecule_label_loss, pred_acc, stop_acc, template_acc, label_acc, kl_loss, molecule_distance_loss = model(batch, beta, temp=temp)
				scaler.scale(t_loss).backward()
				scaler.step(optimizer)
				scaler.update()
			else:
				t_loss, pred_loss, stop_loss, template_loss, molecule_label_loss, pred_acc, stop_acc, template_acc, label_acc, kl_loss, molecule_distance_loss = model(batch, beta, temp=temp)
				t_loss.backward()
				optimizer.step()
			
			# [tqdm] Update batch progress with current loss
			batch_pbar.set_postfix({
				'loss': f"{t_loss.item():.4f}", 
				'kl': f"{kl_loss.item():.4f}",
				'beta': f"{beta:.3f}"
			})
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

				
		# [tqdm] Update epoch progress bar with validation loss
		val_loss = validate(val_pairs, model, config_args)
		epoch_pbar.set_postfix({
			'val_loss': f"{val_loss:.4f}", 
			'patience': f"{patience_counter}/{patience}"
		})
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

		# Early stopping logic
		if val_loss < best_val_loss - min_delta:
			print(f"Best model updated at epoch {epoch}, val_loss = {val_loss:.6f}")
			best_val_loss = val_loss
			best_model_state = model.state_dict().copy()  # Deep copy current state
			patience_counter = 0
		else:
			patience_counter += 1
			print(f"No improvement for {patience_counter}/{patience} epochs")
			
		# Check for early stopping
		if patience_counter >= patience:
			print(f"Early stopping triggered at epoch {epoch}")
			epoch_pbar.close()  # [tqdm] Close progress bar on early stop
			break
	
	# Save only the best model at the end
	if best_model_state is not None:
		best_model_path = "{}/bvae_best_model_with{}.pt".format(save_path, TaskID)  # [Fix] Changed .npy to .pt for PyTorch compatibility
		torch.save(best_model_state, best_model_path)
		print(f"Best model saved: {best_model_path}")
	else:
		print("Warning: No best model state to save")

def validate(data_pairs, model, args):
	model.eval()  # Set model to evaluation mode
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
	
	# Calculate validation loss (using the total loss instead of just the last batch)
	val_loss = (total_pred_loss + total_stop_loss + total_template_loss + total_molecule_label_loss).item() / len(dataloader)
	model.train()  # Set model back to training mode
	return val_loss



print("cuda is ", torch.cuda.is_available())


batch_size = 1000


vocab_path = "./weights/data.txt_fragmentvocab.txt"
data_filename = "./data/data.txt"
epochs = 100
# [DEV] Override epochs for subset testing
if args.subset is not None:
    epochs = min(5, epochs)  # Limit epochs for dev testing
    print(f"[DEV] Subset mode: limiting epochs to {epochs}")

legacy_args={}

legacy_args['batch_size'] = batch_size
legacy_args['datasetname'] = data_filename
legacy_args['epochs'] = epochs

print("loading data.....")

routes, scores = read_multistep_rxns(data_filename)

# debug 
# routes = routes[:3000]

rxn_trees = [ReactionTree(route) for route in routes]
molecules = [rxn_tree.molecule_nodes[0].smiles for rxn_tree in rxn_trees]

# [ECC] Apply subset filtering if requested  
if args.subset is not None and len(rxn_trees) > args.subset:
    print(f"Using subset of {args.subset} reactions (out of {len(rxn_trees)})")
    rxn_trees = rxn_trees[:args.subset]
    molecules = molecules[:args.subset]
    
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


i = params[int(args.params_num)]

hidden_size = i[0]
latent_size = i[1]
depth = i[2]
beta = i[3]
lr = i[4]
save_path = "weights" + i[5]

# [ECC] Validate ECC parameters after latent_size is defined
if args.ecc_type == "repetition" and latent_size % args.ecc_R != 0:
    raise ValueError(f"ECC repetition requires latent_size % ecc_R == 0. Got {latent_size} % {args.ecc_R} != 0")

config_args = {}
config_args['save_path'] = save_path
config_args['beta'] = beta
config_args['lr'] = lr
config_args['batch_size'] = batch_size
config_args['epochs'] = epochs
config_args['device'] = device  # [GPU] Add device to config
print("hidden size:", hidden_size, "latent_size:", latent_size, "batch size:", batch_size, "depth:", depth)
print("beta:", beta, "lr:", lr)
mpn = MPN(hidden_size, depth)
# [ECC] Pass ECC parameters to model for training-time integration
model = bFTRXNVAE(fragmentDic, reactantDic, templateDic, hidden_size, latent_size, depth, device=device, 
                  fragment_embedding=None, reactant_embedding=None, template_embedding=None,
                  ecc_type=args.ecc_type, ecc_R=args.ecc_R).to(device)
print("size of data pairs:", len(data_pairs))
train(data_pairs, model, config_args, args)




