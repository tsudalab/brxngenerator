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
seed = 4
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
    # tqdm
    for i, data_pair in tqdm(enumerate(data_pairs)):
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
# move to cpu first
latents = latents.detach().cpu().numpy()
n = latents.shape[0]
print('===================latent shape', latents.shape)
latents = latents[:, : latent_size // 2]
print('===================latent shape', latents.shape)
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



# main(X_train, y_train, X_test, y_test, molecules, -scores, model, parameters, configs, metric, seed)
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--seed-start", dest="seed_start", type="int", default=seed,
                  help="起始 seed（包含）")
parser.add_option("--seed-end",   dest="seed_end",   type="int", default=seed,
                  help="结束   seed（包含）")
(options, args) = parser.parse_args()

# 3. 对指定范围内的每个 seed 依次调用 main()
for sd in tqdm(range(options.seed_start, options.seed_end + 1), desc="Seeds"):
    print(f"\n=== Running optimization with seed = {sd} ===")
    seed_all(sd)   # 重设随机种子
    main(
        X_train, y_train,
        X_test,  y_test,
        molecules, -scores,
        model, parameters,
        configs, metric,
        sd
    )