from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
import rdchiral
from rdchiral.main import rdchiralRunText, rdchiralRun
from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from .mlp_policies import load_parallel_model , preprocess
from collections import defaultdict, OrderedDict

def merge(reactant_d):
    ret = []
    for reactant, l in reactant_d.items():
        ss, ts = zip(*l)
        ret.append((reactant, sum(ss), list(ts)[0]))
    reactants, scores, templates = zip(*sorted(ret,key=lambda item : item[1], reverse=True))
    return list(reactants), list(scores), list(templates)



class MLPModel(object):
    def __init__(self,state_path, template_path, device=-1, fp_dim=2048):
        super(MLPModel, self).__init__()
        self.fp_dim = fp_dim
        self.net, self.idx2rules = load_parallel_model(state_path,template_path, fp_dim)
        self.net.eval()
        self.device = device
        if device >= 0:
            self.net.to(device)

    def run(self, x, topk=10):
        arr = preprocess(x, self.fp_dim)
        arr = np.reshape(arr,[-1, arr.shape[0]])
        arr = torch.tensor(arr, dtype=torch.float32)
        if self.device >= 0:
            arr = arr.to(self.device)
        preds = self.net(arr)
        preds = F.softmax(preds,dim=1)
        if self.device >= 0:
            preds = preds.cpu()
        probs, idx = torch.topk(preds,k=topk)
        rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]
        reactants = []
        scores = []
        templates = []
        for i , rule in enumerate(rule_k):
            out1 = []
            try:
                out1 = rdchiralRunText(rule, x)
                if len(out1) == 0: continue
                out1 = sorted(out1)
                for reactant in out1:
                    reactants.append(reactant)
                    scores.append(probs[0][i].item()/len(out1))
                    templates.append(rule)
            except ValueError:
                pass
        if len(reactants) == 0: return None
        reactants_d = defaultdict(list)
        for r, s, t in zip(reactants, scores, templates):
            if '.' in r:
                str_list = sorted(r.strip().split('.'))
                reactants_d['.'.join(str_list)].append((s, t))
            else:
                reactants_d[r].append((s, t))

        reactants, scores, templates = merge(reactants_d)
        total = sum(scores)
        scores = [s / total for s in scores]
        return {'reactants':reactants,
                'scores' : scores,
                'template' : templates}



if __name__ == '__main__':
    import argparse
    from pprint import pprint
    parser = argparse.ArgumentParser(description="Policies for retrosynthesis Planner")
    parser.add_argument('--template_rule_path', default='../data/uspto_all/template_rules_1.dat',
                        type=str, help='Specify the path of all template rules.')
    parser.add_argument('--model_path', default='../model/saved_rollout_state_1_2048.ckpt',
                        type=str, help='specify where the trained model is')
    args = parser.parse_args()
    state_path = args.model_path
    template_path = args.template_rule_path
    model =  MLPModel(state_path,template_path,device=-1)
    x = '[F-:1]'
    x = 'S=C(Cl)(Cl)'
    x = 'O=C1Nc2ccccc2C12COc1cc3c(cc12)OCCO3'
    y = model.run(x,10)
    pprint(y)
