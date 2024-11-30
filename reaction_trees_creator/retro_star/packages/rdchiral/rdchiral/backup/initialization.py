import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem.rdchem import ChiralType, BondType, BondDir, BondStereo

from rdchiral.chiral import template_atom_could_have_been_tetra
from rdchiral.utils import vprint, PLEVEL
from rdchiral.bonds import enumerate_possible_cistrans_defs, bond_dirs_by_mapnum, \
    get_atoms_across_double_bonds

BondDirOpposite = {AllChem.BondDir.ENDUPRIGHT: AllChem.BondDir.ENDDOWNRIGHT,
                   AllChem.BondDir.ENDDOWNRIGHT: AllChem.BondDir.ENDUPRIGHT}

class rdchiralReaction():
    '''
    Class to store everything that should be pre-computed for a reaction. This
    makes library application much faster, since we can pre-do a lot of work
    instead of doing it for every mol-template pair
    '''
    def __init__(self, reaction_smarts):
        self.reaction_smarts = reaction_smarts
        
        self.rxn = initialize_rxn_from_smarts(reaction_smarts)

        self.template_r, self.template_p = get_template_frags_from_rxn(self.rxn)

        self.atoms_rt_map = {a.GetAtomMapNum(): a \
            for a in self.template_r.GetAtoms() if a.GetAtomMapNum()}
        self.atoms_pt_map = {a.GetAtomMapNum(): a \
            for a in self.template_p.GetAtoms() if a.GetAtomMapNum()}
        
        self.atoms_rt_idx_to_map = {a.GetIdx(): a.GetAtomMapNum()
            for a in self.template_r.GetAtoms()}
        self.atoms_pt_idx_to_map = {a.GetIdx(): a.GetAtomMapNum()
            for a in self.template_p.GetAtoms()}

        if any(self.atoms_rt_map[i].GetAtomicNum() != self.atoms_pt_map[i].GetAtomicNum() \
                for i in self.atoms_rt_map if i in self.atoms_pt_map):
            raise ValueError('Atomic identity should not change in a reaction!')

        [template_atom_could_have_been_tetra(a) for a in self.template_r.GetAtoms()]
        [template_atom_could_have_been_tetra(a) for a in self.template_p.GetAtoms()]

        self.rt_bond_dirs_by_mapnum = bond_dirs_by_mapnum(self.template_r)
        self.pt_bond_dirs_by_mapnum = bond_dirs_by_mapnum(self.template_p)

        self.required_rt_bond_defs, self.required_bond_defs_coreatoms = \
            enumerate_possible_cistrans_defs(self.template_r)

    def reset(self):
        for (idx, mapnum) in self.atoms_rt_idx_to_map.items():
            self.template_r.GetAtomWithIdx(idx).SetAtomMapNum(mapnum)
        for (idx, mapnum) in self.atoms_pt_idx_to_map.items():
            self.template_p.GetAtomWithIdx(idx).SetAtomMapNum(mapnum)

class rdchiralReactants():
    '''
    Class to store everything that should be pre-computed for a reactant mol
    so that library application is faster
    '''
    def __init__(self, reactant_smiles):
        self.reactant_smiles = reactant_smiles  

        self.reactants = initialize_reactants_from_smiles(reactant_smiles)

        self.atoms_r = {a.GetAtomMapNum(): a for a in self.reactants.GetAtoms()}
        self.idx_to_mapnum = lambda idx: self.reactants.GetAtomWithIdx(idx).GetAtomMapNum()

        self.reactants_achiral = initialize_reactants_from_smiles(reactant_smiles)
        [a.SetChiralTag(ChiralType.CHI_UNSPECIFIED) for a in self.reactants_achiral.GetAtoms()]
        [(b.SetStereo(BondStereo.STEREONONE), b.SetBondDir(BondDir.NONE)) \
            for b in self.reactants_achiral.GetBonds()]

        self.bonds_by_mapnum = [
            (b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum(), b) \
            for b in self.reactants.GetBonds()
        ]

        self.bond_dirs_by_mapnum = {}
        for (i, j, b) in self.bonds_by_mapnum:
            if b.GetBondDir() != BondDir.NONE:
                self.bond_dirs_by_mapnum[(i, j)] = b.GetBondDir()
                self.bond_dirs_by_mapnum[(j, i)] = BondDirOpposite[b.GetBondDir()]

        self.atoms_across_double_bonds = get_atoms_across_double_bonds(self.reactants)


def initialize_rxn_from_smarts(reaction_smarts):
    rxn = AllChem.ReactionFromSmarts(reaction_smarts)
    rxn.Initialize()
    if rxn.Validate()[1] != 0:
        raise ValueError('validation failed')
    if PLEVEL >= 2: print('Validated rxn without errors')


    prd_maps = [a.GetAtomMapNum() for prd in rxn.GetProducts() for a in prd.GetAtoms() if a.GetAtomMapNum()]

    unmapped = 700
    for rct in rxn.GetReactants():
        rct.UpdatePropertyCache()
        Chem.AssignStereochemistry(rct)
        for a in rct.GetAtoms():
            if not a.GetAtomMapNum() or a.GetAtomMapNum() not in prd_maps:
                a.SetAtomMapNum(unmapped)
                unmapped += 1
    if PLEVEL >= 2: print('Added {} map nums to unmapped reactants'.format(unmapped-700))
    if unmapped > 800:
        raise ValueError('Why do you have so many unmapped atoms in the template reactants?')

    return rxn

def initialize_reactants_from_smiles(reactant_smiles):
    reactants = Chem.MolFromSmiles(reactant_smiles)
    Chem.AssignStereochemistry(reactants, flagPossibleStereoCenters=True)
    reactants.UpdatePropertyCache()
    [a.SetAtomMapNum(i+1) for (i, a) in enumerate(reactants.GetAtoms())]
    if PLEVEL >= 2: print('Initialized reactants, assigned map numbers, stereochem, flagpossiblestereocenters')
    return reactants

def get_template_frags_from_rxn(rxn):
    for i, rct in enumerate(rxn.GetReactants()):
        if i == 0:
            template_r = rct
        else:
            template_r = AllChem.CombineMols(template_r, rct)
    for i, prd in enumerate(rxn.GetProducts()):
        if i == 0:
            template_p = prd
        else:
            template_p = AllChem.CombineMols(template_p, prd)
    return template_r, template_p