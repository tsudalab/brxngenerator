from __future__ import print_function
import sys
import os
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem.rdchem import ChiralType, BondType, BondDir

from rdchiral.old.utils import vprint
from rdchiral.old.initialization import rdchiralReaction, rdchiralReactants
from rdchiral.old.chiral import template_atom_could_have_been_tetra, copy_chirality, atom_chirality_matches
from rdchiral.old.clean import canonicalize_outcome_smiles, combine_enantiomers_into_racemic

def rdchiralRunText(reaction_smarts, reactant_smiles, **kwargs):
    '''Run from SMARTS string and SMILES string. This is NOT recommended
    for library application, since initialization is pretty slow. You should
    separately initialize the template and molecules and call run()'''
    rxn = rdchiralReaction(reaction_smarts)
    reactants = rdchiralReactants(reactant_smiles)
    return rdchiralRun(rxn, reactants, **kwargs)

def rdchiralRun(rxn, reactants, keep_isotopes=False, combine_enantiomers=True):
    '''
    rxn = rdchiralReaction (rdkit reaction + auxilliary information)
    reactants = rdchiralReactants (rdkit mol + auxilliary information)

    note: there is a fair amount of initialization (assigning stereochem), most
    importantly assigning isotope numbers to the reactant atoms. It is
    HIGHLY recommended to use the custom classes for initialization.
    '''

    final_outcomes = set()

    atoms_r = reactants.atoms_r

    template_r, template_p = rxn.template_r, rxn.template_p

    atoms_rt_map = rxn.atoms_rt_map
    atoms_pt_map = rxn.atoms_pt_map


    outcomes = rxn.rxn.RunReactants((reactants.reactants_achiral,))
    vprint(2, 'Using naive RunReactants, {} outcomes', len(outcomes))
    if not outcomes:
        return []


    for outcome in outcomes:
        vprint(2, 'Processing {}', str([Chem.MolToSmiles(x, True) for x in outcome]))
        unmapped = 900
        for m in outcome:
            for a in m.GetAtoms():
                if not a.GetIsotope():
                    a.SetIsotope(unmapped)
                    unmapped += 1
        vprint(2, 'Added {} map numbers to product', unmapped-900)



        atoms_rt =  {a.GetIsotope(): atoms_rt_map[a.GetIntProp('old_mapno')] \
            for m in outcome for a in m.GetAtoms() if a.HasProp('old_mapno')}

        [a.SetIsotope(i) for (i, a) in atoms_rt.items()]

        if not all(atom_chirality_matches(atoms_rt[i], atoms_r[i]) for i in atoms_rt):
            vprint(2, 'Chirality violated! Should not have gotten this match')
            continue
        vprint(2, 'Chirality matches! Just checked with atom_chirality_matches')






        isotopes = [a.GetIsotope() for m in outcome for a in m.GetAtoms() if a.GetIsotope()]
        if len(isotopes) != len(set(isotopes)): # duplicate?
            vprint(1, 'Found duplicate isotopes in product - need to stitch')
            merged_mol = Chem.RWMol(outcome[0])
            merged_iso_to_id = {a.GetIsotope(): a.GetIdx() for a in outcome[0].GetAtoms() if a.GetIsotope()}
            for j in range(1, len(outcome)):
                new_mol = outcome[j]
                for a in new_mol.GetAtoms():
                    if a.GetIsotope() not in merged_iso_to_id:
                        merged_iso_to_id[a.GetIsotope()] = merged_mol.AddAtom(a)
                for b in new_mol.GetBonds():
                    bi = b.GetBeginAtom().GetIsotope()
                    bj = b.GetEndAtom().GetIsotope()
                    vprint(10, 'stitching bond between {} and {} in stich has chirality {}, {}'.format(
                        bi, bj, b.GetStereo(), b.GetBondDir()
                    ))
                    if not merged_mol.GetBondBetweenAtoms(
                            merged_iso_to_id[bi], merged_iso_to_id[bj]):
                        merged_mol.AddBond(merged_iso_to_id[bi],
                            merged_iso_to_id[bj], b.GetBondType())
                        merged_mol.GetBondBetweenAtoms(
                            merged_iso_to_id[bi], merged_iso_to_id[bj]
                        ).SetStereo(b.GetStereo())
                        merged_mol.GetBondBetweenAtoms(
                            merged_iso_to_id[bi], merged_iso_to_id[bj]
                        ).SetBondDir(b.GetBondDir())
            outcome = merged_mol.GetMol()
            vprint(1, 'Merged editable mol, converted back to real mol, {}', Chem.MolToSmiles(outcome, True))
        else:
            new_outcome = outcome[0]
            for j in range(1, len(outcome)):
                new_outcome = AllChem.CombineMols(new_outcome, outcome[j])
            outcome = new_outcome
        vprint(2, 'Converted all outcomes to single molecules')




        atoms_pt = {a.GetIsotope(): atoms_pt_map[a.GetIntProp('old_mapno')] \
            for a in outcome.GetAtoms() if a.HasProp('old_mapno')}
        atoms_p = {a.GetIsotope(): a for a in outcome.GetAtoms() if a.GetIsotope()}

        [a.SetIsotope(i) for (i, a) in atoms_pt.items()]



        missing_bonds = []
        for (i, j, b) in reactants.bonds_by_isotope:
            if i in atoms_p and j in atoms_p:
                if not outcome.GetBondBetweenAtoms(atoms_p[i].GetIdx(), atoms_p[j].GetIdx()):
                    if i not in atoms_rt or j not in atoms_rt or not template_r.GetBondBetweenAtoms(atoms_rt[i].GetIdx(), atoms_rt[j].GetIdx()):
                        missing_bonds.append((i, j, b))
        if missing_bonds:
            vprint(1, 'Product is missing non-reacted bonds that were present in reactants!')
            outcome = Chem.RWMol(outcome)
            rwmol_iso_to_id = {a.GetIsotope(): a.GetIdx() for a in outcome.GetAtoms() if a.GetIsotope()}
            for (i, j, b) in missing_bonds:
                outcome.AddBond(rwmol_iso_to_id[i], rwmol_iso_to_id[j])
                new_b = outcome.GetBondBetweenAtoms(rwmol_iso_to_id[i], rwmol_iso_to_id[j])
                new_b.SetBondType(b.GetBondType())
                new_b.SetBondDir(b.GetBondDir())
                new_b.SetIsAromatic(b.GetIsAromatic())
            outcome = outcome.GetMol()
        else:
            vprint(3, 'No missing bonds')


        try:
            outcome.UpdatePropertyCache()
        except ValueError as e:
            vprint(1, '{}, {}'.format(Chem.MolToSmiles(outcome, True), e))
            continue



        for a in outcome.GetAtoms():
            if not a.HasProp('old_mapno'):

                if not a.HasProp('react_atom_idx'):
                    vprint(4, 'Atom {} created by product template, should have right chirality', a.GetIsotope())

                else:
                    vprint(4, 'Atom {} outside of template, copy chirality from reactants', a.GetIsotope())
                    copy_chirality(atoms_r[a.GetIsotope()], a)
            else:

                if template_atom_could_have_been_tetra(atoms_rt[a.GetIsotope()]):
                    vprint(3, 'Atom {} was in rct template (could have been tetra)', a.GetIsotope())

                    if template_atom_could_have_been_tetra(atoms_pt[a.GetIsotope()]):
                        vprint(3, 'Atom {} in product template could have been tetra, too', a.GetIsotope())


                        if atoms_pt[a.GetIsotope()].GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
                            vprint(3, '...but it is not specified in product, so destroy chirality')
                            a.SetChiralTag(ChiralType.CHI_UNSPECIFIED)

                        else:
                            vprint(3, '...and product is specified')


                            if atoms_rt[a.GetIsotope()].GetChiralTag() == ChiralType.CHI_UNSPECIFIED:
                                vprint(3, '...but reactant template was not, so copy from product template')
                                copy_chirality(atoms_pt[a.GetIsotope()], a)

                            else:
                                vprint(3, '...and reactant template was, too! copy from reactants')
                                copy_chirality(atoms_r[a.GetIsotope()], a)
                                if not atom_chirality_matches(atoms_pt[a.GetIsotope()], atoms_rt[a.GetIsotope()]):
                                    vprint(3, 'but! reactant template and product template have opposite stereochem, so invert')
                                    a.InvertChirality()

                    else:
                        vprint(3, 'If reactant template could have been ' +
                            'chiral, but the product template could not, then we dont need ' +
                            'to worry about specifying product atom chirality')

                else:
                    vprint(3, 'Atom {} could not have been chiral in reactant template', a.GetIsotope())

                    if not template_atom_could_have_been_tetra(atoms_pt[a.GetIsotope()]):
                        vprint(3, 'Atom {} also could not have been chiral in product template', a.GetIsotope())
                        vprint(3, '...so, copy chirality from reactant instead')
                        copy_chirality(atoms_r[a.GetIsotope()], a)

                    else:
                        vprint(3, 'Atom could/does have product template chirality!', a.GetIsotope())
                        vprint(3, '...so, copy chirality from product template')
                        copy_chirality(atoms_pt[a.GetIsotope()], a)

            vprint(3, 'New chiral tag {}', a.GetChiralTag())
        vprint(2, 'After attempting to re-introduce chirality, outcome = {}',
            Chem.MolToSmiles(outcome, True))




        if not keep_isotopes:
            [a.SetIsotope(0) for a in outcome.GetAtoms()]

        smiles = canonicalize_outcome_smiles(outcome)
        if smiles is not None:
            final_outcomes.add(smiles)

    if combine_enantiomers:
        final_outcomes = combine_enantiomers_into_racemic(final_outcomes)

    return list(final_outcomes)


if __name__ == '__main__':
    reaction_smarts = '[C:1][OH:2]>>[C:1][O:2][C]'
    reactant_smiles = 'CC(=O)OCCCO'
    outcomes = rdchiralRunText(reaction_smarts, reactant_smiles)
    print(outcomes)
