"""
Enhanced molecular metrics with canonicalization, SAS, novelty, and uniqueness.

Provides improved canonicalization using RDKit standardization and implements
MOSES-compatible novelty and uniqueness metrics plus SAS scoring.

References:
- MOSES: https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.565644/full
- SAS: https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8
- RDKit standardization: https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.rdMolStandardize.html
"""

from typing import Optional, List
import sys
import os

# Add rxnft_vae to path for sascorer import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rxnft_vae'))

try:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDKIT_AVAILABLE = True
except ImportError:
    print("[Warning] RDKit not available")
    RDKIT_AVAILABLE = False

# Import SAS scorer from project  
try:
    import sascorer
    # Test that it can load the fragment scores
    sascorer.readFragmentScores()
    SAS_AVAILABLE = True
    print("✅ SAS scorer loaded successfully")
except Exception as e:
    print(f"[Warning] SAS scorer not available: {e}")
    SAS_AVAILABLE = False


def to_mol(smi: str):
    """Convert SMILES to RDKit Mol object with error handling."""
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        m = Chem.MolFromSmiles(smi)
        if not m:
            return None
        
        # Apply standardization and cleanup
        m = rdMolStandardize.Cleanup(m)
        return m
    except Exception:
        return None


def canon_smi(smi: str) -> Optional[str]:
    """
    Convert SMILES to canonical form with standardization.
    
    Uses RDKit standardization to reduce tautomer/charge noise.
    """
    m = to_mol(smi)
    if not m:
        return None
    
    try:
        return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def mean_sas(mols: List) -> Optional[float]:
    """
    Compute mean SAS score for list of RDKit Mol objects.
    
    Args:
        mols: List of RDKit Mol objects (None values are skipped)
        
    Returns:
        Mean SAS score or None if no valid molecules
    """
    if not SAS_AVAILABLE:
        return None
        
    vals = []
    for m in mols:
        if m is None:
            continue
        try:
            score = sascorer.calculateScore(m)
            vals.append(score)
        except Exception:
            continue
    
    return float(sum(vals) / len(vals)) if vals else None


def novelty(gen_can: List[str], train_can_set: set) -> Optional[float]:
    """
    Compute novelty as fraction of generated molecules not in training set.
    
    Following MOSES definition: novelty = |unique_generated - training_set| / |unique_generated|
    
    Args:
        gen_can: List of canonical SMILES (None values are skipped)
        train_can_set: Set of training canonical SMILES
        
    Returns:
        Novelty fraction or None if no valid generated molecules
    """
    if not gen_can:
        return None
    
    # Filter out None values and get unique set
    valid_gen = set(s for s in gen_can if s is not None)
    if not valid_gen:
        return None
    
    # Count molecules not in training set
    novel_count = sum(1 for s in valid_gen if s not in train_can_set)
    return novel_count / len(valid_gen)


def uniqueness(gen_can: List[str]) -> Optional[float]:
    """
    Compute uniqueness as fraction of unique molecules among valid generated ones.
    
    Following MOSES definition: uniqueness = |unique_valid| / |valid|
    
    Args:
        gen_can: List of canonical SMILES (None values are skipped)
        
    Returns:
        Uniqueness fraction or None if no valid molecules
    """
    # Filter out None values
    valid_gen = [s for s in gen_can if s is not None]
    if not valid_gen:
        return None
    
    unique_set = set(valid_gen)
    return len(unique_set) / len(valid_gen)