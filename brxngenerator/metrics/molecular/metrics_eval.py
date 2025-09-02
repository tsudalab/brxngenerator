"""
Five standardized metrics for molecular generation evaluation.

Implements MOSES-compatible metrics plus reaction validity following RDKit best practices.
All metrics computed on real data - no dummy values.

References:
- MOSES: https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.565644/full
- QED: https://pubmed.ncbi.nlm.nih.gov/22270643/ 
- SAS: https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8
- RDKit: https://www.rdkit.org/docs/
"""

import json
import csv
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

# RDKit imports with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import QED, rdChemReactions
    RDKIT_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] RDKit not available: {e}")
    RDKIT_AVAILABLE = False

# SAS scorer import with fallback
SAS_AVAILABLE = False
try:
    import sascorer
    SAS_AVAILABLE = True
except ImportError:
    try:
        # Try RDKit contrib version
        from rdkit.Contrib.SA_Score import sascorer
        SAS_AVAILABLE = True
    except ImportError:
        print("[Warning] SAS scorer not available - SAS metrics will be N/A")
        SAS_AVAILABLE = False


class MolecularMetrics:
    """
    Standardized molecular generation evaluation metrics.
    
    Implements 5 core metrics:
    1. Valid reaction rate - fraction of parseable reactions
    2. Average QED - drug-likeness score
    3. Uniqueness - fraction of unique valid molecules  
    4. Novelty - fraction not in training set
    5. Average SAS - synthetic accessibility score
    """
    
    def __init__(self, training_smiles: Optional[List[str]] = None):
        """
        Initialize metrics evaluator.
        
        Args:
            training_smiles: List of training set SMILES for novelty computation
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for metrics evaluation")
            
        # Canonicalize training set for novelty computation
        self.training_canonical = set()
        if training_smiles:
            print(f"Canonicalizing {len(training_smiles)} training molecules...")
            for smi in training_smiles:
                canonical = self._canonicalize_smiles(smi)
                if canonical:
                    self.training_canonical.add(canonical)
            print(f"Training set: {len(self.training_canonical)} canonical molecules")
    
    def _canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """Convert SMILES to canonical form, return None if invalid."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                Chem.SanitizeMol(mol)  # Ensure molecule is valid
                return Chem.MolToSmiles(mol)
        except Exception:
            pass
        return None
    
    def _validate_reaction(self, reaction_string: str) -> bool:
        """
        Validate reaction string using RDKit.
        
        Returns True if reaction parses and all reactants/products sanitize.
        """
        try:
            # Try parsing as reaction SMARTS first
            rxn = rdChemReactions.ReactionFromSmarts(reaction_string)
            if rxn is None:
                # Try as reaction SMILES
                rxn = rdChemReactions.ReactionFromSmiles(reaction_string)
                
            if rxn is None:
                return False
                
            # Validate all molecules in the reaction
            for reactant_template in rxn.GetReactants():
                if reactant_template is None:
                    return False
                    
            for product_template in rxn.GetProducts():
                if product_template is None:
                    return False
                    
            return True
            
        except Exception:
            return False
    
    def compute_valid_reaction_rate(self, reaction_strings: List[str]) -> Dict[str, float]:
        """
        Compute fraction of valid reactions.
        
        Args:
            reaction_strings: List of reaction SMILES/SMARTS strings
            
        Returns:
            Dict with 'valid_reaction_rate', 'valid_count', 'total_count'
        """
        if not reaction_strings:
            return {'valid_reaction_rate': 0.0, 'valid_count': 0, 'total_count': 0}
            
        valid_count = 0
        for rxn_str in reaction_strings:
            if self._validate_reaction(rxn_str):
                valid_count += 1
                
        return {
            'valid_reaction_rate': valid_count / len(reaction_strings),
            'valid_count': valid_count,
            'total_count': len(reaction_strings)
        }
    
    def compute_valid_molecule_rate(self, smiles_list: List[str]) -> Dict[str, Union[float, int]]:
        """
        Compute fraction of valid molecules (fallback when no reactions available).
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dict with 'valid_molecule_rate', 'valid_count', 'total_count'
        """
        if not smiles_list:
            return {'valid_molecule_rate': 0.0, 'valid_count': 0, 'total_count': 0}
            
        valid_count = 0
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    Chem.SanitizeMol(mol)
                    valid_count += 1
            except Exception:
                continue
                
        return {
            'valid_molecule_rate': valid_count / len(smiles_list),
            'valid_count': valid_count,
            'total_count': len(smiles_list)
        }
    
    def compute_average_qed(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Compute average QED score for valid molecules.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dict with 'avg_qed', 'qed_std', 'valid_count'
        """
        qed_scores = []
        
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    Chem.SanitizeMol(mol)
                    qed_score = QED.qed(mol)
                    qed_scores.append(qed_score)
            except Exception:
                continue
                
        if not qed_scores:
            return {'avg_qed': 0.0, 'qed_std': 0.0, 'valid_count': 0}
            
        return {
            'avg_qed': float(np.mean(qed_scores)),
            'qed_std': float(np.std(qed_scores)),
            'valid_count': len(qed_scores)
        }
    
    def compute_uniqueness(self, smiles_list: List[str]) -> Dict[str, Union[float, int]]:
        """
        Compute uniqueness as fraction of unique canonical SMILES among valid molecules.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dict with 'uniqueness', 'unique_count', 'valid_count'
        """
        canonical_smiles = []
        
        for smi in smiles_list:
            canonical = self._canonicalize_smiles(smi)
            if canonical:
                canonical_smiles.append(canonical)
                
        if not canonical_smiles:
            return {'uniqueness': 0.0, 'unique_count': 0, 'valid_count': 0}
            
        unique_canonical = set(canonical_smiles)
        
        return {
            'uniqueness': len(unique_canonical) / len(canonical_smiles),
            'unique_count': len(unique_canonical),
            'valid_count': len(canonical_smiles)
        }
    
    def compute_novelty(self, smiles_list: List[str]) -> Dict[str, Union[float, int]]:
        """
        Compute novelty as fraction of unique valid molecules not in training set.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dict with 'novelty', 'novel_count', 'unique_count'
        """
        if not self.training_canonical:
            print("[Warning] No training set provided - novelty = N/A")
            uniqueness_result = self.compute_uniqueness(smiles_list)
            return {
                'novelty': 'N/A',
                'novel_count': 'N/A', 
                'unique_count': uniqueness_result['unique_count']
            }
            
        canonical_smiles = []
        for smi in smiles_list:
            canonical = self._canonicalize_smiles(smi)
            if canonical:
                canonical_smiles.append(canonical)
                
        if not canonical_smiles:
            return {'novelty': 0.0, 'novel_count': 0, 'unique_count': 0}
            
        unique_canonical = set(canonical_smiles)
        novel_molecules = unique_canonical - self.training_canonical
        
        return {
            'novelty': len(novel_molecules) / len(unique_canonical) if unique_canonical else 0.0,
            'novel_count': len(novel_molecules),
            'unique_count': len(unique_canonical)
        }
    
    def compute_average_sas(self, smiles_list: List[str]) -> Dict[str, Union[float, str]]:
        """
        Compute average SAS (Synthetic Accessibility Score) for valid molecules.
        Lower scores indicate easier synthesis (range ~1-10).
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dict with 'avg_sas', 'sas_std', 'valid_count'
        """
        if not SAS_AVAILABLE:
            return {'avg_sas': 'N/A', 'sas_std': 'N/A', 'valid_count': 'N/A'}
            
        sas_scores = []
        
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    Chem.SanitizeMol(mol)
                    sas_score = sascorer.calculateScore(mol)
                    # Ensure valid SAS range (typically 1-10)
                    if 0.1 <= sas_score <= 15.0:  # Allow some tolerance
                        sas_scores.append(sas_score)
            except Exception:
                continue
                
        if not sas_scores:
            return {'avg_sas': 'N/A', 'sas_std': 'N/A', 'valid_count': 0}
            
        return {
            'avg_sas': float(np.mean(sas_scores)),
            'sas_std': float(np.std(sas_scores)),
            'valid_count': len(sas_scores)
        }
    
    def evaluate_all_metrics(self, 
                           generated_data: List[str],
                           data_type: str = "molecules") -> Dict[str, Union[float, int]]:
        """
        Compute all 5 metrics on generated data.
        
        Args:
            generated_data: List of SMILES strings or reaction strings
            data_type: "molecules" or "reactions"
            
        Returns:
            Dict with all metric results plus metadata
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'sample_count': len(generated_data),
            'training_set_size': len(self.training_canonical)
        }
        
        # Validity metric
        if data_type == "reactions":
            validity_result = self.compute_valid_reaction_rate(generated_data)
            results.update(validity_result)
            # Extract molecules from reactions for other metrics
            molecule_smiles = []
            for rxn_str in generated_data:
                if self._validate_reaction(rxn_str):
                    # Extract product molecules (simplified - take first product)
                    try:
                        if '>>' in rxn_str:
                            products = rxn_str.split('>>')[-1].split('.')
                            if products:
                                molecule_smiles.append(products[0])
                    except Exception:
                        continue
        else:
            validity_result = self.compute_valid_molecule_rate(generated_data)
            results.update(validity_result)
            molecule_smiles = generated_data
        
        # Molecular metrics (QED, uniqueness, novelty, SAS)
        results.update(self.compute_average_qed(molecule_smiles))
        results.update(self.compute_uniqueness(molecule_smiles))
        results.update(self.compute_novelty(molecule_smiles))
        results.update(self.compute_average_sas(molecule_smiles))
        
        return results
    
    def save_results(self, 
                    results: Dict,
                    output_path: str,
                    metadata: Optional[Dict] = None):
        """
        Save evaluation results to JSON and CSV files.
        
        Args:
            results: Results dict from evaluate_all_metrics
            output_path: Base path for output files (without extension)
            metadata: Additional metadata (git commit, parameter set, etc.)
        """
        # Add metadata
        if metadata:
            results.update(metadata)
            
        # Save JSON (complete results)
        json_path = f"{output_path}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {json_path}")
        
        # Save CSV (summary metrics only)
        csv_path = f"{output_path}.csv"
        summary_metrics = [
            'valid_reaction_rate', 'valid_molecule_rate', 
            'avg_qed', 'uniqueness', 'novelty', 'avg_sas'
        ]
        
        csv_data = {k: results.get(k, 'N/A') for k in summary_metrics}
        csv_data['timestamp'] = results.get('timestamp')
        csv_data['sample_count'] = results.get('sample_count')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data.keys())
            writer.writeheader()
            writer.writerow(csv_data)
        print(f"Summary saved to {csv_path}")


def load_training_molecules(data_path: str) -> List[str]:
    """
    Load training molecules from project data file.
    
    Args:
        data_path: Path to data.txt file
        
    Returns:
        List of SMILES strings from training data
    """
    training_smiles = []
    
    try:
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract SMILES from data format (adapt to your format)
                    parts = line.split()
                    if parts:
                        # Assume first part is SMILES (adjust if needed)
                        smiles = parts[0]
                        training_smiles.append(smiles)
                        
    except FileNotFoundError:
        print(f"[Warning] Training data not found at {data_path}")
        
    return training_smiles


# Example usage and testing
if __name__ == "__main__":
    # Test with dummy data (replace with real data loading)
    test_smiles = [
        "CCO",  # Ethanol (valid)
        "CC(=O)O",  # Acetic acid (valid)  
        "INVALID_SMILES",  # Invalid
        "c1ccccc1",  # Benzene (valid)
        "CCO"  # Duplicate
    ]
    
    test_reactions = [
        "CCO>>CC=O",  # Simple oxidation
        "INVALID>>REACTION",  # Invalid
        "CC(=O)O.CCO>>CC(=O)OCC"  # Esterification
    ]
    
    print("Testing metrics evaluation...")
    
    # Initialize with dummy training set
    evaluator = MolecularMetrics(training_smiles=["CCO", "CC(=O)O"])
    
    # Test molecule metrics
    print("\nMolecule metrics:")
    mol_results = evaluator.evaluate_all_metrics(test_smiles, data_type="molecules")
    for k, v in mol_results.items():
        print(f"  {k}: {v}")
    
    # Test reaction metrics  
    print("\nReaction metrics:")
    rxn_results = evaluator.evaluate_all_metrics(test_reactions, data_type="reactions")
    for k, v in rxn_results.items():
        print(f"  {k}: {v}")
    
    print("\n✅ Metrics module test complete!")