#!/usr/bin/env python3
"""
FM Hyperparameter Testing for Specific VAE Models

This script tests different FM hyperparameter combinations on real latent vectors
from trained VAE models to find optimal settings for different target metrics.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import json
from pathlib import Path
from itertools import product
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brxngenerator.core.binary_vae_utils import FactorizationMachineSurrogate, prepare_dataset, create_ecc_codec
from rxnft_vae.vae import bFTRXNVAE

class FMHyperparamTester:
    def __init__(self, 
                 vae_model_path,
                 data_pairs_path=None, 
                 device='cpu',
                 output_dir='./fm_testing/hyperparams_results'):
        """
        Initialize FM hyperparameter tester
        
        Args:
            vae_model_path: Path to trained VAE model (.pt file)
            data_pairs_path: Path to molecular data pairs (if None, uses default)
            device: Device for computations
            output_dir: Directory for saving results
        """
        self.vae_model_path = vae_model_path
        self.data_pairs_path = data_pairs_path
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load VAE model
        self.vae_model = self.load_vae_model()
        self.model_name = Path(vae_model_path).stem
        
    def load_vae_model(self):
        """Load trained VAE model"""
        print(f"Loading VAE model from: {self.vae_model_path}")
        
        # Load model state
        checkpoint = torch.load(self.vae_model_path, map_location=self.device)
        
        # Extract model parameters from checkpoint
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint
            
        # Initialize VAE model (you may need to adjust these parameters based on your model)
        vae_model = bFTRXNVAE(
            device=self.device,
            hidden_size=200,  # Adjust based on your model
            latent_size=100,  # Adjust based on your model  
            depth=2,          # Adjust based on your model
            beta=1.0
        )
        
        vae_model.load_state_dict(model_state)
        vae_model.eval()
        
        print(f"✓ VAE model loaded successfully")
        return vae_model
        
    def load_molecular_data(self, subset_size=None):
        """Load molecular data pairs for testing"""
        if self.data_pairs_path:
            # Load custom data pairs
            print(f"Loading molecular data from: {self.data_pairs_path}")
            # Implement your data loading logic here
            raise NotImplementedError("Custom data loading not implemented yet")
        else:
            # Use default data from brxngenerator
            print("Loading default molecular data...")
            try:
                from brxngenerator.core.data_loader import get_data_pairs
                data_pairs = get_data_pairs()
                
                if subset_size and len(data_pairs) > subset_size:
                    # Use random subset for faster testing
                    np.random.seed(42)
                    indices = np.random.choice(len(data_pairs), subset_size, replace=False)
                    data_pairs = [data_pairs[i] for i in indices]
                    
                print(f"✓ Loaded {len(data_pairs)} molecular data pairs")
                return data_pairs
                
            except ImportError:
                print("⚠️  Default data loader not available, generating synthetic molecular-like data")
                return self.generate_synthetic_molecular_data(subset_size or 1000)
    
    def generate_synthetic_molecular_data(self, n_samples=1000):
        """Generate synthetic data that mimics molecular latent vectors"""
        print(f"Generating {n_samples} synthetic molecular-like samples...")
        
        # This is a placeholder - in real usage, you'd load actual molecular data
        # For testing purposes, we'll create data that has similar properties to real molecules
        latent_size = self.vae_model.latent_size
        
        # Generate synthetic latent vectors
        synthetic_latents = []
        synthetic_qed_scores = []
        synthetic_logp_scores = []
        
        for _ in range(n_samples):
            # Generate latent vector with molecular-like properties
            latent = np.random.binomial(1, 0.3, size=latent_size).astype(np.float32)
            
            # Generate realistic QED scores (0-1, higher is better)
            qed_score = np.random.beta(2, 3)  # Skewed towards lower values like real QED
            
            # Generate realistic logP scores (typically -3 to 7)
            logp_score = np.random.normal(2.0, 1.5)
            
            synthetic_latents.append(latent)
            synthetic_qed_scores.append(qed_score)
            synthetic_logp_scores.append(logp_score)
        
        return {
            'latents': np.array(synthetic_latents),
            'qed_scores': np.array(synthetic_qed_scores),
            'logp_scores': np.array(synthetic_logp_scores)
        }
    
    def prepare_fm_dataset(self, data_pairs, metric='qed', ecc_type='none', ecc_R=3):
        """Prepare dataset for FM training using the VAE model"""
        print(f"Preparing FM dataset for metric: {metric}")
        
        if isinstance(data_pairs, dict):  # Synthetic data
            latents = torch.FloatTensor(data_pairs['latents'])
            if metric == 'qed':
                scores = data_pairs['qed_scores'].reshape(-1, 1)
            elif metric == 'logp':
                scores = data_pairs['logp_scores'].reshape(-1, 1)
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            # Use first half of latent vector as per original logic
            half_latents = latents[:, :latents.shape[1] // 2]
            
            # Apply ECC decoding if enabled
            if ecc_type != 'none':
                ecc_codec = create_ecc_codec(ecc_type, R=ecc_R)
                if ecc_codec is not None:
                    from brxngenerator.core.binary_vae_utils import extract_latent_info_bits
                    half_latents = extract_latent_info_bits(half_latents, ecc_codec)
                    print(f"[ECC] Extracted {half_latents.shape[1]} info bits from {latents.shape[1] // 2} code bits")
            
            latents_np = half_latents.detach().cpu().numpy()
            n = latents_np.shape[0]
            
            # Train/test split
            permutation = np.random.permutation(n)
            train_idx, test_idx = permutation[:int(0.9 * n)], permutation[int(0.9 * n):]
            
            X_train, X_test = latents_np[train_idx, :], latents_np[test_idx, :]
            y_train, y_test = -scores[train_idx], -scores[test_idx]  # Negative for maximization
            
        else:  # Real molecular data pairs
            # Use the prepare_dataset function from binary_vae_utils
            logp_paths = {}  # You'd need to provide actual logp paths for logp metric
            X_train, y_train, X_test, y_test = prepare_dataset(
                model=self.vae_model,
                data_pairs=data_pairs,
                latent_size=self.vae_model.latent_size,
                metric=metric,
                logp_paths=logp_paths,
                ecc_type=ecc_type,
                ecc_R=ecc_R
            )
        
        print(f"Dataset prepared: X_train={X_train.shape}, y_train={y_train.shape}")
        return X_train, y_train, X_test, y_test
    
    def test_hyperparameter_combination(self, X_train, y_train, X_test, y_test, 
                                      k_factors, lr, decay_weight, batch_size, 
                                      max_epoch, patience, param_init, metric, seed=42):
        """Test a single hyperparameter combination"""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Initialize FM
        fm_surrogate = FactorizationMachineSurrogate(
            n_binary=X_train.shape[1],
            k_factors=k_factors,
            lr=lr,
            decay_weight=decay_weight,
            batch_size=batch_size,
            max_epoch=max_epoch,
            patience=patience,
            param_init=param_init,
            cache_dir=str(self.output_dir / "temp_models"),
            prop=f"{metric}_test",
            client="hyperparam_test",
            random_seed=seed,
            device=self.device
        )
        
        # Train the model
        try:
            fm_surrogate.train(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
            
            # Evaluate
            model = fm_surrogate.model
            model.eval()
            with torch.no_grad():
                y_pred_train = model(X_train_tensor).cpu().numpy()
                y_pred_test = model(X_test_tensor).cpu().numpy()
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            return {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'converged': True,
                'final_epoch': len(fm_surrogate.train_errors) if hasattr(fm_surrogate, 'train_errors') else max_epoch
            }
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            return {
                'train_mse': float('inf'),
                'test_mse': float('inf'),
                'train_r2': -float('inf'),
                'test_r2': -float('inf'),
                'converged': False,
                'error': str(e),
                'final_epoch': 0
            }
    
    def run_hyperparameter_search(self, 
                                metrics=['qed', 'logp'],
                                ecc_configs=[('none', 3), ('repetition', 2)],
                                subset_size=2000,
                                n_seeds=3):
        """Run comprehensive hyperparameter search"""
        
        print("="*60)
        print("FM HYPERPARAMETER SEARCH")
        print("="*60)
        print(f"VAE Model: {self.model_name}")
        print(f"Target Metrics: {metrics}")
        print(f"ECC Configurations: {ecc_configs}")
        print(f"Dataset Size: {subset_size}")
        print(f"Random Seeds: {n_seeds}")
        
        # Define hyperparameter grid
        hyperparams = {
            'k_factors': [10, 20, 40],
            'lr': [0.001, 0.01, 0.03],
            'decay_weight': [0.001, 0.01],
            'batch_size': [64, 128, 256],
            'max_epoch': [200, 500],
            'patience': [20, 50],
            'param_init': [0.01, 0.1]
        }
        
        # Generate all combinations
        param_names = list(hyperparams.keys())
        param_values = list(hyperparams.values())
        param_combinations = list(product(*param_values))
        
        print(f"Total hyperparameter combinations: {len(param_combinations)}")
        
        # Load molecular data
        data_pairs = self.load_molecular_data(subset_size)
        
        all_results = []
        
        # Test each metric and ECC configuration
        for metric in metrics:
            for ecc_type, ecc_R in ecc_configs:
                print(f"\n--- Testing {metric.upper()} with ECC={ecc_type}, R={ecc_R} ---")
                
                # Prepare dataset for this configuration
                X_train, y_train, X_test, y_test = self.prepare_fm_dataset(
                    data_pairs, metric=metric, ecc_type=ecc_type, ecc_R=ecc_R
                )
                
                # Test each hyperparameter combination
                for i, param_combo in enumerate(tqdm(param_combinations, desc=f"Testing {metric}")):
                    param_dict = dict(zip(param_names, param_combo))
                    
                    # Test with multiple seeds
                    seed_results = []
                    for seed in range(n_seeds):
                        result = self.test_hyperparameter_combination(
                            X_train, y_train, X_test, y_test, 
                            seed=seed, metric=metric, **param_dict
                        )
                        result.update(param_dict)
                        result.update({
                            'metric': metric,
                            'ecc_type': ecc_type,
                            'ecc_R': ecc_R,
                            'seed': seed,
                            'vae_model': self.model_name,
                            'combo_id': i
                        })
                        seed_results.append(result)
                    
                    all_results.extend(seed_results)
        
        # Save results
        results_df = pd.DataFrame(all_results)
        results_file = self.output_dir / f"{self.model_name}_hyperparams_results.csv"
        results_df.to_csv(results_file, index=False)
        
        # Create summary statistics
        summary_df = self.create_results_summary(results_df)
        summary_file = self.output_dir / f"{self.model_name}_hyperparams_summary.csv" 
        summary_df.to_csv(summary_file, index=False)
        
        # Create visualizations
        self.create_visualizations(results_df, summary_df)
        
        print(f"\n✓ Results saved to: {results_file}")
        print(f"✓ Summary saved to: {summary_file}")
        
        return results_df, summary_df
    
    def create_results_summary(self, results_df):
        """Create summary statistics across multiple seeds"""
        
        # Group by all parameters except seed
        group_cols = [col for col in results_df.columns if col not in ['seed', 'train_mse', 'test_mse', 'train_r2', 'test_r2', 'final_epoch', 'converged', 'error']]
        
        summary_stats = []
        for name, group in results_df.groupby(group_cols):
            if len(group) > 0:
                stats = {
                    'test_r2_mean': group['test_r2'].mean(),
                    'test_r2_std': group['test_r2'].std(), 
                    'test_mse_mean': group['test_mse'].mean(),
                    'test_mse_std': group['test_mse'].std(),
                    'train_r2_mean': group['train_r2'].mean(),
                    'convergence_rate': group['converged'].mean(),
                    'avg_epochs': group['final_epoch'].mean(),
                    'n_seeds': len(group)
                }
                
                # Add parameter values
                for i, col in enumerate(group_cols):
                    if isinstance(name, tuple):
                        stats[col] = name[i]
                    else:
                        stats[col] = name
                
                summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Rank by test R² performance
        summary_df = summary_df.sort_values('test_r2_mean', ascending=False).reset_index(drop=True)
        summary_df['rank'] = range(1, len(summary_df) + 1)
        
        return summary_df
    
    def create_visualizations(self, results_df, summary_df):
        """Create visualization plots for hyperparameter analysis"""
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Best hyperparameters per metric
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'FM Hyperparameter Analysis - {self.model_name}', fontsize=16)
        
        metrics = summary_df['metric'].unique()
        ecc_configs = summary_df[['ecc_type', 'ecc_R']].drop_duplicates()
        
        for i, metric in enumerate(metrics[:2]):  # Limit to 2 metrics for layout
            for j, (ecc_type, ecc_R) in enumerate(ecc_configs.values[:2]):  # Limit to 2 ECC configs
                
                subset = summary_df[
                    (summary_df['metric'] == metric) & 
                    (summary_df['ecc_type'] == ecc_type) & 
                    (summary_df['ecc_R'] == ecc_R)
                ]
                
                if len(subset) > 0:
                    # Plot top 10 configurations
                    top_configs = subset.head(10)
                    
                    ax = axes[i, j]
                    ax.barh(range(len(top_configs)), top_configs['test_r2_mean'])
                    ax.set_yticks(range(len(top_configs)))
                    ax.set_yticklabels([f"k={row['k_factors']}, lr={row['lr']}" for _, row in top_configs.iterrows()])
                    ax.set_xlabel('Test R²')
                    ax.set_title(f'{metric.upper()} | ECC={ecc_type}, R={ecc_R}')
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.model_name}_best_hyperparams.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Parameter sensitivity analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Parameter Sensitivity Analysis - {self.model_name}', fontsize=16)
        
        params_to_plot = ['k_factors', 'lr', 'decay_weight', 'batch_size', 'patience', 'param_init']
        
        for i, param in enumerate(params_to_plot):
            ax = axes[i // 3, i % 3]
            
            # Average performance per parameter value across all metrics
            param_perf = summary_df.groupby(param)['test_r2_mean'].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(param_perf[param], param_perf['mean'], yerr=param_perf['std'], 
                       marker='o', capsize=5, capthick=2)
            ax.set_xlabel(param)
            ax.set_ylabel('Test R² (mean ± std)')
            ax.set_title(f'Sensitivity to {param}')
            ax.grid(True, alpha=0.3)
            
            if param in ['lr', 'decay_weight', 'param_init']:
                ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.model_name}_sensitivity.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Select numeric columns for correlation
        numeric_cols = ['k_factors', 'lr', 'decay_weight', 'batch_size', 'patience', 'param_init', 'test_r2_mean', 'test_mse_mean']
        corr_data = summary_df[numeric_cols].corr()
        
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title(f'Hyperparameter Correlations - {self.model_name}')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.model_name}_correlations.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualizations saved to {self.output_dir}")
    
    def print_best_configs(self, summary_df, top_k=5):
        """Print the best hyperparameter configurations"""
        
        print(f"\n{'='*80}")
        print(f"TOP {top_k} HYPERPARAMETER CONFIGURATIONS")
        print(f"{'='*80}")
        
        for metric in summary_df['metric'].unique():
            for ecc_type in summary_df['ecc_type'].unique():
                
                subset = summary_df[
                    (summary_df['metric'] == metric) & 
                    (summary_df['ecc_type'] == ecc_type)
                ]
                
                if len(subset) > 0:
                    print(f"\n--- {metric.upper()} with ECC={ecc_type} ---")
                    
                    top_configs = subset.head(top_k)
                    
                    for idx, (_, config) in enumerate(top_configs.iterrows(), 1):
                        print(f"{idx}. Test R²={config['test_r2_mean']:.4f} (±{config['test_r2_std']:.4f})")
                        print(f"   k_factors={config['k_factors']}, lr={config['lr']}, decay={config['decay_weight']}")
                        print(f"   batch_size={config['batch_size']}, patience={config['patience']}, init={config['param_init']}")
                        print()

def main():
    """Main function to run FM hyperparameter testing"""
    
    # Configuration
    vae_model_path = "/Users/guanjiezou/CliecyPro/brxngenerator-2/weights/compare_baseline/best_model.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Initialize tester
    tester = FMHyperparamTester(
        vae_model_path=vae_model_path,
        device=device
    )
    
    # Run hyperparameter search
    results_df, summary_df = tester.run_hyperparameter_search(
        metrics=['qed'],  # Start with QED only for faster testing
        ecc_configs=[('none', 3), ('repetition', 2)],
        subset_size=1000,  # Smaller for initial testing
        n_seeds=2  # Fewer seeds for faster testing
    )
    
    # Print best configurations
    tester.print_best_configs(summary_df)

if __name__ == "__main__":
    main()