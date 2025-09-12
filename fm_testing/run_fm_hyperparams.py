#!/usr/bin/env python3
"""
Simple runner script to test FM hyperparameters for different VAE models and metrics
"""

import torch
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fm_testing.test_fm_hyperparams import FMHyperparamTester

def run_comprehensive_fm_testing():
    """Run FM hyperparameter testing for all available VAE models"""
    
    print("="*60)
    print("COMPREHENSIVE FM HYPERPARAMETER TESTING")
    print("="*60)
    
    # Find available VAE models
    weights_dir = Path("/Users/guanjiezou/CliecyPro/brxngenerator-2/weights")
    vae_models = [
        weights_dir / "compare_baseline" / "best_model.pt",
        weights_dir / "compare_ecc" / "best_model.pt",
        weights_dir / "hidden_size_100_latent_size_100_depth_2_beta_1.0_lr_0.001" / "bvae_best_model_with.pt"
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test configurations
    metrics = ['qed', 'logp'] 
    ecc_configs = [('none', 3), ('repetition', 2)]
    
    all_results = {}
    
    for vae_model_path in vae_models:
        if not vae_model_path.exists():
            print(f"⚠️  Model not found: {vae_model_path}")
            continue
            
        print(f"\n{'='*40}")
        print(f"Testing VAE Model: {vae_model_path.name}")
        print(f"{'='*40}")
        
        try:
            # Initialize tester
            tester = FMHyperparamTester(
                vae_model_path=str(vae_model_path),
                device=device,
                output_dir=f"./fm_testing/results_{vae_model_path.parent.name}"
            )
            
            # Run hyperparameter search
            results_df, summary_df = tester.run_hyperparameter_search(
                metrics=metrics,
                ecc_configs=ecc_configs,
                subset_size=1500,  # Reasonable size for thorough testing
                n_seeds=3
            )
            
            # Print best configurations
            tester.print_best_configs(summary_df, top_k=3)
            
            # Store results
            all_results[vae_model_path.name] = {
                'results': results_df,
                'summary': summary_df,
                'best_config': summary_df.iloc[0].to_dict()
            }
            
            print(f"✅ Completed testing for {vae_model_path.name}")
            
        except Exception as e:
            print(f"❌ Failed to test {vae_model_path.name}: {str(e)}")
            continue
    
    # Create cross-model comparison
    if len(all_results) > 1:
        create_cross_model_comparison(all_results)
    
    print(f"\n{'='*60}")
    print("✅ COMPREHENSIVE TESTING COMPLETED")
    print(f"{'='*60}")
    
    return all_results

def create_cross_model_comparison(all_results):
    """Create comparison across different VAE models"""
    print(f"\n{'='*60}")
    print("CROSS-MODEL FM PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    import pandas as pd
    
    # Collect best configurations from each model
    comparison_data = []
    
    for model_name, results in all_results.items():
        best_config = results['best_config']
        comparison_data.append({
            'vae_model': model_name,
            'best_test_r2': best_config['test_r2_mean'],
            'best_test_r2_std': best_config['test_r2_std'],
            'metric': best_config['metric'],
            'ecc_type': best_config['ecc_type'],
            'k_factors': best_config['k_factors'],
            'lr': best_config['lr'],
            'decay_weight': best_config['decay_weight']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('best_test_r2', ascending=False)
    
    print("\nBest FM Performance by VAE Model:")
    print("-" * 80)
    for _, row in comparison_df.iterrows():
        print(f"{row['vae_model']:50} | R²={row['best_test_r2']:.4f} (±{row['best_test_r2_std']:.4f}) | {row['metric']} | ECC={row['ecc_type']}")
        print(f"{'':50} | k={row['k_factors']}, lr={row['lr']}, decay={row['decay_weight']}")
        print()
    
    # Save comparison
    output_dir = Path("./fm_testing/cross_model_comparison")
    output_dir.mkdir(exist_ok=True)
    comparison_df.to_csv(output_dir / "vae_model_comparison.csv", index=False)
    
    print(f"✅ Cross-model comparison saved to: {output_dir}")

def run_quick_fm_test():
    """Quick test with one model to verify everything works"""
    print("Running quick FM hyperparameter test...")
    
    # Use the most reliable model
    vae_model_path = "/Users/guanjiezou/CliecyPro/brxngenerator-2/weights/compare_baseline/best_model.pt"
    
    if not Path(vae_model_path).exists():
        print(f"❌ Model not found: {vae_model_path}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        tester = FMHyperparamTester(
            vae_model_path=vae_model_path,
            device=device,
            output_dir="./fm_testing/quick_test"
        )
        
        # Quick test with reduced parameters
        results_df, summary_df = tester.run_hyperparameter_search(
            metrics=['qed'],  # Only QED for speed
            ecc_configs=[('none', 3)],  # Only baseline
            subset_size=500,  # Small dataset
            n_seeds=2  # Fewer seeds
        )
        
        print(f"\n✅ Quick test completed!")
        print(f"Best R²: {summary_df.iloc[0]['test_r2_mean']:.4f}")
        print(f"Best config: k={summary_df.iloc[0]['k_factors']}, lr={summary_df.iloc[0]['lr']}")
        
        return results_df, summary_df
        
    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")
        return None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FM hyperparameter testing")
    parser.add_argument("--quick", action="store_true", help="Run quick test with one model")
    parser.add_argument("--full", action="store_true", help="Run comprehensive test with all models")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_fm_test()
    elif args.full:
        run_comprehensive_fm_testing()
    else:
        print("Usage:")
        print("  python run_fm_hyperparams.py --quick    # Quick test")
        print("  python run_fm_hyperparams.py --full     # Full test")