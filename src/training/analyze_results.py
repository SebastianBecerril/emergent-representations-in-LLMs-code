"""
Analyze experiment results and find the best hyperparameters.
"""

import json
import os
from pathlib import Path
import argparse


def load_all_results(results_dir="experiments/results"):
    """Load all experiment results from JSON files."""
    results = []
    
    if not os.path.exists(results_dir):
        print(f"No results directory found: {results_dir}")
        return results
    
    for file in Path(results_dir).glob("*.json"):
        # Skip summary files
        if "summary" in file.name:
            continue
            
        with open(file, "r") as f:
            result = json.load(f)
            results.append(result)
    
    return results


def print_results_table(results, sort_by="eval_loss"):
    """Print results in a nice table format."""
    if not results:
        print("No results to display.")
        return
    
    # Sort results
    results_sorted = sorted(results, key=lambda x: x.get(sort_by, float('inf')))
    
    print(f"\n{'='*80}")
    print(f"Experiment Results (sorted by {sort_by})")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Experiment':<30} {'Train Loss':<12} {'Eval Loss':<12} {'LR':<10}")
    print("-" * 80)
    
    for i, result in enumerate(results_sorted, 1):
        name = result.get('experiment_name', 'Unknown')
        train_loss = result.get('train_loss', float('nan'))
        eval_loss = result.get('eval_loss', float('nan'))
        lr = result.get('config', {}).get('learning_rate', 'N/A')
        
        # Highlight best result
        marker = "🏆" if i == 1 else f"{i}."
        
        print(f"{marker:<6} {name:<30} {train_loss:<12.4f} {eval_loss:<12.4f} {lr:<10}")
    
    print("=" * 80)


def analyze_hyperparameter(results, param_name):
    """Analyze the effect of a specific hyperparameter."""
    param_results = {}
    
    for result in results:
        param_value = result.get('config', {}).get(param_name)
        if param_value is not None:
            eval_loss = result.get('eval_loss')
            if param_value not in param_results:
                param_results[param_value] = []
            param_results[param_value].append(eval_loss)
    
    if not param_results:
        print(f"No results found for parameter: {param_name}")
        return
    
    print(f"\n{'='*60}")
    print(f"Analysis of {param_name}")
    print(f"{'='*60}")
    print(f"{param_name:<30} {'Avg Eval Loss':<15} {'# Runs':<10}")
    print("-" * 60)
    
    for value in sorted(param_results.keys()):
        losses = param_results[value]
        avg_loss = sum(losses) / len(losses)
        print(f"{str(value):<30} {avg_loss:<15.4f} {len(losses):<10}")
    
    print("=" * 60)
    
    # Find best value
    best_value = min(param_results.keys(), 
                    key=lambda x: sum(param_results[x]) / len(param_results[x]))
    best_loss = sum(param_results[best_value]) / len(param_results[best_value])
    print(f"\n✨ Best {param_name}: {best_value} (Avg Eval Loss: {best_loss:.4f})")


def get_best_config(results):
    """Extract the best configuration."""
    if not results:
        return None
    
    best = min(results, key=lambda x: x.get('eval_loss', float('inf')))
    return best


def compare_to_baseline(results, baseline_name="baseline"):
    """Compare all experiments to baseline."""
    baseline = None
    for result in results:
        if result.get('experiment_name') == baseline_name:
            baseline = result
            break
    
    if not baseline:
        print(f"Baseline experiment '{baseline_name}' not found.")
        return
    
    baseline_loss = baseline.get('eval_loss')
    
    print(f"\n{'='*80}")
    print(f"Comparison to Baseline (eval_loss: {baseline_loss:.4f})")
    print(f"{'='*80}")
    print(f"{'Experiment':<35} {'Eval Loss':<12} {'Improvement':<15}")
    print("-" * 80)
    
    improvements = []
    for result in results:
        if result.get('experiment_name') == baseline_name:
            continue
        
        name = result.get('experiment_name', 'Unknown')
        eval_loss = result.get('eval_loss', float('nan'))
        improvement = ((baseline_loss - eval_loss) / baseline_loss) * 100
        
        improvements.append((name, eval_loss, improvement))
    
    # Sort by improvement
    improvements.sort(key=lambda x: x[2], reverse=True)
    
    for name, eval_loss, improvement in improvements:
        sign = "↓" if improvement > 0 else "↑"
        print(f"{name:<35} {eval_loss:<12.4f} {sign} {abs(improvement):<.2f}%")
    
    print("=" * 80)


def print_recommendations(results):
    """Print recommendations based on results."""
    if not results:
        return
    
    best = get_best_config(results)
    
    print(f"\n{'='*80}")
    print("🎯 RECOMMENDATIONS")
    print(f"{'='*80}")
    print(f"\nBest Experiment: {best['experiment_name']}")
    print(f"Eval Loss: {best['eval_loss']:.4f}")
    print(f"\nBest Configuration:")
    
    config = best['config']
    for key, value in config.items():
        if key != 'name':
            print(f"  {key}: {value}")
    
    print(f"\n{'='*80}")
    print("Next Steps:")
    print("  1. Update BEST_CONFIG in experiment_config.py with these values")
    print("  2. Run longer training (e.g., 10 epochs) with best config")
    print("  3. Monitor for overfitting by watching eval_loss")
    print("  4. Try fine-tuning around the best learning rate")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--param",
        type=str,
        help="Analyze specific hyperparameter (e.g., learning_rate, per_device_train_batch_size)"
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare all experiments to baseline"
    )
    
    args = parser.parse_args()
    
    # Load all results
    results = load_all_results()
    
    if not results:
        print("No experiment results found. Run some experiments first!")
        print("Example: python run_experiment.py --suite lr")
        exit(0)
    
    # Print results table
    print_results_table(results)
    
    # Compare to baseline if requested
    if args.compare_baseline:
        compare_to_baseline(results)
    
    # Analyze specific parameter if requested
    if args.param:
        analyze_hyperparameter(results, args.param)
    
    # Print recommendations
    print_recommendations(results)

