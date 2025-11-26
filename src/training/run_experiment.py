"""
Experiment runner for hyperparameter tuning.
Run experiments and automatically log results.
"""

from datasets import Dataset
from transformers import (
    GPT2TokenizerFast,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import torch
import torch.utils.data
import os
import json
from datetime import datetime
import argparse

# Import experiment configurations
from experiment_config import (
    BASELINE,
    LR_EXPERIMENTS,
    BATCH_SIZE_EXPERIMENTS,
    ADVANCED_EXPERIMENTS,
    CONTEXT_EXPERIMENTS,
    BEST_CONFIG
)


def prepare_data(tokenizer, block_size=256):
    """Prepare the Shakespeare dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "tiny_shakespeare.txt")
    
    with open(data_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    tokens = tokenizer(
        text,
        return_attention_mask=False,
        return_tensors="pt",
    )["input_ids"][0]
    
    num_blocks = tokens.size(0) // block_size
    tokens = tokens[: num_blocks * block_size]
    tokens = tokens.view(num_blocks, block_size)
    
    dataset = Dataset.from_dict({"input_ids": tokens})
    
    # 80/20 train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset


def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
    }


def run_experiment(config, experiment_name=None):
    """Run a single training experiment with given config."""
    
    if experiment_name is None:
        experiment_name = config.get("name", "unnamed")
    
    print(f"\n{'='*60}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*60}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"{'='*60}\n")
    
    # Initialize model and tokenizer
    model_name = "distilgpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    block_size = config.get("block_size", 256)
    train_dataset, val_dataset = prepare_data(tokenizer, block_size)
    
    # Output directory
    output_dir = f"experiments/{experiment_name}"
    
    # Training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
        num_train_epochs=config.get("num_train_epochs", 3),
        learning_rate=config.get("learning_rate", 5e-5),
        warmup_steps=config.get("warmup_steps", 0),
        weight_decay=config.get("weight_decay", 0.0),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        logging_steps=20,
        eval_steps=100,
        save_steps=500, 
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to=None,
        logging_dir=f"{output_dir}/logs",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # Train
    train_result = trainer.train()
    
    # Evaluate
    eval_result = trainer.evaluate()
    
    # Save results
    results = {
        "experiment_name": experiment_name,
        "config": config,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save results to JSON
    os.makedirs("experiments/results", exist_ok=True)
    results_file = f"experiments/results/{experiment_name}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiment Complete: {experiment_name}")
    print(f"Train Loss: {train_result.training_loss:.4f}")
    print(f"Eval Loss: {eval_result['eval_loss']:.4f}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}\n")
    
    return results


def run_experiment_suite(suite_name):
    """Run a suite of experiments."""
    
    suites = {
        "lr": LR_EXPERIMENTS,
        "batch_size": BATCH_SIZE_EXPERIMENTS,
        "advanced": ADVANCED_EXPERIMENTS,
        "context": CONTEXT_EXPERIMENTS,
    }
    
    if suite_name not in suites:
        print(f"Unknown suite: {suite_name}")
        print(f"Available suites: {list(suites.keys())}")
        return
    
    experiments = suites[suite_name]
    all_results = []
    
    print(f"\n{'#'*60}")
    print(f"Running Experiment Suite: {suite_name}")
    print(f"Total experiments: {len(experiments)}")
    print(f"{'#'*60}\n")
    
    for i, config in enumerate(experiments, 1):
        print(f"\nExperiment {i}/{len(experiments)}")
        results = run_experiment(config)
        all_results.append(results)
    
    # Save suite summary
    summary_file = f"experiments/results/suite_{suite_name}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'#'*60}")
    print(f"Suite Complete: {suite_name}")
    print(f"{'#'*60}")
    print("\nResults Summary:")
    print(f"{'Experiment':<30} {'Train Loss':<12} {'Eval Loss':<12}")
    print("-" * 60)
    for result in all_results:
        print(f"{result['experiment_name']:<30} "
              f"{result['train_loss']:<12.4f} "
              f"{result['eval_loss']:<12.4f}")
    print(f"\nSummary saved to: {summary_file}")
    print(f"{'#'*60}\n")
    
    # Find best experiment
    best = min(all_results, key=lambda x: x['eval_loss'])
    print(f"🏆 Best Experiment: {best['experiment_name']}")
    print(f"   Eval Loss: {best['eval_loss']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training experiments")
    parser.add_argument(
        "--suite",
        type=str,
        choices=["lr", "batch_size", "advanced", "context", "baseline", "best"],
        help="Experiment suite to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Custom config name from experiment_config.py"
    )
    
    args = parser.parse_args()
    
    if args.suite:
        if args.suite == "baseline":
            run_experiment(BASELINE)
        elif args.suite == "best":
            run_experiment(BEST_CONFIG)
        else:
            run_experiment_suite(args.suite)
    else:
        print("Usage:")
        print("  python run_experiment.py --suite lr              # Run learning rate experiments")
        print("  python run_experiment.py --suite batch_size      # Run batch size experiments")
        print("  python run_experiment.py --suite advanced        # Run advanced experiments")
        print("  python run_experiment.py --suite baseline        # Run baseline")
        print("  python run_experiment.py --suite best            # Run best config")

