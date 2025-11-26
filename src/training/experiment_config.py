"""
Experiment configurations for hyperparameter tuning.
Each experiment is a dictionary of hyperparameters to try.
"""

BASELINE = {
    "name": "baseline",
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 4,
    "num_train_epochs": 3,
    "block_size": 256,
    "warmup_steps": 0,
    "weight_decay": 0.0,
    "lr_scheduler_type": "linear",
}

# Experiment 1: Learning Rate Sweep
LR_EXPERIMENTS = [
    {
        "name": "lr_1e-5",
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 4,
        "num_train_epochs": 3,
        "block_size": 256,
    },
    {
        "name": "lr_3e-5",
        "learning_rate": 3e-5,
        "per_device_train_batch_size": 4,
        "num_train_epochs": 3,
        "block_size": 256,
    },
    {
        "name": "lr_5e-5",  # baseline
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 4,
        "num_train_epochs": 3,
        "block_size": 256,
    },
    {
        "name": "lr_1e-4",
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 4,
        "num_train_epochs": 3,
        "block_size": 256,
    },
]

# Experiment 2: Batch Size 
BATCH_SIZE_EXPERIMENTS = [
    {
        "name": "bs_2",
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 2,
        "num_train_epochs": 3,
        "block_size": 256,
    },
    {
        "name": "bs_8",
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 8,
        "num_train_epochs": 3,
        "block_size": 256,
    },
    {
        "name": "bs_16",
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 16,
        "num_train_epochs": 3,
        "block_size": 256,
    },
]

# Experiment 3: Advanced - Warmup + Weight Decay
ADVANCED_EXPERIMENTS = [
    {
        "name": "warmup_100_wd_0.01",
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 4,
        "num_train_epochs": 5,
        "block_size": 256,
        "warmup_steps": 100,
        "weight_decay": 0.01,
    },
    {
        "name": "warmup_200_wd_0.01",
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 4,
        "num_train_epochs": 5,
        "block_size": 256,
        "warmup_steps": 200,
        "weight_decay": 0.01,
    },
    {
        "name": "cosine_schedule",
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 4,
        "num_train_epochs": 5,
        "block_size": 256,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
    },
]

# Experiment 4: Longer Context
CONTEXT_EXPERIMENTS = [
    {
        "name": "context_512",
        "learning_rate": 1e-4,
        "per_device_train_batch_size": 2,
        "num_train_epochs": 3,
        "block_size": 512,
    },
]

# Best practices configuration
BEST_CONFIG = {
    "name": "best_model",
    "learning_rate": 1e-4, 
    "per_device_train_batch_size": 4,
    "num_train_epochs": 10, 
    "block_size": 256,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "gradient_accumulation_steps": 2,
}

