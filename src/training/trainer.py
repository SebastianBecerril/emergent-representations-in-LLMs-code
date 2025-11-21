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

model_name = "distilgpt2"

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.to("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, "tiny_shakespeare.txt")

with open(data_file, "r", encoding="utf-8") as f:
    text = f.read()

block_size = 256

tokens = tokenizer(
    text,
    return_attention_mask=False,
    return_tensors="pt",
)["input_ids"][0]

num_blocks = tokens.size(0) // block_size
tokens = tokens[: num_blocks * block_size]
tokens = tokens.view(num_blocks, block_size)

dataset = Dataset.from_dict({"input_ids": tokens})

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
    }

# Split dataset into train/validation (80/20 split)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)
)

# Training configuration
args = TrainingArguments(
    output_dir="out-distilgpt2",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=20,           
    eval_steps=100,             
    save_steps=200,
    eval_strategy="steps",      
    load_best_model_at_end=True, 
    metric_for_best_model="eval_loss",
    greater_is_better=False,    
    fp16=True,                  
    report_to=None,             
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

trainer.train()

trainer.save_model("distilgpt2-tiny-shakespeare")
tokenizer.save_pretrained("distilgpt2-tiny-shakespeare")
