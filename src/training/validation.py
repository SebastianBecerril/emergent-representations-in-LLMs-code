from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "distilgpt2-tiny-shakespeare")

if not os.path.exists(model_dir):
    raise FileNotFoundError(
        f"Model directory not found: {model_dir}\n"
        "Please run trainer.py first to train and save the model."
    )

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "ROMEO:\n"
device = model.device
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

output = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    do_sample=True,
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
