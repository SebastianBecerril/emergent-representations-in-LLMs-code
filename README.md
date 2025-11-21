# Emergent Representations in LLMs

## Project Overview

The goal of this project is to investigate world or environment representations that develop in large language models during training. Motivated by recent LLM probing literature, we intend to focus on domains of language tasks that relate program traces to changes in some external state, such as chess moves and corresponding changes in a chess board's state, or actions in a planning language and corresponding changes in a task planning environment. This process will allow us to research what are the inner operations and processes that language models possess when dealing with tasks that heavily rely on underlying state/world understanding.

In order to probe a trained LLM, we will start with a small base model and develop a training and probing pipeline using an initial toy dataset.

## First Experiment

Training a DistilGPT-2 model (from Hugging Face) on the Little Shakespeare dataset to get acquainted with PyTorch and model training workflows.


## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

