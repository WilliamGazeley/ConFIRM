# ConFIRM

This repo contains two things:

1. a pip-installable package that can be used to generate questions
2. scripts that were used to generate the results of the ConFIRM paper

## Package
### Usage

To install this package:

```bash
pip install git+https://github.com/WilliamGazeley/ConFIRM.git
```

For some examples of how this package can be used, it's recommended to take a look at the `tests/`` folder

## Scripts

In order to keep the package light-weight and usable for general cases, we separate the features/formatting that is specific to the ConFIRM paper.

Usage:
```bash
# Make sure the ConFIRM package is installed
pip install -r scripts/requirements.txt
bash scripts/run_question_generation.sh #  Add your LLM provider credentials

# NOTE: download and place the Personage dataset under /datasets/personage-nlg
bash scripts/run_rephrasing.sh #  Update with your filenames

# Finetuning
python wandb_scripts/train/lora_train.py
```

To run your own experiments, just use the package instead.
