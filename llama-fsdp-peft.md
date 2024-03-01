```markdown
# Fine-Tuning and Inference with LLaMA Model on DGX A100

This tutorial will guide you through the steps to fine-tune and perform inference with the LLaMA model on a DGX A100 server.

## Prerequisites

- Access to a DGX A100 server
- Basic knowledge of terminal commands

## Step 1: Log Into DGX A100

First, establish an SSH connection to the DGX A100 server:

```bash
ssh username@10.33.2.36


Replace `username` with your actual username on the DGX A100 server.

## Step 2: Prepare the Environment

Navigate to your preferred working directory and create a new directory for the LLaMA fine-tuning project:

```bash
cd /raid/username
mkdir llama-ft
cd llama-ft
```

Clone the LLaMA recipes repository:

```bash
git clone https://github.com/facebookresearch/llama-recipes.git
```

Create a Docker volume for the project:

```bash
docker volume create --driver local --opt type=none --opt device=/raid/username/llama-ft --opt o=bind llama-ft
```

Check if the Docker volume was successfully created:

```bash
docker volume ls
```

## Step 3: Run Docker Container

Launch a Docker container with the necessary configurations:

```bash
docker run --gpus all --net=host -it -v llama-ft:/workspace/llama-ft nvcr.io/nvidia/pytorch:23.12-py3
```

## Step 4: Install Dependencies

Upgrade `pip` and install required packages:

```bash
python -m pip install --upgrade pip
pip install huggingface-cli
```

Navigate to the `llama-recipes` directory:

```bash
cd llama-ft/llama-recipes
```

Update `pip` and `setuptools`, and install the project dependencies:

```bash
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .
```

## Step 5: Login to Hugging Face

Authenticate with Hugging Face CLI:

```bash
huggingface-cli login
```

## Step 6: Fine-Tune the Model

Run the fine-tuning script with the desired configuration:

```bash
torchrun --nnodes 1 --nproc_per_node 8 examples/finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name meta-llama/Llama-2-7b-hf --fsdp_config.pure_bf16 --output_dir results
```

## Step 7: Prepare for Inference

Create a Python script for inference:

```bash
touch inference.py
```

Edit `inference.py` with your preferred text editor and include the following code:

```python
device_map = {"": 0}
base_model_name = "meta-llama/Llama-2-7b-hf"
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoPeftModelForCausalLM.from_pretrained('results', device_map=device_map, torch_dtype=torch.bfloat16)
text = "How much wood can a woodchuck chuck?"
inputs = tokenizer(text, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), attention_mask=inputs["attention_mask"], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Install the TRL package:

```bash
pip install trl
```

## Step 8: Run Inference

Execute the inference script:

```bash
python inference.py
```

You should now see the model's generated text based on your input.

Congratulations! You have successfully fine-tuned and performed inference with a LLaMA model on a DGX A100 server.
```

Make sure to replace `username` with your actual username and adjust paths or configurations as necessary for your specific environment.
```
