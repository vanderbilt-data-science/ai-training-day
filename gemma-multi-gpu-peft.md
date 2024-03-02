# Multi-GPU Fine-Tuning of Gemma on 8 DGX A100s

This tutorial will guide you through the steps to fine-tune Gemma-7b on a DGX A100 server.

## Prerequisites

- Access to a DGX A100 server
- Basic knowledge of terminal commands

## Step 1: Log Into DGX A100

First, establish an SSH connection to the DGX A100 server:

```bash
ssh username@10.33.2.36
```

Replace `username` with your actual username on the DGX A100 server.

## Step 2: Prepare the Environment

Navigate to your preferred working directory and create a new directory for the LLaMA fine-tuning project:

```bash
cd /raid/username
mkdir gemma-ft
cd gemma-ft
```

Clone the trl repository by HuggingFace:

```bash
git clone https://github.com/huggingface/trl
```

Create a Docker volume for the project:

```bash
docker volume create --driver local --opt type=none --opt device=/raid/username/gemma-ft --opt o=bind gemma-ft
```

Check if the Docker volume was successfully created:

```bash
docker volume ls
```

## Step 3: Run Docker Container

Launch a Docker container with the necessary configurations:

```bash
docker run --gpus all --net=host -it -v gemma-ft:/workspace/gemma-ft nvcr.io/nvidia/pytorch:23.12-py3
```

## Step 4: Install Dependencies

Upgrade `pip` and install required packages:

```bash
python -m pip install --upgrade pip
pip install huggingface-cli
pip install -U transformers trl peft bitsandbytes
```

## Step 5: Login to Hugging Face

Provide your HF Token:

```bash
huggingface-cli login
```

## Step 6: Fine tune Gemma

Navigate to the `trl` directory:

```bash
cd gemma-ft/trl
```

Run the following command with modifications as necessary: 

```
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml --num_processes=8 \
	examples/scripts/sft.py \
	--model_name google/gemma-7b \
	--dataset_name OpenAssistant/oasst_top1_2023-08-25 \
	--per_device_train_batch_size 2 \
	--gradient_accumulation_steps 1 \
	--learning_rate 2e-4 \
	--save_steps 20_000 \
	--use_peft \
	--lora_r 16 --lora_alpha 32 \
	--lora_target_modules q_proj k_proj v_proj o_proj \
	--load_in_4bit \
	--output_dir gemma-finetuned-openassistant
```

The script can be easily parallelized (or serialized) by tweaking the ```num_processses``` flag to the number of GPUs on your machine. 
