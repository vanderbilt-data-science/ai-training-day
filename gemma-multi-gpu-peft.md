# Multi-GPU Fine-Tuning of Gemma on 8 DGX A100s

This tutorial will guide you through the steps to fine-tune and inference Gemma-7b on a DGX A100 server. Note that these instructions are easily adapated to any server with availble GPUs. 

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

Navigate to your preferred working directory and create a new directory for the Gemma fine-tuning project:

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

Congratulations! You have successfully fine-tuned Gemma on a DGX A100 server.

Make sure to replace username with your actual username and adjust paths or configurations as necessary for your specific environment.

## Step 7: Save Docker Container with all installed dependencies

Any packages installed using pip will be lost once the docker container is ended. To avoid having to install dependencies each time, you can save a new image of your docker container with all the dependencies included. You may do this prior to step 6 as well. 

1. Make a second SSH connection into the DGX.

2. From the second connection, check running docker images by running `docker ps`. Note the CONTAINER_ID of the PyTorch container running in the other SSH connection.

5. Run the following docker command to create a new image:
```
docker commit <d4576eda8284/CONTAINER_ID> gemma_trl_ft
```

6. To run the newly created container run:
```
docker run --gpus all --net=host -it -v gemma_ft:/workspace/gemma_ft gemma_trl_ft
```
## Step 8: Inference on your fine tuned model: 

1. Create an inferencing script:

```
touch inference.py
vim inference.py >>>

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "google/gemma-7b" #path/to/your/model/or/name/on/hub"
adapter_model_name = "gemma-finetuned-openassistant"

model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_model_name) #combine peft adapters and base model

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

device = "cpu"

inputs = tokenizer.encode("This movie was really", return_tensors='pt').to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

In the above script, we first load the original model weights, and then combine the peft adapters and the base model. Note that this is different from what we saw in the notebook - using this method, we only save the peft adapters to the outputs folder which must be combined with the base model in order to inference. 

2. Run Inferencing Script:

```
python inference.py
```
