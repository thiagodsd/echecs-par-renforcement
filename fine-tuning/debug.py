# This command is used to install and upgrade several Python packages using pip, Python's package installer.
# The '!' at the beginning allows us to run shell commands in the notebook.
# The '-qqq' option is used to make the installation process less verbose.
# '--upgrade' is used to ensure that the packages are upgraded to their latest versions if they are already installed.
# The packages being installed are:
# 'bitsandbytes' for efficient gradient accumulation,
# 'transformers' for using transformer models like Phi-3,
# 'peft' for efficient fine-tuning,
# 'accelerate' for easy distributed training,
# 'datasets' for loading and preprocessing datasets,
# 'trl' for reinforcement learning,
# 'flash_attn' for attention-based models.
# 'wandb' stands for Weights & Biases. It is a tool for machine learning experiment tracking, dataset versioning, and model management. It allows you to log and visualize metrics from your code, share findings, and reproduce experiments.
# 'torch' is a package that provides an open-source machine learning library used for building deep learning models.
!pip install -qqq --upgrade bitsandbytes transformers peft accelerate datasets trl flash_attn torch wandb
     

# These commands are used to install two Python packages using pip, Python's package installer.
# The '!' at the beginning allows us to run shell commands in the notebook.

# 'huggingface_hub' is a library developed by Hugging Face that allows you to interact with the Hugging Face Model Hub.
# It provides functionalities to download and upload models, as well as other utilities.
!pip install huggingface_hub

# 'python-dotenv' is a library that allows you to specify environment variables in a .env file.
# It's useful for managing secrets and configuration settings for your application.
!pip install python-dotenv
     

# This command is used to install three Python packages using pip, Python's package installer.
# The '!' at the beginning allows us to run shell commands in the notebook.

# 'absl-py' is a library developed by Google that provides several utilities for Python development, such as logging and command line argument parsing.

# 'nltk' stands for Natural Language Toolkit. It is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources.

# 'rouge_score' is a library for calculating the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score, which is commonly used for evaluating automatic summarization and machine translation systems.
!pip install absl-py nltk rouge_score
     

# This command is used to list all installed Python packages and filter for the 'transformers' package.
# The '!' at the beginning allows us to run shell commands in the notebook.

# 'pip list' lists all installed Python packages.

# The '|' character is a pipe. It takes the output from the command on its left (in this case, 'pip list') and passes it as input to the command on its right.

# 'grep' is a command-line utility for searching plain-text data for lines that match a regular expression. Here it's used to filter the output of 'pip list' for lines that contain 'transformers.'.

# So, this command will list details of the 'transformers' package if it's installed.
!pip list | grep transformers.
     
Importing the libraries

# This code block is importing necessary modules and functions for fine-tuning a language model.

# 'randrange' is a function from the 'random' module that generates a random number within the specified range.
from random import randrange

# 'torch' is the PyTorch library, a popular open-source machine learning library for Python.
import torch

# 'load_dataset' is a function from the 'datasets' library by Hugging Face which allows you to load a dataset.
from datasets import load_dataset

# 'LoraConfig' and 'prepare_model_for_kbit_training' are from the 'peft' library. 
# 'LoraConfig' is used to configure the LoRA (Learning from Random Architecture) model.
# 'prepare_model_for_kbit_training' is a function that prepares a model for k-bit training.
# 'TaskType' contains differenct types of tasks supported by PEFT
# 'PeftModel' base model class for specifying the base Transformer model and configuration to apply a PEFT method to.
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType, PeftModel

# Several classes and functions are imported from the 'transformers' library by Hugging Face.
# 'AutoModelForCausalLM' is a class that provides a generic transformer model for causal language modeling.
# 'AutoTokenizer' is a class that provides a generic tokenizer class.
# 'BitsAndBytesConfig' is a class for configuring the Bits and Bytes optimizer.
# 'TrainingArguments' is a class that defines the arguments used for training a model.
# 'set_seed' is a function that sets the seed for generating random numbers.
# 'pipeline' is a function that creates a pipeline that can process data and make predictions.
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    pipeline
)

# 'SFTTrainer' is a class from the 'trl' library that provides a trainer for soft fine-tuning.
from trl import SFTTrainer
     
Setting Global Parameters

# This code block is setting up the configuration for fine-tuning a language model.

# 'model_id' and 'model_name' are the identifiers for the pre-trained model that you want to fine-tune. 
# In this case, it's the 'Phi-3-mini-4k-instruct' model from Microsoft.
# Model Names 
# microsoft/Phi-3-mini-4k-instruct
# microsoft/Phi-3-mini-128k-instruct
# microsoft/Phi-3-small-8k-instruct
# microsoft/Phi-3-small-128k-instruct
# microsoft/Phi-3-medium-4k-instruct
# microsoft/Phi-3-medium-128k-instruct
# microsoft/Phi-3-vision-128k-instruct
# microsoft/Phi-3-mini-4k-instruct-onnx
# microsoft/Phi-3-mini-4k-instruct-onnx-web
# microsoft/Phi-3-mini-128k-instruct-onnx
# microsoft/Phi-3-small-8k-instruct-onnx-cuda
# microsoft/Phi-3-small-128k-instruct-onnx-cuda
# microsoft/Phi-3-medium-4k-instruct-onnx-cpu
# microsoft/Phi-3-medium-4k-instruct-onnx-cuda
# microsoft/Phi-3-medium-4k-instruct-onnx-directml
# microsoft/Phi-3-medium-128k-instruct-onnx-cpu
# microsoft/Phi-3-medium-128k-instruct-onnx-cuda
# microsoft/Phi-3-medium-128k-instruct-onnx-directml
# microsoft/Phi-3-mini-4k-instruct-gguf

model_id = "microsoft/Phi-3-mini-4k-instruct"
model_name = "microsoft/Phi-3-mini-4k-instruct"

# 'dataset_name' is the identifier for the dataset that you want to use for fine-tuning. 
# In this case, it's the 'python_code_instructions_18k_alpaca' dataset from iamtarun (Ex: iamtarun/python_code_instructions_18k_alpaca).
# Update Dataset Name to your dataset name
dataset_name = "Insert your dataset name here"

# 'dataset_split' is the split of the dataset that you want to use for training. 
# In this case, it's the 'train' split.
dataset_split= "train"

# 'new_model' is the name that you want to give to the fine-tuned model.
new_model = "Name of your new model"

# 'hf_model_repo' is the repository on the Hugging Face Model Hub where the fine-tuned model will be saved. Update UserName to your Hugging Face Username
hf_model_repo="UserName/"+new_model

# 'device_map' is a dictionary that maps the model to the GPU device. 
# In this case, the entire model is loaded on GPU 0.
device_map = {"": 0}

# The following are parameters for the LoRA (Learning from Random Architecture) model.

# 'lora_r' is the dimension of the LoRA attention.
lora_r = 16

# 'lora_alpha' is the alpha parameter for LoRA scaling.
lora_alpha = 16

# 'lora_dropout' is the dropout probability for LoRA layers.
lora_dropout = 0.05

# 'target_modules' is a list of the modules in the model that will be replaced with LoRA layers.
target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]

# 'set_seed' is a function that sets the seed for generating random numbers, 
# which is used for reproducibility of the results.
set_seed(1234)

     
Connect to Huggingface Hub
IMPORTANT: The upcoming section's execution will vary based on your code execution environment and the configuration of your API Keys.

Interactive login to Hugging Face Hub is possible.


# This code block is used to log in to the Hugging Face Model Hub from a notebook.

# 'notebook_login' is a function from the 'huggingface_hub' library that opens a new browser window 
# where you can log in to your Hugging Face account. After logging in, 
# your Hugging Face token will be stored in a configuration file on your machine, 
# which allows you to interact with the Hugging Face Model Hub from your notebook.
from huggingface_hub import notebook_login

# Call the 'notebook_login' function to start the login process.
notebook_login()
     
Alternatively, you can supply a .env file that contains the Hugging Face token.


# This code block is used to log in to the Hugging Face Model Hub using an API token stored in an environment variable.

# 'login' is a function from the 'huggingface_hub' library that logs you in to the Hugging Face Model Hub using an API token.
from huggingface_hub import login

# 'load_dotenv' is a function from the 'python-dotenv' library that loads environment variables from a .env file.
from dotenv import load_dotenv

# 'os' is a standard Python library that provides functions for interacting with the operating system.
import os

# Call the 'load_dotenv' function to load the environment variables from the .env file.
load_dotenv()

# Call the 'login' function with the 'HF_HUB_TOKEN' environment variable to log in to the Hugging Face Model Hub.
# 'os.getenv' is a function that gets the value of an environment variable.
login(token=os.getenv("HF_HUB_TOKEN"))
     
Load the dataset with the instruction set

# This code block is used to load a dataset from the Hugging Face Dataset Hub, print its size, and show a random example from the dataset.

# 'load_dataset' is a function from the 'datasets' library that loads a dataset from the Hugging Face Dataset Hub.
# 'dataset_name' is the name of the dataset to load, and 'dataset_split' is the split of the dataset to load (e.g., 'train', 'test').
dataset = load_dataset(dataset_name, split=dataset_split)

# The 'len' function is used to get the size of the dataset, which is then printed.
print(f"dataset size: {len(dataset)}")

# 'randrange' is a function from the 'random' module that generates a random number within the specified range.
# Here it's used to select a random example from the dataset, which is then printed.
print(dataset[randrange(len(dataset))])
     

# This line of code is used to display the structure of the 'dataset' object.
# By simply writing the name of the object, Python will call its 'repr' (representation) method, 
# which returns a string that describes the object. 
# For a Hugging Face 'Dataset' object, this will typically show information such as the number of rows, 
# the column names, and the types of the data in each column.
dataset
     

# This line of code is used to print a random example from the 'dataset'.

# 'randrange' is a function from the 'random' module that generates a random number within the specified range.
# Here it's used to generate a random index within the range of the dataset size (i.e., 'len(dataset)').

# This random index is then used to select a corresponding example from the 'dataset'. 
# The selected example is printed to the console.
print(dataset[randrange(len(dataset))])
     
Load the tokenizer to prepare the dataset

# This code block is used to load a tokenizer from the Hugging Face Model Hub.

# 'tokenizer_id' is set to the 'model_id', which is the identifier for the pre-trained model.
# This assumes that the tokenizer associated with the model has the same identifier as the model.
tokenizer_id = model_id

# 'AutoTokenizer.from_pretrained' is a method that loads a tokenizer from the Hugging Face Model Hub.
# 'tokenizer_id' is passed as an argument to specify which tokenizer to load.
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

# 'tokenizer.padding_side' is a property that specifies which side to pad when the input sequence is shorter than the maximum sequence length.
# Setting it to 'right' means that padding tokens will be added to the right (end) of the sequence.
# This is done to prevent warnings that can occur when the padding side is not explicitly set.
tokenizer.padding_side = 'right'
     
Function to create the appropiate format for our model. We are going to adapt our dataset to the ChatML format.


# This code block defines two functions that are used to format the dataset for training a chat model.

# 'create_message_column' is a function that takes a row from the dataset and returns a dictionary 
# with a 'messages' key and a list of 'user' and 'assistant' messages as its value.
def create_message_column(row):
    # Initialize an empty list to store the messages.
    messages = []
    
    # Create a 'user' message dictionary with 'content' and 'role' keys.
    user = {
        "content": f"{row['instruction']}\n Input: {row['input']}",
        "role": "user"
    }
    
    # Append the 'user' message to the 'messages' list.
    messages.append(user)
    
    # Create an 'assistant' message dictionary with 'content' and 'role' keys.
    assistant = {
        "content": f"{row['output']}",
        "role": "assistant"
    }
    
    # Append the 'assistant' message to the 'messages' list.
    messages.append(assistant)
    
    # Return a dictionary with a 'messages' key and the 'messages' list as its value.
    return {"messages": messages}

# 'format_dataset_chatml' is a function that takes a row from the dataset and returns a dictionary 
# with a 'text' key and a string of formatted chat messages as its value.
def format_dataset_chatml(row):
    # 'tokenizer.apply_chat_template' is a method that formats a list of chat messages into a single string.
    # 'add_generation_prompt' is set to False to not add a generation prompt at the end of the string.
    # 'tokenize' is set to False to return a string instead of a list of tokens.
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}
     
Apply the ChatML format to our dataset

The code block is used to prepare a dataset for training a chat model.

The dataset.map(create_message_column) line applies the create_message_column function to each example in the dataset. This function takes a row from the dataset and transforms it into a dictionary with a 'messages' key. The value of this key is a list of 'user' and 'assistant' messages.

The 'user' message is created by combining the 'instruction' and 'input' fields from the row, while the 'assistant' message is created from the 'output' field of the row. These messages are appended to the 'messages' list in the order of 'user' and 'assistant'.

The dataset_chatml.map(format_dataset_chatml) line then applies the format_dataset_chatml function to each example in the updated dataset. This function takes a row from the dataset and transforms it into a dictionary with a 'text' key. The value of this key is a string of formatted chat messages.

The tokenizer.apply_chat_template method is used to format the list of chat messages into a single string. The 'add_generation_prompt' parameter is set to False to avoid adding a generation prompt at the end of the string, and the 'tokenize' parameter is set to False to return a string instead of a list of tokens.

The result of these operations is a dataset where each example is a dictionary with a 'text' key and a string of formatted chat messages as its value. This format is suitable for training a chat model.


# This code block is used to prepare the 'dataset' for training a chat model.

# 'dataset.map' is a method that applies a function to each example in the 'dataset'.
# 'create_message_column' is a function that formats each example into a 'messages' format suitable for a chat model.
# The result is a new 'dataset_chatml' with the formatted examples.
dataset_chatml = dataset.map(create_message_column)

# 'dataset_chatml.map' is a method that applies a function to each example in the 'dataset_chatml'.
# 'format_dataset_chatml' is a function that further formats each example into a single string of chat messages.
# The result is an updated 'dataset_chatml' with the further formatted examples.
dataset_chatml = dataset_chatml.map(format_dataset_chatml)
     

# This line of code is used to access and display the first example from the 'dataset_chatml'.

# 'dataset_chatml[0]' uses indexing to access the first example in the 'dataset_chatml'.
# In Python, indexing starts at 0, so 'dataset_chatml[0]' refers to the first example.
# The result is a dictionary with a 'text' key and a string of formatted chat messages as its value.
dataset_chatml[0]
     
Split the dataset into a train and test sets


# This code block is used to split the 'dataset_chatml' into training and testing sets.

# 'dataset_chatml.train_test_split' is a method that splits the 'dataset_chatml' into a training set and a testing set.
# 'test_size' is a parameter that specifies the proportion of the 'dataset_chatml' to include in the testing set. Here it's set to 0.05, meaning that 5% of the 'dataset_chatml' will be included in the testing set.
# 'seed' is a parameter that sets the seed for the random number generator. This is used to ensure that the split is reproducible. Here it's set to 1234.
dataset_chatml = dataset_chatml.train_test_split(test_size=0.05, seed=1234)

# This line of code is used to display the structure of the 'dataset_chatml' after the split.
# It will typically show information such as the number of rows in the training set and the testing set.
dataset_chatml
     
Instruction fine-tune a Phi-3-mini model using LORA and trl
First, we try to identify out GPU


# This code block is used to set the compute data type and attention implementation based on whether bfloat16 is supported on the current CUDA device.

# 'torch.cuda.is_bf16_supported()' is a function that checks if bfloat16 is supported on the current CUDA device.
# If bfloat16 is supported, 'compute_dtype' is set to 'torch.bfloat16' and 'attn_implementation' is set to 'flash_attention_2'.
if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
  attn_implementation = 'flash_attention_2'
# If bfloat16 is not supported, 'compute_dtype' is set to 'torch.float16' and 'attn_implementation' is set to 'sdpa'.
else:
  compute_dtype = torch.float16
  attn_implementation = 'sdpa'

# This line of code is used to print the value of 'attn_implementation', which indicates the chosen attention implementation.
print(attn_implementation)
     
Load the tokenizer and model to finetune

# This code block is used to load a pre-trained model and its associated tokenizer from the Hugging Face Model Hub.

# 'model_name' is set to the identifier of the pre-trained model.
model_name = "microsoft/Phi-3-mini-4k-instruct"

# 'AutoTokenizer.from_pretrained' is a method that loads a tokenizer from the Hugging Face Model Hub.
# 'model_id' is passed as an argument to specify which tokenizer to load.
# 'trust_remote_code' is set to True to trust the remote code in the tokenizer files.
# 'add_eos_token' is set to True to add an end-of-sentence token to the tokenizer.
# 'use_fast' is set to True to use the fast version of the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, add_eos_token=True, use_fast=True)

# The padding token is set to the unknown token.
tokenizer.pad_token = tokenizer.unk_token

# The ID of the padding token is set to the ID of the unknown token.
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

# The padding side is set to 'left', meaning that padding tokens will be added to the left (start) of the sequence.
tokenizer.padding_side = 'left'

# 'AutoModelForCausalLM.from_pretrained' is a method that loads a pre-trained model for causal language modeling from the Hugging Face Model Hub.
# 'model_id' is passed as an argument to specify which model to load.
# 'torch_dtype' is set to the compute data type determined earlier.
# 'trust_remote_code' is set to True to trust the remote code in the model files.
# 'device_map' is passed as an argument to specify the device mapping for distributed training.
# 'attn_implementation' is set to the attention implementation determined earlier.
model = AutoModelForCausalLM.from_pretrained(
          model_id, torch_dtype=compute_dtype, trust_remote_code=True, device_map=device_map,
          attn_implementation=attn_implementation
)
     
Configure the LoRA properties

The SFTTrainer offers seamless integration with peft, simplifying the process of instruction tuning LLMs. All we need to do is create our LoRAConfig and supply it to the trainer. However, before initiating the training process, we must specify the hyperparameters we intend to use, which are defined in TrainingArguments.


# This code block is used to define the training arguments for the model.

# 'TrainingArguments' is a class that holds the arguments for training a model.
# 'output_dir' is the directory where the model and its checkpoints will be saved.
# 'evaluation_strategy' is set to "steps", meaning that evaluation will be performed after a certain number of training steps.
# 'do_eval' is set to True, meaning that evaluation will be performed.
# 'optim' is set to "adamw_torch", meaning that the AdamW optimizer from PyTorch will be used.
# 'per_device_train_batch_size' and 'per_device_eval_batch_size' are set to 8, meaning that the batch size for training and evaluation will be 8 per device.
# 'gradient_accumulation_steps' is set to 4, meaning that gradients will be accumulated over 4 steps before performing a backward/update pass.
# 'log_level' is set to "debug", meaning that all log messages will be printed.
# 'save_strategy' is set to "epoch", meaning that the model will be saved after each epoch.
# 'logging_steps' is set to 100, meaning that log messages will be printed every 100 steps.
# 'learning_rate' is set to 1e-4, which is the learning rate for the optimizer.
# 'fp16' is set to the opposite of whether bfloat16 is supported on the current CUDA device.
# 'bf16' is set to whether bfloat16 is supported on the current CUDA device.
# 'eval_steps' is set to 100, meaning that evaluation will be performed every 100 steps.
# 'num_train_epochs' is set to 3, meaning that the model will be trained for 3 epochs.
# 'warmup_ratio' is set to 0.1, meaning that 10% of the total training steps will be used for the warmup phase.
# 'lr_scheduler_type' is set to "linear", meaning that a linear learning rate scheduler will be used.
# 'report_to' is set to "wandb", meaning that training and evaluation metrics will be reported to Weights & Biases.
# 'seed' is set to 42, which is the seed for the random number generator.

# LoraConfig object is created with the following parameters:
# 'r' (rank of the low-rank approximation) is set to 16,
# 'lora_alpha' (scaling factor) is set to 16,
# 'lora_dropout' dropout probability for Lora layers is set to 0.05,
# 'task_type' (set to TaskType.CAUSAL_LM indicating the task type),
# 'target_modules' (the modules to which LoRA is applied) choosing linear layers except the output layer..


args = TrainingArguments(
        output_dir="./phi-3-mini-LoRA",
        evaluation_strategy="steps",
        do_eval=True,
        optim="adamw_torch",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        log_level="debug",
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        eval_steps=100,
        num_train_epochs=3,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        report_to="wandb",
        seed=42,
)

peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
)
     
Establish Connection with wandb and Initiate the Project and Experiment

# This code block is used to initialize Weights & Biases (wandb), a tool for tracking and visualizing machine learning experiments.

# 'import wandb' is used to import the wandb library.
import wandb

# 'wandb.login()' is a method that logs you into your Weights & Biases account.
# If you're not already logged in, it will prompt you to log in.
# Once you're logged in, you can use Weights & Biases to track and visualize your experiments.
wandb.login()
     

# This code block is used to initialize a Weights & Biases (wandb) run.

# 'project_name' is set to the name of the project in Weights & Biases.
project_name = "Phi3-mini-ft-python-code"

# 'wandb.init' is a method that initializes a new Weights & Biases run.
# 'project' is set to 'project_name', meaning that the run will be associated with this project.
# 'name' is set to "phi-3-mini-ft-py-3e", which is the name of the run.
# Each run has a unique name which can be used to identify it in the Weights & Biases dashboard.
wandb.init(project=project_name, name = "phi-3-mini-ft-py-3e")
     
We now possess all the necessary components to construct our SFTTrainer and commence the training of our model.


# This code block is used to initialize the SFTTrainer, which is used to train the model.

# 'model' is the model that will be trained.
# 'train_dataset' and 'eval_dataset' are the datasets that will be used for training and evaluation, respectively.
# 'peft_config' is the configuration for peft, which is used for instruction tuning.
# 'dataset_text_field' is set to "text", meaning that the 'text' field of the dataset will be used as the input for the model.
# 'max_seq_length' is set to 512, meaning that the maximum length of the sequences that will be fed to the model is 512 tokens.
# 'tokenizer' is the tokenizer that will be used to tokenize the input text.
# 'args' are the training arguments that were defined earlier.

trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_chatml['train'],
        eval_dataset=dataset_chatml['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=args,
)
     
Initiate the model training process by invoking the train() method on our Trainer instance.


# This code block is used to train the model and save it locally.

# 'trainer.train()' is a method that starts the training of the model.
# It uses the training dataset, evaluation dataset, and training arguments that were provided when the trainer was initialized.
trainer.train()

# 'trainer.save_model()' is a method that saves the trained model locally.
# The model will be saved in the directory specified by 'output_dir' in the training arguments.
trainer.save_model()
     
Store the adapter on the Hugging Face Hu


# This code block is used to save the adapter to the Hugging Face Model Hub.

# 'trainer.push_to_hub' is a method that pushes the trained model (or adapter in this case) to the Hugging Face Model Hub.
# The argument "edumunozsala/adapter-phi-3-mini-py_code" is the name of the repository on the Hugging Face Model Hub where the adapter will be saved.
trainer.push_to_hub("HuggingFaceUser/adapter-name")
     
Merge the model and the adapter and save it
Combine the model and the adapter, then save it. It's necessary to clear the memory when operating on a T4 instance.


# This code block is used to free up GPU memory.

# 'del model' and 'del trainer' are used to delete the 'model' and 'trainer' objects. 
# This removes the references to these objects, allowing Python's garbage collector to free up the memory they were using.

del model
del trainer

# 'import gc' is used to import Python's garbage collector module.
import gc

# 'gc.collect()' is a method that triggers a full garbage collection, which can help to free up memory.
# It's called twice here to ensure that all unreachable objects are collected.
gc.collect()
gc.collect()
     

# 'torch.cuda.empty_cache()' is a PyTorch method that releases all unoccupied cached memory currently held by 
# the caching allocator so that those can be used in other GPU application and visible in nvidia-smi.
torch.cuda.empty_cache()
     

# 'gc.collect()' is a method that triggers a full garbage collection in Python.
# It forces the garbage collector to release unreferenced memory, which can be helpful in managing memory usage, especially in a resource-constrained environment.
gc.collect()
     
Load the previously trained and stored model, combine it, and then save the complete model.


# This code block is used to load the trained model, merge it, and save the merged model.

# 'AutoPeftModelForCausalLM' is a class from the 'peft' library that provides a causal language model with PEFT (Performance Efficient Fine-Tuning) support.

from peft import AutoPeftModelForCausalLM

# 'AutoPeftModelForCausalLM.from_pretrained' is a method that loads a pre-trained model (adapter model) and its base model.
#  The adapter model is loaded from 'args.output_dir', which is the directory where the trained model was saved.
# 'low_cpu_mem_usage' is set to True, which means that the model will use less CPU memory.
# 'return_dict' is set to True, which means that the model will return a 'ModelOutput' (a named tuple) instead of a plain tuple.
# 'torch_dtype' is set to 'torch.bfloat16', which means that the model will use bfloat16 precision for its computations.
# 'trust_remote_code' is set to True, which means that the model will trust and execute remote code.
# 'device_map' is the device map that will be used by the model.

new_model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16, #torch.float16,
    trust_remote_code=True,
    device_map=device_map,
)

# 'new_model.merge_and_unload' is a method that merges the model and unloads it from memory.
# The merged model is stored in 'merged_model'.

merged_model = new_model.merge_and_unload()

# 'merged_model.save_pretrained' is a method that saves the merged model.
# The model is saved in the directory "merged_model".
# 'trust_remote_code' is set to True, which means that the model will trust and execute remote code.
# 'safe_serialization' is set to True, which means that the model will use safe serialization.

merged_model.save_pretrained("merged_model", trust_remote_code=True, safe_serialization=True)

# 'tokenizer.save_pretrained' is a method that saves the tokenizer.
# The tokenizer is saved in the directory "merged_model".

tokenizer.save_pretrained("merged_model")
     

# This code block is used to push the merged model and the tokenizer to the Hugging Face Model Hub.

# 'merged_model.push_to_hub' is a method that pushes the merged model to the Hugging Face Model Hub.
# 'hf_model_repo' is the name of the repository on the Hugging Face Model Hub where the model will be saved.
merged_model.push_to_hub(hf_model_repo)

# 'tokenizer.push_to_hub' is a method that pushes the tokenizer to the Hugging Face Model Hub.
# 'hf_model_repo' is the name of the repository on the Hugging Face Model Hub where the tokenizer will be saved.
tokenizer.push_to_hub(hf_model_repo)
     
Model Inference and evaluation
For model inference and evaluation, we will download the model we created from the Hugging Face Hub and test it to ensure its functionality.


# 'hf_model_repo' is a variable that holds the name of the repository on the Hugging Face Model Hub.
# This is where the trained and merged model, as well as the tokenizer, have been saved.
hf_model_repo
     

# 'hf_model_repo' is a variable that holds the name of the repository on the Hugging Face Model Hub.
# This is where the trained and merged model, as well as the tokenizer, have been saved.
# If 'hf_model_repo' is not defined, it is set to 'username/modelname'.
# This is the default repository where the model and tokenizer will be saved if no other repository is specified.
hf_model_repo = 'username/modelname' if not hf_model_repo else hf_model_repo
     
Retrieve the model and tokenizer from the Hugging Face Hub.


# This code block is used to load the model and tokenizer from the Hugging Face Model Hub.

# 'torch' is a library that provides a wide range of functionalities for tensor computations with strong GPU acceleration support.
# 'AutoTokenizer' and 'AutoModelForCausalLM' are classes from the 'transformers' library that provide a tokenizer and a causal language model, respectively.
# 'set_seed' is a function from the 'transformers' library that sets the seed for generating random numbers, which can be used for reproducibility.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# 'set_seed(1234)' sets the seed for generating random numbers to 1234.
set_seed(1234)  # For reproducibility

# 'AutoTokenizer.from_pretrained' is a method that loads a pre-trained tokenizer.
# The tokenizer is loaded from 'hf_model_repo', which is the name of the repository on the Hugging Face Model Hub where the tokenizer was saved.
# 'trust_remote_code' is set to True, which means that the tokenizer will trust and execute remote code.

tokenizer = AutoTokenizer.from_pretrained(hf_model_repo,trust_remote_code=True)

# 'AutoModelForCausalLM.from_pretrained' is a method that loads a pre-trained causal language model.
# The model is loaded from 'hf_model_repo', which is the name of the repository on the Hugging Face Model Hub where the model was saved.
# 'trust_remote_code' is set to True, which means that the model will trust and execute remote code.
# 'torch_dtype' is set to "auto", which means that the model will automatically choose the data type for its computations.
# 'device_map' is set to "cuda", which means that the model will use the CUDA device for its computations.

model = AutoModelForCausalLM.from_pretrained(hf_model_repo, trust_remote_code=True, torch_dtype="auto", device_map="cuda")
     
We arrange the dataset in the same manner as before.


# This code block is used to prepare the dataset for model training.

# 'dataset.map(create_message_column)' applies the 'create_message_column' function to each element in the 'dataset'.
# This function is used to create a new column in the dataset.
dataset_chatml = dataset.map(create_message_column)

# 'dataset_chatml.map(format_dataset_chatml)' applies the 'format_dataset_chatml' function to each element in 'dataset_chatml'.
# This function is used to format the dataset in a way that is suitable for chat ML.
dataset_chatml = dataset_chatml.map(format_dataset_chatml)

# 'dataset_chatml.train_test_split(test_size=0.05, seed=1234)' splits 'dataset_chatml' into a training set and a test set.
# 'test_size=0.05' means that 5% of the data will be used for the test set.
# 'seed=1234' is used for reproducibility.
dataset_chatml = dataset_chatml.train_test_split(test_size=0.05, seed=1234)

# 'dataset_chatml' is printed to the console to inspect its contents.
dataset_chatml
     

# 'dataset_chatml['test'][0]' is used to access the first element of the test set in the 'dataset_chatml' dataset.
# This can be used to inspect the first test sample to understand its structure and contents.
dataset_chatml['test'][0]
     
Create a text generation pipeline to run the inference


# 'pipeline' is a function from the 'transformers' library that creates a pipeline for text generation.
# 'text-generation' is the task that the pipeline will perform.
# 'model' is the pre-trained model that the pipeline will use.
# 'tokenizer' is the tokenizer that the pipeline will use to tokenize the input text.
# The created pipeline is stored in the 'pipe' variable.
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
     

# This code block is used to test the chat template.

# 'pipe.tokenizer.apply_chat_template' is a method that applies the chat template to a list of messages.
# The list of messages is [{"role": "user", "content": dataset_chatml['test'][0]['messages'][0]['content']}], which is the first message in the test set of 'dataset_chatml'.
# 'tokenize' is set to False, which means that the method will not tokenize the messages.
# 'add_generation_prompt' is set to True, which means that the method will add a generation prompt to the messages.
pipe.tokenizer.apply_chat_template([{"role": "user", "content": dataset_chatml['test'][0]['messages'][0]['content']}], tokenize=False, add_generation_prompt=True)
     
Develop a function that organizes the input and performs inference on an individual sample.


# This code block defines a function 'test_inference' that performs inference on a given prompt.

# 'prompt' is the input to the function. It is the text that the model will generate a response to.

# 'pipe.tokenizer.apply_chat_template' is a method that applies the chat template to the prompt.
# The prompt is wrapped in a list and formatted as a dictionary with "role" set to "user" and "content" set to the prompt.
# 'tokenize' is set to False, which means that the method will not tokenize the prompt.
# 'add_generation_prompt' is set to True, which means that the method will add a generation prompt to the prompt.
# The formatted prompt is stored back in the 'prompt' variable.

# 'pipe' is the text generation pipeline that was created earlier.
# It is called with the formatted prompt and several parameters that control the text generation process.
# 'max_new_tokens=256' limits the maximum number of new tokens that can be generated.
# 'do_sample=True' enables sampling, which means that the model will generate diverse responses.
# 'num_beams=1' sets the number of beams for beam search to 1, which means that the model will generate one response.
# 'temperature=0.3' controls the randomness of the responses. Lower values make the responses more deterministic.
# 'top_k=50' limits the number of highest probability vocabulary tokens to consider for each step.
# 'top_p=0.95' enables nucleus sampling and sets the cumulative probability of parameter tokens to 0.95.
# 'max_time=180' limits the maximum time for the generation process to 180 seconds.
# The generated responses are stored in the 'outputs' variable.

# The function returns the first generated response.
# The response is stripped of the prompt and any leading or trailing whitespace.
def test_inference(prompt):
    prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95,
                   max_time= 180) #, eos_token_id=eos_token)
    return outputs[0]['generated_text'][len(prompt):].strip()
     

# This code block calls the 'test_inference' function with the first message in the test set of 'dataset_chatml' as the prompt.
# 'test_inference' performs inference on the prompt and returns a generated response.
# The response is printed to the console.
test_inference(dataset_chatml['test'][0]['messages'][0]['content'])
     
Evaluate the performance

# 'load_metric' is a function from the 'datasets' library that loads a metric for evaluating the model.
# Metrics are used to measure the performance of the model on certain tasks.
from datasets import load_metric
     
We'll employ the ROUGE metric to assess performance. While it may not be the optimal metric, it's straightforward and convenient to utilize.


# 'load_metric("rouge", trust_remote_code=True)' loads the ROUGE metric from the 'datasets' library.
# ROUGE is a set of metrics used to evaluate automatic summarization and machine translation.
# 'trust_remote_code' is set to True, which means that the metric will trust and execute remote code.
# The loaded metric is stored in the 'rouge_metric' variable.
rouge_metric = load_metric("rouge", trust_remote_code=True)
     
Develop a function for performing inference and assessing an instance.


# This code block defines a function 'calculate_rogue' that calculates the ROUGE score for a given row in the dataset.

# 'row' is the input to the function. It is a row in the dataset that contains a message and its corresponding output.

# 'test_inference(row['messages'][0]['content'])' calls the 'test_inference' function with the first message in the row as the prompt.
# 'test_inference' performs inference on the prompt and returns a generated response.
# The response is stored in the 'response' variable.

# 'rouge_metric.compute' is a method that calculates the ROUGE score for the generated response and the corresponding output in the row.
# 'predictions' is set to the generated response and 'references' is set to the output in the row.
# 'use_stemmer' is set to True, which means that the method will use a stemmer to reduce words to their root form.
# The calculated ROUGE score is stored in the 'result' variable.

# The 'result' dictionary is updated to contain the F-measure of each ROUGE score multiplied by 100.
# The F-measure is a measure of a test's accuracy that considers both the precision and the recall of the test.

# The 'response' is added to the 'result' dictionary.

# The function returns the 'result' dictionary.
def calculate_rogue(row):
    response = test_inference(row['messages'][0]['content'])
    result = rouge_metric.compute(predictions=[response], references=[row['output']], use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result['response']=response
    return result
     
Now, we have the ability to execute inference on a collection of samples. For simplicity, the process isn't optimized at this stage. In the future, we plan to perform inference in batches to enhance performance. However, for the time being,


# '%%time' is a magic command in Jupyter notebooks that measures the execution time of the cell.

# 'dataset_chatml['test'].select(range(0,500))' selects the first 500 elements from the test set in the 'dataset_chatml' dataset.

# '.map(calculate_rogue, batched=False)' applies the 'calculate_rogue' function to each element in the selected subset.
# 'calculate_rogue' calculates the ROUGE score for each element.
# 'batched' is set to False, which means that the function will be applied to each element individually, not in batches.

# The results are stored in the 'metricas' variable.
%%time
metricas = dataset_chatml['test'].select(range(0,500)).map(calculate_rogue, batched=False)
     

# 'numpy' is a library in Python that provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
# 'import numpy as np' imports the 'numpy' library and gives it the alias 'np'. This allows us to use 'np' instead of 'numpy' when calling its functions.
import numpy as np
     
Now, we have the ability to compute the metric for the sample.


# This code block prints the mean of the ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores in the 'metricas' dictionary.

# 'np.mean(metricas['rouge1'])' calculates the mean of the ROUGE-1 scores.
# 'np.mean(metricas['rouge2'])' calculates the mean of the ROUGE-2 scores.
# 'np.mean(metricas['rougeL'])' calculates the mean of the ROUGE-L scores.
# 'np.mean(metricas['rougeLsum'])' calculates the mean of the ROUGE-Lsum scores.

# 'print' is used to print the calculated means to the console.
print("Rouge 1 Mean: ",np.mean(metricas['rouge1']))
print("Rouge 2 Mean: ",np.mean(metricas['rouge2']))
print("Rouge L Mean: ",np.mean(metricas['rougeL']))
print("Rouge Lsum Mean: ",np.mean(metricas['rougeLsum']))
     