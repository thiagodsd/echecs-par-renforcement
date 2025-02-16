# /home/dusoudeth/Documentos/github/echecs-par-renforcement/fine-tuning/phi2_finetuning_v2.py
import os
from dataclasses import dataclass
from typing import Dict, List  # noqa: F401

import torch
from datasets import Dataset, load_dataset  # noqa: F401
from peft import (
    LoraConfig,
    PeftConfig,  # noqa: F401
    PeftModel,  # noqa: F401
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from docx import Document
import dotenv

dotenv.load_dotenv("../.env", override=True)

print(os.environ["HF_HOME"])
print(os.environ["TRANSFORMERS_CACHE"])

# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name()
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.memory_summary()

@dataclass
class TrainingConfig:
    #
    base_model_name: str = "microsoft/phi-2"
    model_max_length: int = 1024
    #
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    #
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 2
    max_steps: int = 10
    learning_rate: float = 1e-5
    output_dir: str = "output"
    logging_steps: int = 10
    save_steps: int = 200
    #
    chunk_size: int = 256
    chunk_overlap: int = 25


def prepare_model():
    """
    `todo`
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
    )
    # loading the model
    model = AutoModelForCausalLM.from_pretrained(
        TrainingConfig.base_model_name,
        quantization_config = bnb_config,
        device_map = "auto",
        trust_remote_code = True,
        torch_dtype = torch.float16,
        low_cpu_mem_usage = True
    )
    # loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TrainingConfig.base_model_name,
        model_max_length = TrainingConfig.model_max_length,
        padding_side = "right",
        trust_remote_code = True
    )
    tokenizer.pad_token = tokenizer.eos_token
    # preparing the model for kbit training
    model = prepare_model_for_kbit_training(model)
    # lora configuration
    lora_config = LoraConfig(
        r = TrainingConfig.lora_r,
        lora_alpha = TrainingConfig.lora_alpha,
        target_modules = ["q_proj", "v_proj"],
        lora_dropout = TrainingConfig.lora_dropout,
        bias = "none",
        task_type = "CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def process_documents(file_paths: List[str], tokenizer) -> Dataset:
    """
    `todo`
    """
    # function definitions
    def chunk_text(text: str) -> List[str]:
        """
        here we chunk the text
        """
        chunks = list()
        for i in range(0, len(text), TrainingConfig.chunk_size - TrainingConfig.chunk_overlap):
            chunk = text[i:i + TrainingConfig.chunk_size]
            if len(chunk) >= TrainingConfig.chunk_size // 2:
                chunks.append(chunk)
        return chunks
    def extract_text_from_docx(file_path: str) -> str:
        """
        here we read the text from a docx file
        """
        doc = Document(file_path)
        full_text = list()
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text)
        return "\n".join(full_text)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation = True,
            max_length = TrainingConfig.model_max_length,
            padding = "max_length",
            return_tensors = None
        )
    # processing the documents
    processed_data = list()
    for file_path in file_paths:
        try:
            if file_path.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            chunks = chunk_text(text)
            processed_data.extend([{"text": chunk} for chunk in chunks])
            print(f"Successfully processed {file_path}".lower())
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}".lower())
    # generating the dataset from the processed data
    dataset = Dataset.from_list(processed_data)
    # tokenizing the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched = True,
        remove_columns = dataset.column_names
    )
    return tokenized_dataset


def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    output_dir: str = "output"
):
    """
    `todo`
    """
    # preparing the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TrainingConfig.per_device_train_batch_size,
        gradient_accumulation_steps=TrainingConfig.gradient_accumulation_steps,
        warmup_steps=TrainingConfig.warmup_steps,
        max_steps=TrainingConfig.max_steps,
        learning_rate=TrainingConfig.learning_rate,
        fp16=True,  # Enable fp16 training
        logging_steps=TrainingConfig.logging_steps,
        save_strategy="steps",
        save_steps=TrainingConfig.save_steps,
        ddp_find_unused_parameters=False,
        remove_unused_columns=True,
        optim="paged_adamw_8bit",  # Changed to paged_adamw_8bit
        gradient_checkpointing=True,
        evaluation_strategy="no",
        report_to=["none"],  # Disabled wandb reporting
        max_grad_norm=0.3,  # Added from v1
        warmup_ratio=0.03,  # Added from v1
    )
    # preparing the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    # preparing the trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=data_collator,
    )
    # fine-tuning the model
    trainer.train()
    # saving the model
    trainer.save_model(output_dir)


def main():
    # setting up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}".lower())
    # model & tokenizer
    model, tokenizer = prepare_model()
    print("Model and tokenizer prepared".lower())
    # documents
    file_paths = [
        "/home/dusoudeth/Documentos/github/echecs-par-renforcement/data/docx/isadora_urel_DO_PLANO_DE_CONVIÊNCIA_IDEAL.docx",
        "/home/dusoudeth/Documentos/github/echecs-par-renforcement/data/docx/isadora_urel_tese_doutorado.docx",
    ]
    train_dataset = process_documents(
        file_paths, 
        tokenizer
    )
    print(f"Processed {len(train_dataset)} training examples".lower())
    # training
    train_model(
        model,
        tokenizer,
        train_dataset
    )
    print("Training completed".lower())

if __name__ == "__main__":
    main()