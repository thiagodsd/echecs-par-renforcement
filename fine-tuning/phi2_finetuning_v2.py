import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
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

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512   
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name()
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.memory_summary()

@dataclass
class TrainingConfig:
    # Model configuration
    base_model_name: str = "microsoft/phi-2"
    model_max_length: int = 1024
    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    # Training configuration
    per_device_train_batch_size: int = 1 # per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 50 # warmup_steps: int = 100
    max_steps: int = 1000
    learning_rate: float = 2e-4
    output_dir: str = "output"
    logging_steps: int = 10
    save_steps: int = 200
    # Data processing
    chunk_size: int = 256 # chunk_size: int = 512
    chunk_overlap: int = 25 # chunk_overlap: int = 50

def prepare_model():
    """
    Initialize and prepare the Phi-2 model with 4-bit quantization and LoRA.
    """
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16
    )
    
    # Load base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        TrainingConfig.base_model_name,
        quantization_config = bnb_config,
        device_map = "auto",
        trust_remote_code = True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TrainingConfig.base_model_name,
        model_max_length=TrainingConfig.model_max_length,
        padding_side="right",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=TrainingConfig.lora_r,
        lora_alpha=TrainingConfig.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Phi-2 specific attention layers
        lora_dropout=TrainingConfig.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer


def process_documents(file_paths: List[str], tokenizer) -> Dataset:
    """
    Process and chunk DOCX documents into a format suitable for training.
    """
    def chunk_text(text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), TrainingConfig.chunk_size - TrainingConfig.chunk_overlap):
            chunk = text[i:i + TrainingConfig.chunk_size]
            if len(chunk) >= TrainingConfig.chunk_size // 2:  # Only keep chunks of reasonable size
                chunks.append(chunk)
        return chunks

    def extract_text_from_docx(file_path: str) -> str:
        doc = Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # Skip empty paragraphs
                full_text.append(paragraph.text)
        return "\n".join(full_text)
    
    processed_data = []
    for file_path in file_paths:
        try:
            if file_path.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            chunks = chunk_text(text)
            processed_data.extend([{"text": chunk} for chunk in chunks])
            print(f"Successfully processed {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Create dataset from processed data
    dataset = Dataset.from_list(processed_data)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=TrainingConfig.model_max_length,
            padding="max_length",
            return_tensors=None  # Return lists instead of tensors
        )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    output_dir: str = "output"
):
    """
    Train the model using the prepared dataset.
    """
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TrainingConfig.per_device_train_batch_size,
        gradient_accumulation_steps=TrainingConfig.gradient_accumulation_steps,
        warmup_steps=TrainingConfig.warmup_steps,
        max_steps=TrainingConfig.max_steps,
        learning_rate=TrainingConfig.learning_rate,
        fp16=True,
        logging_steps=TrainingConfig.logging_steps,
        save_strategy="steps",
        save_steps=TrainingConfig.save_steps,
        ddp_find_unused_parameters=False,
        remove_unused_columns=True,  # This is the default, but let's be explicit
        optim="adamw_torch",
        gradient_checkpointing=True,
        # group_by_length=True,  # Group similar length sequences
        # dataloader_num_workers=8,  # Adjust based on your CPU cores
        # dataloader_pin_memory=True,
        evaluation_strategy="no",  # Since we're not doing evaluation
        report_to=["wandb"],  # Keep only wandb for logging
        # torch_compile=False,  # Disable torch compilation for faster startup

    )
    
    # Prepare data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model()
    print("Model and tokenizer prepared")
    
    # Process your documents - replace with your file paths
    file_paths = [
        "/home/dusoudeth/Documentos/github/echecs-par-renforcement/data/docx/isadora_urel_DO_PLANO_DE_CONVIÊNCIA_IDEAL.docx",
        "/home/dusoudeth/Documentos/github/echecs-par-renforcement/data/docx/isadora_urel_tese_doutorado.docx",
    ]  # Add your document paths here
    train_dataset = process_documents(file_paths, tokenizer)
    print(f"Processed {len(train_dataset)} training examples")
    
    # Train the model
    train_model(model, tokenizer, train_dataset)
    print("Training completed")

if __name__ == "__main__":
    main()