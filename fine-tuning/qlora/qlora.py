import os
import torch
from typing import List, Dict
import gc
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import PyPDF2
from docx import Document
import logging
import numpy as np
from tqdm import tqdm
import dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv(".env", override=True)

print(os.environ["HF_HOME"])
print(os.environ["TRANSFORMERS_CACHE"])

# Configure PyTorch memory allocation
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

@dataclass
class TrainingConfig:
    # Model configuration
    base_model_name: str = "maritaca-ai/sabia-7b"
    model_max_length: int = 2048
    
    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Training configuration
    per_device_train_batch_size: int = 2  # Reduced from 4
    gradient_accumulation_steps: int = 16  # Increased from 8
    warmup_steps: int = 50
    max_steps: int = 128
    learning_rate: float = 1e-4
    output_dir: str = "output"
    logging_steps: int = 1
    save_steps: int = 100
    
    # Data processing
    chunk_size: int = 1024
    chunk_overlap: int = 100
    
    # Additional settings
    gradient_checkpointing: bool = True
    flash_attention: bool = False
    compute_dtype: torch.dtype = torch.float16

class DocumentProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.chunk_size = TrainingConfig.chunk_size
        self.chunk_overlap = TrainingConfig.chunk_overlap

    def read_pdf(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            return ""

    def read_docx(self, file_path: str) -> str:
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {str(e)}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if start > 0:
                start = start - self.chunk_overlap
                
            chunk = text[start:end]
            if len(chunk) >= self.chunk_size // 2:
                chunks.append(chunk)
            
            start = end
            
        return chunks

    def process_file(self, file_path: str) -> List[str]:
        if file_path.lower().endswith('.pdf'):
            text = self.read_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            text = self.read_docx(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return []
            
        return self.chunk_text(text)

def prepare_training_data(file_paths: List[str], tokenizer) -> Dataset:
    processor = DocumentProcessor(tokenizer)
    all_chunks = []
    
    for file_path in tqdm(file_paths, desc="Processing files"):
        chunks = processor.process_file(file_path)
        if not chunks:
            logger.warning(f"No chunks generated for file: {file_path}")
            continue
        logger.info(f"Generated {len(chunks)} chunks from {file_path}")
        all_chunks.extend(chunks)
    
    if not all_chunks:
        raise ValueError("No text chunks were generated from the input files!")
    
    logger.info(f"Total chunks generated: {len(all_chunks)}")
    logger.info(f"Average chunk length: {sum(len(chunk) for chunk in all_chunks) / len(all_chunks)}")
    
    dataset_dict = {"text": all_chunks}
    dataset = Dataset.from_dict(dataset_dict)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=TrainingConfig.model_max_length,
            padding="max_length",
            return_tensors=None  # Changed from "pt" to None
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy()  # Using copy() instead of clone()
        }
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    return tokenized_dataset

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()

def setup_model():
    """Setup model with optimized QLoRA configuration"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.float16  # Added for stability
    )
    
    # Load model with updated configuration
    model = AutoModelForCausalLM.from_pretrained(
        TrainingConfig.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        max_memory={0: "40GB"},
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2" if TrainingConfig.flash_attention else "eager"
    )
    
    # Freeze model parameters and ensure correct dtype
    for param in model.parameters():
        param.requires_grad = False
        if param.dtype == torch.float16:
            param.data = param.data.to(torch.float16)
    
    tokenizer = AutoTokenizer.from_pretrained(
        TrainingConfig.base_model_name,
        model_max_length=TrainingConfig.model_max_length,
        padding_side="right",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=TrainingConfig.lora_r,
        lora_alpha=TrainingConfig.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=TrainingConfig.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)

    logger.info(f"Model device map: {model.hf_device_map}")
    logger.info(f"Model dtype: {model.dtype}")
    logger.info(f"Model parameter count: {model.num_parameters()}")
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset):
    """Train the model with updated configuration"""
    logger.info("Starting training setup...")
    
    training_args = TrainingArguments(
        output_dir=TrainingConfig.output_dir,
        per_device_train_batch_size=TrainingConfig.per_device_train_batch_size,
        gradient_accumulation_steps=TrainingConfig.gradient_accumulation_steps,
        warmup_steps=TrainingConfig.warmup_steps,
        max_steps=TrainingConfig.max_steps,
        learning_rate=TrainingConfig.learning_rate,
        fp16=True,
        logging_steps=TrainingConfig.logging_steps,
        save_strategy="steps",
        save_steps=TrainingConfig.save_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        logging_dir="logs",
        report_to=["wandb"],
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        torch_compile=False,  # Disabled to prevent compilation issues
        ddp_find_unused_parameters=False,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Batch size: {TrainingConfig.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {TrainingConfig.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {TrainingConfig.per_device_train_batch_size * TrainingConfig.gradient_accumulation_steps}")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully")
        
        logger.info("Saving final model...")
        trainer.save_model(TrainingConfig.output_dir)
        logger.info(f"Model saved to {TrainingConfig.output_dir}")
        
        metrics = trainer.state.log_history
        logger.info(f"Final training metrics: {metrics[-1] if metrics else 'No metrics available'}")
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        logger.error("GPU Memory Usage:")
        logger.error(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.error(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        raise e

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
    
    clear_memory()
    
    try:
        model, tokenizer = setup_model()
        logger.info("Model and tokenizer prepared")
        
        file_paths = [
            "/root/isadora_urel_acordo .docx",
            "/root/isadora_urel_DO_PLANO_DE_CONVIÊNCIA_IDEAL.docx",
            "/root/isadora_urel_plano_de_convivencia .docx",
            "/root/isadora_urel_plano_de_convivencia_modelo.docx",
            "/root/isadora_urel_plano_de_parentalidade.docx",
            "/root/isadora_urel_Proposta_de_acordo.docx",
            "/root/isadora_urel_tese_doutorado.docx",
            "/root/plano_de_parentalidade_da_isa.docx",
        ]
        
        train_dataset = prepare_training_data(file_paths, tokenizer)
        logger.info(f"Processed {len(train_dataset)} training examples")
        
        if len(train_dataset) == 0:
            raise ValueError("No training examples were generated!")
            
        logger.info("Dataset structure:")
        logger.info(f"Dataset columns: {train_dataset.column_names}")
        logger.info(f"First example keys: {list(train_dataset[0].keys())}")
        
        logger.info("Starting model training...")
        train_model(model, tokenizer, train_dataset)
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise e
    finally:
        clear_memory()

if __name__ == "__main__":
    main()