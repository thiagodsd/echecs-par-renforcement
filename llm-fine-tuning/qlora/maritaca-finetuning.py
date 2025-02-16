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

dotenv.load_dotenv("../../.env", override=True)

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
    # Model configuration
    base_model_name: str = "maritaca-ai/sabia-7b"
    model_max_length: int = 1024  # Reduced for memory efficiency
    
    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Training configuration
    per_device_train_batch_size: int = 1  # Small batch size for 6GB VRAM
    gradient_accumulation_steps: int = 16  # Increased for effective batch size
    warmup_steps: int = 50
    max_steps: int = 500
    learning_rate: float = 2e-4
    output_dir: str = "output"
    logging_steps: int = 10
    save_steps: int = 100
    
    # Data processing
    chunk_size: int = 512  # Reduced chunk size
    chunk_overlap: int = 50

class DocumentProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.chunk_size = TrainingConfig.chunk_size
        self.chunk_overlap = TrainingConfig.chunk_overlap

    def read_pdf(self, file_path: str) -> str:
        """Read PDF file safely"""
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
        """Read DOCX file safely"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {str(e)}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the first chunk, start from the overlap point
            if start > 0:
                start = start - self.chunk_overlap
                
            chunk = text[start:end]
            if len(chunk) >= self.chunk_size // 2:  # Only keep substantial chunks
                chunks.append(chunk)
            
            start = end
            
        return chunks

    def process_file(self, file_path: str) -> List[str]:
        """Process a single file"""
        if file_path.lower().endswith('.pdf'):
            text = self.read_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            text = self.read_docx(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return []
            
        return self.chunk_text(text)

def prepare_training_data(file_paths: List[str], tokenizer) -> Dataset:
    """Prepare training data from files"""
    processor = DocumentProcessor(tokenizer)
    all_chunks = []
    
    for file_path in tqdm(file_paths, desc="Processing files"):
        chunks = processor.process_file(file_path)
        all_chunks.extend(chunks)
    
    # Create dataset
    dataset_dict = {"text": all_chunks}
    dataset = Dataset.from_dict(dataset_dict)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=TrainingConfig.model_max_length,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()

def setup_model():
    """Setup model with QLoRA configuration"""
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        TrainingConfig.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False  # Disable KV cache for training
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TrainingConfig.base_model_name,
        model_max_length=TrainingConfig.model_max_length,
        padding_side="right",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=TrainingConfig.lora_r,
        lora_alpha=TrainingConfig.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=TrainingConfig.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset):
    """Train the model"""
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
        gradient_checkpointing=True,  # Enable gradient checkpointing
        optim="paged_adamw_8bit",  # Use 8-bit optimizer
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train
    trainer.train()
    
    # Save the final model
    trainer.save_model(TrainingConfig.output_dir)

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Clear memory before starting
    clear_memory()
    
    try:
        # Setup model and tokenizer
        model, tokenizer = setup_model()
        logger.info("Model and tokenizer prepared")
        
        # Process your documents - replace with your file paths
        file_paths = [
            # "/home/dusoudeth/Documentos/github/crew-scientific-research/research/context/docx/plano_de_parentalidade_da_isa.docx",
            # "/home/dusoudeth/Documentos/github/crew-scientific-research/research/context/docx/isadora_urel_Proposta_de_acordo.docx",
            # "/home/dusoudeth/Documentos/github/crew-scientific-research/research/context/docx/isadora_urel_plano_de_convivencia_modelo.docx",
            # "/home/dusoudeth/Documentos/github/crew-scientific-research/research/context/docx/isadora_urel_DO_PLANO_DE_CONVIÊNCIA_IDEAL.docx",
            "/home/dusoudeth/Documentos/github/crew-scientific-research/research/context/docx/isadora_urel_tese_doutorado.docx",
        ]
        
        # Prepare training data
        train_dataset = prepare_training_data(file_paths, tokenizer)
        logger.info(f"Processed {len(train_dataset)} training examples")
        
        # Train model
        train_model(model, tokenizer, train_dataset)
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
    finally:
        clear_memory()

if __name__ == "__main__":
    main()
