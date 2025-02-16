import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
from docx import Document
import os
import re
import dotenv

dotenv.load_dotenv("../.env", override=True)

print(os.environ["HF_HOME"])
print(os.environ["TRANSFORMERS_CACHE"])

torch.cuda.empty_cache()
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name()
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.memory_summary()

# ====================
# Data Processing
# ====================

def docx_to_text(docx_path):
    """Convert DOCX file to plain text"""
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def process_legal_documents(docx_dir, chunk_size=512, overlap=50):
    """Process directory of DOCX files into chunked text"""
    texts = []
    
    # Convert DOCX to raw text
    for file in os.listdir(docx_dir):
        if file.endswith(".docx"):
            text = docx_to_text(os.path.join(docx_dir, file))
            # Simple cleaning of common legal formatting
            text = re.sub(r'\n{2,}', '\n', text)  # Remove excessive newlines
            text = re.sub(r'\s{2,}', ' ', text)   # Remove excessive whitespace
            texts.append(text)
    
    # Tokenization and chunking
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", cache_dir=os.environ["HF_HOME"])
    
    chunks = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk = tokens[i:i + chunk_size]
            chunks.append(tokenizer.decode(chunk))
    
    return Dataset.from_dict({"text": chunks})

# ====================
# Model Setup
# ====================

# Configuration
model_name = "mistralai/Mistral-7B-v0.1"
device_map = {"": 0}  # Force model to use GPU

# 4-bit Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # For auto-regressive models

# ====================
# PEFT Configuration
# ====================

peft_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]  # For Mistral architecture
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ====================
# Training Setup
# ====================

# Prepare dataset (replace with your DOCX directory path)
dataset = process_legal_documents("/home/dusoudeth/Documentos/github/echecs-par-renforcement/data/docx")

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=1000,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir="./legal_finetuned",
    save_strategy="steps",
    save_steps=200,
    optim="paged_adamw_8bit",
    report_to="none"
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=lambda x: [x["text"]]
)

# ====================
# Training & Saving
# ====================

print("Starting training...")
trainer.train()

print("Saving final model...")
model.save_pretrained("./legal_finetuned/final_model")
tokenizer.save_pretrained("./legal_finetuned/final_model")