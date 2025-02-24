import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
from docx import Document
import os

# Document processing functions remain the same
def docx_to_text(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def create_qa_pairs(text, window_size=512, stride=256):
    qa_pairs = []
    words = text.split()
    
    for i in range(0, len(words), stride):
        chunk = " ".join(words[i:i+window_size])
        qa_pairs.append({
            "question": f"Summarize the following text: {chunk[:100]}...",
            "context": chunk,
            "answer": chunk[:512]
        })
    return qa_pairs

def prepare_dataset(docx_folder, output_json="dataset.json"):
    dataset = []
    
    for file in os.listdir(docx_folder):
        if file.endswith(".docx"):
            text = docx_to_text(os.path.join(docx_folder, file))
            dataset.extend(create_qa_pairs(text))
    
    with open(output_json, "w") as f:
        json.dump(dataset, f)
    
    return Dataset.from_json(output_json)

# Updated model and tokenizer setup with proper gradient handling for quantized models
def setup_model_and_tokenizer(model_name="microsoft/phi-2"):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False
    )
    
    # Configure special tokens
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    
    # Load model with proper configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure model for training
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def preprocess_function(examples, tokenizer, max_length=512):
    formatted_prompts = [
        f"""<|im_start|>system
You are a helpful document assistant.<|im_end|>
<|im_start|>user
Context: {context}
Question: {question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""
        for question, context, answer in zip(examples['question'], examples['context'], examples['answer'])
    ]
    
    tokenized = tokenizer(
        formatted_prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
        return_attention_mask=True
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def setup_training(model, processed_dataset, output_dir="./phi2-finetuned"):
    # Configure LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["Wqkv", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    
    # Ensure model is in training mode
    peft_model.train()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # Changed optimizer
        remove_unused_columns=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03
    )
    
    # Custom data collator
    def collate_fn(examples):
        batch = {}
        for key in examples[0].keys():
            batch[key] = torch.tensor([example[key] for example in examples], dtype=torch.long)
        return batch
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        data_collator=collate_fn
    )
    
    return trainer

def main():
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare dataset
    raw_dataset = prepare_dataset("/home/dusoudeth/Documentos/github/crew-scientific-research/research/context/docx")
    
    # Process dataset
    processed_dataset = raw_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=raw_dataset.column_names
    )
    train_test_split = processed_dataset.train_test_split(test_size=0.1)
    
    # Setup and run training
    trainer = setup_training(model, train_test_split)
    
    try:
        torch.cuda.empty_cache()
        trainer.train()
        trainer.model.save_pretrained("./phi2-finetuned-lora")
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise

if __name__ == "__main__":
    main()