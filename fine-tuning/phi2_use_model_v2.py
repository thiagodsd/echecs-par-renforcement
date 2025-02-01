import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import textwrap
import os

def load_fine_tuned_model(base_model_name="microsoft/phi-2", adapter_path="./phi2-finetuned-lora"):
    """Load the fine-tuned model and tokenizer with proper dtype handling."""
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            trust_remote_code=True,
            padding_side="right",
            use_fast=False
        )
    except Exception as e:
        print(f"Error loading tokenizer from adapter path, falling back to base model: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="right",
            use_fast=False
        )
        special_tokens = {
            "pad_token": "<|pad|>",
            "eos_token": "<|endoftext|>",
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
        }
        tokenizer.add_special_tokens(special_tokens)
    
    # Load base model with float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,  # Changed to float32
            bnb_4bit_use_double_quant=True
        ),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Changed to float32
        low_cpu_mem_usage=True
    )
    
    model.resize_token_embeddings(len(tokenizer))
    
    try:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            device_map="auto",
            torch_dtype=torch.float32,  # Changed to float32
            is_trainable=False
        )
        model.eval()
        
        # Convert specific layers to float32
        for name, module in model.named_modules():
            if "norm" in name.lower():
                module.to(torch.float32)
    except Exception as e:
        print(f"Error loading adapter: {str(e)}")
        raise
    
    return model, tokenizer

def generate_response(model, tokenizer, question, context, max_length=512):
    """Generate a response with proper response extraction."""
    try:
        context = context[:500]
        
        # Modified prompt format to better guide the model's response
        prompt = f"""<|im_start|>system
You are a helpful assistant that provides clear and informative answers based on the given context. If no context is provided or if it's marked as "NONE", provide a general answer based on your knowledge.
<|im_end|>
<|im_start|>user
Context: {context}
Question: {question}
<|im_end|>
<|im_start|>assistant
Let me help you with that.
"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        if 'attention_mask' in inputs:
            inputs['attention_mask'] = inputs['attention_mask'].to(dtype=torch.float32)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.2,  # Added to prevent repetition
                top_p=0.95,  # Added for better response diversity
                no_repeat_ngram_size=3  # Prevent repeating triplets of tokens
            )
        
        # Decode the response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the assistant's response
        try:
            response_parts = full_response.split('<|im_start|>assistant')
            if len(response_parts) > 1:
                assistant_response = response_parts[-1].split('<|im_end|>')[0]
            else:
                assistant_response = full_response
            
            # Clean up the response
            assistant_response = assistant_response.replace('Let me help you with that.', '').strip()
            if not assistant_response:
                assistant_response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            assistant_response = full_response
        
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return assistant_response
    
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error generating response. Please try with a shorter context."

def main():
    try:
        model, tokenizer = load_fine_tuned_model()
        
        print("\nModel loaded successfully! You can now ask questions.")
        print("Type 'quit' to exit.")
        
        while True:
            context = input("\nEnter the document context (or 'NONE' if not applicable): ")
            if context.lower() == 'quit':
                break
                
            question = input("Enter your question: ")
            if question.lower() == 'quit':
                break
            
            print("\nGenerating response...")
            response = generate_response(model, tokenizer, question, context)
            print("\nResponse:")
            print("\n".join(textwrap.wrap(response, width=80)))
            
            torch.cuda.empty_cache()
            gc.collect()
    
    finally:
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()