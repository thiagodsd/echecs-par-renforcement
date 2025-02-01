import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import textwrap
import os

torch.cuda.empty_cache()
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name()
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.memory_summary()

def load_fine_tuned_model(base_model_name="microsoft/phi-2", adapter_path="./phi2-finetuned-lora"):
    """Load the fine-tuned model and tokenizer with memory optimizations."""
    # Set memory management environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Configure quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load base model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,  # Load from saved path
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    try:
        # Load LoRA adapter with memory optimizations
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            torch_dtype=torch.float16,
            device_map="auto",
            is_trainable=False,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"Error loading adapter: {str(e)}")
        raise
    
    return model, tokenizer

def generate_response(model, tokenizer, question, context, max_length=256):
    """Generate a response with memory management."""
    try:
        # Truncate context if too long
        context = context[:500]
        
        prompt = f"""<|im_start|>system
You are a helpful document assistant.<|im_end|>
<|im_start|>user
Context: {context}
Question: {question}<|im_end|>
<|im_start|>assistant
"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("assistant\n")[-1].strip()
        
        # Clear memory
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return response
    
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Error generating response. Please try with a shorter context."

def main():
    try:
        model, tokenizer = load_fine_tuned_model()
        
        while True:
            context = input("\nEnter the document context (or 'quit' to exit): ")
            if context.lower() == 'quit':
                break
                
            question = input("Enter your question: ")
            if question.lower() == 'quit':
                break
            
            print("\nGenerating response...")
            response = generate_response(model, tokenizer, question, context)
            print("\nResponse:")
            print("\n".join(textwrap.wrap(response, width=80)))
            
            # Clear memory after each interaction
            torch.cuda.empty_cache()
            gc.collect()
    
    finally:
        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()