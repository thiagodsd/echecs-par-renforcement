import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import textwrap

def load_fine_tuned_model(base_model_name="microsoft/phi-2", adapter_path="./phi2-finetuned-lora"):
    """Load the fine-tuned model and tokenizer."""
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, question, context, max_length=512):
    """Generate a response from the model."""
    # Format the prompt
    prompt = f"""<|im_start|>system
You are a helpful document assistant.<|im_end|>
<|im_start|>user
Context: {context}
Question: {question}<|im_end|>
<|im_start|>assistant
"""
    
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode and clean the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    response = response.split("assistant\n")[-1].strip()
    
    return response

def main():
    torch.cuda.empty_cache()
    # Load the model and tokenizer
    print("Loading model...")
    model, tokenizer = load_fine_tuned_model()
    
    print("\nModel loaded! You can now ask questions about your documents.")
    print("Type 'quit' to exit.\n")
    
    while True:
        # Get user input
        context = input("\nEnter the document context (or part of it): ")
        if context.lower() == 'quit':
            break
            
        question = input("Enter your question: ")
        if question.lower() == 'quit':
            break
        
        # Generate and print response
        print("\nGenerating response...")
        response = generate_response(model, tokenizer, question, context)
        
        # Print the wrapped response for better readability
        print("\nResponse:")
        print("\n".join(textwrap.wrap(response, width=80)))

if __name__ == "__main__":
    main()
