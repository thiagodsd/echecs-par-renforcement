# import os
# from typing import Tuple

# import torch
# from peft import PeftModel
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# def load_fine_tuned_model(
#     base_model_name: str = "microsoft/phi-2",
#     adapter_path: str = "output",
#     model_max_length: int = 1024,
# ) -> Tuple[PeftModel, AutoTokenizer]:
#     """
#     Load the fine-tuned model and tokenizer.
    
#     Args:
#         base_model_name (str): Name of the base model
#         adapter_path (str): Path to the fine-tuned adapter weights
#         model_max_length (int): Maximum sequence length for the model
        
#     Returns:
#         Tuple[PeftModel, AutoTokenizer]: The loaded model and tokenizer
#     """
#     # Configure quantization
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16,
#     )
    
#     # Load base model
#     model = AutoModelForCausalLM.from_pretrained(
#         base_model_name,
#         quantization_config=bnb_config,
#         device_map="auto",
#         trust_remote_code=True,
#         torch_dtype=torch.float16,
#         low_cpu_mem_usage=True
#     )
    
#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         base_model_name,
#         model_max_length=model_max_length,
#         padding_side="right",
#         trust_remote_code=True
#     )
#     tokenizer.pad_token = tokenizer.eos_token
    
#     # Load adapter weights
#     model = PeftModel.from_pretrained(model, adapter_path)
#     model.eval()
    
#     return model, tokenizer


# def generate_response(
#     model: PeftModel,
#     tokenizer: AutoTokenizer,
#     question: str,
#     context: str = "",
#     max_length: int = 512,
#     temperature: float = 0.7,
#     top_p: float = 0.9,
#     top_k: int = 50,
#     num_return_sequences: int = 1,
#     repetition_penalty: float = 1.1,
# ) -> str:
#     """
#     Generate a response using the fine-tuned model.
    
#     Args:
#         model (PeftModel): The fine-tuned model
#         tokenizer (AutoTokenizer): The tokenizer
#         question (str): The question to answer
#         context (str): Optional context to provide
#         max_length (int): Maximum length of the generated response
#         temperature (float): Sampling temperature
#         top_p (float): Nucleus sampling parameter
#         top_k (int): Top-k sampling parameter
#         num_return_sequences (int): Number of responses to generate
#         repetition_penalty (float): Penalty for repetition
        
#     Returns:
#         str: The generated response
#     """
#     # Prepare input text
#     input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    
#     # Tokenize input
#     inputs = tokenizer(
#         input_text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=max_length,
#         padding=True,
#     ).to(model.device)
    
#     # Generate response
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_length=max_length,
#             temperature=temperature,
#             top_p=top_p,
#             top_k=top_k,
#             num_return_sequences=num_return_sequences,
#             repetition_penalty=repetition_penalty,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#         )
    
#     # Decode response
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     # Clean up response by removing the input prompt
#     response = response.replace(input_text, "").strip()
    
#     return response


# if __name__ == "__main__":
#     # Example usage
#     model, tokenizer = load_fine_tuned_model()
    
#     question = "O que é plano de parentalidade?"
#     response = generate_response(model, tokenizer, question)
#     print(f"Question: {question}")
#     print(f"Response: {response}")

import os
from typing import Tuple

import torch
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_fine_tuned_model(
    base_model_name: str = "microsoft/phi-2",
    adapter_path: str = "output",
    model_max_length: int = 1024
) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Load the fine-tuned model and tokenizer with the same configuration used during training.
    
    Args:
        base_model_name (str): Name of the base model
        adapter_path (str): Path to the fine-tuned adapter weights
        model_max_length (int): Maximum sequence length for the model
        
    Returns:
        Tuple[PeftModel, AutoTokenizer]: The loaded model and tokenizer
    """
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load the base model with the same configuration used in training
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # Load the tokenizer with the same configuration
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        model_max_length=model_max_length,
        padding_side="right",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the trained LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Set to evaluation mode
    model.eval()
    return model, tokenizer


def generate_response(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    question: str,
    context: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    repetition_penalty: float = 1.1
) -> str:
    """
    Generate a response using the fine-tuned model.
    
    Args:
        model (PeftModel): The fine-tuned model
        tokenizer (AutoTokenizer): The tokenizer
        question (str): The question to answer
        context (str): Optional context to provide
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature (higher = more creative)
        top_p (float): Nucleus sampling parameter
        do_sample (bool): Whether to use sampling or greedy generation
        repetition_penalty (float): Penalty for repetition
        
    Returns:
        str: The generated response
    """
    # Prepare the prompt
    if context:
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"
    
    # Tokenize the prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.max_position_embeddings - max_new_tokens,
        padding=True
    ).to(model.device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and clean up the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    
    return response


if __name__ == "__main__":
    # Example usage
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    
    model, tokenizer = load_fine_tuned_model()
    
    response = generate_response(
        model,
        tokenizer,
        question="O que é plano de parentalidade?",
        context="O que é plano de parentalidade"
    )
    print(f"Response: {response}")