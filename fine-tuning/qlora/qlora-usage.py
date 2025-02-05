import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceModel:
    def __init__(self, base_model_path: str = "maritaca-ai/sabia-7b", adapter_path: str = "output"):
        """
        Initialize the fine-tuned model for inference
        Args:
            base_model_path: Path to the base model
            adapter_path: Path to the trained LoRA adapter weights
        """
        logger.info("Loading model...")
        
        # Load base model with same quantization as training
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True
        )
        
        # Load adapter weights
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            torch_dtype=torch.float16,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Set evaluation mode
        self.model.eval()
        logger.info("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs
    ) -> str:
        """
        Generate text based on the prompt
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,  # Changed from max_length
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            # Decode and return response
            response = self.tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True
            )[0]
            
            # Remove the prompt from the response
            response = response[len(prompt):]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise e

def main():
    # Initialize model
    model = InferenceModel(
        base_model_path="maritaca-ai/sabia-7b",
        adapter_path="output"  # Path to your trained adapter
    )
    
    # Example prompts
    prompts = [
        "Quem é Isadora Urel",
        "Explique o conceito de plano de parentalidade:",
        "Quais são os principais aspectos de um acordo de convivência?",
        "Como elaborar um plano de convivência ideal?",
    ]
    
    # Generate responses
    print("\n" + "= "*50 + "\n")
    for prompt in prompts:
        try:
            print(f"Prompt: {prompt}\n")
            response = model.generate(
                prompt,
                max_new_tokens=2048,
                temperature=0.75,
                top_p=0.9,
                do_sample=True
            )
            print(f"""
            
            RESPONSE
            {response}\n

            """)
            print("="*50 + "\n")
        except Exception as e:
            logger.error(f"Error processing prompt '{prompt}': {str(e)}")
            continue

if __name__ == "__main__":
    main()