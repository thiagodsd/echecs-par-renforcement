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
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    warmup_steps: int = 50
    max_steps: int = 256  # Increased from previous value
    learning_rate: float = 1e-4
    output_dir: str = "output"
    logging_steps: int = 1
    save_steps: int = 50  # More frequent checkpoints
    
    # Data processing
    chunk_size: int = 1024
    chunk_overlap: int = 100
    
    # Additional settings
    gradient_checkpointing: bool = True
    flash_attention: bool = False
    compute_dtype: torch.dtype = torch.float16