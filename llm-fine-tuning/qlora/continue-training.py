def train_model(model, tokenizer, train_dataset):
    """Train the model with updated configuration"""
    logger.info("Starting training setup...")
    
    # Check for existing checkpoint
    last_checkpoint = None
    if os.path.exists(TrainingConfig.output_dir):
        checkpoints = [f for f in os.listdir(TrainingConfig.output_dir) if f.startswith('checkpoint-')]
        if checkpoints:
            last_checkpoint = os.path.join(
                TrainingConfig.output_dir,
                sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
            )
            logger.info(f"Found checkpoint: {last_checkpoint}")
    
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
        torch_compile=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Add resume settings
        resume_from_checkpoint=last_checkpoint,
        overwrite_output_dir=True
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Batch size: {TrainingConfig.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {TrainingConfig.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {TrainingConfig.per_device_train_batch_size * TrainingConfig.gradient_accumulation_steps}")
    
    if last_checkpoint:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
    
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
        trainer.train(resume_from_checkpoint=last_checkpoint)
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