import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(verbose=True)

from typing import Optional, List
from pathlib import Path
from dataclasses import asdict
from pprint import pprint

import torch

from transformers import (IdeficsForVisionText2Text,
                          Trainer,
                          TrainingArguments,
                          BitsAndBytesConfig)
from transformers import set_seed
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model

import wandb

from callbacks.eval_first_step_callback import EvalFirstStepCallback
from dataloader.dataset_group_loader import DATASET_NAME_TO_LOAD_FUNC
from models.idefics_config import IDEFICSConfig
from trainer.finetune_config import FinetuneConfig


def main(idefics_config_path: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True,
                                                  help="Path to the IDEFICSConfig file."),
         finetune_config_path: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True,
                                                   help="Path to FinetuneConfig."),
         seed: Optional[int] = typer.Option(None, help="Random seed."),
         debug: bool = typer.Option(False, help="Whether to run in debug mode or not.")):
    """
    Run inference on the IDEFICS model.
    """
    
    print("\n-----------------------\n")
    
    if seed is not None:
        print(f"Setting random seed to {seed}...")
        set_seed(seed)
        print("\n-----------------------\n")
    
    
    # ======== Load configs ========
    idefics_config = IDEFICSConfig.from_yaml(str(idefics_config_path))
    finetune_config = FinetuneConfig.from_yaml(str(finetune_config_path))
    
    
    # Create config for wandb:
    wandb_config = {
        "idefics_configs": asdict(idefics_config),
        "finetune_config": asdict(finetune_config)
    }
    print("W&B config:")
    pprint(wandb_config)
    
    print("\n-----------------------\n")

    # Initialize W&B:
    wandb.login()
    wandb.init(project=os.environ["WANDB_PROJECT_FINETUNING"],
               job_type="finetuning",
               tags=[idefics_config.checkpoint],
               name=finetune_config.experiment_name,
               config=wandb_config,
               mode="disabled" if debug else None)
    
    
    # ======== Load model ========
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if idefics_config.load_in_4_bits:
        assert device == "cuda", "4bit quantization is only supported on CPU."
        print("4bit quantization enabled. Loading BitsAndBytesConfig...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=idefics_config.load_in_4_bits,
            bnb_4bit_use_double_quant=idefics_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=idefics_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=idefics_config.bnb_4bit_compute_dtype,
            llm_int8_skip_modules=idefics_config.llm_int8_skip_modules
        )
    else:
        bnb_config = None
    
    model = IdeficsForVisionText2Text.from_pretrained(idefics_config.checkpoint,
                                                      quantization_config=bnb_config,
                                                      device_map="auto")
    
    print("Model loaded with the following config:")
    print(model.config)
    print("\n-----------------------\n")
    
    
    # ======== Add LoRA adapters ========
    print("Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=finetune_config.lora_rank,
        lora_alpha=finetune_config.lora_alpha,
        target_modules=finetune_config.target_modules,
        lora_dropout=finetune_config.lora_dropout,
        bias=finetune_config.bias
    )
    model = get_peft_model(model, lora_config)
    
    
    # ======== Load dataset ========
    print("Loading dataset...")
    ds_group = DATASET_NAME_TO_LOAD_FUNC[finetune_config.dataset_name]()
    
    
    # ======== Define callbacks ========
    callbacks: List[TrainerCallback] = []
    
    if finetune_config.eval_first_step:
        callbacks.append(EvalFirstStepCallback())
    if finetune_config.early_stopping_patience != -1:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=finetune_config.early_stopping_patience))
    
    
    # ======== Fine-tuning ========
    training_args = TrainingArguments(
        output_dir=finetune_config.model_dir,
        learning_rate=finetune_config.learning_rate,
        warmup_steps=finetune_config.warmup_steps,
        num_train_epochs=finetune_config.num_train_epochs,
        lr_scheduler_type=finetune_config.lr_scheduler_type,
        per_device_train_batch_size=finetune_config.train_batch_size,
        per_device_eval_batch_size=finetune_config.eval_batch_size,
        gradient_accumulation_steps=finetune_config.gradient_accumulation_steps,
        dataloader_pin_memory=finetune_config.dataloader_pin_memory,
        save_total_limit=finetune_config.save_total_limit,
        evaluation_strategy="steps",
        fp16=idefics_config.fp16,
        fp16_full_eval=idefics_config.fp16,
        bf16=idefics_config.bf16,
        bf16_full_eval=idefics_config.bf16,
        save_strategy="steps",
        save_steps=finetune_config.save_steps,
        eval_steps=finetune_config.eval_steps,
        logging_steps=finetune_config.logging_steps,
        remove_unused_columns=False,
        push_to_hub=False,
        label_names=["labels"],
        load_best_model_at_end=False if finetune_config.save_total_limit == 1 else True,
        report_to="wandb",
        optim=finetune_config.optim
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_group["train"],
        eval_dataset=ds_group["validation"]
    )

    
    print("\n-----------------------\n")
    print("Starting fine-tuning...")
    trainer.train()
    print("Training finished.")
    
    
    print("\n-----------------------\n")
    print("Pushing model to HuggingFace Hub...")
    try:
        trainer.push_to_hub()
        print("Model pushed to HuggingFace Hub.")
    except Exception as e:
        print(f"Error when pushing model to HuggingFace Hub: {e}")
        print("Saving model locally instead...")
        trainer.save_model(finetune_config.model_dir)
    print("\n-----------------------\n")
    print("Done.")
    
    return


if __name__ == "__main__":
    typer.run(main)
