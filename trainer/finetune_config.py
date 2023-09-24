from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class FinetuneConfig:
    """
    Config class for the IDEFICS finetuning.
    """
    
    experiment_name: str
    
    # ======== Training args ========
    model_dir: str  # must end with a slash
    dataset_name: str
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int  # https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#gradient-accumulation
    gradient_checkpointing: bool  # https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#gradient-checkpointing
    optim: str
    learning_rate: float
    warmup_steps: int
    eval_steps: int
    save_steps: int
    logging_steps: int
    num_train_epochs: int
    
    # ======== LoRA (optional) ========
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj"]
    lora_dropout: float = 0.05
    bias: str = "none"
    
    # ======== Other training args (optional) ========
    eval_first_step: bool = True
    lr_scheduler_type: str = "linear_with_warmup"  # see possible values at https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/optimizer_schedules#transformers.SchedulerType
    eval_num_beams: int = 1  # greedy search by default
    save_total_limit: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    save_final_model: bool = True
    
    
    def __post_init__(self) -> None:
        """Set default values and run sanity checks after initialization."""
        if self.early_stopping_patience is None:
            self.early_stopping_patience = -1
        return
    
    
    @staticmethod
    def from_yaml(config_file: str) -> "FinetuneConfig":
        """Parse the YAML config file and return a Config object"""
        assert Path(config_file).exists(), f"Fine-tune config file `{config_file}` does not exist."
        
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Convert types:
        config_dict["learning_rate"] = float(config_dict["learning_rate"])  # str -> float
        
        try:
            config = FinetuneConfig(**config_dict)
        except TypeError as e:
            raise TypeError(f"Error when parsing the fine-tune config file `{config_file}`: {e}")
        
        return config
