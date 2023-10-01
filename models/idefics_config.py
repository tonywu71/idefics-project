from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List
import yaml


@dataclass
class IDEFICSConfig:
    """
    Config class for the IDEFICS model.
    """
    # ======== Generic ========
    checkpoint: str
    fp16: bool = False
    bf16: bool = False
    
    # ======== Quantization ========
    load_in_4_bits: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    llm_int8_skip_modules: List[str] = field(default_factory=lambda: ["lm_head", "embed_tokens"])  # despite the name, for 4bit quantization
    
    
    def __post_init__(self) -> None:
        """Set default values and run sanity checks after initialization."""
        pass
    
    
    def sanity_check(self) -> None:
        """Run sanity checks for the IDEFICSConfig instance."""
        assert not (self.fp16 and self.bf16), "fp16 and bf16 cannot be both enabled."
        return
    
    
    @staticmethod
    def from_yaml(config_file: str) -> "IDEFICSConfig":
        """Parse a YAML config file into an IDEFICSConfig object."""
        assert Path(config_file).exists(), f"IDEFICS config file `{config_file}` does not exist."
        
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        try:
            config = IDEFICSConfig(**config_dict)
        except TypeError as e:
            raise TypeError(f"Error when parsing the IDEFICS config file `{config_file}`: {e}")
        
        return config
