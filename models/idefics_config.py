from dataclasses import dataclass
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
    torch_dtype: str = "float16"
    device_map: str = "auto"
    
    # ======== Quantization ========
    load_in_4_bits: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bf16"
    llm_int8_skip_modules: List[str] = ["lm_head", "embed_tokens"]  # despite the name, for 4bit quantization
    
    
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
