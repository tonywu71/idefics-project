from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class InferenceConfig:
    """
    Config class for the IDEFICS inference.
    """    
    # ======== Optional ========
    max_new_tokens: int = 50
    num_beams: int = 1  # greedy search by default
    seed: Optional[int] = None
    
    @staticmethod
    def from_yaml(config_file: str) -> "InferenceConfig":
        """Parse the YAML config file and return a Config object"""
        assert Path(config_file).exists(), f"Inference config file `{config_file}` does not exist."
        
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        try:
            config = InferenceConfig(**config_dict)
        except TypeError as e:
            raise TypeError(f"Error when parsing the inference config file `{config_file}`: {e}")
        
        return config
