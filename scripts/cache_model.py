import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from pprint import pprint

from transformers import IdeficsForVisionText2Text, BitsAndBytesConfig

from models.idefics_config import IDEFICSConfig


def main(idefics_config_path: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True,
                                                  help="Path to the IDEFICSConfig file.")):
    """
    Load a given model in order to download and save it in the HuggingFace `hub` cachedir.
    """
    
    print("\n-----------------------\n")
    
    # ======== Load configs ========
    idefics_config = IDEFICSConfig.from_yaml(str(idefics_config_path))
    print("Configs loaded.")
    print()
    
    print("IDEFICSConfig:")
    pprint(idefics_config)
    
    print("\n-----------------------\n")
    
    
    # ======== Load model ========
    if idefics_config.load_in_4_bits:
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
    
    print(f"Model `{idefics_config.checkpoint}` successfully cached.")
    print("\n-----------------------\n")
    
    return


if __name__ == "__main__":
    typer.run(main)
