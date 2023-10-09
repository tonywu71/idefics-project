import typer

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from typing import List
from pathlib import Path
from pprint import pprint

import torch
from transformers import (IdeficsForVisionText2Text,
                          AutoProcessor,
                          BitsAndBytesConfig)

from models.idefics_config import IDEFICSConfig
from models.inference_config import InferenceConfig
from utils.constants import BAD_WORDS, EOS_TOKEN


def main(idefics_config_path: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True,
                                                  help="Path to the IDEFICSConfig file."),
         inference_config_path: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True,
                                                    help="Path to the InferenceConfig file."),
         prompts: List[str] = typer.Option(..., help="Prompt to use for inference. Images and text " + \
                                           "must be separated in a list format.")):
    """
    Run inference on the IDEFICS model.
    """
    
    print("\n-----------------------\n")
    
    # ======== Load configs ========
    idefics_config = IDEFICSConfig.from_yaml(str(idefics_config_path))
    inference_config = InferenceConfig.from_yaml(str(inference_config_path))
    print("Configs loaded.")
    print()
    
    print("IDEFICSConfig:")
    pprint(idefics_config)
    
    print("InferenceConfig:")
    pprint(inference_config)
    
    print("\n-----------------------\n")
    
    
    # ======== Load model ========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if idefics_config.load_in_4_bits:
        assert device == "cuda", "4bit quantization is only supported on CUDA GPUs."
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
    
    processor = AutoProcessor.from_pretrained(idefics_config.checkpoint, use_auth_token=True)
    tokenizer = processor.tokenizer  # alias
    model = IdeficsForVisionText2Text.from_pretrained(idefics_config.checkpoint,
                                                      quantization_config=bnb_config,
                                                      device_map="auto")
    
    print(f"Model loaded from `{idefics_config.checkpoint}`.")
    print("\n-----------------------\n")
    
    
    # ======== Inference ========
    bad_words_ids = tokenizer(BAD_WORDS, add_special_tokens=False).input_ids
    eos_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    inputs = processor(prompts, return_tensors="pt").to(device)
    
    print("Inference...")
    generated_ids = model.generate(**inputs,
                                    eos_token_id=[eos_token_id],
                                    bad_words_ids=bad_words_ids,
                                    max_new_tokens=inference_config.max_new_tokens,
                                    num_beams=inference_config.num_beams)
    # NOTE: `generated_ids` is the concatenation of the prompt (without the images) with the completion text.
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n")
    print("Result:")
    print(f"```\n{generated_text}\n```")
    
    return


if __name__ == "__main__":
    typer.run(main)
