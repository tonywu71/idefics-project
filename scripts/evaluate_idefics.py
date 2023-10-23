import typer

import os, sys

from dataloader.dataset_group_loader import DATASET_NAME_TO_LOAD_FUNC
from evaluation.evaluator import eval_idefics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from pprint import pprint

import torch
from transformers import (IdeficsForVisionText2Text,
                          AutoProcessor,
                          BitsAndBytesConfig)
from peft.peft_model import PeftModel

from models.idefics_config import IDEFICSConfig
from models.inference_config import InferenceConfig
from utils.constants import BAD_WORDS, EOS_TOKEN, OUTPUT_DIRPATH


def main(idefics_config_path: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True,
                                                  help="Path to the IDEFICSConfig file."),
         inference_config_path: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True,
                                                    help="Path to the InferenceConfig file."),
         dataset_name: str = typer.Option(..., help="Name of the dataset to evaluate on."),
         split: str = typer.Option("test", help="Split of the dataset to evaluate on. Defaults to `test`."),
         prompt_template: str = typer.Option(..., help="Prompt template to use for inference."),
         save_preds: bool = typer.Option(True, help="Whether to save predictions or not.")):
    """
    Run evaluation on the IDEFICS model.
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
    
    print("Prompt template:")
    print(prompt_template)
    
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
    
    
    if idefics_config.lora_checkpoint:
        if idefics_config.load_in_4_bits:
            raise NotImplementedError("LoRA adapters are not supported with 4bit quantization.")
        print(f"Loading LoRA adapters from `{idefics_config.lora_checkpoint}`...")
        model = PeftModel.from_pretrained(model, model_id=idefics_config.lora_checkpoint)
        model = model.merge_and_unload()
        print("LoRA adapters loaded.")
    
    
    # ======== Load dataset ========
    ds = DATASET_NAME_TO_LOAD_FUNC[dataset_name]()[split]
    print(f"Dataset `{dataset_name}` ({split} split) loaded.")
    
    
    # ======== Evaluation ========
    bad_words_ids = tokenizer(BAD_WORDS, add_special_tokens=False).input_ids
    eos_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    
    generate_kwargs = dict(
        eos_token_id=[eos_token_id],
        bad_words_ids=bad_words_ids,
        max_new_tokens=inference_config.max_new_tokens,
        num_beams=inference_config.num_beams
    )
    
    if idefics_config.lora_checkpoint:
        savedir = OUTPUT_DIRPATH / f"{idefics_config.lora_checkpoint}-{dataset_name}_{split}"
    else:
        savedir = OUTPUT_DIRPATH / f"{idefics_config.checkpoint}-{dataset_name}_{split}"
    savedir.mkdir(parents=True, exist_ok=True)
    
    print("Evaluation...")
    results = eval_idefics(model=model,
                           processor=processor,
                           ds=ds,
                           prompt_template=prompt_template,
                           task="text_generation",
                           save_preds=save_preds,
                           savepath=str(savedir / "preds.json"),
                           generate_kwargs=generate_kwargs)

    print("\n-----------------------\n")
    
    print("Results:")
    print(results)
    
    print("\n-----------------------\n")
    
    results.to_csv(savedir / "results.csv")
    print(f"Results saved to `{savedir / 'results.csv'}`.")
    
    return


if __name__ == "__main__":
    typer.run(main)
