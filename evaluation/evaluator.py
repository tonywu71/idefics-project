from typing import Dict, List, Any, Optional
from tqdm.auto import tqdm

import json
import pandas as pd

from transformers.models.idefics.modeling_idefics import IdeficsForVisionText2Text
from transformers.models.idefics.processing_idefics import IdeficsProcessor

from datasets import Dataset
import evaluate


def eval_idefics(model: IdeficsForVisionText2Text,
                 processor: IdeficsProcessor,
                 ds: Dataset,
                 prompt_template: str,
                 task: str = "text_generation",
                 save_preds: bool = False,
                 savepath: Optional[str] = None,
                 generate_kwargs: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Evaluate an IDEFICS model on a DatasetGroup.
    """
    
    SUPPORTED_TASKS = ["text_generation"]
    
    if task == "text_generation":
        metric = evaluate.load("rouge")
    else:
        raise NotImplementedError(f"Task {task} not supported. Supported tasks are: {SUPPORTED_TASKS}")
    
    # Create placeholders for predictions and references:
    predictions, references = [], []
    
    # Iterate over the dataset:
    for sample in tqdm(ds, total=ds.num_rows, desc="Evaluating"):
        try:
            prompt = prompt_template.format(image=sample["image"])
        except KeyError:
            raise KeyError("Prompt template must contain the `{image}` keyword.")
        
        inputs = processor(prompt, return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(**inputs, **generate_kwargs) if generate_kwargs else model.generate(**inputs)
        # NOTE: `generated_ids` is the concatenation of the prompt (without the images) with the completion text.
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        predictions.append(generated_text)
        references.append(sample["text"])
    
    # Save predictions and references:
    if save_preds:
        print()
        save_preds_to_json(references, predictions, savepath)
        print(f"Exported references and predictions to `{savepath}`.")
    
    # Compute scores:
    results = pd.DataFrame(metric.compute(predictions=predictions, references=references))
    
    return results


def save_preds_to_json(references: List[str],
                       predictions: List[str],
                       savepath: str) -> None:
    """
    Export `references` and `predictions` to a JSON file.
    """
    data = {'references': references, 'predictions': predictions}
    with open(savepath, 'w') as file:
        json.dump(data, file)
    return
