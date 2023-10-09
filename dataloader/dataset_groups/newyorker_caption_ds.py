from typing import Dict, Any

import torchvision.transforms as transforms
from transformers import BatchEncoding
from transformers.models.idefics.processing_idefics import IdeficsProcessor
from datasets import Dataset, load_dataset

from dataloader.base_dataset_group import DatasetGroup
from dataloader.image_utils import convert_to_rgb


class NewYorkerCaption(DatasetGroup):
    """
    Dataset group for the NewYorker Caption Contest dataset.
    Original dataset: `jmhessel/newyorker_caption_contest`.
    """
    def __init__(self,
                 processor_checkpoint: str = "HuggingFaceM4/idefics-9b",
                 streaming: bool = False,
                 verbose: bool = False) -> None:
        super().__init__(processor_checkpoint=processor_checkpoint,
                         streaming=streaming,
                         verbose=verbose)
        self.name = "explanation"
        
        if verbose:
            print(f"Loading dataset splits from {self.path}...")
        self.load_dataset_splits()
        
        if verbose:
            print("Preparing dataset...")
        self.prepare_data()
        
        return
    
    
    @property
    def path(self) -> str:
        """
        Return the model associated to the class.
        """
        if not hasattr(self, "_path"):
            self._path = "jmhessel/newyorker_caption_contest"
        return self._path

    
    def load_dataset_splits(self) -> None:
        self.splits = {
            "train": load_dataset(self.path, name=self.name, streaming=self.streaming, split="train"),
            "validation": load_dataset(self.path, name=self.name, streaming=self.streaming, split="validation"),
            "test": load_dataset(self.path, name=self.name, streaming=self.streaming, split="test")
        }
        return
    
    
    @staticmethod
    def prepare_dataset_fn(batch: Dict[str, Any], processor: IdeficsProcessor) -> BatchEncoding:
        """
        Preprocess the data for the `PokemonCardsDataset`.
        """
        image_size = processor.image_processor.image_size
        image_mean = processor.image_processor.image_mean
        image_std = processor.image_processor.image_std

        image_transform = transforms.Compose([
            convert_to_rgb,
            transforms.RandomResizedCrop((image_size, image_size),
                                         scale=(0.9, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])

        prompts = []
        for i in range(len(batch["image_uncanny_description"])):
            caption = batch["image_uncanny_description"][i].split(".")[0]
            prompts.append(
                [
                    batch["image"][i],
                    f"Question: How is this picture uncanny? Answer: {caption}"
                ],
            )

        inputs = processor(prompts, transform=image_transform, return_tensors="pt")

        # We use the same input and label IDs because we are using a decoding autoregressive model:
        inputs["labels"] = inputs["input_ids"]

        return inputs
