from typing import Dict, Any

import torchvision.transforms as transforms
from transformers import BatchEncoding
from transformers.models.idefics.processing_idefics import IdeficsProcessor
from datasets import Dataset, load_dataset

from dataloader.base_dataset_group import DatasetGroup
from dataloader.image_utils import convert_to_rgb


class PokemonCards(DatasetGroup):
    """
    Dataset group for the PokemonCards dataset.
    Original dataset: `TheFusion21/PokemonCards`.
    """
    def __init__(self,
                 processor_checkpoint: str = "HuggingFaceM4/idefics-9b",
                 streaming: bool = False,
                 verbose: bool = False,
                 train_size: float = 0.89,
                 validation_size: float = 0.01,
                 test_size: float = 0.10,
                 seed: int = 42) -> None:
        super().__init__(processor_checkpoint=processor_checkpoint,
                         streaming=streaming,
                         verbose=verbose)
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.seed = seed
        
        assert self.train_size + self.validation_size + self.test_size == 1.0, \
            "The sum of the splits must be equal to 1."
        
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
            self._path = "tonywu71/PokemonCards_fixed"
        return self._path

    
    def load_dataset_splits(self) -> None:
        ds = load_dataset(self.path, streaming=self.streaming, split="train")
        assert isinstance(ds, Dataset)  # only one split is loaded here
        
        # Split the dataset into train, validation and test sets:
        ds = ds.train_test_split(test_size=self.test_size, shuffle=True, seed=self.seed)
        ds_test = ds["test"]
        ds_train_val = ds["train"].train_test_split(test_size=(1/self.test_size) * self.validation_size,
                                                    shuffle=True,
                                                    seed=self.seed)
        
        self.splits = {
            "train": ds_train_val["train"],
            "validation": ds_train_val["test"],
            "test": ds_test,
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
        for i in range(len(batch['caption'])):
            # NOTE: We split the captions to avoid having very long examples, which would require more GPU RAM during training.
            caption = batch['caption'][i].split(".")[0]
            prompts.append(
                [
                    batch['image_url'][i],
                    f"Question: What's on the picture? Answer: This is {batch['name'][i]}. {caption}</s>",
                ],
            )

        inputs = processor(prompts, transform=image_transform, return_tensors="pt")

        # We use the same input and label IDs because we are using a decoding autoregressive model:
        inputs["labels"] = inputs["input_ids"]

        return inputs
