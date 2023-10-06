from typing import Dict, Optional, Any
from abc import ABC, abstractmethod
from functools import partial
from tqdm.auto import tqdm

from datasets import Dataset
from transformers import AutoProcessor, BatchEncoding
from transformers.models.idefics.processing_idefics import IdeficsProcessor


class BaseDatasetGroup(ABC):
    """
    Base class for dataset groups.
    
    A dataset group is a collection of preprocessed splits from a dataset. They can
    be used to train a model using the HuggingFace Trainer out-of-the-box.
    
    Note: This class can be seen as a custom `DatasetDict` class.
    """
    
    def __init__(self,
                 processor_checkpoint: str = "HuggingFaceM4/idefics-9b",
                 streaming: bool = False) -> None:
        self.processor_checkpoint = processor_checkpoint
        self.streaming = streaming
        self.splits: Dict[str, Dataset] = {}
        self._processor: Optional[IdeficsProcessor] = None
        return
    
    
    @property
    def processor(self) -> IdeficsProcessor:
        """
        Return the processor associated to the class.
        """
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.processor_checkpoint)
        return self._processor
    
    
    @property
    @abstractmethod
    def path(self) -> str:
        """
        Return the model associated to the class.
        """
        return self.path
    
    
    def prepare_data(self, num_proc: int = 1, verbose: bool = False) -> None:
        """
        Prepare the data for all splits in the dataset group.
        """
        prepare_dataset = partial(self.prepare_dataset_fn, processor=self.processor)
        
        tbar = tqdm(self.splits.items(), desc="Preparing data...", disable=not(verbose))
        for split_name, ds in tbar:
            self.splits[split_name] = ds.map(prepare_dataset, batched=True, num_proc=num_proc)
        return
    
    
    @staticmethod
    @abstractmethod
    def prepare_dataset_fn(batch: Dict[str, Any], processor: IdeficsProcessor) -> BatchEncoding:
        """
        Preprocess the data for the input dataset. This method should be used with
        `functools.partial` and then passed to `Dataset.map`.
        """
        pass
    
    
    def __getitem__(self, split: str):
        return self.splits[split]
    
    
    def keys(self):
        return self.splits.keys()
    
    
    def items(self):
        return self.splits.items()
