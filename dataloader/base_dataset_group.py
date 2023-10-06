from typing import Dict, Optional, Any
from abc import ABC, abstractmethod
from functools import partial
from tqdm.auto import tqdm

from transformers import AutoProcessor, BatchEncoding
from transformers.models.idefics.processing_idefics import IdeficsProcessor
from datasets import Dataset

class BaseDatasetGroup(ABC):
    """
    Base class for dataset groups.
    
    A dataset group is a collection of preprocessed splits from a dataset. They can
    be used to train a model using the HuggingFace Trainer out-of-the-box.
    
    Note: This class can be seen as a custom `DatasetDict` class.
    """
    
    def __init__(self,
                 processor_checkpoint: str = "HuggingFaceM4/idefics-9b",
                 streaming: bool = False,
                 verbose: bool = False) -> None:
        self.processor_checkpoint = processor_checkpoint
        self.streaming = streaming
        self.verbose = verbose
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
    
    
    @abstractmethod
    def load_dataset_splits(self) -> None:
        """
        Load the dataset splits.
        NOTE: This method should be implemented by the child class as some datasets
        require manual splitting.
        """
        pass
    
    
    def prepare_data(self) -> None:
        """
        Prepare the data for all splits in the dataset group.
        """
        prepare_dataset = partial(self.prepare_dataset_fn, processor=self.processor)
        
        tbar = tqdm(self.splits.items(), desc="Preparing data...", disable=not(self.verbose))
        for split_name, ds in tbar:
            self.splits[split_name].set_transform(prepare_dataset)
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

    
    def __str__(self) -> str:
        return f"DatasetGroup({self.path})"


    def __repr__(self) -> str:
        repr_str = f"{type(self).__name__}({self.path}):\n"
        for split, ds in self.splits.items():
            repr_str += f"{split}: {len(ds)} samples\n"
        return repr_str
