from typing import List, Iterator, Dict, Any

import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
from datasets import Dataset
from transformers import PreTrainedTokenizer

from lm_dataset.util import load_python_edu_text


class LanguageModelingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer: PreTrainedTokenizer, max_length, continuous=True, 
                 buffer_size=2 ** 22, seed=42, transforms: list = None, global_shuffling=True, 
                 local_shuffling=True, add_bos_token=False,):
        # Note that types of dataset and dataset_text_field are list to handle combined pretraining corpus.
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.continuous = continuous
        self.buffer_size = buffer_size
        self.seed = seed
        self.transforms = transforms
        self.global_shuffling = global_shuffling
        self.local_shuffling = local_shuffling
        self.rng = random.Random(seed)

        if self.tokenizer.padding_side == "left":
            raise ValueError("The tokenizer padding side must be right.")

        if self.tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must have an eos token")
        
        self.add_bos_token = add_bos_token
        if self.add_bos_token:
            self.max_length -= 1
            
        self._loaded_state = None

    def __len__(self):
        # Not the actual number of packed samples
        return len(self.dataset)

    def state_dict(self) -> Dict[str, Any]:
        """Save the iteration state"""
        return {
            "epoch": getattr(self, 'epoch', 0),
            "iter_step": getattr(self, 'iter_step', 0),
            "token_buffer": getattr(self, 'token_buffer', []),
            "full_samples": getattr(self, 'full_samples', None),
            "full_samples_idx": getattr(self, 'full_samples_idx', 0),
            "total_yield_index": getattr(self, 'total_yield_index', 0),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the iteration state"""
        self._loaded_state = state_dict

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:       
        data_remaining = True
        dataset = self.dataset
        if self.local_shuffling:
            local_rng = np.random.RandomState(self.seed)
        buffer: List[str] = []
        current_buffer_size: int = 0  # in number of characters
            
        if self._loaded_state is None:
            self.epoch, self.iter_step = 0, 0
            self.token_buffer: List[int] = []
            self.full_samples = None
            self.full_samples_idx, self.total_yield_index = 0, 0
            self.resume = False
        else:
            self.epoch = self._loaded_state["epoch"]
            self.iter_step = self._loaded_state["iter_step"]
            self.token_buffer = self._loaded_state["token_buffer"]
            self.full_samples = self._loaded_state["full_samples"]
            self.full_samples_idx = self._loaded_state["full_samples_idx"]
            self.total_yield_index = self._loaded_state["total_yield_index"]
            self.resume = True
                
        if self.global_shuffling:
            dataset = self.dataset.suffle(seed=self.seed + self.epoch)
        iterator = iter(dataset)    
        
        if self.iter_step > 0:            
            # Skip to the last loaded sample
            for _ in range(self.iter_step):
                next(iterator)
        
        while data_remaining:
            if not self.resume:
                # Fill text buffer
                while current_buffer_size < self.buffer_size:
                    try:
                        sample = next(iterator); self.iter_step += 1
                        
                        if "blob_id" in sample:
                            # in case of python-edu dataset, we need to load decoded text files
                            sample = load_python_edu_text(sample)
                            if sample["download_success"]:
                                sample = sample["text"]
                            else:
                                continue
                        else:
                            sample = sample["text"]
                        
                        if not self.add_bos_token:
                            sample = sample + self.tokenizer.eos_token
                        else:
                            sample = sample + self.tokenizer.bos_token
                        buffer.append(sample)
                        current_buffer_size += len(sample)
                    except StopIteration:
                        if self.continuous:
                            self.epoch += 1
                            if self.global_shuffling:
                                dataset = self.dataset.shuffle(seed=self.seed + self.epoch)
                            iterator = iter(dataset)
                            self.iter_step = 0
                        else:
                            data_remaining = False
                            break

                # Tokenize (move data from buffer to token_buffer)
                # TODO: optimize this (tokenization could be handled in a background thread)
                tokenized = self.tokenizer(buffer, add_special_tokens=False)["input_ids"]

                buffer = []
                current_buffer_size = 0

                for sample in tokenized:
                    self.token_buffer.extend(sample)

                if not self.add_bos_token:
                    effective_max_length = self.max_length
                else:
                    effective_max_length = self.max_length - 1
                
                # Stack full samples from token buffer
                n_full_samples = len(self.token_buffer) // effective_max_length
                if n_full_samples == 0:
                    continue
                full_samples = torch.LongTensor(self.token_buffer[:n_full_samples * effective_max_length])
                full_samples = full_samples.reshape(n_full_samples, effective_max_length)
                if self.add_bos_token:
                    bos_tokens = torch.LongTensor([self.tokenizer.bos_token_id] * n_full_samples).unsqueeze(1)
                    full_samples = torch.cat([bos_tokens, full_samples], dim=1)
                if self.local_shuffling:
                    full_samples = full_samples[local_rng.permutation(range(n_full_samples))]
                self.full_samples = full_samples
                self.token_buffer = self.token_buffer[n_full_samples * effective_max_length:]
                
            else:
                # yield dummay sample
                final_sample = None
                for _ in range(self.total_yield_index):
                    if final_sample is None:
                        input_ids = self.full_samples[0]
                        attention_mask = torch.ones_like(input_ids)
                        final_sample = {"input_ids": input_ids, "attention_mask": attention_mask}
                        if self.transforms is not None:
                            for transform in self.transforms:
                                final_sample = transform(final_sample)
                    yield final_sample
            
            for i, input_ids in enumerate(self.full_samples):
                if self.resume:
                    if i < self.full_samples_idx:
                        continue
                    elif i == self.full_samples_idx:
                        self.resume = False
                        continue
                self.full_samples_idx = i
                self.total_yield_index += 1
                
                attention_mask = torch.ones_like(input_ids)
                final_sample = {"input_ids": input_ids, "attention_mask": attention_mask}
                if self.transforms is not None:
                    for transform in self.transforms:
                        final_sample = transform(final_sample)
                yield final_sample

