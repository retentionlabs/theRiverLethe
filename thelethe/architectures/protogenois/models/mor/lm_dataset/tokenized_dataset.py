import os

import numpy as np
import torch

from lm_dataset.util import MMapIndexedDataset


class TokenizedCorpus:
    # Load pre-tokenized corpus from memory.
    def __init__(self, token_data: np.ndarray, document_lengths: np.ndarray, document_indices: np.ndarray):
        """
        :param token_data: uint16 recommended
        :param document_lengths: uint16 recommended
        """
        self.token_data = token_data
        self.document_lengths = document_lengths
        self.document_indices = document_indices
        self.total_length = document_indices[-1] + document_lengths[-1]

    def __len__(self):
        # document count
        return self.document_indices.shape[0]

    def __getitem__(self, i: int):
        return self.token_data[self.document_indices[i]:self.document_indices[i] + self.document_lengths[i]].copy()


class TokenizedCorpusDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_corpus: TokenizedCorpus, length: int, eos_token: int, add_bos_token: bool = False, 
                 bos_token = None, transforms: list = None, seed: int = 42):
        """
        :param tokenized_corpus:
        :param length:
        :param eos_token: token to place after documents
        iteration may be packed together with documents from the previous iteration.
        :param transforms: to apply to each sample
        :param seed:
        """
        self.tokenized_corpus = tokenized_corpus
        self.length = length
        self.eos_token = eos_token
        self.add_bos_token = add_bos_token
        if self.add_bos_token:
            assert bos_token is not None
        self.bos_token = bos_token
        self.transforms = transforms
        self.seed = seed

        self._prepare_indices()

    def __len__(self):
        # the remaining sequences of the last document are not used
        if not self.add_bos_token:
            return self.padded_total_length // self.length
        else:
            return self.padded_total_length // (self.length - 1)

    def __getitem__(self, idx):
        input_ids = torch.full([self.length], -1, dtype=torch.long)
        attention_mask = torch.full([self.length], -1, dtype=torch.long)
        
        if not self.add_bos_token:
            sample_length = 0  # current length of the sample
            corpus_index = idx * self.length  # current index in the corpus
        else:
            input_ids[0] = self.bos_token
            attention_mask[0] = 1
            
            sample_length = 1
            corpus_index = idx * (self.length - 1)

        document_index = np.searchsorted(self.padded_document_indices, corpus_index, side="right") - 1
        assert 0 <= document_index < self.tokenized_corpus.document_indices.shape[0]

        # fill sample
        while sample_length < self.length:        
            # current_index_in_document refers to the current location with respect to the start of the actual document
            # not the start of the padded document
            current_index_in_document = corpus_index - self.padded_document_indices[document_index]
            sample_remaining = self.length - sample_length

            # assert current_index_in_document > self.tokenized_corpus.document_lengths[document_index]
            
            if current_index_in_document < self.tokenized_corpus.document_lengths[document_index]:
                # insert document
                document_remaining = self.tokenized_corpus.document_lengths[document_index] - current_index_in_document
                sample_remaining = self.length - sample_length
                copy_length = min(document_remaining, sample_remaining)

                copy_start_index = int(self.tokenized_corpus.document_indices[document_index] + current_index_in_document)
                data = self.tokenized_corpus.token_data[copy_start_index:copy_start_index + copy_length]
                input_ids[sample_length:sample_length + copy_length] = torch.from_numpy(np.array(data, dtype=np.int64))
                attention_mask[sample_length:sample_length + copy_length] = 1

                corpus_index += copy_length
                sample_length += copy_length
                
            elif current_index_in_document == self.tokenized_corpus.document_lengths[document_index]:
                # insert eos or bos token
                if not self.add_bos_token:
                    assert self.eos_token is not None
                    input_ids[sample_length] = self.eos_token
                else:
                    input_ids[sample_length] = self.bos_token
                    
                attention_mask[sample_length] = 1
                sample_length += 1
                corpus_index += 1                
                document_index += 1

        assert (input_ids == -1).sum() == 0
        assert (attention_mask == -1).sum() == 0
        assert (attention_mask == 1).all()
        sample = {
            "index": idx,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)
        return sample

    def _prepare_indices(self):        
        # Takes about 3 seconds for deduped The Pile (~134M documents)
        self.padded_document_lengths = self.tokenized_corpus.document_lengths + 1

        cumsum = np.cumsum(np.concatenate([[0], self.padded_document_lengths]), dtype=np.int64)
        self.padded_total_length = cumsum[-1]
        self.padded_document_indices = cumsum[:-1]
        
        
class PythiaPileTokenizedCorpus(TokenizedCorpus):
    # tokenized corpus of Pile dataset using Pythia tokenizer
    def __init__(self, pythia_pile_idxmaps_path: str):
        self.path = os.path.join(pythia_pile_idxmaps_path, "pile_0.87_deduped_text_document")
        self.dataset = MMapIndexedDataset(self.path, skip_warmup=True)

        token_data = np.memmap(self.path + ".bin", dtype="uint16", mode="r", order="C")
        document_lengths = self.dataset._index._sizes
        document_indices = self.dataset._index._pointers // 2
        super().__init__(token_data, document_lengths, document_indices)