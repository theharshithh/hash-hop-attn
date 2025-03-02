import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import numpy as np
from evals.eval_utils import MultiHopEval
from transformers import AutoTokenizer

class HashHopDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        seq_length: int,
        hops: int,
        hash_pair_str_length: int,
        chain_of_thought: bool,
        tokenizer
    ):
        self.samples = []
        self.tokenizer = tokenizer
        
        for _ in range(n_samples):
            sample = MultiHopEval.make_one(
                n_chars_problem=seq_length,
                num_queries=1,
                hops=hops,
                hash_pair_str_length=hash_pair_str_length,
                chain_of_thought=chain_of_thought
            )
            
            src_text = sample.prompt
            tgt_text = sample.completion.split("\n")[-1]
            
            src_tokens = self.tokenizer(src_text, add_special_tokens=True)["input_ids"]
            tgt_tokens = self.tokenizer(tgt_text, add_special_tokens=True)["input_ids"]
            
            self.samples.append((src_tokens, tgt_tokens))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_tokens, tgt_tokens = self.samples[idx]
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

def collate_fn(batch, tokenizer):
    src_tensors, tgt_tensors = zip(*batch)
    
    padded_src = torch.nn.utils.rnn.pad_sequence(
        src_tensors, 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    )
    padded_tgt = torch.nn.utils.rnn.pad_sequence(
        tgt_tensors, 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    )
    
    return padded_src, padded_tgt

def create_dataloaders(
    n_train_samples: int,
    n_val_samples: int,
    seq_length: int,
    hops: int,
    hash_pair_str_length: int,
    chain_of_thought: bool,
    batch_size: int,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token 
    
    def batch_collate(batch):
        return collate_fn(batch, tokenizer)
    
    train_dataset = HashHopDataset(
        n_samples=n_train_samples,
        seq_length=seq_length,
        hops=hops,
        hash_pair_str_length=hash_pair_str_length,
        chain_of_thought=chain_of_thought,
        tokenizer=tokenizer
    )
    
    val_dataset = HashHopDataset(
        n_samples=n_val_samples,
        seq_length=seq_length,
        hops=hops,
        hash_pair_str_length=hash_pair_str_length,
        chain_of_thought=chain_of_thought,
        tokenizer=tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=batch_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=batch_collate
    )
    
    return train_loader, val_loader, tokenizer

if __name__ == "__main__":
    train_loader, val_loader, tokenizer = create_dataloaders(
        n_train_samples=1000,
        n_val_samples=100,
        seq_length=1000,
        hops=2,
        hash_pair_str_length=16,
        chain_of_thought=False, 
        batch_size=16,
        num_workers=0
    )
    
    test_text = "Hey how u doing? Let's test some special chars: →='"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    batch = next(iter(train_loader))
    print(f"\nBatch shape: {batch[0].shape}") #b,seq_len