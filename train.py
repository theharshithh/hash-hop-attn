import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import math
import time
from pathlib import Path
import wandb
import numpy as np
from model import HashHopTransformer
from dataset import create_dataloaders
from evals.eval_utils import MultiHopEval

def train_epoch(model, train_loader, optimizer, device, hop_count, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        src, tgt = src.to(device), tgt.to(device)
        batch_size = src.size(0)
        
        src_mask = (src != model.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        optimizer.zero_grad()
        
        output, attention_maps = model(src, mask=src_mask, hop_count=hop_count)
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        loss = F.cross_entropy(
            output[:, :-1].reshape(-1, output.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=model.pad_token_id
        )
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        total_tokens += (tgt_output != model.pad_token_id).sum().item()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / total_tokens

def validate(model, val_loader, device, hop_count):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            batch_size = src.size(0)
            
            src_mask = (src != model.pad_token_id).unsqueeze(1).unsqueeze(2)
            
            output, _ = model(src, mask=src_mask, hop_count=hop_count)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            loss = F.cross_entropy(
                output[:, :-1].reshape(-1, output.size(-1)),
                tgt_output.reshape(-1),
                ignore_index=model.pad_token_id
            )
            
            predictions = output[:, :-1].argmax(dim=-1)
            mask = tgt_output != model.pad_token_id
            correct_predictions += (predictions == tgt_output)[mask].sum().item()
            total_tokens += mask.sum().item()
            
            total_loss += loss.item() * batch_size
    
    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = correct_predictions / total_tokens
    return avg_loss, accuracy

def evaluate_hash_hopping(model, tokenizer, device, num_samples=100):
    model.eval()
    results = []
    
    for hop_count in [1, 2, 3]:
        correct = 0
        total = 0
        
        for _ in range(num_samples):
            sample = MultiHopEval.make_one(
                n_chars_problem=1000,
                num_queries=1,
                hops=hop_count,
                hash_pair_str_length=16,
                chain_of_thought=False
            )
            
            inputs = tokenizer(sample.prompt, return_tensors='pt', add_special_tokens=True)
            src_tokens = inputs['input_ids'].to(device)
            src_mask = (src_tokens != model.pad_token_id).unsqueeze(1).unsqueeze(2)
            
            with torch.no_grad():
                output, _ = model(src_tokens, mask=src_mask, hop_count=hop_count)
                pred_tokens = output[0].argmax(dim=-1)
                prediction = tokenizer.decode(pred_tokens.tolist())
            
            for query, target in sample.targets.items():
                if target in prediction:
                    correct += 1
                total += 1
        
        accuracy = correct / total
        results.append((hop_count, accuracy))
        print(f"Hop count: {hop_count}, Accuracy: {accuracy:.4f}")
    
    return results

def train():
    config = {
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'warmup_steps': 4000,
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,
        'max_seq_len': 1024,
        'hash_pair_str_length': 16,
        'n_train_samples': 10000,
        'n_val_samples': 1000,
        'initial_hops': 1,
        'max_hops': 3
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader, tokenizer = create_dataloaders(
        n_train_samples=config['n_train_samples'],
        n_val_samples=config['n_val_samples'],
        seq_length=config['max_seq_len'],
        hops=config['initial_hops'],
        hash_pair_str_length=config['hash_pair_str_length'],
        chain_of_thought=False,
        batch_size=config['batch_size']
    )
    
    model = HashHopTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        pad_token_id=tokenizer.pad_token_id
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    wandb.init(project="hash-hop-attention", config=config)
    
    best_val_accuracy = 0
    current_hop_count = config['initial_hops']
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, current_hop_count)
        
        val_loss, val_accuracy = validate(model, val_loader, device, current_hop_count)
        
        hop_results = evaluate_hash_hopping(model, tokenizer, device)
        
        scheduler.step()
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }, 'best_hash_hop_model.pt')
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'current_hop_count': current_hop_count,
            **{f'hop_{hop}_accuracy': acc for hop, acc in hop_results}
        })
        
        if epoch > 0 and epoch % 10 == 0 and val_accuracy > 0.8:
            current_hop_count = min(current_hop_count + 1, config['max_hops'])
            print(f"Increasing hop count to {current_hop_count}")

if __name__ == '__main__':
    train()