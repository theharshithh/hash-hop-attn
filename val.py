import torch

def evaluate_hash_hopping(model, test_dataloader, device='cuda'):
    model.eval()
    results = {1: [], 2: [], 3: []}
    
    with torch.no_grad():
        for hop_count in [1, 2, 3]:
            correct = 0
            total = 0
            
            for src, tgt in test_dataloader:
                src, tgt = src.to(device), tgt.to(device)
                output, attention_maps = model(src, hop_count=hop_count)
                
                pred = output.argmax(dim=-1)
                
                correct += (pred == tgt).sum().item()
                total += tgt.numel()
                
                results[hop_count].append({
                    'attention_maps': [attn.cpu() for attn in attention_maps],
                    'accuracy': correct/total
                })
    
    return results