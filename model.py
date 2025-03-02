import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer_model import FeedForward, LayerNorm, InputEmbedding, PE, ProjectionLayer
import random

class HashHopAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.hash_gate = nn.Linear(d_model, 1)
        self.hash_projection = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None, hop_count=1):
        batch_size = q.size(0)
        
        Q = self.W_q(q)  # (batch, seq_len, d_model)
        K = self.W_k(k)
        V = self.W_v(v)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        hash_gates = torch.sigmoid(self.hash_gate(q))  # (batch, seq_len, 1)
        hash_gates = hash_gates.view(batch_size, 1, -1, 1)  # Reshape for broadcasting
        
        current_state = Q
        attention_weights_list = []
        # for hashhops. 
        for hop in range(hop_count):
            # scores = Q * K.T / sqrt(d_k)
            scores = torch.matmul(current_state, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = attention_weights * hash_gates
            attention_weights = self.dropout(attention_weights)
            attention_weights_list.append(attention_weights)
            
            context = torch.matmul(attention_weights, V)
            
            if hop < hop_count - 1:
                current_state = self.hash_projection(
                    context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
                ).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attention_weights_list

class HashHopBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.hash_attention = HashHopAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, hop_count=1):
        attn_out, attention_weights = self.hash_attention(
            self.norm1(x), self.norm1(x), self.norm1(x), 
            mask=mask, hop_count=hop_count
        )
        x = x + self.dropout(attn_out)
        
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x, attention_weights

class HashHopTransformer(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        d_model=512, 
        num_heads=8, 
        num_layers=6, 
        d_ff=2048, 
        max_seq_len=1024,
        dropout=0.1,
        pad_token_id=None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.embedding = InputEmbedding(d_model, vocab_size)
        self.positional_encoding = PE(d_model, max_seq_len, dropout)
        
        # Hash Hop specific layers
        self.layers = nn.ModuleList([
            HashHopBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = LayerNorm(d_model)
        self.projection = ProjectionLayer(d_model, vocab_size)
        
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None, hop_count=1):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        attention_maps = []
        
        for layer in self.layers:
            x, attention_weights = layer(x, mask, hop_count)
            attention_maps.append(attention_weights)
        
        x = self.final_norm(x)
        output = self.projection(x)
        
        return output, attention_maps


