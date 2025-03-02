import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PE(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0,d_model, 2).float() * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term) # every alternative terms skipped by 2
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, seq_len, d_model) (b,t,c)
        self.register_buffer('pe', pe) #saved in state, not calc always

    @torch.no_grad()
    def forward(self, x):
        # self.pe -> (1, t, c)
        pe_slice = self.pe[:, :x.shape[1], :]
        x = x + pe_slice
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, d_model=512, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # mul
        self.bias = nn.Parameter(torch.zeros(d_model)) #added

    def forward(self, x):
        # x -> (b, t, c)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps) + self.bias

# this is a simple vanilla nn
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1,b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2,b2

    def forward(self, x):
        # x -> b,t,c = b, seq_len, d_model
        x = self.linear_1(x) # (b,t, d_ff)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x) # (b,t,c)
        return x

# multihead attn will have both single head, masked head
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(d_model, d_model) # (d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        #q,k,v dims -> (b, num_heads, seq_len, d_k)
        d_k = query.shape[-1]
        # attention is of dims (b, num_heads, seq_len, seq_len) -> for every batch, and head, we get a sq mtx of attn bw diff seqs.
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)# (b,num_heads, seq_len, dk) * (d,num_heads,dk, seq_len) -> (b,num_heads, seq_len, seq_len)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, float('-inf'))

        attention_scores = attention_scores.softmax(dim=-1) # batch, num_heads, seq_len, seq_len
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        attention_matrix = (attention_scores @ value) # (b, num_heads, seq_len, d_k) * (b, num_heads, seq_len, seq_len) ->
        # attention_matrix -> (b, num_heads, seq_len, d_k)
        return attention_matrix, attention_scores

    def forward(self,q,k,v, mask):
        #q -> (b,seq_len,d_model)
        query = self.W_q(q) # (b, seq_len, d_model) * (d_model, d_model) ->  (b, seq_len, d_model)
        key = self.W_k(k) # (b, seq_len, d_model) * (d_model, d_model) ->  (b, seq_len, d_model)
        value = self.W_v(v)
        # (b, seq_len, d_model) -> (b, seq_len, num_heads, dk) ->  transpose-> (batch, h, seq_len, dk)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1,2) # (b, seq_len, num_heads, d_k)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1,2) # (b, seq_len, num_heads, d_k)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1,2) # (b, seq_len, num_heads, d_k)

        x, self.attention_scores = Attention.attention(query, key, value, mask, self.dropout)
        # x -> b, num_heads, seq_len, d_k -> (b, seq_len, num_heads, d_k) -> b, seq_len, d_model
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.d_model)
        x = self.w_o(x)
        # x-> b,seq_len, d_model
        return x

class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sub_layer):
        residual_con = x + self.dropout(sub_layer(self.norm(x)))
        return residual_con


# residual block -> res_conn += attn+ dropout(norm(x))
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: Attention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self,x,src_mask):
        x = self.residual_connection[0](x, lambda x: self.attention_block(x,x,x, src_mask))
        x = self.residual_connection[1](x, lambda x: self.feed_forward_block(x))
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: Attention, cross_attention_block: Attention, feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward
        self.dropout = dropout
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_op, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x,encoder_op, encoder_op, src_mask))
        x = self.residual_connection[2](x, lambda x: self.feed_forward_block(x))

        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers= layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_op, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_op, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1) # (b,t,c) -> (b, t, vocab_size)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PE, tgt_pos: PE, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed, self.tgt_embed = src_embed, tgt_embed
        self.src_pos, self.tgt_pos = src_pos, tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)

    def decode(self, encoder_op, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_op, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, no_of_layers=6, no_of_heads=8,dropout=0.1, d_ff=2048):
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    src_pos = PE(d_model, src_seq_len,dropout)
    tgt_pos = PE(d_model, tgt_seq_len, dropout)

    #encoder blk
    encoder_blocks = []
    for _ in range(no_of_layers):
        encoder_self_attn = Attention(d_model, no_of_heads, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attn, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    #decoder blk
    decoder_blocks = []
    for _ in range(no_of_layers):
        decoder_self_attention = Attention(d_model,no_of_heads, dropout)
        decoder_cross_attention = Attention(d_model, no_of_heads, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

def build_decoder_only_transformer(vocab_size, seq_len, d_model=512, no_of_layers=6, no_of_heads=8, dropout=0.1, d_ff=2048):
    embed = InputEmbedding(d_model, vocab_size)
    pos = PE(d_model, seq_len, dropout)
    
    decoder_blocks = []
    for _ in range(no_of_layers):
        # self attn is what we need, encoder blk has the structure we want
        # so we use encoder blk as decoder blk
        self_attention = Attention(d_model, no_of_heads, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        
        decoder_block = EncoderBlock(self_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)
    
    decoder = Encoder(decoder_blocks)
    projection_layer = ProjectionLayer(d_model, vocab_size)
    
    class DecoderOnlyTransformer(nn.Module):
        def __init__(self, decoder, embed, pos, projection_layer):
            super().__init__()
            self.decoder = decoder
            self.embed = embed
            self.pos = pos
            self.projection_layer = projection_layer
        
        def forward(self, x, mask=None):
            x = self.embed(x)
            x = self.pos(x)
            x = self.decoder(x, mask)        
            return self.projection_layer(x)
    
    transformer = DecoderOnlyTransformer(decoder, embed, pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer

