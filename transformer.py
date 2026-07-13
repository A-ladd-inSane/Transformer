import torch
import math
from torch import nn
import torch.nn.functional as F


class Embeddings(nn.Module):
    def __init__(self, vocab, dim):
        super(Embeddings, self).__init__()
        self.table = nn.Embedding(vocab, dim)
        self.dim = dim

    def forward(self, x):
        return self.table(x) * math.sqrt(self.dim)


class PE(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=1000):
        super(PE, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0., max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0., dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, h, dim, dropout=0.1):
        super(Attention, self).__init__()
        self.h = h
        self.dim = dim
        self.d_k = dim // h
        self.dropout = nn.Dropout(dropout)

        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.fc = nn.Linear(dim, dim, bias=False)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2).contiguous()

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

    def scaled_dot_product_attention(self, Q, K, V, mask=None, scale=None):
        if scale is None:
            scale = 1 / (math.sqrt(Q.size(-1)))
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn_weight = F.softmax(attn, dim=-1)
        attn_weight = self.dropout(attn_weight)
        out = torch.matmul(attn_weight, V)
        return out, attn_weight

    def forward(self, query, key_value=None, mask=None):
        if key_value is None:
            key_value = query

        Q = self.Wq(query)
        K = self.Wk(key_value)
        V = self.Wv(key_value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        out, attn_weight = self.scaled_dot_product_attention(Q, K, V, mask)

        out = self.combine_heads(out)
        out = self.fc(out)
        out = self.dropout(out)

        return out, attn_weight


class SelfAttention(Attention):
    def __init__(self, d_model, h=8, dropout=0.1):
        super().__init__(h, d_model, dropout)

    def forward(self, x, mask=None):
        return super().forward(x, None, mask)


class MultiHead(Attention):
    def __init__(self, h, dim, dropout=0.1):
        super(MultiHead, self).__init__(h, dim, dropout)

    def forward(self, query, key_value=None, mask=None):
        return super().forward(query, key_value, mask)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, dim, h, ff_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHead(h, dim, dropout)
        self.feed = FeedForward(dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, None, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ff_out = self.feed(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim, h, ff_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.s_attn = MultiHead(h, dim, dropout)
        self.c_attn = MultiHead(h, dim, dropout)
        self.feed = FeedForward(dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        s_attn_out, _ = self.s_attn(x, None, tgt_mask)
        x = self.norm1(x + self.dropout1(s_attn_out))

        c_attn_out, _ = self.c_attn(x, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(c_attn_out))

        ff_out = self.feed(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x


class Encoder(nn.Module):
    def __init__(self, dim, h, ff_dim, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(dim, h, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, dim, h, ff_dim, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(dim, h, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, dim=512, h=8, ff_dim=2048,
                 num_layers=6, dropout=0.1, max_len=1000):
        super(Transformer, self).__init__()

        self.encoder_embed = Embeddings(src_vocab, dim)
        self.decoder_embed = Embeddings(tgt_vocab, dim)
        self.pe = PE(dim, dropout, max_len)
        self.encoder = Encoder(dim, h, ff_dim, num_layers, dropout)
        self.decoder = Decoder(dim, h, ff_dim, num_layers, dropout)
        self.output_proj = nn.Linear(dim, tgt_vocab)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_tgt_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return ~mask

    def forward(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = self.generate_tgt_mask(tgt.size(1), tgt.device)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask & tgt_pad_mask

        # 修正：必须先 embedding 再加位置编码！
        src_embedded = self.encoder_embed(src)
        src_embedded = self.pe(src_embedded)

        tgt_embedded = self.decoder_embed(tgt)
        tgt_embedded = self.pe(tgt_embedded)

        encoder_output = self.encoder(src_embedded, src_mask)
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)

        output = self.output_proj(decoder_output)
        return output