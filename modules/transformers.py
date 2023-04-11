"""
Referenced from https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51
"""

import torch
from torch import nn
import torch.nn.functional as F
import math

class SinPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        if not batch_first:
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        self.batch_first = batch_first
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if self.batch_first:
            return self.pe[:x.size(1)]
        else:
            return self.pe[:x.size(0)]
        #return self.dropout(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, learnable: bool, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor):
        position_ids = torch.arange(x.shape[1], dtype=torch.long, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        position_ids = position_ids.unsqueeze(0).expand(x.shape)
        position_embeddings = self.embeddings(position_ids)
        return position_embeddings

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    """_summary_
    https://miro.medium.com/v2/resize:fit:720/format:webp/1*BzhKcJJxv974OxWOVqUuQQ.png

    Args:
        query (torch.Tensor): _description_
        key (torch.Tensor): _description_
        value (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    # MatMul( Q * K ^ T)
    # q_k: torch.Tensor = torch.bmm(query, key.transpose(1, 2))
    q_k: torch.Tensor = torch.matmul(query, key.transpose(1, 2))
    # Scale # (divide above by sqrt(dk))
    scale = query.size(-1) ** 0.5
    q_k /= scale

    # mask
    if mask is not None:
        # q_k = q_k.masked_fill(mask=mask, value=float('-inf'))
        # print(f'{q_k.shape = }')
        # print(f'{q_k = }')
        # raise Exception
        # q_k = q_k * mask
        batch_size = q_k.shape[0]
        index = (mask == 0).repeat(batch_size, 1, 1)
        q_k[index] = -1e9#-float('inf')
        # q_k[index] = -float('inf')
        # print(f'{torch.argmax(q_k, dim=-1) = }')
        # print(f'{q_k = }')
        # raise Exception

    # softmax (above)
    softmax = F.softmax(q_k, dim=-1)
    # MatMul (softmax res * V)
    # result = torch.bmm(softmax, value)
    result = torch.matmul(softmax, value)
    return result

class AttentionHead(nn.Module):
    """_summary_
    
    https://miro.medium.com/v2/resize:fit:720/format:webp/1*BzhKcJJxv974OxWOVqUuQQ.png

    Args:
        nn (_type_): _description_
    """
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value), mask)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        return self.linear(
            torch.cat([head(query, key, value, mask) for head in self.heads], dim=-1)
        )

def position_enccoding(
    seq_len: int, dim_model: int, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> torch.Tensor:
    """_summary_

    https://miro.medium.com/v2/resize:fit:640/format:webp/1*C3a9RL6-SFC6fW8NGpJg5A.png

    Args:
        seq_len (int): _description_
        dim_model (int): _description_
        device (torch.device, optional): _description_. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
        torch.Tensor: _description_
    """
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1,1, -1)
    phase = pos / (1e4 ** (dim // dim_model))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_input: int = 512, dim_feedforward: int =2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input)
    )

class AddNorm(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float=0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        dim_k = max(dim_model// num_heads, 1)
        dim_q = dim_k
        self.attention = AddNorm(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension = dim_model,
            dropout=dropout
        )
        self.feed_forward = AddNorm(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        seq_len, dimension= src.size(1), src.size(2)
        src += position_enccoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        dim_q = max(dim_model // num_heads, 1)
        dim_k = dim_q
        self.attention_1 = AddNorm(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout
        )
        self.attention_2 = AddNorm(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout
        )
        self.feed_forward = AddNorm(
            feed_forward(dim_model, dim_feedforward),
            dim_model,
            dropout,
        )

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor=None, debug: bool=False) -> torch.Tensor:
        if debug:
            print(f'{tgt.shape = }, {mask.shape = }, {memory.shape = }')
            print(f'starting first attention')
        tgt = self.attention_1(tgt, tgt, tgt, mask)
        if debug:
            print(f'finished first attention')
            print(f'{tgt.shape = }, {mask.shape = }, {memory.shape = }')
        tgt = self.attention_2(tgt, memory, memory)
        if debug:
            print(f'finished second attention')
        return self.feed_forward(tgt)
    
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        seq_len, dimension = tgt.size(1), tgt.size(2)
        tgt += position_enccoding(seq_len, dimension)
        for layer in self.layers:
            tgt = layer(tgt, memory, mask)
        
        return torch.softmax(self.linear(tgt), dim=-1)

class Transformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(src)
        return self.decoder(tgt, memory)

if __name__ == "__main__":
    src = torch.rand(64, 32, 512)
    tgt = torch.rand(64, 16, 512)
    out: torch.Tensor = Transformer()(src, tgt)
    print(out.shape)
    # torch.Size([64, 16, 512])

    encoder_out: torch.Tensor = TransformerEncoder()(
        src=src
    )

    print(f'{encoder_out.shape = }')

    decoder_out: torch.Tensor = TransformerDecoder()(
        tgt= torch.rand(1, 1, 512),
        memory= encoder_out[0].unsqueeze(0)
    )

    print(f'{decoder_out.shape = }')