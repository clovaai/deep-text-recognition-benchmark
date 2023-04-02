"""
Referenced from https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51
"""

import torch
from torch import nn
import torch.nn.functional as F

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
        q_k[index] = -float('inf')

    # softmax (above)
    softmax = F.softmax(q_k, dim=-1)
    # MatMul (softmax res * V)
    result = torch.bmm(softmax, value)
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
    seq_len: int, dim_model: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """_summary_

    https://miro.medium.com/v2/resize:fit:640/format:webp/1*C3a9RL6-SFC6fW8NGpJg5A.png

    Args:
        seq_len (int): _description_
        dim_model (int): _description_
        device (torch.device, optional): _description_. Defaults to torch.device("cpu").

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

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        # print(f'{tgt.shape = }, {mask.shape = }, {memory.shape = }')
        # print(f'starting first attention')
        tgt = self.attention_1(tgt, tgt, tgt, mask)
        # print(f'finished first attention')
        # print(f'{tgt.shape = }, {mask.shape = }, {memory.shape = }')
        tgt = self.attention_2(tgt, memory, memory)
        # print(f'finished second attention')
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