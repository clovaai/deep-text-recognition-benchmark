import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers import position_enccoding as sinosial_positional_encoding
from .transformers import TransformerDecoderLayer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha

class PositionalEmbedding(nn.Module):
    def __init__(self, learnable: bool, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, position_ids):
        position_embeddings = self.embeddings(position_ids)
        return position_embeddings


class TransformerDecoder(nn.Module):
    def __init__(
            self, learnable_embeddings: bool,
            num_output: int, seq_length: int, embedding_dim: int,
            # num_chars: int, 
            num_layers: int = 6, dim_model: int = 512,
            num_heads: int = 8, dim_feedforward: int = 2048,
            dropout: float = 0.1, device: torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ) -> None:
        super().__init__()
        self.device = device
        self.position_embeddings = PositionalEmbedding(learnable_embeddings, num_embeddings=seq_length, embedding_dim=embedding_dim)
        self.word_embeddings = nn.Embedding(num_embeddings=num_output, embedding_dim=embedding_dim)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                # nn.TransformerDecoderLayer(
                #     d_model=dim_model, nhead=num_heads, 
                #     dim_feedforward=dim_feedforward, dropout=dropout,
                #     batch_first=True
                # )
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(in_features=embedding_dim,out_features=num_output)
        self.learnable_embeddings = learnable_embeddings
        self.dim_model = dim_model

    def forward(self, input_ids: torch.Tensor, encoded_memory: torch.Tensor, mask: torch.Tensor=None):
        input_shape = input_ids.shape
        seq_length = input_shape[1]

        if self.learnable_embeddings:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
            positional_embeddings = self.position_embeddings(position_ids)
        else:
            positional_embeddings = sinosial_positional_encoding(seq_len=seq_length, dim_model=self.dim_model)

        input_embeddings = self.word_embeddings(input_ids)

        input_embeddings += positional_embeddings

        # print(f'starting first decoder layer {input_embeddings.shape = }')
        for layer in self.layers:
            input_embeddings = layer(input_embeddings, encoded_memory, mask)
            # print(f'going next decoder layer')
        
        output = self.linear(input_embeddings)
        output = torch.softmax(output, dim=-1)
        return output
        # print(f'finished layer, {input_embeddings.shape = }')
        # return torch.softmax(self.linear(input_embeddings), dim=-1)

    def generate_attn_mask(self, seq_len: int, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        if type(self.layers[0]) is nn.TransformerDecoderLayer:
            return nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        else:
            mask = torch.tril(torch.ones(seq_len, seq_len), device=device)
            # mask = mask.unsqueeze(0)
            return mask
