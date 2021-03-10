import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import *


class EncoderBlock(nn.Module):
    """Starts with positional encoding, followed by multiple convolutional layers.
        Then, a self-attention, and finally a feefoward layer.
        Between each is a layer-norm.
        Note: Each of these is placed within a residual block
        Note: filters is the dimensionality of this block, i.e., input and output size

        Args:
            inp_dim (int): Dimension of the input (after positional encoding)
            num_conv (int): Number of initial convolutional layers
            kernel (int): Kernel size of each convolution
            filters (int): Number of filters
            num_heads (int): Number of heads for self-attention
            dropout_p (float): Dropout probability for positional encoding
            dropout (float): Dropout probability for feedforward layer
            max_len (int): Maximum length for positional encoding

       Note: output of this layer should equal the number of filters
     """

    def __init__(self, inp_dim, num_conv=2, kernel=7, num_heads=1, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.pos_enc = PositionalEncoding(inp_dim)
        self.num_conv = num_conv
        self.dropout = dropout
        # depthwise separable cnn layer for fewer parameters
        self.conv_layers = nn.ModuleList([nn.Sequential(ds_conv(input_channel=inp_dim, output_channel=inp_dim, k_size=kernel),
                                              nn.ReLU()) for _ in range(num_conv)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(inp_dim) for _ in range(num_conv)])
        self.conv_layer1 = nn.Sequential(ds_conv(input_channel=inp_dim, output_channel=inp_dim, k_size=kernel), nn.ReLU())
        self.conv_layer2 = nn.Sequential(ds_conv(input_channel=inp_dim, output_channel=inp_dim, k_size=kernel), nn.ReLU())
        self.att_layer_norm = nn.LayerNorm(inp_dim)
        self.self_attention = MHA(inp_dim, num_heads, attn_pdrop=dropout, resid_pdrop=dropout)
        self.feed_layer_norm = nn.LayerNorm(inp_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, padding_mask, current_layer, total_layer, test=False):
        x = self.pos_enc(x)
        for i in range(self.num_conv):
            y = self.layer_norms[i](x)
            if i % 2 == 0:
                y = self.dropout(y)
            y = y.permute(0, 2, 1)
            y = self.conv_layers[i](y)
            y = y.permute(0, 2, 1)
            x = stochastic_dropout(x, y, current_layer+i, total_layer, test=test)
        # Self-Attention
        y = self.att_layer_norm(x)
        y = self.dropout(y)
        y = self.self_attention(y, padding_mask)
        x = x + y
        # Feedforward
        y = self.feed_layer_norm(x)
        y = self.dropout(y)
        y = y.permute(0, 2, 1)
        y = self.conv_layer1(y)
        y = self.conv_layer2(y)
        y = y.permute(0, 2, 1)

        x = stochastic_dropout(x, y, current_layer+self.num_conv+1, total_layer, test=test)
        return x


class ds_conv(nn.Module):
    """ Depthwise Separable Convolution for input into encoder block within
            embedding encoder layer
            In the original paper, kernel size is set to 7.

            Args:
                input_channel: number of input channel
                output_channel: number of output channel
                k_size: kernel size
    """

    def __init__(self, input_channel, output_channel, k_size):
        super(ds_conv, self).__init__()
        if k_size % 2 == 0:
            raise Exception("kernel size doesn't guarantee same input volume and output volume")
        self.depthwise = nn.Conv1d(input_channel, input_channel, kernel_size=k_size, padding=k_size // 2,
                                   groups=input_channel)
        self.pointwise = nn.Conv1d(input_channel, output_channel, kernel_size=1, groups=1)
        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity="relu")
        nn.init.zeros_(self.depthwise.weight)
        nn.init.zeros_(self.pointwise.weight)

    def forward(self, x):
        x = self.pointwise(self.depthwise(x))
        return x


class PositionalEncoding(nn.Module):
    """Positional Encoding module for input into encoder block within
       embedding encoder layer.

       Reference:  https://pytorch.org/tutorials/beginner/transformer_tutorial.html

       Args:
           emb_dim (int): Embedding dimension for positional encoding.
                          Equals dimension of input.
    """

    def __init__(self, hidden_size, seq_len = 450, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        table = create_sinusoidal_table(seq_len, hidden_size).unsqueeze(0)
        self.register_buffer("pos_table", table)

    def forward(self, x):
        x = x + self.pos_table[:, x.size(1), :]
        x = self.dropout(x)
        return x


def create_sinusoidal_table(seq_len, hidden_size):
    table = torch.zeros(seq_len, hidden_size)
    pos = torch.tensor(list(range(seq_len))).unsqueeze(1)
    half_dim = torch.tensor(list(range(hidden_size // 2)))
    half_dim = torch.exp(-2 / hidden_size * np.log(10000) * half_dim.reshape(1, hidden_size // 2))
    table[:, 0::2] = torch.sin(pos / half_dim)
    table[:, 1::2] = torch.cos(pos / half_dim)
    return table


class MHA(nn.Module):
    """
    multiheaded attention with mask adapted from hw 5 Causal MHA.
    """

    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.layer_norm = nn.LayerNorm(n_embd)
        self.n_head = n_head

    def forward(self, x, mask):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = mask.unsqueeze(2)
        mask = mask.repeat(1, 1, mask.shape[1])
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, self.n_head, 1, 1)
        att = att.masked_fill(mask == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_drop(y)

        return y

def stochastic_dropout(identity, output, cur_depth, total_depth, test=False, p_last=0.9):
    if test:
        return output + identity

    prop = 1 - (cur_depth/total_depth) * (1 - p_last)
    b = np.random.binomial(1, prop)
    if b == 0:
        return identity
    return output + identity


