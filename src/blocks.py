import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import *
import math


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

    def __init__(self, inp_dim, num_conv=2, kernel=7, num_heads=8,
                 dropout_p=0.1, dropout=0.5, max_len=5000):
        super(EncoderBlock, self).__init__()
        self.pos_enc = PositionalEncoding(inp_dim, dropout_p, max_len)
        self.num_conv = num_conv
        self.dropout = dropout
        # depthwise separable cnn layer for fewer parameters
        self.first_conv_layer = ds_conv(input_channel=inp_dim, output_channel=inp_dim, k_size=kernel)
        self.conv_layers = nn.ModuleList([ds_conv(input_channel=inp_dim, output_channel=inp_dim,
                                                  k_size=kernel) for _ in range(num_conv-1)])
        self.first_layer_norm = nn.LayerNorm(inp_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(inp_dim) for _ in range(num_conv-1)])
        self.att_layer_norm = nn.LayerNorm(inp_dim)
        self.self_attention = CausalSelfAttention(inp_dim, num_heads, dropout_p)
        self.feed_layer_norm = nn.LayerNorm(inp_dim)
        self.proj1 = nn.Linear(inp_dim, inp_dim)
        self.nonLinear = nn.ReLU()
        self.proj2 = nn.Linear(inp_dim, inp_dim)

    def forward(self, x, num_blocks):
        # TODO Figure out mask in attention and every other dropout
        num_layers = (self.num_conv + 1) * num_blocks
        x = self.pos_enc(x)
        j = 1
        for i in range(self.num_conv):
            if i == 0:
                conv, layer_norm = self.first_conv_layer, self.first_layer_norm
            else:
                conv, layer_norm = self.conv_layers[i-1], self.layer_norms[i-1]
            y = x
            x = x.permute(0, 2, 1)
            x = layer_norm(x).permute(0, 2, 1)
            x = conv(out)
            x = layer_dropout(x, y, self.dropout * float(j) / num_layers)
            j += 1

        # Self-Attention
        y = x
        x = x.permute(0, 2, 1)
        x = self.att_layer_norm(x).permute(0, 2, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.self_attention(x)
        x = self.layer_dropout(x, y, self.dropout * float(j) / num_layers)
        j += 1

        # Feedforward
        y = x
        x = x.permute(0, 2, 1)
        x = self.feed_layer_norm(x).permute(0, 2, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.proj1(x)
        y = self.nonLinear(y)
        y = self.proj2(y)
        x = self.layer_dropout(x, y, self.dropout * float(j) / num_layers)
        return x

    def layer_dropout(self, x, res, dropout):
    if self.training:
        if torch.empty(1).uniform_(0,1) < dropout:
            return res
        else:
            return F.dropout(x, dropout, training=self.training) + res
    else:
        return x + res


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

    def __init__(self, emb_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-np.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(1, 0, 2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class CausalSelfAttention(nn.Module):
    """
    The implementation comes from hw5. We don't take credit for the implementation of this class.
    """

    def __init__(self, n_embd, n_head, pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(pdrop)
        self.resid_drop = nn.Dropout(pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.n_head = n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
        