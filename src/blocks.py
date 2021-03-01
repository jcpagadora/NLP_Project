import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

    def __init__(self, inp_dim, num_conv=4, kernel=7, filters=128, num_heads=8,
                 dropout_p=0.1, dropout=0.5, max_len=5000):
        super(EncoderBlock, self).__init__()
        self.pos_enc = PositionalEncoding(inp_dim, dropout_p, max_len)
        self.num_conv = num_conv
        self.dropout = dropout

        # depthwise separable cnn layer for fewer parameters
        self.first_conv_layer = ds_conv(input_channel=inp_dim,
                                        output_channel=filters,
                                        k_size=kernel)
        self.conv_layer = ds_conv(input_channel=filters,
                                  output_channel=filters,
                                  k_size=kernel)
        self.self_attention = nn.MultiheadAttention(filters, num_heads)
        self.feed_forward = nn.Linear(filters, filters)

    def forward(self, x):
        # Positional encoding
        x = self.pos_enc(x)
        # Conv layers
        # First convolution
        x = x.transpose(1, 2)
        y = nn.LayerNorm(x.size()[1:])(x)
        x = self.first_conv_layer(y)
        for i in range(self.num_conv - 1):
            y = nn.LayerNorm(x.size()[1:])(x)
            y = self.conv_layer(y)
            x = x + y
        # Self-Attention
        y = nn.LayerNorm(x.size()[1:])(x)
        y = y.permute(2, 0, 1)
        x = x.permute(2, 0, 1)
        y, _ = self.self_attention(query=y, key=y, value=y)
        x = x + y
        # Feedforward
        y = nn.LayerNorm(x.size()[1:])(x)
        y = self.feed_forward(y)
        y = nn.Dropout(self.dropout)(y)
        x = x + y
        x = x.permute(1, 0, 2)
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
        self.pointwise = nn.Conv1d(input_channel, output_channel, k_size=1, groups=1)

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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
