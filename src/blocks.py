
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EncoderBlock(nn.Module):
	"""Starts with positional encoding, followed by multiple convolutional layers.
	   Then, a self-attention, and finally a feefoward layer.
	   Between each is a layer-norm.
	   Note: Each of these is placed within a residual block

	   Args:
	    	inp_dim (int): Dimension of the input (after positional encoding)
	     	num_conv (int): Number of initial convolutional layers
	    	kernel (int): Kernel size of each convolution
	    	filters (int): Number of filters
	    	num_heads (int): Number of heads for self-attention
	    	dropout (float): Dropout probability
	    	max_len (int): Maximum length for positional encoding

	   Note: output of this layer should equal the number of filters 
	"""
	def __init__(self, inp_dim, num_conv=4, kernel=7, filters=128, num_heads=8, dropout=0.1, max_len=5000):
		self.pos_enc = PositionalEncoding(inp_dim, dropout, max_len)
		self.num_conv = num_conv

		# TODO: Figure out depthwise separable
		self.conv_layer = nn.Conv1D(filters, filters, kernel)


	def forward(self, x):
		# Positional encoding
		x = self.pos_enc(x)
		# Conv layers
		y = nn.LayerNorm(x.size()[1:])
		for i in range(self.num_conv):
			y = self.conv_layer(y)
		x = x + y # Res connection
		# TODO: Self-attention, & feedforward layer



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





















