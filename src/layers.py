
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
from blocks import *


class Embedding(nn.Module):
    """Input embedding layer for QANet.

   	Includes character embeddings and a 2-layer Highway Encoder

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_size (int): Number of characters in the character-vocabulary
        char_emb_dim (int): Dimension of character embeddings
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_size, char_emb_dim, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding(char_size, char_emb_dim)
        
        # TODO: figure out highway encoder, if this is right
        self.hwy = HighwayEncoder(2, word_vectors.size(1) + char_emb_dim)

    def forward(self, word_idxs, char_idxs):
        word_emb = self.word_embed(word_idxs)   # (batch_size, seq_len, embed_size)
       	char_emb = self.char_embed(char_idxs)	# (batch_size, seq_len, word_len, char_emb_dim)

       	# Need to take maximum over rows of embedding matrix for each word
       	char_emb = char_emb.max(-2).values 		# (batch_size, seq_len, char_emb_dim)
       	# Concatenate the embedding
       	emb = torch.cat((char_emb.T, word_emb.T)).T

        emb = F.dropout(emb, self.drop_prob, self.training)

        return emb

# TODO: Fill out classes for all other layers

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x
        return x


class EmbeddingEncoder(nn.Module):
	"""Embedding Encoder layer consists of a stack of encoder blocks,
		each encoder block consists of several convolutional layers, then
		self-attention, then a feed forward layer
	"""

	def __init__(self, inp_dim, num_conv=4, kernel=7, filters=128, num_heads=8, 
					dropout_p=0.1, dropout=0.5, max_len=5000):
		super(EmbeddingEncoder, self).__init__()
		self.block1 = EncoderBlock(inp_dim, num_conv, kernel, filters, num_heads, 
					dropout_p, dropout, max_len)

	def forward(self, x):
		return self.block1(x)


class ContextQueryAttention(nn.Module):
    """This layer "combines" the context and query encoded embeddings from the
        previous layers through attention. It is a standard context-query
        attention layer that is used in other architectures, such as BiDAF.
        For now, this was basically taken from the given starter code (BiDAF)
	"""
    def __init__(self, hidden_size, drop_prob=0.1):
        super(ContextQueryAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, context, query, c_mask, q_mask):
        batch_size, c_len, _ = context.size()
        q_len = query.size(1)
        s = self.get_similarity_matrix(context, query)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, query)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), context)

        x = torch.cat([context, a, context * a, context * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, context, query):
        c_len, q_len = context.size(1), query.size(1)
        c = F.dropout(context, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(query, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class ModelEncoder(nn.Module):
	"""Consists of a stack of encoder blocks. The QANet paper uses 7 of these blocks
		per ModelEncoder, and a dimension of 512 (takes as input the previous attention
		layer, which is of dimension 4 x 128 = 512).
	"""

	def __init__(self, inp_dim, num_conv=2, kernel=7, filters=128, num_heads=8,
                 dropout_p=0.1, dropout=0.5, max_len=5000, num_blocks=7):
		super(ModelEncoder, self).__init__()

		# First block, in case of different input dimension
		self.blocks = [EncoderBlock(inp_dim, num_conv=2, 
									kernel=7, filters=128, 
									num_heads=8, dropout_p=0.1, 
									dropout=0.5, max_len=5000)]
		# Add the other blocks
		for i in range(num_blocks - 1):
			self.blocks.append(EncoderBlock(inp_dim=filters, num_conv=2, 
									kernel=7, filters=128, 
									num_heads=8, dropout_p=0.1, 
									dropout=0.5, max_len=5000))

	def forward(self, x):
		for enc_block in self.blocks:
			x = enc_block(x)
		return x



class OutputLayer(nn.Module):
    """Final layer of the QANet model. Takes as input the three model encoders in
        the previous layer, m0, m1, m2. For predicting the start-probability, computes
        p1 = softmax(W1[m0 : m1]).
        Similarly, predicts end-probability by computing
        p2 = softmax(W2[m0 : m2])
    """

    def __init__(self, in_dim, out_dim=1):
        super(OutputLayer, self).__init__()
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim)

    def forward(self, m0, m1, m2):
        concat1 = torch.cat([model_enc0, model_enc1], dim=2)
        concat2 = torch.cat([model_enc0, model_enc1], dim=2)
        lin_out1 = self.W1(concat1)
        lin_out2 = self.W2(concat2)
        lin_out1 = lin_out1.view(lin_out1.shape[:2])
        lin_out2 = lin_out1.view(lin_out2.shape[:2])
        start_prob = F.softmax(lin_out1)
        end_prob = F.softmax(lin_out2)
        return start_prob, end_prob
        
