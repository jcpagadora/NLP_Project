
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


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
        self.hwy = HighwayEncoder(2, word_embed.size(1) + char_emb_dim)

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

	




































