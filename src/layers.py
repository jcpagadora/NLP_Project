
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
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_size, char_emb_dim, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding(char_size, char_emb_dim)

        # TODO: figure out highway encoder
        self.proj = nn.Linear(word_vectors.size(1) + char_emb_dim, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, word_idxs, char_idxs):
        word_emb = self.word_embed(word_idxs)   # (batch_size, seq_len, embed_size)
       	char_emb = self.char_embed(char_idxs)	# (batch_size, seq_len, word_len, char_emb_dim)

       	# Need to take maximum over rows of embedding matrix for each word
       	char_emb = char_emb.max(-2).values 		# (batch_size, seq_len, char_emb_dim)
       	# Concatenate the embedding
       	emb = torch.cat((char_emb.T, word_emb.T)).T

        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


# TODO: Fill out classes for all other layers


















