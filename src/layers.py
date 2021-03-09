import math

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

    def __init__(self, word_vectors, char_vectors, hidden_size, word_drop=0.1, char_drop=0.05):
        super(Embedding, self).__init__()
        self.word_drop = word_drop
        self.char_drop = char_drop
        self.hidden_size = hidden_size
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        self.proj = nn.Linear(word_vectors.size(-1) + char_vectors.size(-1), self.hidden_size, bias=False)
        self.conv = nn.Conv1d(char_vectors.size(1), char_vectors.size(1), kernel_size=5)
        self.hwy = HighwayEncoder(2, self.hidden_size)

    def forward(self, word_idxs, char_idxs):
        # word emb
        word_emb = self.word_embed(word_idxs)  # (batch_size, seq_len, embed_size)
        word_emb = F.dropout(word_emb, self.word_drop)

        # char emb
        char_emb = self.char_embed(char_idxs)  # (batch_size, seq_len, word_len, char_emb_dim)
        char_dim = char_emb.size()

        # reshape to fit 1d conv
        char_emb = char_emb.reshape(char_dim[0] * char_dim[1], char_dim[2], char_dim[3])
        char_emb = char_emb.permute(0, 2, 1)

        # conv and max pool
        char_emb = self.conv(char_emb)
        char_emb, _ = torch.max(char_emb, dim=-1)
        char_emb = char_emb.reshape(char_dim[0], char_dim[1], char_dim[3])
        char_emb = F.dropout(char_emb, self.char_drop)

        emb = torch.cat([char_emb, word_emb], dim=2)
        emb = self.proj(emb)
        emb = self.hwy(emb)

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

    def __init__(self, inp_dim, num_conv=4, kernel=7, num_heads=1,
                 dropout_p=0.1, dropout=0.5, max_len=5000):
        super(EmbeddingEncoder, self).__init__()
        self.block1 = EncoderBlock(inp_dim, num_conv, kernel, num_heads,
                                   dropout_p, dropout, max_len)

    def forward(self, x, key_padding_mask):
        return self.block1(x, key_padding_mask)


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
        s = self.get_similarity_matrix(context, query)  # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)  # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)  # (batch_size, c_len, q_len)

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
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2) \
            .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class ModelEncoder(nn.Module):
    """Consists of a stack of encoder blocks. The QANet paper uses 7 of these blocks
        per ModelEncoder, and a dimension of 512 (takes as input the previous attention
        layer, which is of dimension 4 x 128 = 512).
    """

    def __init__(self, inp_dim, num_conv=2, kernel=7, num_heads=8,
                 dropout_p=0.1, dropout=0.5, max_len=5000, num_blocks=7):
        super(ModelEncoder, self).__init__()

        self.blocks = nn.ModuleList([EncoderBlock(inp_dim, num_conv=2, kernel=7,
                                    num_heads=1, dropout_p=0.1,
                                    dropout=0.5, max_len=5000) for _ in range(num_blocks)])

    def forward(self, x, key_padding_mask):
        for enc_block in self.blocks:
            x = enc_block(x, key_padding_mask)
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

    def forward(self, m0, m1, m2, mask):
        concat1 = torch.cat([m0, m1], dim=2)
        concat2 = torch.cat([m1, m2], dim=2)
        lin_out1 = self.W1(concat1)
        lin_out2 = self.W2(concat2)
        start_prob = masked_softmax(lin_out1.squeeze(), mask, log_softmax=True)
        end_prob = masked_softmax(lin_out2.squeeze(), mask, log_softmax=True)
        return start_prob, end_prob


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.
    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the output from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).
    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)  # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)  # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)  # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, context, query):
        c_len, q_len = context.size(1), query.size(1)
        c = F.dropout(context, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(query, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2) \
            .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s
