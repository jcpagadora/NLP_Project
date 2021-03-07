import layers
import torch
import torch.nn as nn
from layers import *


class QANet(nn.Module):

    """
    Outline of Model:
        1. Input Embedding Layer
            - Word embeddings (GloVe)
            - Character embeddings

        2. Embedding Encoder Layer
            - Stack of encoder blocks, each consists of
                i. Convolution layers
                * In paper, use depthwise separable convolutions,
                      kernel = 7,  # filters = 128,  # conv. layers = 4
                ii. Self-attention layer
                iii. Feed forward layer

        3. Context-Query Attention Layer

        4. Model Encoder Layer

        5. Output Layer


    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        TODO: double check char vectors
        char_size (int): Number of characters in char-level vocabulary
        char_emb_dim (int): Dimensionality of character-level embedding
        conv_dim (int): For conv layers, this is number of filters, but used throughout as hidden_size
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, conv_dim=128, drop_prob=0.5):

        super(QANet, self).__init__()
        self.ember = Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size)

        in_dim = hidden_size

        self.emb_encer = EmbeddingEncoder(in_dim, num_conv=4, kernel=7, 
                                                    num_heads=8,
                                                   dropout_p=0.1, dropout=0.5, 
                                                   max_len=5000)

        self.cq_att_layer = BiDAFAttention(in_dim)
        self.att_cnn = nn.Conv1d(in_dim*4, in_dim, kernel_size=7, padding=7//2)

        self.model_encoder = ModelEncoder(in_dim, num_conv=2, kernel=7,
                                                    num_heads=8,
                                                   dropout_p=0.1, dropout=0.5, 
                                                   max_len=5000)
        self.output = OutputLayer(in_dim*2)


    def forward(self, cword_idxs, cchar_idxs, qword_idxs, qchar_idxs):
        cword_mask = torch.zeros_like(cword_idxs) != cword_idxs
        qword_mask = torch.zeros_like(qword_idxs) != qword_idxs
        c_len_words, q_len_words = cword_mask.sum(-1), qword_mask.sum(-1)

        # Input embedding
        c_emb = self.ember(cword_idxs, cchar_idxs)
        q_emb = self.ember(qword_idxs, qchar_idxs)

        # Embedding encoder
        c_emb_enc = self.emb_encer(c_emb, 1, 1)
        q_emb_enc = self.emb_encer(q_emb, 1, 1)
        # Context-Query Attention
        cq_att = self.cq_att_layer(c_emb_enc, q_emb_enc, cword_mask, qword_mask)

        cq_att = cq_att.permute(0, 2, 1)
        cq_att = self.att_cnn(cq_att)
        cq_att = cq_att.permute(0, 2, 1)

        # Model Encoder Layer
        model_enc0 = self.model_encoder(cq_att, 7)
        model_enc1 = self.model_encoder(model_enc0, 7)
        model_enc2 = self.model_encoder(model_enc1, 7)

        # Output Layer
        start_probs, end_probs = self.output(model_enc0, model_enc1, model_enc2, cword_idxs)
        return start_probs, end_probs

