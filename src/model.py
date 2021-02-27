
import layers
import torch
import torch.nn as nn


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
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
	"""
	def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
        
        # TODO: More layers!!


    def forward(self, cword_idxs, cchar_idxs, qword_idxs, qchar_idxs):
    	# Might use this?
    	cword_mask = torch.zeros_like(cword_idxs) != cword_idxs
        qword_mask = torch.zeros_like(qword_idxs) != qword_idxs
        c_len_words, q_len_words = cword_mask.sum(-1), qword_mask.sum(-1)

        cchar_mask = torch.zeros_like(cchar_idxs) != cchar_idxs
        qchar_mask = torch.zeros_like(qchar_idxs) != qchar_idxs
        c_len_chars, q_len_chars = cchar_mask.sum(-1), qchar_mask.sum(-1)

        # Input embedding
        c_emb = self.emb(cw_idxs)
        q_emb = self.emb(qw_idxs)

        # TODO: Complete QANET forward computation


