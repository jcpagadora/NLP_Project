
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
        char_size (int): Number of characters in char-level vocabulary
        char_emb_dim (int): Dimensionality of character-level embedding
        drop_prob (float): Dropout probability.
	"""
	def __init__(self, word_vectors, char_size, char_emb_dim=200, drop_prob=0.5):
        super(QANet, self).__init__()
        self.c_ember = layers.Embedding(word_vectors=word_vectors,
                                    char_size=char_size,
                                    char_emb_dim=char_emb_dim,
                                    drop_prob=drop_prob)
        self.q_ember = layers.Embedding(word_vectors=word_vectors,
                                    char_size=char_size,
                                    char_emb_dim=char_emb_dim,
                                    drop_prob=drop_prob)

       	in_dim = word_vectors.size(1) + char_emb_dim

        self.c_emb_encer = layers.EmbeddingEncoder(in_dim, num_conv=4, kernel=7, filters=128, num_heads=8, 
					dropout_p=0.1, dropout=0.5, max_len=5000)

        self.q_emb_encer = layers.EmbeddingEncoder(in_dim, num_conv=4, kernel=7, filters=128, num_heads=8, 
					dropout_p=0.1, dropout=0.5, max_len=5000)
        
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
        c_emb = self.c_ember(cword_idxs, cchar_idxs)
        q_emb = self.q_ember(qword_idxs, qchar_idxs)

        # Embedding encoder
        c_emb_enc = self.c_emb_encer(c_emb)
        q_emb_enc = self.q_emb_encer(c_emb)

        # TODO: Complete the rest of QANET forward computation


