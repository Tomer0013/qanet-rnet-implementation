import layers
import torch
import torch.nn as nn


class RNet(nn.Module):
    """
    R-Net model as described in
    https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
    without the dynamic (calculation at every rnn step) attention.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0):
        super(RNet, self).__init__()
        self.emb = layers.RNetEmbedding(word_vectors, char_vectors, hidden_size, drop_prob)
        self.gated_att = layers.GatedAttentionRNN(2 * hidden_size, hidden_size, drop_prob)
        self.self_att = layers.GatedAttentionRNN(2 * hidden_size, hidden_size, drop_prob)
        self.out = layers.RNetOutput(2 * hidden_size, hidden_size)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, cq_idxs):
        cw_mask = torch.zeros_like(cw_idxs) != cw_idxs
        qw_mask = torch.zeros_like(qw_idxs) != qw_idxs
        cw_len, qw_len = cw_mask.sum(-1), qw_mask.sum(-1)

        up = self.emb(cw_idxs, cc_idxs, cw_len)  # (batch_size, c_len, 2 * h_dim)
        uq = self.emb(qw_idxs, cq_idxs, qw_len)  # (batch_size, q_len, 2 * h_dim)
        vp = self.gated_att(up, uq, qw_mask, cw_len)  # (batch_size, c_len, 2 * h_dim)
        hp = self.self_att(vp, vp, cw_mask, cw_len)  # (batch_size, c_len, 2 * h_dim)
        out = self.out(hp, uq, cw_mask, qw_mask, cw_len)  # 2 x (batch_size, c_len)

        return out


class QANet(nn.Module):
    """
    QANet model as described in the QANet paper.
    https://arxiv.org/pdf/1804.09541.pdf

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained char vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
        drop_prob_w (float): Dropout probability on word embeddings.
        drop_prob_c (float): Dropout probability on character embeddings.
        p_last_sublayer (float): Survival rate of last sublayer in encoder blocks. Used for
                                 layer dropout.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, char_dim, max_seq_len, drop_prob=0.0, drop_prob_w=0.0,
                 drop_prob_c=0.0, p_last_sublayer=1.0):
        super(QANet, self).__init__()
        self.drop_prob = drop_prob
        self.emb = layers.QANetEmbedding(word_vectors, char_vectors, char_dim, drop_prob_w, drop_prob_c)
        self.emb_dim_map = layers.DimMapConv1D(self.emb.word_emb.embedding_dim + self.emb.char_emb.embedding_dim,
                                               hidden_size, kernel_size=7, padding=7 // 2)
        self.stacked_emb_enc_blocks = layers.StackedEncoderBlocks(num_blocks=1,
                                                                  input_size=hidden_size, conv_kernel_size=7,
                                                                  num_conv_blocks=4, num_heads=8,
                                                                  p_last_sublayer=p_last_sublayer,
                                                                  max_seq_len=max_seq_len)
        self.c2q = layers.QANetC2QAttention(hidden_size)
        self.mod_dim_map = layers.DimMapConv1D(4 * hidden_size, hidden_size, kernel_size=5, padding=5 // 2)
        self.stacked_mod_enc_blocks = layers.StackedEncoderBlocks(num_blocks=7,
                                                                  input_size=hidden_size, conv_kernel_size=5,
                                                                  num_conv_blocks=2, num_heads=8,
                                                                  p_last_sublayer=p_last_sublayer,
                                                                  max_seq_len=max_seq_len)
        self.out = layers.QANetOutput(2 * hidden_size)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, cq_idxs):
        cw_mask = torch.zeros_like(cw_idxs) != cw_idxs
        qw_mask = torch.zeros_like(qw_idxs) != qw_idxs

        context_emb = self.emb(cw_idxs, cc_idxs)
        question_emb = self.emb(qw_idxs, cq_idxs)

        context_emb = self.emb_dim_map(context_emb)
        question_emb = self.emb_dim_map(question_emb)

        context_enc = self.stacked_emb_enc_blocks(context_emb, cw_mask)
        question_enc = self.stacked_emb_enc_blocks(question_emb, qw_mask)

        context_enc = torch.dropout(context_enc, self.drop_prob, self.training)
        question_enc = torch.dropout(question_enc, self.drop_prob, self.training)

        c2q_enc = self.c2q(context_enc, question_enc, cw_mask, qw_mask)

        c2q_enc = self.mod_dim_map(c2q_enc)

        mod_enc_a = self.stacked_mod_enc_blocks(c2q_enc, cw_mask)
        mod_enc_b = self.stacked_mod_enc_blocks(mod_enc_a, cw_mask)
        mod_enc_c = self.stacked_mod_enc_blocks(mod_enc_b, cw_mask)

        mod_enc_a = torch.dropout(mod_enc_a, self.drop_prob, self.training)
        mod_enc_b = torch.dropout(mod_enc_b, self.drop_prob, self.training)
        mod_enc_c = torch.dropout(mod_enc_c, self.drop_prob, self.training)

        out = self.out(mod_enc_a, mod_enc_b, mod_enc_c, cw_mask)

        return out
