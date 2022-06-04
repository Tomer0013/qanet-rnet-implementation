import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from util import masked_softmax, get_positional_encoding


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
    def __init__(self, num_layers: int, hidden_size: int) -> None:
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 use_gru: bool = False, drop_prob: float = 0.) -> None:
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        if use_gru:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True,
                              bidirectional=True,
                              dropout=drop_prob if num_layers > 1 else 0.)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               bidirectional=True,
                               dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class RNetEmbedding(nn.Module):
    """Embedding layer used by R-Net, with the character-level component.

    See paper for details, as described in the
    "3.1 Question and Passage Encoder" section.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained character vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors: torch.Tensor, char_vectors: torch.Tensor,
                 hidden_size: int, drop_prob: float) -> None:
        super(RNetEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors)
        self.char_rnn = RNNEncoder(self.char_embed.embedding_dim, hidden_size, 1, True)
        self.birnn = RNNEncoder(self.word_embed.embedding_dim + 2 * hidden_size, hidden_size, 3, True)

    def forward(self, word_idx: torch.Tensor, char_idx: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:

        # Needed for flow
        batch_size = char_idx.shape[0]
        char_mask = torch.zeros_like(char_idx) != char_idx
        char_len = char_mask.flatten(0, 1).sum(-1)
        lengths, sort_idx = char_len.sort(0, descending=True)
        _, unsort_idx = sort_idx.sort(0)

        # Char embeddings
        char_emb = self.char_embed(char_idx)
        char_emb = char_emb.flatten(0, 1)
        char_emb = char_emb[sort_idx]
        char_emb = char_emb[lengths > 0]
        char_emb = self.char_rnn(char_emb, lengths[lengths > 0])
        char_emb = char_emb[torch.arange(char_emb.shape[0]), lengths[lengths > 0] - 1, :]
        char_emb = (torch.cat([char_emb,
                              torch.zeros((lengths[lengths == 0].size()[0], 2 * self.char_rnn.rnn.hidden_size),
                                          device=self.char_embed.weight.device)],
                              dim=0)[unsort_idx]).view(batch_size, -1, 2 * self.char_rnn.rnn.hidden_size)

        # Word embeddings
        word_emb = self.word_embed(word_idx)

        # Concat and encode in BiRNN
        x = torch.cat([word_emb, char_emb], dim=-1)
        x = self.birnn(x, seq_lengths)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class GatedAttentionRNN(nn.Module):
    """
    Gated attention RNN as described in the R-Net paper
    in section 3.2. The vp_t-1 element is left out of the sj_t equation,
    in order to compute the attention just once, and not at every rnn
    step. Computing the attention at every step was too demanding for
    my pc.

    Once the vp_t-1 is removed, this becomes the self-matching attention
    layer as described in the paper in section 3.3.

    Args:
        input_size (int): Size of input dim.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, input_size: int, hidden_size: int, drop_prob: float = 0) -> None:
        super(GatedAttentionRNN, self).__init__()
        self.drop_prob = drop_prob
        self.w_value_lin = nn.Linear(input_size, hidden_size, bias=False)
        self.w_query_lin = nn.Linear(input_size, hidden_size, bias=False)
        self.v_lin = nn.Linear(hidden_size, 1, bias=False)
        self.g_lin = nn.Linear(2 * input_size, 2 * input_size, bias=False)
        self.rnn = RNNEncoder(2 * input_size, hidden_size, 1, True)

    def forward(self, u_query: torch.Tensor, u_value: torch.Tensor,
                u_value_lengths_mask: torch.Tensor, u_query_lengths: torch.Tensor) -> torch.Tensor:
        s1 = self.w_value_lin(u_value).unsqueeze(dim=1).expand(-1, u_query.shape[1], -1, -1)
        s2 = self.w_query_lin(u_query).unsqueeze(dim=2)
        s = self.v_lin(torch.tanh(s1 + s2)).squeeze()
        a = masked_softmax(s, u_value_lengths_mask.unsqueeze(dim=1))
        c = torch.bmm(a, u_value)
        rnn_input = torch.cat((u_query, c), dim=-1)
        g = torch.sigmoid(self.g_lin(rnn_input))
        rnn_input = g * rnn_input
        x = self.rnn(rnn_input, u_query_lengths)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class RNetOutput(nn.Module):
    """
    Output layer of the R-Net model, as described in
    section 3.4 of the paper. Same as in section 3.2,
    the element regarding the last rnn hidden state in s_t is removed.
    Because of that, I simply feed hp through the RNN, instead of c_t which
    is originally calculated every rnn step.

    Args:
        input_size (int): Size of input dim.
        hidden_size (int): Size of hidden dim.
    """
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(RNetOutput, self).__init__()
        self.v_lin = nn.Linear(hidden_size, 1)
        self.wuq_lin = nn.Linear(input_size, hidden_size)
        self.wvq_lin = nn.Linear(input_size, hidden_size)
        self.whp_lin = nn.Linear(input_size, hidden_size)
        self.wha_lin = nn.Linear(input_size, hidden_size)
        self.rnn = RNNEncoder(input_size, hidden_size, 1, True)
        self.vrq_param = nn.Parameter(torch.zeros(1, input_size))
        nn.init.xavier_uniform_(self.vrq_param)

    def forward(self, hp: torch.Tensor, uq: torch.Tensor, cw_mask: torch.Tensor,
                qw_mask: torch.Tensor, cw_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Get r_q
        s_rq = self.v_lin(torch.tanh(self.wuq_lin(uq) + self.wvq_lin(self.vrq_param))).squeeze()
        a_rq = masked_softmax(s_rq, qw_mask)
        r_q = a_rq.unsqueeze(dim=1).matmul(uq).squeeze()

        # Get p1
        s_1 = self.v_lin(torch.tanh(self.whp_lin(hp) + self.wha_lin(r_q).unsqueeze(dim=1))).squeeze()
        log_p1 = masked_softmax(s_1, cw_mask, log_softmax=True)

        # Feed through rnn and get last hidden state
        ha_t = self.rnn(hp, cw_len)[torch.arange(hp.shape[0]), cw_len - 1, :]

        # Get p2
        s_2 = self.v_lin(torch.tanh(self.whp_lin(hp) + self.wha_lin(ha_t).unsqueeze(dim=1))).squeeze()
        log_p2 = masked_softmax(s_2, cw_mask, log_softmax=True)

        return log_p1, log_p2


class QANetEmbedding(nn.Module):
    """
    Embedding layer as described in the QANet paper.
    In this version I decided to avoid training of char vectors and use the existing
    ones. I take the max of each dim, as noted in the paper.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained character vectors.
        char_dim (int): Dim size of character embeddings.
        drop_prob_w (float): Dropout probability on word embeddings.
        drop_prob_c (float): Dropout probability on character embeddings.
    """
    def __init__(self, word_vectors: torch.Tensor, char_vectors: torch.Tensor,
                 char_dim: int, drop_prob_w: float = 0.0, drop_prob_c: float = 0.0) -> None:
        super(QANetEmbedding, self).__init__()
        self.drop_prob_w = drop_prob_w
        self.drop_prob_c = drop_prob_c
        self.word_emb = nn.Embedding.from_pretrained(word_vectors)
        self.char_emb = nn.Embedding(char_vectors.shape[0], char_dim)
        self.char_conv = DepthWiseSeparableConv1D(self.char_emb.embedding_dim, self.char_emb.embedding_dim,
                                                  kernel_size=5, padding=5 // 2)
        self.highway = HighwayEncoder(2, self.word_emb.embedding_dim + self.char_emb.embedding_dim)
        self.oov_w = nn.Parameter(torch.zeros((1, self.word_emb.embedding_dim), dtype=torch.float32))

        nn.init.xavier_uniform_(self.oov_w)

    def forward(self, w_idxs: torch.Tensor, c_idxs: torch.Tensor, oov_token: int = 1) -> torch.Tensor:
        w_vecs = self.word_emb(w_idxs)
        w_vecs = torch.dropout(w_vecs, self.drop_prob_w, self.training)
        w_vecs[w_idxs == oov_token] += self.oov_w
        c_vecs = self.char_emb(c_idxs)
        c_vecs = torch.dropout(c_vecs, self.drop_prob_c, self.training)
        c_vecs = c_vecs.max(dim=2)[0]
        c_vecs = self.char_conv(c_vecs)
        x = torch.cat((w_vecs, c_vecs), dim=-1)
        x = self.highway(x)

        return x


class DepthWiseSeparableConv1D(nn.Module):
    """
    Depth wise separable 1d convolution.

    Args:
        channels_in (int): Number of channels in the input.
        channels_out (int): Number of channels in the output.
        seq_length_first (bool): If true, expects input to be (batch_size, seq_length, dim_size).
    """
    def __init__(self, channels_in: int, channels_out: int, seq_length_first: bool = True, **kwargs) -> None:
        super(DepthWiseSeparableConv1D, self).__init__()
        self.seq_length_first = seq_length_first
        self.depthwise_conv_1 = nn.Conv1d(channels_in, channels_in, groups=channels_in, bias=False, **kwargs)
        self.depthwise_conv_2 = nn.Conv1d(channels_in, channels_out, kernel_size=(1,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.seq_length_first:
            x = self.depthwise_conv_1(x.permute(0, 2, 1))
            x = self.depthwise_conv_2(x).permute(0, 2, 1)
        else:
            x = self.depthwise_conv_1(x)
            x = self.depthwise_conv_2(x)

        return x


class QANetResidualBlock(nn.Module):
    """
    A residual block which can be one of three types:
    Conv1d, self-attention, or linear.
    Each block begins with a layer normalization and then
    proceeds to chosen type.

    Args:
        block_type (str): Can be either 'covd1d', 'att', 'linear'
        input_size (int): Size of input dim.
        survival_prob (float): Probability of the residual block being effective.
                               Used for layer dropout.
    """
    def __init__(self, block_type: str, input_size: int, survival_prob: float = 1.0, **kwargs) -> None:
        super(QANetResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.layer_norm = nn.LayerNorm(input_size)
        self.block_type = block_type
        if block_type == 'conv1d':
            self.func = nn.Sequential(
                            nn.ReLU(),
                            DepthWiseSeparableConv1D(input_size, **kwargs)
                        )
        elif block_type == 'att':
            self.func = MultiHeadAttention(input_size, **kwargs)
        elif block_type == 'ff':
            self.func = nn.Sequential(
                            nn.Linear(input_size, input_size),
                            nn.ReLU(),
                            nn.Linear(input_size, input_size)
                         )
        else:
            raise Exception("Invalid block type.")

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        if self.training:
            if random.random() <= self.survival_prob:
                x_og = x
                x = self.layer_norm(x)
                x = self.func(x, *args)
                x = x + x_og
        else:
            x_og = x
            x = self.layer_norm(x)
            x = self.func(x, *args)
            x = self.survival_prob * x + x_og
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi headed attention as described in the paper
    "Attention is All You Need".

    Args:
        input_size (int): Dim size of input.
        num_heads (int): Number of heads.
    """
    def __init__(self, input_size: int, num_heads: int) -> None:
        assert input_size % num_heads == 0, "Input size divided by num heads must be a whole number."
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.q_proj = torch.nn.Linear(input_size, input_size)
        self.k_proj = torch.nn.Linear(input_size, input_size)
        self.v_proj = torch.nn.Linear(input_size, input_size)
        self.o_proj = torch.nn.Linear(input_size, input_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n, t, d = x.shape

        q = self.q_proj(x).view(n, t, self.num_heads, d // self.num_heads).transpose(1, 2)
        k = self.k_proj(x).view(n, t, self.num_heads, d // self.num_heads).transpose(1, 2)
        v = self.v_proj(x).view(n, t, self.num_heads, d // self.num_heads).transpose(1, 2)

        s = q.matmul(k.transpose(-2, -1)) / math.sqrt(d // self.num_heads)
        s = masked_softmax(s, mask.view(n, 1, 1, -1))
        a = s.matmul(v).transpose(1, 2).contiguous().view(n, t, -1)
        x = self.o_proj(a)

        return x


class QANetEncoderBlock(nn.Module):
    """
    Encoder block as described in the QANet paper.

    Args:
        input_size (int): Dim size of input.
        conv_kernel_size (int): Size of the 1D convolution kernel.
        num_conv_blocks (int): Numboer of convolution blocks within the encoder.
        num_heads (int): Number of heads for multihead attention.
        p_last_sublayer (float): Survival rate of last layer. Used for layer dropout.
        max_seq_len (int): Maximum possible sequence length.
    """
    def __init__(self, input_size: int, conv_kernel_size: int, num_conv_blocks: int, num_heads: int,
                 p_last_sublayer: float, max_seq_len: int) -> None:
        super(QANetEncoderBlock, self).__init__()
        conv_blocks = [QANetResidualBlock('conv1d', input_size, channels_out=input_size,
                                          kernel_size=conv_kernel_size,
                                          padding=conv_kernel_size // 2) for _ in range(num_conv_blocks)]
        self_att_block = [QANetResidualBlock('att', input_size, num_heads=num_heads)]
        ff_block = [QANetResidualBlock('ff', input_size)]
        self.sublayers = nn.ModuleList(conv_blocks + self_att_block + ff_block)
        self.pos_enc = PositionalEncoding(max_seq_len, input_size)

        for i, sublayer in enumerate(self.sublayers):
            sublayer.survival_prob = 1 - ((i + 1) / len(self.sublayers)) * (1 - p_last_sublayer)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        for sublayer in self.sublayers:
            if sublayer.block_type == 'att':
                x = sublayer(x, mask)
            else:
                x = sublayer(x)

        return x


class PositionalEncoding(nn.Module):
    """
    A module which adds positional encoding to the input.
    1 is added to the max_seq_len when creating the positional encoding
    because when SQuAD v2 is used, the first index in the sequence is used
    for "no answer", and the maximum sequence length is increased by 1.

    Args:
        max_seq_len (int): Maximum possible sequence length supported.
        dim (int): Input dimension.
    """
    def __init__(self, max_seq_len: int, dim: int) -> None:
        super(PositionalEncoding, self).__init__()
        self.register_buffer('p', get_positional_encoding(max_seq_len + 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.p[:x.shape[1], :]

        return x


class StackedEncoderBlocks(nn.Module):
    """
    A class to stack encoder blocks and feed the data through.

    Args:
        num_blocks (int): Number of encoder blocks in the stack.
    """
    def __init__(self, num_blocks: int, **kwargs) -> None:
        super(StackedEncoderBlocks, self).__init__()
        self.enc_list = nn.ModuleList([QANetEncoderBlock(**kwargs) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for enc in self.enc_list:
            x = enc(x, mask)

        return x


class QANetOutput(nn.Module):
    """
    Output layer of QANet as described in the paper.

    Args:
        input_size (int): Dim size of input.
    """
    def __init__(self, input_size: int) -> None:
        super(QANetOutput, self).__init__()
        self.w1 = nn.Linear(input_size, 1)
        self.w2 = nn.Linear(input_size, 1)

    def forward(self, m0: torch.Tensor, m1: torch.Tensor,
                m2: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_p1 = masked_softmax(self.w1(torch.cat((m0, m1), dim=-1)).squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(self.w2(torch.cat((m0, m2), dim=-1)).squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class QANetC2QAttention(nn.Module):
    """
    C2Q Attention layer uesed in the QANet model. As the paper notes,
    this is probably the same c2q attention mechanism found in BiDAF.
    Therefore, this is the orginal implementation of that mechanism, and
    not the memory efficient one supplied with this project's skeleton.
    Args:
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, hidden_size: int) -> None:
        super(QANetC2QAttention, self).__init__()
        self.w_proj = nn.Linear(3 * hidden_size, 1)

    def forward(self, c: torch.Tensor, q: torch.Tensor, c_mask: torch.Tensor, q_mask: torch.Tensor) -> torch.Tensor:
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        h = c.unsqueeze(dim=2).expand(-1, -1, q.shape[1], -1)
        u = q.unsqueeze(dim=1).expand(-1, c.shape[1], -1, -1)
        h_u = h * u
        c_q_concat = torch.cat((h, u, h_u), dim=-1)
        s = self.w_proj(c_q_concat).squeeze()
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x


class DimMapConv1D(nn.Module):
    """
    A feed forward through 1d convolutions network for remapping the input dim
    to hidden dim.

    Args:
        input_size (int): Dimension of input.
        hidden_size (int): Hidden dimension.
    """
    def __init__(self, input_size: int, hidden_size: int, **kwargs) -> None:
        super(DimMapConv1D, self).__init__()
        self.map_dim_func = nn.Sequential(
            nn.ReLU(),
            DepthWiseSeparableConv1D(input_size, hidden_size, **kwargs)
        )
        self.w_proj = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.map_dim_func(x) + self.w_proj(x)

        return x
