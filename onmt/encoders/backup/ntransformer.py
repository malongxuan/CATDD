"""
Implementation of "Attention is All You Need"
"""
import torch
import torch.nn as nn

import onmt
from onmt.encoders.encoder import EncoderBase
# from onmt.utils.misc import aeq
from onmt.modules.position_ffn import PositionwiseFeedForward

from onmt.modules.global_attention import GlobalAttention

class NTransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(NTransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_i = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_q = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, key, value, src_mask, mask=None):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm_i(inputs)
        query, _ = self.self_attn(input_norm, input_norm, input_norm,
                                  mask=src_mask)
        query = self.dropout(query) + inputs

        query_norm = self.layer_norm_q(query)
        out, _ = self.attn(key, value, query_norm,
                                    mask=mask)
        out = self.dropout(out) + query

        return self.feed_forward(out)


class STransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(STransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class NTransformerEncoder(EncoderBase):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout):
        super(NTransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.transformer = nn.ModuleList(
            [NTransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query_bank, key_bank=None, value_bank=None, query_mask=None, mask=None):
        out = query_bank
        for i in range(self.num_layers):
            out = self.transformer[i](out, key_bank, value_bank, query_mask, mask)

        out = self.layer_norm(out)
        return out


class STransformerEncoder(EncoderBase):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout):
        super(STransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.transformer = nn.ModuleList(
            [STransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, mask=None):
        out = src
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)

        out = self.layer_norm(out)
        return out


class TransformerEncoder(EncoderBase):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings

        self.last_his_transformer = NTransformerEncoder(num_layers, d_model, heads, d_ff, dropout)
        self.last_knl_transformer = NTransformerEncoder(num_layers, d_model, heads, d_ff, dropout)

        self.last_transformer = STransformerEncoder(num_layers, d_model, heads, d_ff, dropout)
        self.knl_transformer = STransformerEncoder(num_layers, d_model, heads, d_ff, dropout)

        self.his_gate = GlobalAttention(d_model, attn_type='mlp')

    def __preprocess(self, src):
        emb = self.embeddings(src)
        out = emb.transpose(0, 1).contiguous()

        words = src[:, :, 0].transpose(0, 1)
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
        return out, mask

    def forward(self, src, knl=None):
        """ See :obj:`EncoderBase.forward()`"""
        his, last = src[:100, :, :], src[100:, :, :]

        his_out, his_mask = self.__preprocess(his)
        last_out, last_mask = self.__preprocess(last)
        knl_out, knl_mask = self.__preprocess(knl)

        his_bank = self.last_transformer(his_out, his_mask)
        last_bank = self.last_transformer(last_out, last_mask)
        knl_bank = self.knl_transformer(knl_out, knl_mask)

        last_his_out = self.last_his_transformer(last_out, his_bank, his_bank, last_mask, his_mask)
        his_knl_out = self.last_knl_transformer(last_his_out, knl_bank, knl_bank, last_mask, knl_mask)
        last_knl_out = self.last_knl_transformer(last_out, knl_bank, knl_bank, last_mask, knl_mask)

        alpha = self.his_gate.score(last_bank, his_bank)
        alpha = torch.sigmoid(torch.max(alpha, -1, True)[0]).expand(his_knl_out.shape)
        knl_out = torch.mul(his_knl_out, alpha) + torch.mul(last_knl_out, 1-alpha) + last_knl_out
        src_out = torch.mul(last_his_out, alpha) + torch.mul(last_bank, 1-alpha) + last_bank

        return knl_out.transpose(0, 1).contiguous(), src_out.transpose(0, 1).contiguous(), knl_bank.transpose(0, 1).contiguous()
