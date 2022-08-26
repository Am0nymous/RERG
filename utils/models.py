import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import rnn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(self.act(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def inner_(self, x, y, z, flag):
        # referenced by https://github.com/tensorflow/tensor2tensor/blob/b63f2d9803385c494176f7eb5acf728c152d00f8/tensor2tensor/layers/common_attention.py
        batch_size, heads, length, dim = x.size()
        # z is [batch, length, length, dim]
        # xy_matmul is [batch, heads, length, length or dim]
        if flag == True:
            y = y.transpose(2, 3)
        xy_matmul = torch.matmul(x, y)
        # x_t is [length, batch, head, dim or length]
        x_t = x.permute(2, 0, 1, 3)
        # z is [length, batch, length, dim]
        z = z.permute(1, 0, 2, 3)
        if flag == True:
            z = z.transpose(2, 3)
        # x_t_r is [length, batch * head, length]
        # x_t_r = x_t.contiguous().view(length, heads * batch_size, -1)
        # x_tz_matmul is [length, batch * heads, length]
        # z is [length, batch, length, dim]
        # x_tz_matmul is [length, batch, heads, length]
        x_tz_matmul = torch.matmul(x_t, z)
        # x_tz_matmul_r is [length, batch, heads, length]
        x_tz_matmul_r = x_tz_matmul.contiguous().view(length, batch_size, heads, -1)
        # x_tz_matmul_r_t is [batch, heads, length, length]
        x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
        return xy_matmul + x_tz_matmul_r_t

    def forward(self, q, k, v, adj_k, adj_v, mask=None):
        # q: batch x heads x length x dim
        # k: batch x heads x length x dim
        # adj: batch x length x length x dim
        attn = self.inner_(q / self.temperature, k, adj_k, True)
        # attn is [batch, heads, length, length]
        # attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e4)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = self.inner_(attn, v, adj_v, False)
        return output, attn


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class LSTMWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layer, concat=False, bidir=True, dropout=0.3, return_last=True):
        super(LSTMWrapper, self).__init__()
        self.rnns = nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                input_dim_ = input_dim
                output_dim_ = hidden_dim
            else:
                input_dim_ = hidden_dim if not bidir else hidden_dim * 2
                output_dim_ = hidden_dim
            self.rnns.append(nn.LSTM(input_dim_, output_dim_, 1, bidirectional=bidir, batch_first=True))
        self.dropout = dropout
        self.concat = concat
        self.n_layer = n_layer
        self.return_last = return_last

    def forward(self, input, input_lengths=None):
        # input_length must be in decreasing order
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []

        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()

        for i in range(self.n_layer):
            output = F.dropout(output, p=self.dropout, training=self.training)

            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
            self.rnns[i].flatten_parameters()
            output, _ = self.rnns[i](output)

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen - output.size(1), output.size(2))],
                                       dim=1)

            outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]

class MultiHeadAttention_graph(nn.Module):
    ''' Multi-Head Attention module '''

    # Attention + GNN
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_k, n_head * d_k, bias=False)
        self.act = GeLU()
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.retain = nn.Linear(d_model, 1, bias=False)
        self._node_weight_fc = nn.Linear(n_head * d_k, 1, bias=False)
        self._self_node_fc = nn.Linear(n_head * d_k, n_head * d_k, bias=False)
        self._dd_node_fc = nn.Linear(n_head * d_k, n_head * d_k, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_head * d_k, eps=1e-12)
        # self.edge_weight_tensor_k = torch.Tensor(100, 64)
        # self.edge_weight_tensor_k = nn.Parameter(nn.init.xavier_uniform_(self.edge_weight_tensor_k))
        # self.edge_weight_tensor_v = torch.Tensor(100, 64)
        # self.edge_weight_tensor_v = nn.Parameter(nn.init.xavier_uniform_(self.edge_weight_tensor_v))

    def forward(self, q, k, v, vec_adj_k, vec_adj_v, adj_k, adj_v, mask=None):
        # vec_adj_k = F.embedding(adj_k.long(), self.edge_weight_tensor_k, padding_idx=0)
        # vec_adj_v = F.embedding(adj_v.long(), self.edge_weight_tensor_v, padding_idx=0)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        # adj_k : b x node x node x dv
        # adj_v: b x node x node x dv
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, vec_adj_k, vec_adj_v, mask=adj_k.long().unsqueeze(1))
        del k, v
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = q.contiguous().view(sz_b, len_q, -1)
        # adj_k = adj_k.repeat(n_head, 1, 1)
        d_node_len = q.size(1)
        node_neighbor_num = adj_k.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        d_node_neighbor_num = node_neighbor_num.float().masked_fill((1 - node_neighbor_num_mask).to(dtype=torch.bool),
                                                                    1)
        # update itself information
        node_weight = torch.sigmoid(self._node_weight_fc(q)).squeeze(-1)
        self_d_node_info = self._self_node_fc(q)
        # integrate neighbor information
        dd_node_info = self._dd_node_fc(q)
        dd_node_weight = node_weight.unsqueeze(1).expand(-1, d_node_len, -1). \
            masked_fill((1 - adj_k).to(dtype=torch.bool), 0)
        dd_node_info = torch.matmul(dd_node_weight, dd_node_info)

        agg_d_node_info = (dd_node_info) / d_node_neighbor_num.unsqueeze(-1)

        q = self_d_node_info + agg_d_node_info
        q = self.dropout(self.act(self.fc(q)))
        # q = q.contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        # probs = torch.sigmoid(self.retain(q))
        # q = residual * (torch.tensor(1.) - probs) + probs * q
        q = residual + q

        # print(q.shape)

        # q = self.layer_norm(q)

        # q = self.layer_norm(q)
        del self_d_node_info, agg_d_node_info, dd_node_info, residual
        # get gate values
        # gate = torch.sigmoid(self.fa(torch.cat((update, cur_input), -1))) * input_mask.unsqueeze(
        # -1)  # bs x max_node x node_dims

        # apply gate values
        # cur_input = gate * self.act(update) + (1 - gate) * cur_input  # bs x max_node x node_dim
        return q, attn
        # return q, attn


class FC_net(nn.Module):
    def __init__(self):
        super(FC_net, self).__init__()
        for i in range(6):
            setattr(self, "fc{}".format(i), nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Linear(128, 2)))

class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))