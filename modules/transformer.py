import torch
from torch import nn
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.multihead_attention import MultiheadAttention
from torch_geometric.nn import GATConv  # 引入GAT
import math


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, lens, modalities, missing, attn_dropout=0.0, relu_dropout=0.0,
                 res_dropout=0.0, embed_dropout=0.0, attn_mask=False, embed_positions=None):
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.lens = lens  # 各模态长度
        self.modalities = modalities
        self.missing = missing

        if embed_positions is not None:
            self.embed_scale = math.sqrt(embed_dim)
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        else:
            self.embed_scale = 1
            self.embed_positions = embed_positions

        self.attn_mask = attn_mask

        # Transformer层
        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(
                embed_dim, lens=lens, modalities=modalities, missing=missing,
                num_heads=num_heads, attn_dropout=attn_dropout,
                relu_dropout=relu_dropout, res_dropout=res_dropout,
                attn_mask=attn_mask
            )
            self.layers.append(new_layer)

        # 高层GNN模块 - 确保GATConv配置正确
        self.gnn = GATConv(embed_dim, embed_dim, heads=1, concat=False)  # 单头GAT
        self.gnn_layer_norm = LayerNorm(embed_dim)
        self.edge_index = self.build_cross_modal_edges()  # 跨模态边

        # 缓存边索引，避免重复创建
        self.register_buffer('cached_edge_index', self.edge_index)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def build_cross_modal_edges(self):
        """构建跨模态边（排除缺失模态）"""
        modal_nodes = []
        idx = 0
        # 按存在的模态分割节点
        if 'L' in self.modalities and self.missing != 'L':
            modal_nodes.append((idx, idx + self.lens[0] - 1))
            idx += self.lens[0]
        if 'A' in self.modalities and self.missing != 'A':
            modal_nodes.append((idx, idx + self.lens[1] - 1))
            idx += self.lens[1]
        if 'V' in self.modalities and self.missing != 'V':
            modal_nodes.append((idx, idx + self.lens[2] - 1))

        # 构建模态间边
        edges = []
        for i in range(len(modal_nodes)):
            for j in range(i + 1, len(modal_nodes)):
                start_i, end_i = modal_nodes[i]
                start_j, end_j = modal_nodes[j]
                # 采样节点以控制边数量
                sample_i = torch.linspace(start_i, end_i, min(10, end_i - start_i + 1)).long()
                sample_j = torch.linspace(start_j, end_j, min(10, end_j - start_j + 1)).long()
                for u in sample_i:
                    for v in sample_j:
                        edges.append((u, v))
                        edges.append((v, u))

        if not edges:  # 处理没有跨模态边的情况
            return torch.zeros((2, 0), dtype=torch.long)

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        # Transformer底层特征提取
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)

        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)

        # Transformer层传播
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        # GNN全局建模 - 确保输入是2D张量
        # 输入形状: (seq_len, batch_size, embed_dim)
        batch_size = x.size(1)
        seq_len = x.size(0)
        embed_dim = x.size(2)

        # 1. 调整为 [batch_size * seq_len, embed_dim] (严格2D)
        x_gnn = x.permute(1, 0, 2).contiguous()  # (batch, seq_len, dim)
        x_gnn = x_gnn.view(-1, embed_dim)  # (batch*seq_len, dim) - 确保是2D

        # 2. 验证维度并处理边索引
        assert x_gnn.dim() == 2, f"GAT输入必须是2D张量，实际是{x_gnn.dim()}D"

        # 处理批次边索引
        if self.cached_edge_index.numel() > 0 and batch_size > 1:
            # 为每个批次偏移边索引
            edge_index = self.cached_edge_index.repeat(1, batch_size)
            batch_offsets = torch.arange(0, batch_size * seq_len, seq_len, device=x.device)
            for i in range(batch_size):
                start_idx = i * self.cached_edge_index.size(1)
                end_idx = (i + 1) * self.cached_edge_index.size(1)
                edge_index[:, start_idx:end_idx] += batch_offsets[i]
        else:
            edge_index = self.cached_edge_index.to(x.device)

        # 3. 应用GATConv
        x_gnn_out = self.gnn(x_gnn, edge_index)  # 现在输入是2D的

        # 4. 添加残差连接和层归一化
        x_gnn = x_gnn + x_gnn_out  # 残差连接
        x_gnn = self.gnn_layer_norm(x_gnn)

        # 5. 恢复原始维度 (seq_len, batch, dim)
        x_gnn = x_gnn.view(batch_size, seq_len, embed_dim).permute(1, 0, 2)

        return x_gnn

    def max_positions(self):
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, lens, modalities, missing, num_heads=4, attn_dropout=0.1, relu_dropout=0.1,
                 res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads,
            lens=lens, modalities=modalities, missing=missing,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


if __name__ == '__main__':
    # 测试代码，验证GAT输入维度
    encoder = TransformerEncoder(300, 4, 2, lens=(10, 20, 15), modalities='LAV', missing=None)
    x = torch.rand(45, 2, 300)  # (seq_len, batch_size, embed_dim)

    # 检查GAT输入维度
    batch_size = x.size(1)
    seq_len = x.size(0)
    embed_dim = x.size(2)
    x_gnn_test = x.permute(1, 0, 2).contiguous().view(-1, embed_dim)
    print(f"GAT测试输入维度: {x_gnn_test.shape}, 维度数: {x_gnn_test.dim()} (应为2)")

    output = encoder(x)
    print(f"输出维度: {output.shape} (应输出 (45, 2, 300))")
