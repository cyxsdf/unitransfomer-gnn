import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # 引入GAT图神经网络

from modules.unimf import MultimodalTransformerEncoder
from modules.transformer import TransformerEncoder
from transformers import BertTokenizer, BertModel


class TRANSLATEModel(nn.Module):
    def __init__(self, hyp_params, missing=None):
        """
        Construct a Translate model with hierarchical Transformer + GNN architecture.
        """
        super(TRANSLATEModel, self).__init__()
        if hyp_params.dataset == 'meld_senti' or hyp_params.dataset == 'meld_emo':
            self.l_len, self.a_len = hyp_params.l_len, hyp_params.a_len
            self.orig_d_l, self.orig_d_a = hyp_params.orig_d_l, hyp_params.orig_d_a
            self.v_len, self.orig_d_v = 0, 0
        else:
            self.l_len, self.a_len, self.v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
            self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v

        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.trans_layers = hyp_params.trans_layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.trans_dropout = hyp_params.trans_dropout
        self.modalities = hyp_params.modalities  # 输入模态
        self.missing = missing  # 缺失模态标记

        # 位置嵌入和模态类型嵌入
        self.position_embeddings = nn.Embedding(max(self.l_len, self.a_len, self.v_len), self.embed_dim)
        self.modal_type_embeddings = nn.Embedding(4, self.embed_dim)

        # 多模态标记
        self.multi = nn.Parameter(torch.Tensor(1, self.embed_dim))
        nn.init.xavier_uniform_(self.multi)

        # 翻译模块（底层Transformer）
        self.translator = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            lens=(self.l_len, self.a_len, self.v_len),
            layers=self.trans_layers,
            modalities=self.modalities,
            missing=self.missing,
            attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout
        )

        # 高层GNN模块 - 用于跨模态全局语义建模
        self.gnn = GATConv(self.embed_dim, self.embed_dim, heads=1, concat=False)  # 单头GAT
        self.gnn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.edge_index = self.build_cross_modal_edges()  # 预构建跨模态边
        self.register_buffer('cached_edge_index', self.edge_index)  # 缓存边索引

        # 模态投影层
        if 'L' in self.modalities or self.missing == 'L':
            self.proj_l = nn.Linear(self.orig_d_l, self.embed_dim)
        if 'A' in self.modalities or self.missing == 'A':
            self.proj_a = nn.Linear(self.orig_d_a, self.embed_dim)
        if 'V' in self.modalities or self.missing == 'V':
            self.proj_v = nn.Linear(self.orig_d_v, self.embed_dim)

        # 输出层
        if self.missing == 'L':
            self.out = nn.Linear(self.embed_dim, self.orig_d_l)
        elif self.missing == 'A':
            self.out = nn.Linear(self.embed_dim, self.orig_d_a)
        elif self.missing == 'V':
            self.out = nn.Linear(self.embed_dim, self.orig_d_v)
        else:
            raise ValueError('Unknown missing modality type')

    def build_cross_modal_edges(self):
        """构建跨模态边索引（根据输入模态动态生成）"""
        modal_indices = []
        current_idx = 0

        # 记录各模态的索引范围
        if 'L' in self.modalities and self.missing != 'L':
            modal_indices.append((current_idx, current_idx + self.l_len - 1))
            current_idx += self.l_len
        if 'A' in self.modalities and self.missing != 'A':
            modal_indices.append((current_idx, current_idx + self.a_len - 1))
            current_idx += self.a_len
        if 'V' in self.modalities and self.missing != 'V':
            modal_indices.append((current_idx, current_idx + self.v_len - 1))

        # 构建模态间连接（跨模态全连接）
        edges = []
        for i in range(len(modal_indices)):
            for j in range(i + 1, len(modal_indices)):
                start_i, end_i = modal_indices[i]
                start_j, end_j = modal_indices[j]
                # 采样节点以减少边数量，避免计算量过大
                sample_i = torch.linspace(start_i, end_i, min(10, end_i - start_i + 1)).long()
                sample_j = torch.linspace(start_j, end_j, min(10, end_j - start_j + 1)).long()
                for u in sample_i:
                    for v in sample_j:
                        edges.append((u, v))
                        edges.append((v, u))  # 无向图

        if not edges:  # 处理没有跨模态边的情况
            return torch.zeros((2, 0), dtype=torch.long)

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def forward(self, src, tgt, phase='train', eval_start=False):
        """
        src and tgt should have dimension [batch_size, seq_len, n_features]
        """
        # 模态特征投影和转置
        if self.modalities == 'L':
            if self.missing == 'A':
                x_l, x_a = src, tgt
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_l = x_l.transpose(0, 1)  # (seq, batch, embed_dim)
                x_a = x_a.transpose(0, 1)
            elif self.missing == 'V':
                x_l, x_v = src, tgt
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_l = x_l.transpose(0, 1)
                x_v = x_v.transpose(0, 1)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'A':
            if self.missing == 'L':
                x_a, x_l = src, tgt
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_a = x_a.transpose(0, 1)
                x_l = x_l.transpose(0, 1)
            elif self.missing == 'V':
                x_a, x_v = src, tgt
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_a = x_a.transpose(0, 1)
                x_v = x_v.transpose(0, 1)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'V':
            if self.missing == 'L':
                x_v, x_l = src, tgt
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
                x_v = x_v.transpose(0, 1)
                x_l = x_l.transpose(0, 1)
            elif self.missing == 'A':
                x_v, x_a = src, tgt
                x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
                x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
                x_v = x_v.transpose(0, 1)
                x_a = x_a.transpose(0, 1)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'LA':
            (x_l, x_a), x_v = src, tgt
            x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
            x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
            x_l = x_l.transpose(0, 1)
            x_a = x_a.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
        elif self.modalities == 'LV':
            (x_l, x_v), x_a = src, tgt
            x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
            x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
            x_l = x_l.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
            x_a = x_a.transpose(0, 1)
        elif self.modalities == 'AV':
            (x_a, x_v), x_l = src, tgt
            x_a = F.dropout(F.relu(self.proj_a(x_a)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v)), p=self.trans_dropout, training=self.training)
            x_l = F.dropout(F.relu(self.proj_l(x_l)), p=self.trans_dropout, training=self.training)
            x_a = x_a.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
            x_l = x_l.transpose(0, 1)
        else:
            raise ValueError('Unknown modalities type')

        # 模态类型嵌入
        L_MODAL_TYPE_IDX = 0
        A_MODAL_TYPE_IDX = 1
        V_MODAL_TYPE_IDX = 2

        # 准备[Uni]标记
        batch_size = tgt.shape[0]
        multi = self.multi.unsqueeze(1).repeat(1, batch_size, 1)

        # 处理训练/测试阶段的标记拼接
        if phase != 'test':
            if self.missing == 'L':
                x_l = torch.cat((multi, x_l[:-1]), dim=0)
            elif self.missing == 'A':
                x_a = torch.cat((multi, x_a[:-1]), dim=0)
            elif self.missing == 'V':
                x_v = torch.cat((multi, x_v[:-1]), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        else:
            if eval_start:
                if self.missing == 'L':
                    x_l = multi  # 使用[Uni]作为生成缺失模态的起始
                elif self.missing == 'A':
                    x_a = multi
                elif self.missing == 'V':
                    x_v = multi
                else:
                    raise ValueError('Unknown missing modality type')
            else:
                if self.missing == 'L':
                    x_l = torch.cat((multi, x_l), dim=0)
                elif self.missing == 'A':
                    x_a = torch.cat((multi, x_a), dim=0)
                elif self.missing == 'V':
                    x_v = torch.cat((multi, x_v), dim=0)
                else:
                    raise ValueError('Unknown missing modality type')

        # 添加位置嵌入和模态类型嵌入
        if 'L' in self.modalities or self.missing == 'L':
            x_l_pos_ids = torch.arange(x_l.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            l_pos_embeds = self.position_embeddings(x_l_pos_ids)
            l_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_l_pos_ids, L_MODAL_TYPE_IDX))
            l_embeds = l_pos_embeds + l_modal_type_embeds
            x_l = x_l + l_embeds
            x_l = F.dropout(x_l, p=self.embed_dropout, training=self.training)
        if 'A' in self.modalities or self.missing == 'A':
            x_a_pos_ids = torch.arange(x_a.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            a_pos_embeds = self.position_embeddings(x_a_pos_ids)
            a_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_a_pos_ids, A_MODAL_TYPE_IDX))
            a_embeds = a_pos_embeds + a_modal_type_embeds
            x_a = x_a + a_embeds
            x_a = F.dropout(x_a, p=self.embed_dropout, training=self.training)
        if 'V' in self.modalities or self.missing == 'V':
            x_v_pos_ids = torch.arange(x_v.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            v_pos_embeds = self.position_embeddings(x_v_pos_ids)
            v_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_v_pos_ids, V_MODAL_TYPE_IDX))
            v_embeds = v_pos_embeds + v_modal_type_embeds
            x_v = x_v + v_embeds
            x_v = F.dropout(x_v, p=self.embed_dropout, training=self.training)

        # 拼接多模态特征
        if self.modalities == 'L':
            if self.missing == 'A':
                x = torch.cat((x_l, x_a), dim=0)
            elif self.missing == 'V':
                x = torch.cat((x_l, x_v), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'A':
            if self.missing == 'L':
                x = torch.cat((x_a, x_l), dim=0)
            elif self.missing == 'V':
                x = torch.cat((x_a, x_v), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'V':
            if self.missing == 'L':
                x = torch.cat((x_v, x_l), dim=0)
            elif self.missing == 'A':
                x = torch.cat((x_v, x_a), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'LA':
            x = torch.cat((x_l, x_a, x_v), dim=0)
        elif self.modalities == 'LV':
            x = torch.cat((x_l, x_v, x_a), dim=0)
        elif self.modalities == 'AV':
            x = torch.cat((x_a, x_v, x_l), dim=0)
        else:
            raise ValueError('Unknown modalities type')

        # 1. 底层Transformer提取局部依赖
        transformer_output = self.translator(x)

        # 2. 高层GNN建模跨模态全局语义关系
        # 调整维度: (seq_len, batch, embed_dim) -> (batch, seq_len, embed_dim)
        gnn_input = transformer_output.permute(1, 0, 2)

        # 关键修改：将3D输入转换为2D以适应GATConv
        batch_size = gnn_input.size(0)
        seq_len = gnn_input.size(1)
        embed_dim = gnn_input.size(2)

        # 转换为2D张量 [batch_size * seq_len, embed_dim]
        gnn_input_2d = gnn_input.view(-1, embed_dim)

        # 验证维度
        assert gnn_input_2d.dim() == 2, f"GAT输入必须是2D张量，实际是{gnn_input_2d.dim()}D"

        # 处理边索引（为批次偏移）
        edge_index = self.cached_edge_index.to(gnn_input.device)
        if edge_index.numel() > 0 and batch_size > 1:
            # 为每个批次偏移边索引
            edge_index = edge_index.repeat(1, batch_size)
            batch_offsets = torch.arange(0, batch_size * seq_len, seq_len, device=gnn_input.device)
            for i in range(batch_size):
                start_idx = i * self.cached_edge_index.size(1)
                end_idx = (i + 1) * self.cached_edge_index.size(1)
                edge_index[:, start_idx:end_idx] += batch_offsets[i]

        # 应用GATConv
        gnn_output_2d = self.gnn(gnn_input_2d, edge_index)

        # 恢复为3D张量 [batch_size, seq_len, embed_dim]
        gnn_output = gnn_output_2d.view(batch_size, seq_len, embed_dim)
        gnn_output = self.gnn_layer_norm(gnn_output)

        # 恢复维度: (batch, seq_len, embed_dim) -> (seq_len, batch, embed_dim)
        final_output = gnn_output.permute(1, 0, 2)

        # 提取输出特征
        if self.modalities == 'L':
            output = final_output[self.l_len:].transpose(0, 1)  # (batch, seq, embed_dim)
        elif self.modalities == 'A':
            output = final_output[self.a_len:].transpose(0, 1)
        elif self.modalities == 'V':
            output = final_output[self.v_len:].transpose(0, 1)
        elif self.modalities == 'LA':
            output = final_output[self.l_len + self.a_len:].transpose(0, 1)
        elif self.modalities == 'LV':
            output = final_output[self.l_len + self.v_len:].transpose(0, 1)
        elif self.modalities == 'AV':
            output = final_output[self.a_len + self.v_len:].transpose(0, 1)
        else:
            raise ValueError('Unknown modalities type')

        output = self.out(output)
        return output


class UNIMFModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a UniMF model with hierarchical Transformer + GNN architecture.
        """
        super(UNIMFModel, self).__init__()
        if hyp_params.dataset == 'meld_senti' or hyp_params.dataset == 'meld_emo':
            self.orig_l_len, self.orig_a_len = hyp_params.l_len, hyp_params.a_len
            self.orig_d_l, self.orig_d_a = hyp_params.orig_d_l, hyp_params.orig_d_a
        else:
            self.orig_l_len, self.orig_a_len, self.orig_v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
            self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v

        self.l_kernel_size = hyp_params.l_kernel_size
        self.a_kernel_size = hyp_params.a_kernel_size
        if hyp_params.dataset != 'meld_senti' and hyp_params.dataset != 'meld_emo':
            self.v_kernel_size = hyp_params.v_kernel_size
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.multimodal_layers = hyp_params.multimodal_layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.modalities = hyp_params.modalities
        self.dataset = hyp_params.dataset
        self.language = hyp_params.language
        self.use_bert = hyp_params.use_bert

        self.distribute = hyp_params.distribute

        # CLS标记长度
        if self.dataset == 'meld_senti' or self.dataset == 'meld_emo':
            self.cls_len = 33
        else:
            self.cls_len = 1
        self.cls = nn.Parameter(torch.Tensor(self.cls_len, self.embed_dim))
        nn.init.xavier_uniform_(self.cls)

        # 计算卷积后的序列长度
        self.l_len = self.orig_l_len - self.l_kernel_size + 1
        self.a_len = self.orig_a_len - self.a_kernel_size + 1
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            self.v_len = self.orig_v_len - self.v_kernel_size + 1

        output_dim = hyp_params.output_dim  # 输出维度

        # BERT模型（如果使用）
        if self.use_bert:
            self.text_model = BertTextEncoder(language=hyp_params.language, use_finetune=True)

        # 1. 时序卷积块
        self.proj_l = nn.Conv1d(self.orig_d_l, self.embed_dim, kernel_size=self.l_kernel_size)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.embed_dim, kernel_size=self.a_kernel_size)
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            self.proj_v = nn.Conv1d(self.orig_d_v, self.embed_dim, kernel_size=self.v_kernel_size)
        if 'meld' in self.dataset:
            self.proj_cls = nn.Conv1d(self.orig_d_l + self.orig_d_a, self.embed_dim, kernel_size=1)

        # 2. GRU编码器
        self.t = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)
        self.a = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            self.v = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)

        # 3. 多模态融合块
        # 3.1. 位置嵌入和模态类型嵌入
        if self.dataset == 'meld_senti' or self.dataset == 'meld_emo':
            self.position_embeddings = nn.Embedding(max(self.cls_len, self.l_len, self.a_len), self.embed_dim)
        else:
            self.position_embeddings = nn.Embedding(max(self.l_len, self.a_len, self.v_len), self.embed_dim)
        self.modal_type_embeddings = nn.Embedding(4, self.embed_dim)

        # 3.2. UniMF（底层Transformer）
        self.unimf = MultimodalTransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            layers=self.multimodal_layers,
            lens=(self.cls_len, self.l_len, self.a_len),
            modalities=self.modalities,
            attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout
        )

        # 3.3. 高层GNN模块 - 跨模态全局语义建模
        self.gnn = GATConv(self.embed_dim, self.embed_dim, heads=1, concat=False)  # 单头GAT
        self.gnn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.edge_index = self.build_cross_modal_edges()  # 构建跨模态边
        self.register_buffer('cached_edge_index', self.edge_index)  # 缓存边索引

        # 4. 投影层
        combined_dim = self.embed_dim
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def build_cross_modal_edges(self):
        """为UNIMF模型构建跨模态边"""
        modal_indices = []
        current_idx = 0

        # 添加CLS标记的索引范围
        modal_indices.append((current_idx, current_idx + self.cls_len - 1))
        current_idx += self.cls_len

        # 添加各模态的索引范围
        if 'L' in self.modalities:
            modal_indices.append((current_idx, current_idx + self.l_len - 1))
            current_idx += self.l_len
        if 'A' in self.modalities:
            modal_indices.append((current_idx, current_idx + self.a_len - 1))
            current_idx += self.a_len
        if 'V' in self.modalities and self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            modal_indices.append((current_idx, current_idx + self.v_len - 1))

        # 构建模态间连接
        edges = []
        for i in range(len(modal_indices)):
            for j in range(i + 1, len(modal_indices)):
                start_i, end_i = modal_indices[i]
                start_j, end_j = modal_indices[j]
                # 采样节点以减少边数量
                sample_i = torch.linspace(start_i, end_i, min(10, end_i - start_i + 1)).long()
                sample_j = torch.linspace(start_j, end_j, min(10, end_j - start_j + 1)).long()
                for u in sample_i:
                    for v in sample_j:
                        edges.append((u, v))
                        edges.append((v, u))  # 无向图

        if not edges:  # 处理没有跨模态边的情况
            return torch.zeros((2, 0), dtype=torch.long)

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def forward(self, x_l, x_a, x_v=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        if self.distribute:
            self.t.flatten_parameters()
            self.a.flatten_parameters()
            if x_v is not None:
                self.v.flatten_parameters()

        # 模态类型嵌入索引
        L_MODAL_TYPE_IDX = 0
        A_MODAL_TYPE_IDX = 1
        V_MODAL_TYPE_IDX = 2
        MULTI_MODAL_TYPE_IDX = 3

        # 准备[CLS]标记
        batch_size = x_l.shape[0]
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            cls = self.cls.unsqueeze(1).repeat(1, batch_size, 1)
        else:
            cls = self.proj_cls(torch.cat((x_l, x_a), dim=-1).transpose(1, 2)).permute(2, 0, 1)

        # 准备位置嵌入和模态类型嵌入
        cls_pos_ids = torch.arange(self.cls_len, device=x_l.device).unsqueeze(1).expand(-1, batch_size)
        h_l_pos_ids = torch.arange(self.l_len, device=x_l.device).unsqueeze(1).expand(-1, batch_size)
        h_a_pos_ids = torch.arange(self.a_len, device=x_a.device).unsqueeze(1).expand(-1, batch_size)
        if x_v is not None:
            h_v_pos_ids = torch.arange(self.v_len, device=x_v.device).unsqueeze(1).expand(-1, batch_size)

        cls_pos_embeds = self.position_embeddings(cls_pos_ids)
        h_l_pos_embeds = self.position_embeddings(h_l_pos_ids)
        h_a_pos_embeds = self.position_embeddings(h_a_pos_ids)
        if x_v is not None:
            h_v_pos_embeds = self.position_embeddings(h_v_pos_ids)

        cls_modal_type_embeds = self.modal_type_embeddings(torch.full_like(cls_pos_ids, MULTI_MODAL_TYPE_IDX))
        l_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_l_pos_ids, L_MODAL_TYPE_IDX))
        a_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_a_pos_ids, A_MODAL_TYPE_IDX))
        if x_v is not None:
            v_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_v_pos_ids, V_MODAL_TYPE_IDX))

        # 投影文本/视觉/音频特征并压缩序列长度
        if self.use_bert:
            x_l = self.text_model(x_l)

        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        if x_v is not None:
            x_v = x_v.transpose(1, 2)

        proj_x_l = self.proj_l(x_l)
        proj_x_a = self.proj_a(x_a)
        if x_v is not None:
            proj_x_v = self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        if x_v is not None:
            proj_x_v = proj_x_v.permute(2, 0, 1)

        # 使用GRU编码
        h_l, _ = self.t(proj_x_l)
        h_a, _ = self.a(proj_x_a)
        if x_v is not None:
            h_v, _ = self.v(proj_x_v)

        # 添加位置和模态类型嵌入
        cls_embeds = cls_pos_embeds + cls_modal_type_embeds
        l_embeds = h_l_pos_embeds + l_modal_type_embeds
        a_embeds = h_a_pos_embeds + a_modal_type_embeds
        if x_v is not None:
            v_embeds = h_v_pos_embeds + v_modal_type_embeds
        cls = cls + cls_embeds
        h_l = h_l + l_embeds
        h_a = h_a + a_embeds
        if x_v is not None:
            h_v = h_v + v_embeds
        h_l = F.dropout(h_l, p=self.embed_dropout, training=self.training)
        h_a = F.dropout(h_a, p=self.embed_dropout, training=self.training)
        if x_v is not None:
            h_v = F.dropout(h_v, p=self.embed_dropout, training=self.training)

        # 多模态融合
        # 1. 拼接所有特征并输入到底层Transformer
        if x_v is not None:
            x = torch.cat((cls, h_l, h_a, h_v), dim=0)
        else:
            x = torch.cat((cls, h_l, h_a), dim=0)
        transformer_output = self.unimf(x)

        # 2. 高层GNN建模跨模态全局关系
        # 调整维度: (seq_len, batch, embed_dim) -> (batch, seq_len, embed_dim)
        gnn_input = transformer_output.permute(1, 0, 2)

        # 关键修改：将3D输入转换为2D以适应GATConv
        batch_size = gnn_input.size(0)
        seq_len = gnn_input.size(1)
        embed_dim = gnn_input.size(2)

        # 转换为2D张量 [batch_size * seq_len, embed_dim]
        gnn_input_2d = gnn_input.view(-1, embed_dim)

        # 验证维度
        assert gnn_input_2d.dim() == 2, f"GAT输入必须是2D张量，实际是{gnn_input_2d.dim()}D"

        # 处理边索引（为批次偏移）
        edge_index = self.cached_edge_index.to(gnn_input.device)
        if edge_index.numel() > 0 and batch_size > 1:
            # 为每个批次偏移边索引
            edge_index = edge_index.repeat(1, batch_size)
            batch_offsets = torch.arange(0, batch_size * seq_len, seq_len, device=gnn_input.device)
            for i in range(batch_size):
                start_idx = i * self.cached_edge_index.size(1)
                end_idx = (i + 1) * self.cached_edge_index.size(1)
                edge_index[:, start_idx:end_idx] += batch_offsets[i]

        # 应用GATConv
        gnn_output_2d = self.gnn(gnn_input_2d, edge_index)

        # 恢复为3D张量 [batch_size, seq_len, embed_dim]
        gnn_output = gnn_output_2d.view(batch_size, seq_len, embed_dim)
        gnn_output = self.gnn_layer_norm(gnn_output)

        # 恢复维度
        x = gnn_output.permute(1, 0, 2)

        # 获取[CLS]标记用于预测
        if x_v is not None:
            last_hs = x[0]  # 单CLS标记
        else:
            last_hs = x[:self.cls_len]  # 多CLS标记

        # 残差块处理
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        if x_v is None:
            output = output.transpose(0, 1)
        return output, last_hs


class BertTextEncoder(nn.Module):
    def __init__(self, language='en', use_finetune=False):
        """
        language: en / cn
        """
        super(BertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_bert/bert_en', do_lower_case=True)
            self.model = model_class.from_pretrained('pretrained_bert/bert_en')
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_bert/bert_cn')
            self.model = model_class.from_pretrained('pretrained_bert/bert_cn')

        self.use_finetune = use_finetune

    def get_tokenizer(self):
        return self.tokenizer

    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        """
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if self.use_finetune:
            last_hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids
            )[0]
        else:
            with torch.no_grad():
                last_hidden_states = self.model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids
                )[0]
        return last_hidden_states
