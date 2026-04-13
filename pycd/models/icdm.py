import wandb as wb
import numpy as np
import scipy.sparse as sp
import dgl
import dgl.nn as dglnn
# from dgl.nn.pytorch import SAGEConv 原本就是注释
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pycd.models import BaseCDModel
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from tqdm import tqdm
import warnings
from dgl.base import DGLError
from dgl import function as fn
from dgl.utils import check_eq_shape, expand_as_pair
from dgl.nn.pytorch import GATConv, GATv2Conv, GraphConv

# 新增PyTorch Geometric相关导入
from torch_geometric.nn import SAGEConv, GATv2Conv, GCNConv
# 导入图格式转换工具（补充文件中的函数）
from pycd.utils.graph_convert import dgl_to_pyg

from pycd.train.trainer import Trainer, EarlyStopping  #新增早停
from pycd.utils.logging import init_logger, get_experiment_dir, save_experiment_config #新增保存地址
import os

# from runners.commonutils.datautils import transform 需要, get_doa_function, get_r_matrix , get_group_acc
# transform 使用torch的数据加载，照搬还是修改？     get_doa_function 计算DOA 能否替换   get_group_acc 没用上
# get_r_matrix 构建r矩阵，照搬？ 默认用-1表示未作答，新学生需要重新编号，即0，1替换-1
'''
np_test = [
 [0, 1, 1],  # 学生0 做 题目1 得分1
 [0, 2, 0],  # 学生0 做 题目2 得分0
 [1, 0, 1]   # 学生1 做 题目0 得分1
]
第一类学生id，第二列题目id,第三列得分情况
得到的r矩阵是       q1   q2   q3  ...
              s1    -1   0    1   ...
              s2     1   0    -1  ...
              s3
'''
from torch.utils.data import TensorDataset, DataLoader  # transform 需要
from joblib import Parallel, delayed  # get_doa_function 需要


def transform(q: torch.tensor, user, item, score, batch_size, dtype=torch.float64):
    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64),
        torch.tensor(item, dtype=torch.int64),
        q[item, :],
        torch.tensor(score, dtype=dtype)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def get_doa_function(know_num):
    if know_num == 734:
        doa_func = DOA_Junyi
    elif know_num == 835:
        doa_func = DOA_Junyi835
    elif know_num == 123:
        doa_func = DOA_Assist910
    elif know_num == 102:
        doa_func = DOA_Assist17
    elif know_num == 268:
        doa_func = DOA_Nips20
    elif know_num == 95:
        doa_func = DOA_Assist09
    elif know_num == 189:
        doa_func = DOA_EdNet_1
    else:
        doa_func = DOA
    return doa_func


def calculate_doa_k_block(mas_level, q_matrix, r_matrix, k, block_size=50):
    n_students, n_skills = mas_level.shape
    n_questions, _ = q_matrix.shape
    n_attempts = r_matrix.shape[1]
    DOA_k = 0.0
    numerator = 0
    denominator = 0
    question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
    for start in range(0, n_students, block_size):
        end = min(start + block_size, n_students)
        mas_level_block = mas_level[start:end, :]
        delta_matrix_block = mas_level[start:end, k].reshape(-1, 1) > mas_level[start:end, k].reshape(1, -1)
        r_matrix_block = r_matrix[start:end, :]
        for j in question_hask:
            row_vector = (r_matrix_block[:, j].reshape(1, -1) != -1).astype(int)
            columen_vector = (r_matrix_block[:, j].reshape(-1, 1) != -1).astype(int)
            mask = row_vector * columen_vector
            delta_r_matrix = r_matrix_block[:, j].reshape(-1, 1) > r_matrix_block[:, j].reshape(1, -1)
            I_matrix = r_matrix_block[:, j].reshape(-1, 1) != r_matrix_block[:, j].reshape(1, -1)
            numerator_ = np.logical_and(mask, delta_r_matrix)
            denominator_ = np.logical_and(mask, I_matrix)
            numerator += np.sum(delta_matrix_block * numerator_)
            denominator += np.sum(delta_matrix_block * denominator_)
    if denominator == 0:
        DOA_k = 0
    else:
        DOA_k = numerator / denominator
    return DOA_k



def DOA(mastery_level, q_matrix, r_matrix):
    know_n = q_matrix.shape[1]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k)(mastery_level, q_matrix, r_matrix, k) for k in range(know_n))
    return np.mean(doa_k_list)


def DOA_Junyi(mastery_level, q_matrix, r_matrix, concepts=None):
    if concepts is None:
        concepts = [433, 28, 653, 563, 631, 392, 632, 393, 652, 394]
    know_n = q_matrix.shape[1]
    # concepts = np.random.randint(0, know_n, 20)
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k, 2000) for k in concepts)
    return np.mean(doa_k_list)


def DOA_Junyi835(mastery_level, q_matrix, r_matrix, concepts=None):
    if concepts is None:
        concepts = [487, 31, 749, 633, 727, 442, 728, 443, 748, 32]
    know_n = q_matrix.shape[1]
    # concepts = np.random.randint(0, know_n, 20)
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k, 2000) for k in concepts)
    return np.mean(doa_k_list)


def DOA_Assist910(mastery_level, q_matrix, r_matrix):
    concepts = [98, 30, 79, 82, 49, 99, 32, 81, 45, 6]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in concepts)
    return np.mean(doa_k_list)


def DOA_Assist17(mastery_level, q_matrix, r_matrix):
    concepts = [21, 58, 14, 5, 33, 34, 10, 7, 4, 60]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in concepts)
    return np.mean(doa_k_list)


def DOA_OOD(mastery_level, q_matrix, r_matrix):
    know_n = q_matrix.shape[1]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_ood)(mastery_level, q_matrix, r_matrix, k) for k in range(know_n))
    return doa_k_list


def DOA_Nips20(mastery_level, q_matrix, r_matrix):
    concepts = [0, 1, 17, 38, 87, 8, 67, 91, 9, 30]
    # concepts = [0, 1, 36, 16, 7, 78, 62, 39, 77, 82]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in concepts)
    return np.mean(doa_k_list)


def DOA_Assist09(mastery_level, q_matrix, r_matrix):
    concepts = [82, 23, 63, 66, 35, 39, 26, 9, 83, 10]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in concepts)
    return np.mean(doa_k_list)


def DOA_EdNet_1(mastery_level, q_matrix, r_matrix):
    concepts = [181, 179, 180, 182, 177, 183, 24, 52, 2, 30]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in concepts)
    return np.mean(doa_k_list)

def DOA(mastery_level, q_matrix, r_matrix):
    # print("q shape:", q_matrix.shape)
    # print("r shape:", r_matrix.shape)
    # print("mastery_level shape:", mastery_level.shape)
    know_n = q_matrix.shape[1]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in range(know_n))
    return np.mean(doa_k_list)


def get_r_matrix(np_test, stu_num, prob_num, new_idx=None):
    if new_idx is None:
        r = -1 * np.ones(shape=(stu_num, prob_num))
        for i in range(np_test.shape[0]):
            s = int(np_test[i, 0])
            p = int(np_test[i, 1])
            score = np_test[i, 2]
            r[s, p] = int(score)
    else:
        r = -1 * np.ones(shape=(stu_num, prob_num))

        for i in range(np_test.shape[0]):
            s = new_idx.index(int(np_test[i, 0]))
            p = int(np_test[i, 1])
            score = np_test[i, 2]
            r[s, p] = int(score)
    return r


# from runners.commonutils.util import get_number_of_params, NoneNegClipper
# get_number_of_params 输出参数数量，可以不要
# NoneNegClipper 类似层权重初始化 照搬？

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


# from runners.ICDM.utils import l2_loss, dgl2tensor, concept_distill, # get_subgraph
# l2_loss计算loss的照搬？看看能不能修改
# dgl2tensor邻居矩阵转换成张量，照搬？
# concept_distill 图的消息传递，照搬？
# get_subgraph 将节点转换到GPU 照搬？

import networkx as nx  # dgl2tensor需要


def l2_loss(*weights):
    loss = 0.0
    for w in weights:
        loss += torch.sum(torch.pow(w, 2)) / w.shape[0]

    return 0.5 * loss


def dgl2tensor(g):
    nx_graph = g.to_networkx()
    adj_matrix = nx.adjacency_matrix(nx_graph).todense()
    tensor = torch.from_numpy(adj_matrix)
    return tensor


def concept_distill(matrix, concept):
    coeff = 1.0 / torch.sum(matrix, dim=1)
    concept = matrix.to(concept.dtype) @ concept
    concept_distill = concept * coeff[:, None]
    return concept_distill


def get_subgraph(g, id, device):
    return dgl.in_subgraph(g, id).to(device)


# from dgl.nn.pytorch import SAGEConv 原本就是注释的

warnings.filterwarnings('ignore')


# 构建三个图
def build_graph4CE(config: dict):
    q = config['q']
    q = q.detach().cpu().numpy()
    know_num = int(config['know_num'])
    exer_num = int(config['prob_num'])
    node = exer_num + know_num
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []
    indices = np.where(q != 0)
    for exer_id, know_id in zip(indices[0].tolist(), indices[1].tolist()):
        edge_list.append((int(know_id + exer_num), int(exer_id)))
        edge_list.append((int(exer_id), int(know_id + exer_num)))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    return g


def build_graph4SE(config: dict, mode='tl'):
    if mode == 'tl':
        data = config['np_train']
    elif mode == 'ind_train':
        data = config['np_train_old']
    else:
        data = np.vstack((config['np_train_old'], config['np_train_new']))

    stu_num = config['stu_num']
    exer_num = config['prob_num']
    node = stu_num + exer_num
    g_right = dgl.DGLGraph()
    g_right.add_nodes(node)
    g_wrong = dgl.DGLGraph()
    g_wrong.add_nodes(node)

    right_edge_list = []
    wrong_edge_list = []

    for index in range(data.shape[0]):
        stu_id = int(data[index, 0])
        exer_id = int(data[index, 1])
        if int(float(data[index, 2])) == 1:
            if mode == 'tl' or mode == 'ind_train' or int(stu_id) in config['exist_idx']:
                right_edge_list.append((int(stu_id), int(exer_id + stu_num)))
                right_edge_list.append((int(exer_id + stu_num), int(stu_id)))
            else:
                right_edge_list.append((int(exer_id + stu_num), int(stu_id)))
        else:
            if mode == 'tl' or mode == 'ind_train' or int(stu_id) in config['exist_idx']:
                wrong_edge_list.append((int(stu_id), int(exer_id + stu_num)))
                wrong_edge_list.append((int(exer_id + stu_num), int(stu_id)))
            else:
                wrong_edge_list.append((int(exer_id + stu_num), int(stu_id)))
    right_src, right_dst = tuple(zip(*right_edge_list))
    wrong_src, wrong_dst = tuple(zip(*wrong_edge_list))
    g_right.add_edges(right_src, right_dst)
    g_wrong.add_edges(wrong_src, wrong_dst)
    return g_right, g_wrong


def build_graph4SC(config: dict, mode='tl'):
    if mode == 'tl':
        data = config['np_train']
    elif mode == 'ind_train':
        data = config['np_train_old']
    else:
        data = np.vstack((config['np_train_old'], config['np_train_new']))
    stu_num = int(config['stu_num'])
    know_num = int(config['know_num'])
    q = config['q']
    q = q.detach().cpu().numpy()
    node = stu_num + know_num
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []
    sc_matrix = np.zeros(shape=(stu_num, know_num))
    for index in range(data.shape[0]):
        stu_id = data[index, 0]
        exer_id = data[index, 1]
        concepts = np.where(q[int(exer_id)] != 0)[0]
        for concept_id in concepts:
            if mode == 'tl' or mode == 'ind_train' or int(stu_id) in config['exist_idx']:
                if sc_matrix[int(stu_id), int(concept_id)] != 1:
                    edge_list.append((int(stu_id), int(concept_id + stu_num)))
                    edge_list.append((int(concept_id + stu_num), int(stu_id)))
                    sc_matrix[int(stu_id), int(concept_id)] = 1
            else:
                if mode != 'involve':
                    if sc_matrix[int(stu_id), int(concept_id)] != 1:
                        edge_list.append((int(concept_id + stu_num), int(stu_id)))
                        sc_matrix[int(stu_id), int(concept_id)] = 1
                else:
                    if sc_matrix[int(stu_id), int(concept_id)] != 1:
                        edge_list.append((int(stu_id), int(concept_id + stu_num)))
                        sc_matrix[int(stu_id), int(concept_id)] = 1
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    return g


class SAGEConv(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            aggregator_type,
            feat_drop=0.0,
            bias=True,
            norm=None,
            activation=None,
    ):
        super(SAGEConv, self).__init__()
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
            elif self._aggre_type == "gcn":
                check_eq_shape(feat)
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata["h"] = (
                        self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    )
                else:
                    if graph.is_block:
                        graph.dstdata["h"] = graph.srcdata["h"][
                                             : graph.num_dst_nodes()
                                             ]
                    else:
                        graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                        degs.unsqueeze(-1) + 1
                )
                if not lin_before_mp:
                    h_neigh = h_neigh
            elif self._aggre_type == "pool":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, fn.max("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = graph.dstdata["neigh"]
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                rst = h_neigh
                # add bias manually for GCN
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = h_self + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class Attn(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attn, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


class SAGENet(nn.Module):
    def __init__(self, dim, layers_num=2, type='mean', device='cpu', drop=True):
        super(SAGENet, self).__init__()
        self.drop = drop
        self.type = type  # 'mean'(DGL-SAGE)/'gat'(DGL-GAT)/'gatv2'(DGL-GATv2)/'pyg_gatv2'/'pyg_graphsage'/'pyg_gcn'
        self.device = device
        self.dim = dim
        self.layers = nn.ModuleList()  # 用ModuleList统一管理，避免手动append的混乱

        if type == 'gatv2':
            layer_cls = lambda: dglnn.GATv2Conv(in_feats=dim, out_feats=dim, num_heads=4, allow_zero_in_degree=True)
            # 2. S_E_wrong：sage_pool（新增SAGE池化聚合）
        elif type == 'sage_pool':
            layer_cls = lambda: dglnn.SAGEConv(in_feats=dim, out_feats=dim, aggregator_type='pool')
            # 3. E_C_right：tag_conv（新增TAGConv，适配高阶依赖）
        elif type == 'tag_conv':
            layer_cls = lambda: dglnn.TAGConv(in_feats=dim, out_feats=dim, k=3)  # k=3捕捉3阶邻居
            # 4. E_C_wrong：dgl_gcn（GraphConv，即基础GCN）
        elif type == 'dgl_gcn':
            layer_cls = lambda: dglnn.GraphConv(in_feats=dim, out_feats=dim, allow_zero_in_degree=True)
            # 5. S_C：sage_gcn（新增SAGE的GCN聚合模式）
        elif type == 'sage_gcn':
            layer_cls = lambda: dglnn.SAGEConv(in_feats=dim, out_feats=dim, aggregator_type='gcn')
            # 保留原有分支（兼容其他场景）
        elif type == 'mean':
            layer_cls = lambda: dglnn.SAGEConv(in_feats=dim, out_feats=dim, aggregator_type='mean')
        elif type == 'gat':
            layer_cls = lambda: dglnn.GATConv(in_feats=dim, out_feats=dim, num_heads=4)
        # 硬编码：PyG层（完全独立）
        elif type == 'pyg_gatv2':
            layer_cls = lambda: GATv2Conv(in_channels=dim, out_channels=dim, heads=4, concat=False)
        elif type == 'pyg_graphsage':
            layer_cls = lambda: SAGEConv(in_feats=dim, out_feats=dim, aggregator_type='mean')
        elif type == 'pyg_gcn':
            layer_cls = lambda: GCNConv(in_channels=dim, out_channels=dim)
        else:
            raise ValueError(f"Unknown type: {type}")

        # 初始化layers
        for _ in range(layers_num):
            self.layers.append(layer_cls().to(device))

    def forward(self, g, h):
        outs = [h]  # 初始特征：[N, dim]（2维）
        tmp = h.to(self.device)
        from dgl import DropEdge

        for index, layer in enumerate(self.layers):
            drop = DropEdge(p=0.05 + 0.1 * index)
            if self.drop and self.training:
                g = drop(g)

            # ------------------- PyG 卷积逻辑（保留） -------------------
            if self.type.startswith('pyg_'):
                pyg_data = dgl_to_pyg(g, tmp)
                pyg_x = F.dropout(pyg_data.x, p=0.05 + 0.1 * index) if (self.drop and self.training) else pyg_data.x
                tmp = layer(pyg_x, pyg_data.edge_index)
                tmp = tmp.to(self.device)
            # ------------------- DGL 卷积逻辑（核心修正） -------------------
            else:
                # 为需要的卷积层添加自环
                if self.type in ['dgl_gcn', 'tag_conv', 'sage_gcn', 'gatv2', 'gat']:
                    g = dgl.add_self_loop(g)

                # 执行卷积层前向
                tmp = layer(g, tmp)

                # 关键：压缩多头注意力的维度（GAT/GATv2专用）
                if self.type in ['gatv2', 'gat']:
                    # 多头输出：[N, num_heads, dim] → 平均后：[N, dim]
                    tmp = torch.mean(tmp, dim=1)

            # 确保tmp始终是2维（[N, dim]），与初始h维度一致
            assert len(tmp.shape) == 2, f"tmp维度错误：{tmp.shape}，应为[N, {self.dim}]"
            outs.append(tmp / (1 + index))

        # 堆叠并求和（此时所有张量都是2维，尺寸一致）
        res = torch.sum(torch.stack(outs, dim=1), dim=1)
        return res


class IGNet(nn.Module):
    def __init__(self, stu_num, prob_num, know_num, dim, graph, norm_adj=None, inter_layers=3, hidden_dim=512,
                 device='cuda',
                 gcnlayers=3, agg_type='mean', cdm_type='glif', khop=2):
        super().__init__()
        self.stu_num = stu_num
        self.prob_num = prob_num
        self.know_num = know_num
        self.dim = dim
        self.graph = graph
        self.device = device
        self.norm_adj = norm_adj.to(self.device).to(torch.float64)
        self.cdm_type = cdm_type
        self.khop = khop

        self.stu_emb = nn.Embedding(self.stu_num, self.dim)
        self.exer_emb_right = nn.Embedding(self.prob_num, self.dim)
        self.exer_emb_wrong = nn.Embedding(self.prob_num, self.dim)
        self.know_emb = nn.Embedding(self.know_num, self.dim)
        self.global_user = nn.Parameter(torch.zeros(size=(1, self.dim)))
        self.gcn_layers = gcnlayers

        self.S_E_right = SAGENet(dim=self.dim, type='gatv2', device=device, layers_num=self.khop)
        self.S_E_wrong = SAGENet(dim=self.dim, type='sage_pool', device=device, layers_num=self.khop)
        self.E_C_right = SAGENet(dim=self.dim, type='tag_conv', device=device, layers_num=self.khop)
        self.E_C_wrong = SAGENet(dim=self.dim, type='dgl_gcn', device=device, layers_num=self.khop)
        self.S_C = SAGENet(dim=self.dim, type='sage_gcn', device=device, layers_num=self.khop)

        self.attn_S = Attn(self.dim, attn_drop=0.2)
        self.attn_E_right = Attn(self.dim, attn_drop=0.2)
        self.attn_E_wrong = Attn(self.dim, attn_drop=0.2)
        self.attn_E = Attn(self.dim, attn_drop=0.2)
        self.attn_C = Attn(self.dim, attn_drop=0.2)

        self.Involve_Matrix = dgl2tensor(self.graph['I'])[:self.stu_num, self.stu_num:].to(self.device)
        self.transfer_stu_layer = nn.Linear(self.dim, self.know_num).to(torch.float64)
        self.transfer_exer_layer = nn.Linear(self.dim, self.know_num).to(torch.float64)
        self.transfer_concept_layer = nn.Linear(self.dim, self.know_num).to(torch.float64)

        self.change_latent_stu_mirt = nn.Linear(self.know_num, 16)
        self.change_latent_exer_mirt = nn.Linear(self.know_num, 16)

        self.change_latent_stu_irt = nn.Linear(self.know_num, 1)
        self.change_latent_exer_irt = nn.Linear(self.know_num, 1)

        self.disc_emb = nn.Embedding(self.prob_num, 1)
        layers = []
        for i in range(inter_layers):
            layer = nn.Linear(self.know_num if i == 0 else hidden_dim // pow(2, i - 1), hidden_dim // pow(2, i))
            layers.append(layer.to(torch.float64))
            layers.append(nn.Dropout(p=0.3))
            layers.append(nn.Tanh())

        final_layer = nn.Linear(hidden_dim // pow(2, inter_layers - 1), 1)
        layers.append(final_layer.to(torch.float64))  # 转换为 float64
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        BatchNorm_names = ['layers.{}.weight'.format(4 * i + 1) for i in range(3)]
        for index, (name, param) in enumerate(self.named_parameters()):
            if 'weight' in name:
                if name not in BatchNorm_names:
                    nn.init.xavier_normal_(param)

    def compute(self, emb):
        all_emb = emb.to(torch.float64)
        embs = [emb]
        for layer in range(self.gcn_layers):
            # print(self.norm_adj.dtype,all_emb.dtype) 64 32

            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)
        out_embs = torch.mean(torch.stack(embs, dim=1), dim=1)
        return out_embs

    def forward(self, stu_id, exer_id, knowledge_point):
        concept_id = torch.where(knowledge_point != 0)[1].to(self.device)
        concept_id_S = concept_id + torch.full(concept_id.shape, self.stu_num).to(self.device)
        concept_id_E = concept_id + torch.full(concept_id.shape, self.prob_num).to(self.device)
        exer_id_S = exer_id + torch.full(exer_id.shape, self.stu_num).to(self.device)

        subgraph_node_id_Q = torch.cat((exer_id.detach().cpu(), concept_id_E.detach().cpu()), dim=-1)
        subgraph_node_id_R = torch.cat((stu_id.detach().cpu(), exer_id_S.detach().cpu()), dim=-1)
        subgraph_node_id_I = torch.cat((stu_id.detach().cpu(), concept_id_S.detach().cpu()), dim=-1)

        R_subgraph_Right = get_subgraph(self.graph['right'], subgraph_node_id_R, device=self.device)
        R_subgraph_Wrong = get_subgraph(self.graph['wrong'], subgraph_node_id_R, device=self.device)
        I_subgraph = get_subgraph(self.graph['I'], subgraph_node_id_I, device=self.device)
        Q_subgraph = get_subgraph(self.graph['Q'], subgraph_node_id_Q, device=self.device)

        exer_info_right = self.exer_emb_right.weight
        exer_info_wrong = self.exer_emb_wrong.weight
        concept_info = self.know_emb.weight

        E_C_right = torch.cat([exer_info_right, concept_info]).to(self.device)
        E_C_wrong = torch.cat([exer_info_wrong, concept_info]).to(self.device)

        E_C_info_right = self.E_C_right(Q_subgraph, E_C_right)
        E_C_info_wrong = self.E_C_wrong(Q_subgraph, E_C_wrong)
        #
        stu_info = self.stu_emb.weight
        S_C = torch.cat([stu_info, concept_info]).to(self.device)
        S_E_right = torch.cat([stu_info, exer_info_right]).to(self.device)
        S_E_wrong = torch.cat([stu_info, exer_info_wrong]).to(self.device)
        S_E_info_right = self.S_E_right(R_subgraph_Right, S_E_right)
        S_E_info_wrong = self.S_E_wrong(R_subgraph_Wrong, S_E_wrong)
        S_C_info = self.S_C(I_subgraph, S_C)

        E_forward_right = self.attn_E_right.forward(
            [E_C_info_right[:self.prob_num], S_E_info_right[self.stu_num:]])
        E_forward_wrong = self.attn_E_wrong.forward(
            [E_C_info_wrong[:self.prob_num], S_E_info_wrong[self.stu_num:]])
        E_forward = E_forward_right * E_forward_wrong
        C_forward = self.attn_C.forward(
            [E_C_info_right[self.prob_num:], E_C_info_wrong[self.prob_num:], S_C_info[self.stu_num:]])
        S_forward = self.attn_S.forward(
            [S_E_info_right[:self.stu_num], S_E_info_wrong[:self.stu_num], S_C_info[:self.stu_num]])

        emb = torch.cat([S_forward, E_forward]).to(self.device)
        disc = torch.sigmoid(self.disc_emb(exer_id))

        def irf(theta, a, b, D=1.702):
            return torch.sigmoid(torch.mean(D * a * (theta - b), dim=1)).to(self.device).view(-1)

        if self.cdm_type == 'glif':
            out = self.compute(emb)
            # out = out.to(torch.float64)
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                out[:self.stu_num]), self.transfer_exer_layer(out[self.stu_num:]), self.transfer_concept_layer(
                C_forward)
            exer_concept_distill = concept_distill(knowledge_point, C_forward)
            state = disc * (torch.sigmoid(S_forward[stu_id] * exer_concept_distill) - torch.sigmoid(
                E_forward[exer_id] * exer_concept_distill)) * knowledge_point
            return self.layers(state.to(torch.float64)).view(-1)

        elif self.cdm_type == 'ncdm':
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                S_forward), self.transfer_exer_layer(E_forward), self.transfer_concept_layer(C_forward)
            state = disc * (torch.sigmoid(S_forward[stu_id]) - torch.sigmoid(
                E_forward[exer_id])) * knowledge_point
            return self.layers(state).view(-1)

        elif self.cdm_type == 'mirt':
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                S_forward), self.transfer_exer_layer(E_forward), self.transfer_concept_layer(C_forward)
            return irf(S_forward[stu_id], disc, E_forward[exer_id])

        elif self.cdm_type == 'irt':
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                S_forward), self.transfer_exer_layer(E_forward), self.transfer_concept_layer(
                C_forward)

            return irf(S_forward[stu_id], disc, E_forward[exer_id])

        else:
            raise ValueError('We do not support it yet')

    def get_mastery_level(self, pin_memory=False):
        if pin_memory:
            device = 'cpu'
        else:
            device = self.device
        R_subgraph_Right = self.graph['right'].to(device)
        R_subgraph_Wrong = self.graph['wrong'].to(device)
        I_subgraph = self.graph['I'].to(device)
        Q_subgraph = self.graph['Q'].to(device)

        exer_info_right = self.exer_emb_right.weight
        exer_info_wrong = self.exer_emb_wrong.weight
        concept_info = self.know_emb.weight

        E_C_right = torch.cat([exer_info_right, concept_info]).to(device)
        E_C_wrong = torch.cat([exer_info_wrong, concept_info]).to(device)

        E_C_info_right = self.E_C_right(Q_subgraph, E_C_right)
        E_C_info_wrong = self.E_C_wrong(Q_subgraph, E_C_wrong)
        #
        stu_info = self.stu_emb.weight
        S_C = torch.cat([stu_info, concept_info]).to(device)
        S_E_right = torch.cat([stu_info, exer_info_right]).to(device)
        S_E_wrong = torch.cat([stu_info, exer_info_wrong]).to(device)
        S_E_info_right = self.S_E_right(R_subgraph_Right, S_E_right)
        S_E_info_wrong = self.S_E_wrong(R_subgraph_Wrong, S_E_wrong)

        self.attn_S = self.attn_S.to(device)
        self.attn_C = self.attn_C.to(device)
        self.attn_E_right = self.attn_E_right.to(device)
        self.attn_E_wrong = self.attn_E_wrong.to(device)
        self.norm_adj = self.norm_adj.to(device)
        self.transfer_stu_layer = self.transfer_stu_layer.to(device)
        self.transfer_exer_layer = self.transfer_exer_layer.to(device)
        self.transfer_concept_layer = self.transfer_concept_layer.to(device)

        E_forward_right = self.attn_E_right.forward(
            [E_C_info_right[:self.prob_num], S_E_info_right[self.stu_num:]])
        E_forward_wrong = self.attn_E_wrong.forward(
            [E_C_info_wrong[:self.prob_num], S_E_info_wrong[self.stu_num:]])
        E_forward = E_forward_right * E_forward_wrong

        S_C_info = self.S_C(I_subgraph, S_C)
        C_forward = self.attn_C.forward(
            [E_C_info_right[self.prob_num:], E_C_info_wrong[self.prob_num:], S_C_info[self.stu_num:]])
        S_forward = self.attn_S.forward(
            [S_E_info_right[:self.stu_num], S_E_info_wrong[:self.stu_num], S_C_info[:self.stu_num]])

        if self.cdm_type == 'glif':
            emb = torch.cat([S_forward, E_forward]).to(device)
            out = self.compute(emb)
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                out[:self.stu_num]), self.transfer_exer_layer(out[self.stu_num:]), self.transfer_concept_layer(
                C_forward)
            stu_concept_distill = concept_distill(self.Involve_Matrix, C_forward)
            return torch.sigmoid(S_forward * stu_concept_distill).detach().cpu().numpy()
        else:
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                S_forward), self.transfer_exer_layer(E_forward), self.transfer_concept_layer(
                C_forward)
            return torch.sigmoid(S_forward).detach().cpu().numpy()
    def get_all_knowledge_emb(self):
        
        device=self.stu_emb.weight.device
        R_subgraph_Right = self.graph['right'].to(device)
        R_subgraph_Wrong = self.graph['wrong'].to(device)
        I_subgraph = self.graph['I'].to(device)
        Q_subgraph = self.graph['Q'].to(device)

        exer_info_right = self.exer_emb_right.weight
        exer_info_wrong = self.exer_emb_wrong.weight
        concept_info = self.know_emb.weight

        E_C_right = torch.cat([exer_info_right, concept_info]).to(device)
        E_C_wrong = torch.cat([exer_info_wrong, concept_info]).to(device)

        E_C_info_right = self.E_C_right(Q_subgraph, E_C_right)
        E_C_info_wrong = self.E_C_wrong(Q_subgraph, E_C_wrong)
        #
        stu_info = self.stu_emb.weight
        S_C = torch.cat([stu_info, concept_info]).to(device)
        S_E_right = torch.cat([stu_info, exer_info_right]).to(device)
        S_E_wrong = torch.cat([stu_info, exer_info_wrong]).to(device)
        S_E_info_right = self.S_E_right(R_subgraph_Right, S_E_right)
        S_E_info_wrong = self.S_E_wrong(R_subgraph_Wrong, S_E_wrong)

        self.attn_S = self.attn_S.to(device)
        self.attn_C = self.attn_C.to(device)
        self.attn_E_right = self.attn_E_right.to(device)
        self.attn_E_wrong = self.attn_E_wrong.to(device)
        self.norm_adj = self.norm_adj.to(device)
        self.transfer_stu_layer = self.transfer_stu_layer.to(device)
        self.transfer_exer_layer = self.transfer_exer_layer.to(device)
        self.transfer_concept_layer = self.transfer_concept_layer.to(device)

        E_forward_right = self.attn_E_right.forward(
            [E_C_info_right[:self.prob_num], S_E_info_right[self.stu_num:]])
        E_forward_wrong = self.attn_E_wrong.forward(
            [E_C_info_wrong[:self.prob_num], S_E_info_wrong[self.stu_num:]])
        E_forward = E_forward_right * E_forward_wrong

        S_C_info = self.S_C(I_subgraph, S_C)
        C_forward = self.attn_C.forward(
            [E_C_info_right[self.prob_num:], E_C_info_wrong[self.prob_num:], S_C_info[self.stu_num:]])
        S_forward = self.attn_S.forward(
            [S_E_info_right[:self.stu_num], S_E_info_wrong[:self.stu_num], S_C_info[:self.stu_num]])

        if self.cdm_type == 'glif':
            emb = torch.cat([S_forward, E_forward]).to(device)
            out = self.compute(emb)
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                out[:self.stu_num]), self.transfer_exer_layer(out[self.stu_num:]), self.transfer_concept_layer(
                C_forward)
            stu_concept_distill = concept_distill(self.Involve_Matrix, C_forward)
            return torch.sigmoid(S_forward * stu_concept_distill).detach().cpu().numpy()
        else:
            S_forward, E_forward, C_forward = self.transfer_stu_layer(
                S_forward), self.transfer_exer_layer(E_forward), self.transfer_concept_layer(
                C_forward)
            return torch.sigmoid(S_forward).detach().cpu().numpy()

    def apply_clipper(self):
        clipper = NoneNegClipper()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.apply(clipper)


class ICDM(BaseCDModel):
    def __init__(self, stu_num, prob_num, know_num, dim=64, device='cuda:0', graph=None, gcn_layers=3, agg_type='mean',
                 weight_reg=0.05, wandb=True, cdm_type='glif', khop=2, exist_idx=None, new_idx=None):
        super(ICDM, self).__init__()
        self.net = None
        self.know_num = know_num
        self.prob_num = prob_num
        self.stu_num = stu_num
        self.device = device
        self.dim = dim
        self.wandb = wandb
        self.agg_type = agg_type
        self.graph = graph
        self.gcn_layers = gcn_layers
        self.weight_reg = weight_reg
        self.cdm_type = cdm_type
        self.khop = khop
        self.mas_list = []
        self.exist_idx = exist_idx
        self.new_idx = new_idx

        
        

    def train(self, np_train, np_val, np_test, np_test_new=None, q=None, batch_size=None, exp_dir=None, wandb_instance=None, epoch=10, lr=0.0005):

        # 训练集, 验证集, 测试集
        train_data, val_data, test_data = [
            transform(q, _[:, 0], _[:, 1], _[:, 2], batch_size)  
            for _ in [np_train, np_val, np_test]
        ]
        # print(val_data)


        def log_metrics(metrics):
            """记录指标到wandb（如果启用）"""
            if wandb_instance:
                 wandb_instance.log(metrics)


        def get_adj_matrix(tmp_adj):
            adj_mat = tmp_adj + tmp_adj.T
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
            return adj_matrix

        def sp_mat_to_sp_tensor(sp_mat):
            coo = sp_mat.tocoo().astype(np.float64)
            indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
            return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

        def create_adj_mat():
            n_nodes = self.stu_num + self.prob_num
            train_stu = np_train[:, 0]
            train_exer = np_train[:, 1]
            ratings = np.ones_like(train_stu, dtype=np.float64)
            tmp_adj = sp.csr_matrix((ratings, (train_stu, train_exer + self.stu_num)), shape=(n_nodes, n_nodes))
            return sp_mat_to_sp_tensor(get_adj_matrix(tmp_adj))

        norm_adj = create_adj_mat()  # 1
        self.net = IGNet(stu_num=self.stu_num, prob_num=self.prob_num, know_num=self.know_num, dim=self.dim,
                        device=self.device, graph=self.graph, norm_adj=norm_adj, gcnlayers=self.gcn_layers,
                        agg_type=self.agg_type, cdm_type=self.cdm_type, khop=self.khop).to(self.device)   # 2
        self.net = self.net.to(torch.float64)

        r = get_r_matrix(np_val, self.stu_num, self.prob_num)  # 3
        if self.new_idx is not None:
            r_new = get_r_matrix(np_val, len(self.new_idx), self.prob_num, self.new_idx)


        bce_loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        # get_number_of_params('igcdm', self.net) 得到参数数量，用不上

        #学习率调度器定义
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

        #早停设置
        early_stop = EarlyStopping(patience=5, mode='max')

        #模型保存设置
        
        exp_dir = exp_dir # 获取实验目录
        ckpt_path = os.path.join(exp_dir, 'model.pth')
        print(f"model save to: {ckpt_path}")

        best_metric = None
        best_epoch = None


        for epoch_i in range(1, epoch+1):

            losses = []
            bce_losses = []
            reg_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Train"):
                batch_count += 1
                stu_id, exer_id, knowledge_emb, y = batch_data  
                stu_id: torch.Tensor = stu_id.to(self.device)
                exer_id: torch.Tensor = exer_id.to(self.device)
                knowledge_emb = knowledge_emb.to(self.device)
                y: torch.Tensor = y.to(self.device)
                tmp_E_right = self.net.exer_emb_right.weight
                tmp_E_wrong = self.net.exer_emb_wrong.weight
                reg_loss = l2_loss(       
                    tmp_E_right[exer_id],
                    tmp_E_wrong[exer_id]
                )
                pred = self.net.forward(stu_id, exer_id, knowledge_emb)
                bce_loss = bce_loss_function(pred, y)
                bce_losses.append(bce_loss.mean().item())
                total_loss = bce_loss + self.weight_reg * reg_loss  
                reg_losses.append((self.weight_reg * reg_loss).detach().cpu().numpy())
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                self.net.apply_clipper()
                losses.append(total_loss.mean().item())

                #学习率调度器使用 ，使用之后感觉效果很差
                # if scheduler:
                #     try:
                #         scheduler.step(sum(losses) / len(losses))
                #     except TypeError:
                #         scheduler.step()


            loss = sum(losses) / len(losses)

            if wandb_instance:
                log_metrics({
                    'epoch': epoch_i
                })


            # back_p = {
            #     'loss': None,
            #     'old':
            #         {'auc': None, 'accuracy': None, 'accuracy': None, 'rmse': None, 'f1': None, 'doa': None
            #         },
            #     'new':
            #         {'all_auc': None, 'all_acc': None, 'all_doa': None, 'new_auc': None, 'new_acc': None,
            #         'new_doa': None
            #         }
            # }
            # back_p['loss'] = loss

            if self.exist_idx is None:
                
                auc, accuracy, rmse, f1, doa = self.eval(val_data, q=q, r=r, val_test="Eval")

                # back_p['old']['auc'], back_p['old']['accuracy'], back_p['old']['rmse'], back_p['old']['f1'], \
                # back_p['old']['doa'] = auc, accuracy, rmse, f1, doa

                #格式化输出
                # print(f"Train Loss: {loss:.4f}, Val Metric(Auc): {auc}, Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}, Val RMSE: {rmse:.4f}, Val DOA: {doa:.4f}")
                print(f"[Epoch {epoch_i}] Train Loss: {loss:.4f}, Val Metric(Auc): {auc}, Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}, Val RMSE: {rmse:.4f}, Val DOA: {doa:.4f}")

                #模型保存判断
                if ckpt_path is not None:
                    if best_metric is None or auc > best_metric:
                        best_metric = auc
                        best_epoch = epoch_i
                        torch.save(self.net.state_dict(), ckpt_path)
                        print(f"  → Saved best model to {ckpt_path}")

                # 早停判断
                if early_stop is not None:
                    if early_stop.step(auc):
                        print(f"  → Early stopping at epoch {epoch_i}")
                        break




            else: # 如果使用这部分，参数要怎么设置暂时不知道，要的参数格式不清楚
                all_auc, all_acc, all_doa, new_auc, new_acc, new_doa = self.eval(val_data, q=q, r=r, r_new=r_new, val_test="Eval")

                # back_p['new']['all_auc'], back_p['new']['all_acc'], back_p['new']['all_doa'] = all_auc, all_acc, all_doa
                # back_p['new']['new_auc'], back_p['new']['new_acc'], back_p['new']['new_doa'] = new_auc, new_acc, new_doa
                # print(back_p['new'])

                #格式化输出
                print(f"Train Loss: {loss:.4f}, Val Metric(Auc): {all_auc}, Val Accuracy: {all_acc:.4f}, Val RMSE: {rmse:.4f}, Val DOA: {all_doa:.4f}，\
                Val New_Auc: {all_auc}, Val New_Acc: {all_acc:.4f}, Val New_Doa: {all_doa:.4f}")

                #模型保存判断
                if ckpt_path is not None:
                    if best_metric is None or all_auc > best_metric:
                        best_metric = all_auc
                        torch.save(self.net.state_dict(), ckpt_path)
                        print(f"  → Saved best model to {ckpt_path}")

                # 早停判断
                if early_stop is not None:
                    if early_stop.step(all_auc):
                        print(f"  → Early stopping at epoch {epoch}")
                        break

        # 训练结束后，加载最优参数
        if self.exist_idx is None:

            if ckpt_path is not None:
                self.net.load_state_dict(torch.load(ckpt_path))

                r = get_r_matrix(np_test, self.stu_num, self.prob_num) # 因为原来的r矩阵用的是验证集的，这里使用测试集，所以使用测试集构建

                auc, accuracy, rmse, f1, doa = self.eval(test_data, q=q, r=r, val_test="Test")

                # # wandb记录
                # if wandb_instance:
                #     wandb_instance.summary["test_auc"] = auc
                #     wandb_instance.summary["test_accuracy"] = accuracy
                #     wandb_instance.summary["test_rmse"] = rmse
                #     wandb_instance.summary["test_f1"] = f1
                #     wandb_instance.summary["test_doa"] = doa

                
                return auc, accuracy, rmse, f1, doa, ckpt_path, best_metric, best_epoch
        else:
            # 训练结束后，加载最优参数
            if ckpt_path is not None:
                self.net.load_state_dict(torch.load(ckpt_path))
            

                r = get_r_matrix(np_test, self.stu_num, self.prob_num)
                all_auc, all_acc, all_doa, new_auc, new_acc, new_doa = self.eval(test_data, q=q, r=r, r_new=r_new, val_test="Test")

                return all_auc, all_acc, all_doa, new_auc, new_acc, new_doa

    def eval(self, val_data, q=None, r=None, r_new=None, val_test=None):
        self.net = self.net.to(self.device)
        self.net.eval()
        y_true, y_pred = [], []
        y_true_new, y_pred_new = [], []
        mas = self.net.get_mastery_level()
        self.mas_list.append(mas)
        doa_func = get_doa_function(know_num=self.know_num)
        for batch_data in tqdm(val_data, f"{val_test}"):
            # print(len(batch_data))
            user_id, item_id, know_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(self.device)
            item_id: torch.Tensor = item_id.to(self.device)
            know_emb = know_emb.to(self.device)
            pred: torch.Tensor = self.net.forward(user_id, item_id, knowledge_point=know_emb)
            if self.new_idx is not None:
                for index, user in enumerate(user_id.detach().cpu().tolist()):
                    if user in self.new_idx:
                        y_true_new.append(y.tolist()[index])
                        y_pred_new.append(pred.detach().cpu().tolist()[index])
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())
        if self.exist_idx is None:
            return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
                   np.sqrt(mean_squared_error(y_true, y_pred)), f1_score(y_true, np.array(y_pred) >= 0.5,
                                                                         ), doa_func(mas, q.detach().cpu().numpy(),
                                                                                     r)
        else:
            return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
                   doa_func(mas, q.detach().cpu().numpy(), r), roc_auc_score(y_true_new, y_pred_new), accuracy_score(
                y_true_new, np.array(y_pred_new) >= 0.5), doa_func(mas[self.new_idx], q.detach().cpu().numpy(), r_new)
    def get_all_knowledge_emb(self):
        self.net.eval()
        return self.net.get_all_knowledge_emb()
