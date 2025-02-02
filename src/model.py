import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_layer import NR_GraphAttention
from tabulate import tabulate
import logging
from torch_scatter import scatter_add
import torch_geometric.utils as utils
import pickle
from util import *
from hyper_util import *
from model_util import *

from hyper_util import PoincareBall

def getPositionEncoding(seq_len, dim, n=10000):
    PE = np.zeros(shape=(seq_len, dim))
    for pos in range(seq_len):
        for i in range(int(dim / 2)):
            denominator = np.power(n, 2 * i / dim)
            PE[pos, 2 * i] = np.sin(pos / denominator)
            PE[pos, 2 * i + 1] = np.cos(pos / denominator)

    return PE


class ALL_entroy(nn.Module):
    def __init__(self, device):
        super(ALL_entroy, self).__init__()
        self.device = device

    def forward_one(self, train_set, x, e2):
        x1_train, x2_train = x[train_set[:, 0]], x[train_set[:, 1]]
        label = torch.arange(0, x1_train.shape[0]).to(self.device)
        d = {}
        for i in range(e2.shape[0]):
            d[int(e2[i])] = i
        x2 = x[e2]
        # print(x1_train.shape[0])
        pred = torch.matmul(x1_train, x2.transpose(0, 1))
        self.bias_0 = torch.nn.Parameter(torch.zeros(x2.shape[0])).to(self.device)
        pred += self.bias_0.expand_as(pred)
        for i in range(x1_train.shape[0]):
            label[i] = d[int(train_set[i, 1])]
        # label = train_set[:, 1].unsqueeze(1)
        label = label.unsqueeze(1)
        # print(label.shape)
        # print(label)
        # exit(0)
        # print( torch.zeros(x1_train.shape[0], x2.shape[0]).shape)
        soft_targets = torch.zeros(x1_train.shape[0], x2.shape[0]). \
            to(self.device).scatter_(1, label, 1)
        # print(soft_targets[2][train_set[2, 1]])
        soft = 0.8
        soft_targets = soft_targets * soft \
                       + (1.0 - soft_targets) \
                       * ((1.0 - soft) / (x2.shape[0] - 1))
        # print(soft_targets[2][train_set[2, 1]])
        logsoftmax = nn.LogSoftmax(dim=1)
        # exit(0)
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

    def forward(self, train_set, x, e2):
        loss_l = self.forward_one(train_set, x, e2)
        # loss_r = self.forward_one(train_set[[1, 0]], x)
        return loss_l

class LapEncoding:
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, edge_index, num_nodes):
        edge_index, edge_attr = utils.get_laplacian(
            edge_index.long(), normalization=self.normalization,
            num_nodes=num_nodes)
        L = utils.to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()  # increasing order
        EigVal, EigVec = np.real(EigVal[idx]), np.real(EigVec[:, idx])
        return torch.from_numpy(EigVec[:, 1:self.pos_enc_dim + 1]).float()


def normalize_adj(edge_index, edge_weight=None, num_nodes=None):
    edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1),
                                 device=edge_index.device)
    num_nodes = utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return utils.to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)


def norm(x):
    mean = torch.mean(x)
    std = torch.std(x)
    return (x-mean)/std



class ST_Encoder_Module(nn.Module):
    def __init__(self, node_hidden, rel_hidden,
                 device, node_size, rel_size,
                 dropout_rate=0.0, depth=2):
        super(ST_Encoder_Module, self).__init__()
        self.ent_embedding_1 = None
        self.node_hidden = node_hidden

        self.depth = depth
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        self.loss = ALL_entroy(self.device)

        self.m_adj = None
        # original entity_emb
        self.ent_embedding = nn.Embedding(node_size, node_hidden)
        torch.nn.init.xavier_uniform_(self.ent_embedding.weight)

        self.rel_embedding = nn.Embedding(rel_size, rel_hidden)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)

        self.e_encoder = NR_GraphAttention(
            node_dim=self.node_hidden,
            depth=self.depth,
            use_bias=True,
            device=self.device

        )
        self.r_encoder = NR_GraphAttention(
            node_dim=self.node_hidden,
            depth=self.depth,
            use_bias=True,
            device=self.device
        )

        self.i_encoder = NR_GraphAttention(
            node_dim=self.node_hidden,
            depth=self.depth,
            use_bias=True,
            device=self.device
        )
        embed_size = 384
        self.query_layer = nn.Linear(384, 384).to(self.device)
        self.key_layer = nn.Linear(embed_size, embed_size).to(self.device)
        self.value_layer = nn.Linear(embed_size, embed_size).to(self.device)

        self.scale = torch.sqrt(torch.FloatTensor([embed_size]))

        self.mk = nn.Linear(384,384,bias=False)
        self.mv = nn.Linear(384,384,bias=False)

        self.mk_1 = nn.Linear(384,384,bias=False)
        self.mv_1 = nn.Linear(384,384,bias=False)

        self.emb_num = 3
        self.weight = nn.Parameter(torch.ones((self.emb_num, 1)),
                                   requires_grad=True)

    def avg(self, adj, emb, size: int, node_size):
        adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[node_size, size])

        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)

    def avg_r(self, adj, emb, size: int, node_size):
        adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[node_size, size])

        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)


    def fusion(self, embs):
        embs = [self.weight[idx] * F.normalize(embs[idx]) for idx in range(self.emb_num) if embs[idx] is not None]
        cat_emb = torch.cat(embs, dim=-1)
        return cat_emb

    def forward(self, ent_adj, rel_adj, node_size, rel_size, adj_list,
                r_index, r_val, triple_size, mask=None):
        ent_feature = self.avg(ent_adj, self.ent_embedding.weight, node_size, node_size)
        rel_feature = self.avg_r(rel_adj, self.rel_embedding.weight, rel_size, node_size)

        opt = [self.rel_embedding.weight, adj_list, r_index, r_val, triple_size, rel_size, node_size, mask]
        ent_feat = self.e_encoder([ent_feature] + opt)
        rel_feat = self.r_encoder([rel_feature] + opt)
        out_feature = torch.cat([ent_feat, rel_feat], dim=-1)

        out_feature = self.dropout(out_feature)

        return [out_feature]

    def Attention(self, m1, m2):
        Q = self.query_layer(m1)  # [27793, 384]
        K = self.key_layer(m2)  # [27793, 384]
        V = self.value_layer(m2)  # [27793, 384]
        self.window_size = 1024
        # Initialize output
        output = torch.zeros_like(V)

        # Compute local attention
        for i in range(0, Q.size(0), self.window_size):
            q_chunk = Q[i:i + self.window_size]  # Chunk of queries
            k_chunk = K[i:i + self.window_size]  # Corresponding chunk of keys
            v_chunk = V[i:i + self.window_size]  # Corresponding chunk of values

            attention_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / self.scale.to(self.device)  # Local attention scores
            attention_weights = F.softmax(attention_scores, dim=-1)  # Local attention weights

            output[i:i + self.window_size] = torch.matmul(attention_weights, v_chunk)

        return output

    def external_attention(self, queries, f_type='ent'):
        if f_type == 'ent':
            attn = self.mk(queries)  # bs,n,S
        else:
            attn = self.mk_1(queries)

        attn = F.softmax(attn, dim=-1)  # bs,n,S
        attn = attn / torch.sum(attn, dim=1, keepdim=True)  # bs,n,S

        if f_type == 'ent':
            out = self.mv(attn)  # bs,n,d_model
        else:
            out = self.mv_1(attn)

        return out


class VAE_ST_Encoder_Module(nn.Module):
    def __init__(self,  node_hidden, rel_hidden,
                 device, node_size, rel_size,
                 dropout_rate=0.0, depth=2):
        super(VAE_ST_Encoder_Module, self).__init__()
        self.node_size = node_size
        self.rel_size = rel_size
        self.device = device
        self.droprate = dropout_rate
        self.depth = depth
        self.st_mu_encoder = ST_Encoder_Module(
            node_hidden=node_hidden,
            rel_hidden=rel_hidden,
            node_size=self.node_size,
            rel_size=self.rel_size,
            device=self.device,
            dropout_rate=self.droprate,
            depth=self.depth).to(self.device)
        self.st_std_emcoder = ST_Encoder_Module(
            node_hidden=node_hidden,
            rel_hidden=rel_hidden,
            node_size=self.node_size,
            rel_size=self.rel_size,
            device=self.device,
            dropout_rate=self.droprate,
            depth=self.depth).to(self.device)

        # self.gph_layer_norm_mu = nn.LayerNorm(384, elementwise_affine=True)
        self.gph_layer_norm_mu = nn.LayerNorm(768, elementwise_affine=True).to(self.device)

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        from torch.distributions.kl import kl_divergence
        from torch.distributions import Normal

        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_1, 0.4)))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_2, 0.4)))
        mu_1_fixed = mu_1.clone()
        sigma_1_fixed = sigma_1.clone()
        mu_1_fixed[torch.isnan(mu_1_fixed)] = 0
        mu_1_fixed[torch.isinf(mu_1_fixed)] = torch.max(
            mu_1_fixed[~torch.isinf(mu_1_fixed)]
        )
        sigma_1_fixed[torch.isnan(sigma_1_fixed)] = 1
        sigma_1_fixed[torch.isinf(sigma_1_fixed)] = torch.max(
            sigma_1_fixed[~torch.isinf(sigma_1_fixed)]
        )
        sigma_1_fixed[sigma_1_fixed <= 0] = 1
        q_target = Normal(mu_1_fixed, sigma_1_fixed)

        mu_2_fixed = mu_2.clone()
        sigma_2_fixed = sigma_2.clone()
        mu_2_fixed[torch.isnan(mu_2_fixed)] = 0
        mu_2_fixed[torch.isinf(mu_2_fixed)] = torch.max(
            mu_2_fixed[~torch.isinf(mu_2_fixed)]
        )
        sigma_2_fixed[torch.isnan(sigma_2_fixed)] = 1
        sigma_2_fixed[torch.isinf(sigma_2_fixed)] = torch.max(
            sigma_2_fixed[~torch.isinf(sigma_2_fixed)]
        )
        sigma_2_fixed[sigma_2_fixed <= 0] = 1
        q_context = Normal(mu_2_fixed, sigma_2_fixed)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl


    def forward(self, ent_adj, rel_adj, node_size,
                    rel_size, adj_list, r_index, r_val,
                    triple_size):
        st_mu_emb = self.st_mu_encoder(ent_adj, rel_adj, node_size,
                    rel_size, adj_list, r_index, r_val,triple_size)
        # mu = self.gph_layer_norm_mu(st_mu_emb)
        mu = self.gph_layer_norm_mu(st_mu_emb[0])

        st_std_emb = self.st_std_emcoder(ent_adj, rel_adj, node_size,
                    rel_size, adj_list, r_index, r_val,triple_size)
        # std = F.elu(st_std_emb)
        std = F.elu(st_std_emb[0])
        eps = torch.randn_like(std)
        gph_emb = mu + eps * std
        vae_loss = self._kld_gauss(
                        mu, std, torch.zeros_like(mu), torch.ones_like(std)
                    )

        return [gph_emb, vae_loss]


class ST_Encoder_Module_hyper(nn.Module):
    def __init__(self, node_hidden, rel_hidden,
                 device, node_size, rel_size,
                 dropout_rate=0.0, depth=2):
        super(ST_Encoder_Module_hyper, self).__init__()
        self.ent_embedding_1 = None
        self.node_hidden = node_hidden

        self.depth = depth
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        self.loss = ALL_entroy(self.device)
        self.m_adj = None
        # original entity_emb
        self.ent_embedding = nn.Embedding(node_size, node_hidden)
        torch.nn.init.xavier_uniform_(self.ent_embedding.weight)

        self.rel_embedding = nn.Embedding(rel_size, rel_hidden)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)

        self.hyper_rel_embbedding = nn.Embedding(node_size, node_hidden)
        torch.nn.init.xavier_uniform_(self.hyper_rel_embbedding.weight)

        self.e_encoder = NR_GraphAttention(
            node_dim=self.node_hidden,
            depth=self.depth,
            use_bias=True,
            device=self.device

        )
        self.r_encoder = NR_GraphAttention(
            node_dim=self.node_hidden,
            depth=self.depth,
            use_bias=True,
            device=self.device
        )

        self.i_encoder = NR_GraphAttention(
            node_dim=self.node_hidden,
            depth=self.depth,
            use_bias=True,
            device=self.device
        )
        embed_size = 384
        self.query_layer = nn.Linear(384, 384).to(self.device)
        self.key_layer = nn.Linear(embed_size, embed_size).to(self.device)
        self.value_layer = nn.Linear(embed_size, embed_size).to(self.device)

        self.scale = torch.sqrt(torch.FloatTensor([embed_size]))

        self.mk = nn.Linear(384,384,bias=False)
        self.mv = nn.Linear(384,384,bias=False)

        self.mk_1 = nn.Linear(384,384,bias=False)
        self.mv_1 = nn.Linear(384,384,bias=False)

        self.emb_num = 3
        self.weight = nn.Parameter(torch.ones((self.emb_num, 1)),
                                   requires_grad=True)

    def avg(self, adj, emb, size: int, node_size):
        adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[node_size, size])

        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)

    def avg_r(self, adj, emb, size: int, node_size):
        adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[node_size, size])

        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)


    def fusion(self, embs):
        embs = [self.weight[idx] * F.normalize(embs[idx]) for idx in range(self.emb_num) if embs[idx] is not None]
        cat_emb = torch.cat(embs, dim=-1)
        return cat_emb

    def get_dim_act_curv(self):
        act = getattr(F, 'relu')
        acts = [act] * 1
        dims = [256] + ([256] * 1)
        task = "lp"
        if task in ['lp', 'rec']:
            dims += [256]
            acts += [act]
            n_curvatures = 2
        else:
            n_curvatures = 1
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
        return dims, acts, curvatures


    def hyper_encode(self, x, adj):
        dims, acts, self.curvatures = self.get_dim_act_curv()
        self.c = nn.Parameter(torch.Tensor([1.]))
        import hyper_util
        self.manifold = getattr(hyper_util, 'PoincareBall')()
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            # in_dim, out_dim = dims[i], dims[i + 1]
            in_dim, out_dim = self.node_hidden, self.node_hidden
            act = acts[i]
            hgc_layers.append(
                HyperbolicGraphConvolution(
                    self.manifold, in_dim, out_dim, c_in, c_out, 0.2, act, 0, 0, 0).to(self.device)
            )
        self.layers = nn.Sequential(*hgc_layers)
        x_tan = self.manifold.proj_tan0(x, c=self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0].to(self.device))
        self.encode_graph = True
        # rows, cols = adj[0,:], adj[1,:]
        shape = (adj.max().item()+1, adj.max().item()+1)

        adj_features = torch.sparse_coo_tensor(indices=adj,
                                               values=torch.ones(adj.size(1), dtype=torch.float).to(self.device),
                                               size=shape)

        if self.encode_graph:
            # input = (x_hyp, adj)
            input = (x_hyp, adj_features)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x_hyp)
        output = self.manifold.logmap0(output, c=self.curvatures[0])
        return output

    def hyper_rel_encode(self, x, adj):
        dims, acts, self.curvatures = self.get_dim_act_curv()
        self.c = nn.Parameter(torch.Tensor([1.]))
        import hyper_util
        self.manifold = getattr(hyper_util, 'PoincareBall')()
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 2):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            # in_dim, out_dim = dims[i], dims[i + 1]
            in_dim, out_dim = self.node_hidden, self.node_hidden
            act = acts[i]
            hgc_layers.append(
                HyperbolicGraphConvolution(
                    self.manifold, in_dim, out_dim, c_in, c_out, 0.2, act, 0, 0, 0).to(self.device)
            )
        self.layers = nn.Sequential(*hgc_layers)
        x_tan = self.manifold.proj_tan0(x, c=self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0].to(self.device))
        self.encode_graph = True
        # rows, cols = adj[0,:], adj[1,:]
        shape = (adj.max().item()+1, adj[1,:].max().item()+1)

        adj_features = torch.sparse_coo_tensor(indices=adj,
                                               values=torch.ones(adj.size(1), dtype=torch.float).to(self.device),
                                               size=shape)

        if self.encode_graph:
            # input = (x_hyp, adj)
            input = (x_hyp, adj_features) #x_hyp (node_size, node_hidden) adj_feature:(node_size, rel_size)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x_hyp)
        output = self.manifold.logmap0(output, c=self.curvatures[0])
        return output

    def forward(self, ent_adj, rel_adj, node_size, rel_size, adj_list,
                r_index, r_val, triple_size, mask=None):
        hyper_ent_embedding = self.ent_embedding.weight
        # hyper_rel_embedding_0 = self.rel_embedding.weight
        # hyper_rel_embedding = self.hyper_rel_embbedding.weight

        ent_feature = self.avg(ent_adj, self.ent_embedding.weight, node_size, node_size)
        rel_feature = self.avg_r(rel_adj, self.rel_embedding.weight, rel_size, node_size)

        opt = [self.rel_embedding.weight, adj_list, r_index, r_val, triple_size, rel_size, node_size, mask]
        ent_feat = self.e_encoder([ent_feature] + opt)
        rel_feat = self.r_encoder([rel_feature] + opt)
        out_feature = torch.cat([ent_feat, rel_feat], dim=-1)

        out_feature = self.dropout(out_feature)

        hyper_ent_embedding = hyper_ent_embedding + self.hyper_encode(hyper_ent_embedding, ent_adj)
        hyper_out_feature = torch.cat([hyper_ent_embedding, hyper_ent_embedding, hyper_ent_embedding, rel_feat], dim=-1)

        # hyper_rel_embedding = hyper_rel_embedding + self.hyper_rel_encode(hyper_rel_embedding_0, rel_adj)
        # hyper_out_feature = torch.cat([hyper_ent_embedding, hyper_ent_embedding, hyper_ent_embedding,
        #                                hyper_rel_embedding, hyper_rel_embedding, hyper_rel_embedding], dim=-1)

        return [out_feature, hyper_out_feature]
        # return [out_feature]



class DoubleEmbedding(nn.Module):
    def __init__(self, ent_num, embedding_dim, init_type='xavier'):
        super(DoubleEmbedding, self).__init__()
        self.embedding = nn.Embedding(ent_num, embedding_dim,
                                      _weight=torch.zeros(ent_num, embedding_dim), dtype=torch.float)
        if init_type == 'xavier':
            nn.init.xavier_uniform_(self.embedding.weight.data)
        else:
            nn.init.normal_(self.embedding.weight.data, std=1./math.sqrt(ent_num))

    def normalize(self):
        self.embedding.weight.data = F.normalize(self.embedding.weight, dim=-1, p=2)

    def forward(self, data):
        return self.embedding(data)


class Weight_train(nn.Module):
    def __init__(self, N_view, device='cuda:0'):
        super(Weight_train, self).__init__()
        self.N_view = N_view
        # self.fc = nn.Linear(10, 1)
        self.weight = nn.Parameter(torch.ones(self.N_view, 1),
                                     requires_grad=True).to(device)
        # print(self.weight)
        self.device = device

    # def forward(self, data_list):
    #     assert len(data_list) == len(self.weight)
    #     data = [d for d in data_list]
    #     weight_norm = F.softmax(self.weight, dim=0)
    #     fuse_data = [weight_norm[idx] * F.normalize(data[idx]) for idx in range(len(data_list))]
    #     fuse_data_ = sum(fuse_data)/len(data_list)
    #     # weighted_data = [w.retain_grad() * d for w, d in zip(self.weight_list, data_list)]
    #     # fuse_data = sum(weighted_data)/len(data_list)
    #     gold_data = torch.eye(fuse_data_.size(0)).to(self.device)
    #     # mse_loss = F.mse_loss(fuse_data, gold_data)
    #     fuse_data_flat = fuse_data_.view(-1)
    #     gold_data_flat = gold_data.view(-1)
    #     cos_sim = F.cosine_similarity(fuse_data_flat, gold_data_flat, dim=0, eps=1e-8)
    #     cos_loss = 1 - cos_sim
    #     return self.weight, cos_loss

    def forward(self, batch_pair, st_feature, side_modalities_list):
        side_m_list = []
        l, r = batch_pair[:, 0].long(), batch_pair[:, 1].long()
        st_score = torch.mm(st_feature[l], st_feature[r].t())
        side_m_list.append(st_score)
        for i, (_, score) in enumerate(side_modalities_list.items()):
            min_num = score.shape[0]
            r_ = r - min_num
            # print(r_.min(), r_.max())
            side_m_list.append(torch.tensor(score[np.ix_(l.tolist(), r_.tolist())]).to(self.device))
        assert len(side_m_list) == len(self.weight)
        weight_norm = F.softmax(self.weight, dim=0)

        fuse_data = [weight_norm[idx] * F.normalize(side_m_list[idx]) for idx in range(len(side_m_list))]
        fuse_data_ = sum(fuse_data) / len(side_m_list)

        gold_data = torch.eye(fuse_data_.size(0)).to(self.device)
        fuse_data_flat = fuse_data_.view(-1)
        gold_data_flat = gold_data.view(-1)
        cos_sim = F.cosine_similarity(fuse_data_flat, gold_data_flat, dim=0, eps=1e-8)
        cos_loss = 1 - cos_sim
        return self.weight, cos_loss


class Weight_train_2(nn.Module):
    def __init__(self, N_view, device='cuda:0'):
        super(Weight_train_2, self).__init__()
        self.N_view = N_view
        self.device = device
        self.fc = nn.Linear(self.N_view, 1, bias=False, dtype=torch.float).to(device)

    def forward(self, batch_pair, st_feature, side_modalities_list):
        side_m_list = []
        l, r = batch_pair[:, 0].long(), batch_pair[:, 1].long()
        st_score = torch.mm(st_feature[l], st_feature[r].t())
        side_m_list.append(st_score)

        for i, (_, score) in enumerate(side_modalities_list.items()):
            min_num = score.shape[0]
            r_ = r - min_num
            side_m_list.append(torch.tensor(score[np.ix_(l.tolist(), r_.tolist())]).to(self.device))

        # 强制权重为非负 name和Image的权重都会为0
        # self.fc.weight.data = torch.clamp(self.fc.weight.data, min=0)
        # 对fc进行归一化
        # self.fc.weight.data = F.softmax(self.fc.weight.data , dim=0)
        # self.fc.weight.data = F.normalize(self.fc.weight.data, p=1, dim=0)

        assert len(side_m_list) == self.N_view
        side_m_tensor = torch.stack(side_m_list, dim=1)
        # 对输入的数据进行归一化
        side_m_tensor = F.normalize(side_m_tensor)
        reshaped_input = side_m_tensor.permute(0, 2, 1).reshape(-1, self.N_view)
        reshaped_input = reshaped_input.float()
        weighted_sum = self.fc(reshaped_input)
        fuse_data_ = weighted_sum.view(side_m_tensor.shape[0], side_m_tensor.shape[2])

        # 计算损失
        gold_data = torch.eye(fuse_data_.size(0)).to(self.device)
        fuse_data_flat = fuse_data_.view(-1)
        gold_data_flat = gold_data.view(-1)
        cos_sim = F.cosine_similarity(fuse_data_flat, gold_data_flat, dim=0, eps=1e-8)
        cos_loss = 1 - cos_sim

        return self.fc.weight, cos_loss


class MformerFusion(nn.Module):
    def __init__(self, device, num_hidden_layers, s_num, t_num, hidden_size, num_attention_heads, intermediate_size):
        super(MformerFusion, self).__init__()
        self.device = device
        self.fusion_layer = nn.ModuleList([BertLayer(t_num, num_attention_heads, intermediate_size)
                                           for _ in range(num_hidden_layers)])
        self.fusion_layer.to(self.device)
        self.num_attention_heads = num_attention_heads


    def forward(self, st_feature, side_modalities_list):
        side_m_list = []
        assert isinstance(side_modalities_list, dict) and side_modalities_list

        for i, (_, score) in enumerate(side_modalities_list.items()):
            s_num = score.shape[0]
            t_num = score.shape[1]
            side_m_list.append(torch.tensor(score).to(self.device))

        st_score = torch.mm(st_feature[:s_num], st_feature[s_num:].t()).to(self.device)
        side_m_list.append(st_score)
        modal_num = len(side_m_list)
        # 需要较大的计算量
        # hidden_states = torch.stack(side_m_list, dim=1)  # hidden_states: [s_num, modal_num, t_sum]
        # bs = hidden_states.shape[0]
        # for i, layer_module in enumerate(self.fusion_layer):
        #     layer_outputs = layer_module(hidden_states, output_attentions=True)
        #     hidden_states = layer_outputs[0]
        # attention_pro = torch.sum(layer_outputs[1], dim=-3)
        # attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
        # weight_norm = F.softmax(attention_pro_comb, dim=-1)
        # embs = [weight_norm[:, idx].unsqueeze(1) * F.normalize(side_m_list[idx]) for idx in range(modal_num)]
        # joint_emb = torch.cat(embs, dim=1)

        # 进行矩阵分块处理
        # hidden_states = torch.stack(side_m_list, dim=0)  # hidden_states: [modal_num, s_num, t_sum]
        # hidden_states = hidden_states.unsqueeze(0)  # hidden_states: [1, modal_num, s_num, t_sum]
        # chunk_height, chunk_width = 512, 512
        # def calculate_padding(size, chunk_size):
        #     padding = (chunk_size - (size % chunk_size)) % chunk_size
        #     return padding
        # padding_height = calculate_padding(hidden_states.size(2), chunk_height)
        # padding_width = calculate_padding(hidden_states.size(3), chunk_width)
        # padded_features = F.pad(hidden_states, (0, padding_width, 0, padding_height))
        # chunks = []
        # indices = []  # 记录每个块的起始位置
        # for i in range(0, s_num, chunk_height):
        #     for j in range(0, t_num, chunk_width):
        #         chunk = hidden_states[:, :, i:i + chunk_height, j:j + chunk_width]
        #         chunks.append(chunk)
        #         indices.append((i, j))  # 记录块的起始位置
        # for chunk in chunks:
        #     for i, layer_module in enumerate(self.fusion_layer):
        #         chunk_outputs = layer_module(chunk, output_attentions=True)
        #         chunk = chunk_outputs[0]
        #     attention_pro = torch.sum(chunk_outputs[1], dim=-3)
        #     attention_pro_comb = torch.sum(attention_pro, dim=-2)
        #                              / math.sqrt(modal_num * self.args.num_attention_heads)
        # restored_padded_features = torch.zeros_like(padded_features)  # 创建与填充后图像形状相同的空张量
        # for chunk, (i, j) in zip(chunks, indices):
        #     restored_padded_features[:, :, i:i + chunk_height, j:j + chunk_width] = chunk
        # restored_features = restored_padded_features[:, :, :hidden_states.size(2), :hidden_states.size(3)]

        # return joint_emb, hidden_states, weight_norm

        #将特征转为序列，进行序列分块
        hidden_states = torch.stack(side_m_list, dim=1)  # hidden_states: [s_num, modal_num, t_sum]
        del side_m_list
        torch.cuda.empty_cache()

        hs_new = []
        bs = 400
        # for hs in hidden_states:
        for i in range(0, len(hidden_states), bs):
            # print(hs.shape)
            hs = hidden_states[i:i + bs]
            if bs == 1:
                hs = hs.unsqueeze(0)
            for i, layer_module in enumerate(self.fusion_layer):
                hs_outputs = layer_module(hs, output_attentions=True)
                hs = hs_outputs[0]
            attention_pro = torch.sum(hs_outputs[1], dim=-3)
            attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(modal_num * self.num_attention_heads)
            weight_norm = F.softmax(attention_pro_comb, dim=-1)
            if bs > 1:
                hs_emb = [weight_norm[:, idx].unsqueeze(1) * F.normalize(hs[:, idx]) for idx in range(modal_num)]
            else:
                hs_emb = [weight_norm[:, idx] * F.normalize(hs[:, idx]) for idx in range(modal_num)]
            joint_emb = torch.cat(hs_emb, dim=0)
            hs_new.append(joint_emb)
            del hs_outputs, joint_emb, hs_emb, attention_pro_comb, weight_norm, attention_pro
            torch.cuda.empty_cache()
        weighted_emb = torch.stack(hs_new, dim=0)
        return weighted_emb

# mfmodel = MformerFusion(device='cuda:0', num_hidden_layers=1, s_num=12842, t_num=14951,
#                         hidden_size=300, num_attention_heads=1, intermediate_size=400)
# st_matx = torch.rand(27793, 768)
# st_matx = (st_matx - st_matx.min()) / (st_matx.max() - st_matx.min())
#
# side_matx = torch.rand(12842, 14951)
# side_matx = (side_matx - side_matx.min()) / (side_matx.max() - side_matx.min())
# side_modalities_list = {'0': side_matx,
#                         '1': side_matx,
#                         '2': side_matx}
# joint_emb, hidden_states, weight_norm = mfmodel(st_matx, side_modalities_list)


class Loss_Module(nn.Module):
    def __init__(self, node_size, gamma=3):
        super(Loss_Module, self).__init__()
        self.gamma = gamma
        self.node_size = node_size

    def align_loss(self, pairs, emb):

        def squared_dist(A, B):
            row_norms_A = torch.sum(torch.square(A), dim=1)
            row_norms_A = torch.reshape(row_norms_A, [-1, 1])
            row_norms_B = torch.sum(torch.square(B), dim=1)
            row_norms_B = torch.reshape(row_norms_B, [1, -1])
            return row_norms_A + row_norms_B - 2 * torch.matmul(A, B.t())

        # print(pairs)
        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        l_emb, r_emb = emb[l], emb[r]

        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        l_neg_dis = squared_dist(l_emb, emb)
        r_neg_dis = squared_dist(r_emb, emb)

        del l_emb, r_emb

        l_loss = pos_dis - l_neg_dis + self.gamma
        l_loss = l_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))
        r_loss = pos_dis - r_neg_dis + self.gamma
        r_loss = r_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))

        del r_neg_dis, l_neg_dis

        r_loss = (r_loss - torch.mean(r_loss, dim=-1, keepdim=True).detach()) / torch.std(r_loss, dim=-1,
                                                                                          unbiased=False,
                                                                                          keepdim=True).detach()
        l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(l_loss, dim=-1,
                                                                                          unbiased=False,
                                                                                          keepdim=True).detach()

        lamb, tau = 30, 10
        l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
        r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
        return torch.mean(l_loss + r_loss)

    def align_loss_weight(self,  pairs, emb):
        def squared_dist(A, B):
            row_norms_A = torch.sum(torch.square(A), dim=1)
            row_norms_A = torch.reshape(row_norms_A, [-1, 1])
            row_norms_B = torch.sum(torch.square(B), dim=1)
            row_norms_B = torch.reshape(row_norms_B, [1, -1])
            return row_norms_A + row_norms_B - 2 * torch.matmul(A, B.t())

        # pairs 现在是一个 n*3 的矩阵，其中第三列是置信度
        l, r, confidence = pairs[:, 0].long(), pairs[:, 1].long(), pairs[:, 2]
        l_emb, r_emb = emb[l], emb[r]

        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        l_neg_dis = squared_dist(l_emb, emb)
        r_neg_dis = squared_dist(r_emb, emb)

        # 使用置信度作为权重
        confidence = confidence.float()  # 确保置信度是浮点数
        pos_dis_weighted = pos_dis * confidence.unsqueeze(1)
        # pos_dis_weighted = pos_dis

        l_neg_dis_weighted = l_neg_dis
        r_neg_dis_weighted = r_neg_dis

        del l_emb, r_emb
        del r_neg_dis, l_neg_dis

        # 计算损失函数时考虑置信度
        l_loss = pos_dis_weighted - l_neg_dis_weighted + self.gamma
        r_loss = pos_dis_weighted - r_neg_dis_weighted + self.gamma

        l_loss = l_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))
        r_loss = r_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))


        del l_neg_dis_weighted, r_neg_dis_weighted

        r_loss = (r_loss - torch.mean(r_loss, dim=-1, keepdim=True).detach()) / torch.std(r_loss, dim=-1,
                                                                                          unbiased=False,
                                                                                          keepdim=True).detach()
        l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(l_loss, dim=-1,
                                                                                          unbiased=False,
                                                                                          keepdim=True).detach()

        lamb, tau = 30, 10
        l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
        r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)

        total_loss = (l_loss + r_loss) / 2
        total_loss = total_loss * confidence

        final_loss = torch.mean(total_loss)
        del l_loss, r_loss, total_loss

        return final_loss

    def forward(self, train_pairs: torch.Tensor, feature, weight=False):
        if weight:
            loss = self.align_loss_weight(train_pairs, feature)
        else:
            loss = self.align_loss(train_pairs, feature)

        return loss


class Aug_loss(nn.Module):
    def __init__(self):
        super(Aug_loss, self).__init__()

    def forward(self, pairs, emb, aug_emb, sim_method="inner", t=0.08):
        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        l_emb, r_emb = emb[l], emb[r]
        l_aug_emb, r_aug_emb = aug_emb[l], aug_emb[r]

        def logit_compute(sim_method,  emb1, emb2):
            if sim_method == "cosine":
                embeddings1_abs = emb1.norm(dim=1, keepdim=True)  # keepdim=True 保持维度
                embeddings2_abs = emb2.norm(dim=1, keepdim=True)
                logits = torch.mm(emb1, emb2.t())
                logits = logits / (embeddings1_abs * embeddings2_abs.t() + 1e-5)
                logits = logits / t
                del embeddings1_abs, embeddings2_abs
            elif sim_method == "inner":
                logits = torch.mm(emb1, emb2.T)
            return logits
        l_logits = logit_compute(sim_method, l_emb, l_aug_emb)
        r_logits = logit_compute(sim_method, r_emb, r_aug_emb)

        labels = torch.arange(len(pairs)).to(emb.device)
        loss_1 = F.cross_entropy(l_logits, labels)
        loss_1_r = F.cross_entropy(r_logits, labels)
        loss_2 = F.cross_entropy(l_logits.T, labels)
        loss_2_r = F.cross_entropy(r_logits.T, labels)

        # inner loss
        # inner_logits = logit_compute(sim_method, l_emb, l_emb)
        # inner_logits_r = logit_compute(sim_method, r_emb, r_emb)
        # inner_loss_l = F.cross_entropy(inner_logits, labels)
        # inner_loss_r = F.cross_entropy(inner_logits_r, labels)
        # inner_2_l = F.cross_entropy(inner_logits.T, labels)
        # inner_2_r = F.cross_entropy(inner_logits_r.T, labels)

        # return (loss_1 + loss_2 + loss_1_r + loss_2_r) / 4
        # 增加inner_loss,修改loss求和方式
        return (loss_1+loss_2)/2 + (loss_1_r+loss_2_r)/2
        # return (loss_1+loss_2)/2 + (loss_1_r+loss_2_r)/2 + (inner_loss_l + inner_2_l)/2
        # return (loss_1+loss_2)/2+(loss_1_r+loss_2_r)/2 + (inner_loss_l + inner_2_l)/2 + (inner_loss_r+inner_2_r)/2


