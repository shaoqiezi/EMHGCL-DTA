import torch
import torch.nn as nn

from torch_geometric.nn import DenseGCNConv, GCNConv, global_mean_pool as gep
from torch_geometric.utils import dropout_adj
from utils import _similarity
import esm



class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss, mode, intraview_negs=False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None):
        l1 = self.loss(anchor=h1, sample=h2)
        l2 = self.loss(anchor=h2, sample=h1)
        return (l1 + l2) * 0.5

class InfoNCE(object):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob.diag()
        return -loss.mean()

    def __call__(self, anchor, sample) -> torch.FloatTensor:
        loss = self.compute(anchor, sample)
        return loss
class GCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(GCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = GCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, edge_index, edge_weight)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(gep(output, batch))

        return embeddings


class GCNModel(nn.Module):
    def __init__(self, layers_dim):
        super(GCNModel, self).__init__()

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GCNBlock(layers_dim, relu_layers_index=list(range(self.num_layers)))

    def forward(self, graph_batchs):
        embedding_batchs = list(
                map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))
        embeddings = []
        for i in range(self.num_layers):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embeddings

class ESMBlock(nn.Module):
    def __init__(self):
        super(ESMBlock, self).__init__()

        self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_size = 40

    def forward(self, graph):
        truncated = [(label, seq[:1024]) for label, seq in graph]
        embedding = []
        for i in range(0, len(truncated), self.batch_size):
            batch = truncated[i:i + self.batch_size]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
            batch_tokens = batch_tokens.to(self.device)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            # 前向传播
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[self.model.num_layers])

            # 提取特征
            token_representations = results["representations"][self.model.num_layers]
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations = token_representations[i, 1: tokens_len - 1].mean(0)
                embedding.append(sequence_representations)

        return torch.stack(embedding, dim=0)

class ESMModel(nn.Module):
    def __init__(self):
        super(ESMModel, self).__init__()

        self.pro_rep = ESMBlock()

    def forward(self, graph_batchs):
        batchs = list(map(lambda graph: graph.seq, graph_batchs))

        embedding_batchs = self.pro_rep(batchs[0])
        embeddings = embedding_batchs
        return embeddings

class DenseGCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(DenseGCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = DenseGCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, adj, add_loop=True)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))

        return embeddings


class DenseGCNModel(nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0.):
        super(DenseGCNModel, self).__init__()

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, relu_layers_index=list(range(self.num_layers)),
                                        dropout_layers_index=list(range(self.num_layers)))

    def forward(self, graph):
        xs, adj, num_d, num_t = graph.x, graph.adj, graph.num_drug, graph.num_target
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs],
                                                                p=self.edge_dropout_rate, force_undirected=True,
                                                                num_nodes=num_d + num_t, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        embeddings = self.graph_conv(xs, adj_dropout)

        return embeddings


class LinearBlock(nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = nn.Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = []
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)

        return embeddings



class Contrast(nn.Module):
    def __init__(self, tau, lam):
        super(Contrast, self).__init__()

        self.tau = tau
        self.lam = lam
        self.ratio = 1
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = 'cpu'

    def forward(self, z):
        k = torch.tensor(int(z.shape[0] * self.ratio))
        p = (1 / torch.sqrt(k)) * torch.randn(k, z.shape[0]).to(self.device)
        z1 = p @ z
        contrast_model = DualBranchContrast(loss=InfoNCE(
            tau=self.tau), mode='L2L', intraview_negs=True).to(self.device)
        loss = contrast_model(z, z1)

        return self.lam * loss

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=1):
        super().__init__()
        self.in_dim = in_channels
        bottleneck_dim = self.in_dim // reduction_ratio

        # 建立 Squeeze-and-Excitation (SE) 机制的多层感知机
        self.se_mlp = nn.Sequential(
            nn.Linear(self.in_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, self.in_dim),
            nn.Sigmoid()  # 输出权重范围在 (0, 1)
        )


    def forward(self, s1, s2):
        s = torch.concat((s1, s2), 1)
        # 2. 计算通道注意力权重
        attention_weights = self.se_mlp(s)  # 形状: (B, dim1 + dim2)

        # 3. 特征重标定 (权重按元素乘回原特征)
        recalibrated_feat = attention_weights * s  # 形状: (B, dim1 + dim2)

        return recalibrated_feat
class EMHGCLDTA(nn.Module):
    def __init__(self, tau, lam, ns_dims, d_ms_dims, t_ms_dims, embedding_dim=128, dropout_rate=0.2):
        super(EMHGCLDTA, self).__init__()

        self.output_dim = embedding_dim * 2

        self.affinity_graph_conv = DenseGCNModel(ns_dims, dropout_rate)
        self.drug_graph_conv = GCNModel(d_ms_dims)
        self.target_graph_conv = GCNModel(t_ms_dims)
        self.drug_contrast = Contrast(tau, lam)
        self.target_contrast = Contrast(tau, lam)
        self.target_esm = ESMModel()
        self.lin = nn.Linear(320, self.output_dim)
        self.ca = ChannelAttention(in_channels=512)



    def forward(self, affinity_graph, drug_graph_batchs, prot):
        num_d = affinity_graph.num_drug

        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]
        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]

        target_esm_embedding = self.lin(prot)


        drug_embedding = self.ca(affinity_graph_embedding[:num_d], drug_graph_embedding)
        target_embedding = self.ca(affinity_graph_embedding[num_d:], target_esm_embedding)

        dru_loss = self.drug_contrast(drug_embedding)
        tar_loss = self.target_contrast(target_embedding)

        return dru_loss + tar_loss, drug_embedding, target_embedding



class PredictModule(nn.Module):
    def __init__(self, embedding_dim=128, output_dim=1):
        super(PredictModule, self).__init__()

        self.prediction_func, prediction_dim_func = (lambda x, y: torch.cat((x, y), -1), lambda dim: 8 * dim)
        mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]

        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, drug_embedding, target_embedding):
        drug_id, target_id, y = data.drug_id, data.target_id, data.y

        drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
        target_feature = target_embedding[target_id.int().cpu().numpy()]

        concat_feature = self.prediction_func(drug_feature, target_feature)
        mlp_embeddings = self.mlp(concat_feature)
        out = mlp_embeddings[-1]

        return out


