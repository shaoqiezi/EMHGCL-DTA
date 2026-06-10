import pickle
import json
import torch
import numpy as np
import networkx as nx

from torch_geometric import data as DATA
from collections import OrderedDict
from rdkit import Chem
from utils import DTADataset, minMaxNormalize, denseAffinityRefine



def load_data(dataset):
    affinity = pickle.load(open('data/' + dataset + '/affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = -np.log10(affinity / 1e9)
    return affinity


def process_data(affinity_mat, dataset):
    dataset_path = 'data/' + dataset + '/'

    train_file = json.load(open(dataset_path + 'S1_train_set.txt'))

    train_index = []
    for i in range(len(train_file)):
        train_index += train_file[i]
    test_index = json.load(open(dataset_path + 'S1_test_set.txt'))

    rows, cols = np.where(np.isnan(affinity_mat) == False)
    train_rows, train_cols = rows[train_index], cols[train_index]
    train_Y = affinity_mat[train_rows, train_cols]
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)
    test_rows, test_cols = rows[test_index], cols[test_index]
    test_Y = affinity_mat[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

    train_affinity_mat = np.zeros_like(affinity_mat)
    train_affinity_mat[train_rows, train_cols] = train_Y
    affinity_graph = get_affinity_graph(dataset, train_affinity_mat)

    return train_dataset, test_dataset, affinity_graph


def get_affinity_graph(dataset, adj):
    dataset_path = 'data/' + dataset + '/'
    num_drug, num_target = adj.shape[0], adj.shape[1]

    d_d = np.loadtxt(dataset_path + 'drug-drug-sim.txt', delimiter=',')
    t_t = np.loadtxt(dataset_path + 'target-target-sim.txt', delimiter=',')



    if dataset == "davis":
        adj[adj != 0] -= 5
        adj_norm = minMaxNormalize(adj, 0)
    elif dataset == "kiba":
        adj_refine = denseAffinityRefine(adj.T, 150)
        adj_refine = denseAffinityRefine(adj_refine.T, 40)
        adj_norm = minMaxNormalize(adj_refine, 0)
    else:
        adj_refine = denseAffinityRefine(adj.T, 150)
        adj_refine = denseAffinityRefine(adj_refine.T, 40)
        adj_norm = minMaxNormalize(adj_refine, 0)
    adj_1 = adj_norm
    adj_2 = adj_norm.T

    d_d_denoise = adaptive_denoising(d_d, 1.0)
    t_t_denoise = adaptive_denoising(t_t, 1.0)


    adj_1_features = np.zeros_like(adj_1)
    adj_1_features[adj_1 != 0] = 1
    adj_2_features = np.zeros_like(adj_2)
    adj_2_features[adj_2 != 0] = 1


    adj_x = np.concatenate((
        np.concatenate((d_d_denoise, adj_1_features), 1),
        np.concatenate((adj_2_features, t_t_denoise), 1)
    ), 0)



    adj = np.concatenate((
        np.concatenate((np.zeros([num_drug, num_drug]), adj_1), 1),
        np.concatenate((adj_2, np.zeros([num_target, num_target])), 1)
    ), 0)


    train_row_ids, train_col_ids = np.where(adj != 0)
    edge_indexs = np.concatenate((
        np.expand_dims(train_row_ids, 0),
        np.expand_dims(train_col_ids, 0)
    ), 0)
    edge_weights = adj[train_row_ids, train_col_ids]

    affinity_graph = DATA.Data(x=torch.Tensor(adj_x), adj=torch.Tensor(adj),
                               edge_index=torch.LongTensor(edge_indexs))
    affinity_graph.__setitem__("edge_weight", torch.Tensor(edge_weights))
    affinity_graph.__setitem__("num_drug", num_drug)
    affinity_graph.__setitem__("num_target", num_target)

    return affinity_graph


def adaptive_denoising(sim_matrix, alpha=1.0):
    """
    自适应去噪：根据每行分布动态保留显著特征
    Args:
        sim_matrix: 原始相似度矩阵 (N x N)
        alpha: 统计阈值系数 (建议取值范围 0.5 ~ 2.0)
               threshold = mean + alpha * std
    Returns:
        denoised_matrix: 去噪后的自适应特征矩阵
    """
    # 1. 计算每一行的均值和标准差
    # means / stds 形状为 (N, 1)
    means = np.mean(sim_matrix, axis=1, keepdims=True)
    stds = np.std(sim_matrix, axis=1, keepdims=True)

    # 2. 计算每一行的自适应阈值
    thresholds = means + alpha * stds

    # 3. 生成掩码 (Mask)
    # 只有相似度大于阈值的位才为 True
    mask = sim_matrix >= thresholds

    # 4. 应用掩码
    # 将不满足条件的位设为 0
    denoised_matrix = np.where(mask, sim_matrix, 0.0)

    # 5. 可选：确保对角线（自身相似度）始终保留
    # 自身相似度通常是 1.0，肯定大于阈值，但手动设置一下更稳健
    np.fill_diagonal(denoised_matrix, np.diagonal(sim_matrix))

    return denoised_matrix

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):

    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def get_drug_molecule_graph(ligands):
    smile_graph = OrderedDict()

    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        smile_graph[d] = smile_to_graph(lg)

    return smile_graph


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])


    return c_size, features, edge_index, smile







