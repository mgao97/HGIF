
import torch
from sklearn.metrics import roc_auc_score,f1_score
import dgl
import random
import GCL.augmentors as Aug
import numpy as np
from GCL.augmentors.functional import dropout_adj
from GCL.augmentors.functional import sort_edge_index
EOS = 1e-10

from dgl.nn import EdgeWeightNorm
norm = EdgeWeightNorm(norm='both')


def gen_dgl_graph(index1, index2, edge_w=None, ndata=None):
    g = dgl.graph((index1, index2),num_nodes=ndata.shape[0])
    if edge_w is not None:
        g.edata['w'] = edge_w
    if ndata is not None:
        g.ndata['feature'] = ndata
    return g


def normalize(edge_index):
    """ normalizes the edge_index
    """
    adj_t = edge_index.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t




def normalize_adj(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. /(torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. /(torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size()).coalesce()


def eval_func(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred1 = y_pred.softmax(1)[:,1].detach().cpu().numpy()
    mask = np.isin(y_true, [0, 1])
    y_true = y_true[mask]
    y_pred1 = y_pred1[mask]
    auc = roc_auc_score(y_true, y_pred1)
    f1 = f1_score(y_true, y_pred.softmax(1).data[mask].cpu().numpy().argmax(axis=1), average="macro")
    return auc,f1



@torch.no_grad()
def evaluate_whole_graph( model,datasets_te):
    model.eval()
    auc_list=[]
    for i in range(len(datasets_te)):
        train_out = model.inference(datasets_te[i])
        train_auc,f1 = eval_func(datasets_te[i].ndata['label'], train_out)
        auc_list.append(train_auc)

    return auc_list




@torch.no_grad()
def evaluate_graph( model, dataset_tr):
    model.eval()
    train_out = model.inference(dataset_tr)
    train_auc,f1 = eval_func(dataset_tr.ndata['label'], train_out)


    return train_auc,f1




def graph_aug(args,dataset):
    g_list = []
    x = dataset.ndata['feature']
    edge_index = torch.cat((dataset.all_edges()[0].unsqueeze(0), dataset.all_edges()[1].unsqueeze(0)), dim=0)
    for k in range(args.K):
        x_k, edge_index_k = Graph_Editer(x, edge_index,dataset.ndata['label'])
        graph_k = dgl.graph((edge_index_k[0], edge_index_k[1]), num_nodes=x_k.shape[0])
        graph_k = dgl.add_self_loop(graph_k)
        graph_k.ndata['feature']=x_k
        g_list.append(graph_k)

    return g_list

def Graph_Editer(x, edge_index,labels):

    normal_node=torch.where(labels == 0)[0]
    abnormal_node = torch.where(labels == 1)[0]

    rate_feature= random.uniform(0, 0.5)
    rate_normal_normal_remove_edge=random.uniform(0, 1)
    rate_normal_normal_add_edge = random.uniform(0, 1)

    rate_normal_abnormal_remove_edge = random.uniform(0, 1)
    rate_normal_abnormal_add_edge = random.uniform(0, 1)

    rate_abnormal_abnormal_remove_edge = random.uniform(0, 1)
    rate_abnormal_abnormal_add_edge = random.uniform(0, 1)


    normal_index0 = torch.isin(edge_index[0], normal_node)
    normal_index1 = torch.isin(edge_index[1], normal_node)

    normal_index = normal_index0 &  normal_index1

    normal_normal_edge=edge_index[0:,normal_index]

    abnormal_index0 = torch.isin(edge_index[0], abnormal_node)
    abnormal_index1 = torch.isin(edge_index[1], abnormal_node)

    abnormal_index = abnormal_index0 & abnormal_index1

    abnormal_abnormal_edge = edge_index[0:, abnormal_index]

    normal_abnormal_index=(abnormal_index0 & normal_index1) | (abnormal_index1 & normal_index0)

    normal_abnormal_edge = edge_index[0:, normal_abnormal_index]


    normal_normal_new, edge_weights = dropout_adj(normal_normal_edge, edge_attr=None, p=rate_normal_normal_remove_edge)
    abnormal_abnormal_new, edge_weights = dropout_adj(abnormal_abnormal_edge, edge_attr=None, p=rate_abnormal_abnormal_remove_edge)
    normal_abnormal_new, edge_weights = dropout_adj(normal_abnormal_edge, edge_attr=None, p=rate_normal_abnormal_remove_edge)


    normal_normal_new = add_edge(normal_normal_new,normal_normal_edge.shape[1], normal_node,abnormal_node,ratio=rate_normal_normal_add_edge,flag=0)
    abnormal_abnormal_new = add_edge(abnormal_abnormal_new,abnormal_abnormal_edge.shape[1], normal_node,abnormal_node,ratio=rate_abnormal_abnormal_add_edge,flag=1)
    normal_abnormal_new = add_edge(normal_abnormal_new,normal_abnormal_edge.shape[1],normal_node,abnormal_node, ratio=rate_normal_abnormal_add_edge,flag=2)

    edge_index = torch.cat([normal_normal_new, abnormal_abnormal_new,normal_abnormal_new], dim=1)
    edge_index = sort_edge_index(edge_index)
    aug = Aug.Compose([Aug.FeatureMasking(pf=rate_feature)])
    x_aug, edge_index_aug,edge_weight_aug = aug(x, edge_index)
    return x_aug, edge_index



def add_edge(edge_index: torch.Tensor, num_edges1,normal_node: torch.Tensor,abnormal_node: torch.Tensor, ratio: float,flag: int) -> torch.Tensor:
    num_edges = edge_index.size()[1]
    num_add = int(num_edges1 * ratio)
    if flag==0 or flag==1:
        index=torch.randint(0, num_edges - 1, size=(2,num_add)).to(edge_index.device)
        new_edge_index = torch.cat((edge_index[0][index[0]].unsqueeze(0),edge_index[1][index[1]].unsqueeze(0)),dim=0).to(edge_index.device)
        edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        edge_index = sort_edge_index(edge_index)
    else:
        index = torch.randint(0, num_edges - 1, size=(2, 3*num_add)).to(edge_index.device)
        abnormal_index0 = torch.isin(edge_index[0][index[0]], abnormal_node)
        abnormal_index1 = torch.isin(edge_index[1][index[1]], abnormal_node)

        normal_index0 = torch.isin(edge_index[0][index[0]], normal_node)
        normal_index1 = torch.isin(edge_index[1][index[1]], normal_node)

        index_ok=abnormal_index0 & normal_index1 | abnormal_index1 & normal_index0
        index=index[0:, index_ok][0:num_add]
        new_edge_index = torch.cat((edge_index[0][index[0]].unsqueeze(0), edge_index[1][index[1]].unsqueeze(0)),dim=0).to(edge_index.device)
        edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        edge_index = sort_edge_index(edge_index)

    return edge_index




def graph_aug1(args,dataset):
    g_list = []
    x = dataset.ndata['feature']
    edge_index = torch.cat((dataset.all_edges()[0].unsqueeze(0), dataset.all_edges()[1].unsqueeze(0)), dim=0)
    for k in range(args.K):
        x_k, edge_index_k = Graph_Editer1(x, edge_index)
        graph_k = dgl.graph((edge_index_k[0], edge_index_k[1]), num_nodes=x_k.shape[0])
        graph_k = dgl.add_self_loop(graph_k)
        graph_k.ndata['feature'] = x_k

        g_list.append(graph_k)

    return g_list




def Graph_Editer1(x, edge_index):
    rate= random.uniform(0, 0.5)
    aug = Aug.Compose([Aug.EdgeRemoving(pe=rate), Aug.FeatureMasking(pf=rate)])
    x_aug, edge_index_aug,edge_weight_aug = aug(x, edge_index)
    return x_aug, edge_index_aug


###############################
import torch
from torch.utils.data import Dataset

import os
import torch
import dgl
import pandas as pd
from torch.utils.data import Dataset
import torch.utils.data as data_utils

class BotDataset(Dataset):
    def __init__(self, name, text_data, profile_data, social_data, label_data, edge_data=None):
        self.name = name
        self.text_data = text_data
        self.profile_data = profile_data
        self.social_data = social_data
        self.label_data = label_data
        self.edge_data = edge_data  # 新增边数据

        if self.name == "train":
            self.start = 0
            self.end = 8278
        elif self.name == "valid":
            self.start = 8278
            self.end = 10643
        elif self.name == "test":
            self.start = 10643
            self.end = 11826
        
        self.len = self.end - self.start

    def __len__(self):
        return self.len

def process_and_save_graph(args, dataset_name):
    # 加载数据
    text_data = torch.load(args.path + 'tweets_tensor.pt')
    profile_data = torch.load(args.path + 'des_tensor.pt')
    social_data = torch.load(args.path + 'graph_feats2.pt')
    label_data = torch.tensor(torch.load(args.path + 'labels.pt', weights_only=False)).long()

    # 假设 edge.csv 文件的格式为：source, target, relation
    edge_file = args.path + 'edge_new.csv'
    filtered_edges_df = pd.read_csv(edge_file)
    
        # 创建一个唯一的整数 ID 映射
    # unique_nodes = pd.concat([filtered_edges_df['source_id'], filtered_edges_df['target_id']]).unique()
    # node_id_map = {node: idx for idx, node in enumerate(unique_nodes)}
    
    
    
    # max_node_id = 229579  # 最大允许的节点 ID
    unique_nodes = pd.concat([filtered_edges_df['source_id'], filtered_edges_df['target_id']]).unique()
    
    
    node_id_map = {node: idx for idx, node in enumerate(unique_nodes)}
    
    # print(node_id_map)

    # 创建新的整数 ID 列
    filtered_edges_df['source_id_int'] = filtered_edges_df['source_id'].map(node_id_map)
    filtered_edges_df['target_id_int'] = filtered_edges_df['target_id'].map(node_id_map)
    
    
    
    filtered_edges_df = filtered_edges_df[
    (filtered_edges_df['source_id_int'] < 229580) &
    (filtered_edges_df['target_id_int'] < 229580)
    ]
    
    unique_nodes = pd.concat([filtered_edges_df['source_id_int'], filtered_edges_df['target_id_int']]).unique()
    

    # 保存映射后的 DataFrame
    mapped_edge_file = args.path + 'edge_new_mapped.csv'
    filtered_edges_df.to_csv(mapped_edge_file, index=False)
    
    # 构建完整图
    graph_data = {
        ('user', 'follow', 'user'): (filtered_edges_df[filtered_edges_df['relation'] == 'follow']['source_id_int'].values, filtered_edges_df[filtered_edges_df['relation'] == 'follow']['target_id_int'].values),
        ('user', 'friend', 'user'): (filtered_edges_df[filtered_edges_df['relation'] == 'friend']['source_id_int'].values, filtered_edges_df[filtered_edges_df['relation'] == 'friend']['target_id_int'].values),
        ('user', 'similar', 'user'): (filtered_edges_df[filtered_edges_df['relation'] == 'similarity']['source_id_int'].values, filtered_edges_df[filtered_edges_df['relation'] == 'similarity']['target_id_int'].values),
    }

    x = torch.cat([text_data, profile_data, social_data[:229580]], dim=1)
    
    # 构建完整的图
    full_g = dgl.heterograph(graph_data)
    if full_g.number_of_nodes('user') != x.shape[0]:
        full_g.nodes['user'].data['feature'] = x[:full_g.number_of_nodes('user'),:]
        full_g.nodes['user'].data['label'] = label_data[:full_g.number_of_nodes('user')]

    # 保存完整的图
    save_path = f"{args.path}{dataset_name}_graph.bin"
    dgl.save_graphs(save_path, full_g)
    print(f"Saved full graph to {save_path}")
    
    # # 构建训练、验证、测试集
    # dataset_dict = {
    #     "train": (0, 8278),
    #     "valid": (8278, 10643),
    #     "test": (10643, 11826)
    # }

    # for name in ["train", "valid", "test"]:
    #     start, end = dataset_dict[name]
        
    #     # 获取子集数据
    #     subset_text_data = text_data[start:end]
    #     subset_profile_data = profile_data[start:end]
    #     subset_social_data = social_data[start:end]
    #     subset_label_data = label_data[start:end]

    #     # 拼接节点特征
    #     x = torch.cat([subset_text_data, subset_profile_data, subset_social_data], dim=1)
        
    #     subset_edges_df = filtered_edges_df[
    #         (filtered_edges_df['source_id_int'] >= start) &
    #         (filtered_edges_df['source_id_int'] < end) &
    #         (filtered_edges_df['target_id_int'] >= start) &
    #         (filtered_edges_df['target_id_int'] < end)
    #     ]

    #     # 转换为 DGL 的边索引格式
    #     # 调整节点索引为子集内的相对索引
    #     # 例如：source 和 target 都是全局索引，需要减去 start 得到子集内的索引
    #     subset_edges_df['source_id_int'] = subset_edges_df['source_id_int'] - start
    #     subset_edges_df['target_id_int'] = subset_edges_df['target_id_int'] - start

    #     # 分离边类型
    #     follow_edges_df = subset_edges_df[subset_edges_df['relation'] == 'follow']
    #     friend_edges_df = subset_edges_df[subset_edges_df['relation'] == 'friend']
    #     similarity_df = subset_edges_df[subset_edges_df['relation'] == 'similarity']

    #     # 构建图数据
    #     graph_data = {
    #         ('user', 'follow', 'user'): (follow_edges_df['source_id_int'].values, follow_edges_df['target_id_int'].values),
    #         ('user', 'friend', 'user'): (friend_edges_df['source_id_int'].values, friend_edges_df['target_id_int'].values),
    #         ('user', 'similar', 'user'): (similarity_df['source_id_int'].values, similarity_df['target_id_int'].values)
    #     }

    #     # 构建 DGL 图
    #     g = dgl.heterograph(graph_data)
        
    #     print(g.number_of_nodes('user'))  
    #     print(g.number_of_edges())  

    #     print(x.shape)

    #     # 添加节点特征和标签
    #     if g.number_of_nodes('user') != x.shape[0]:
    #         g.nodes['user'].data['feature'] = x[:g.number_of_nodes('user'),:]
    #         g.nodes['user'].data['label'] = subset_label_data[:g.number_of_nodes('user')]
    #     else:
    #         g.nodes['user'].data['feature'] = x
    #         g.nodes['user'].data['label'] = subset_label_data

        
        
    #     # 保存图
    #     save_path = f"{args.path}{dataset_name}_{name}_graph.bin"
    #     dgl.save_graphs(save_path, g)

    #     print(f"Saved {name} graph to {save_path}")



import argparse
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser.add_argument("--path", type=str, default="../data/twibot-20/", help="dataset path")

args = parser.parse_args()


process_and_save_graph(args, "twibot-20")

