import torch
import torch.nn as nn
from dgl.data import FraudYelpDataset, FraudAmazonDataset
import os
import dgl

import GCL.augmentors as Aug
import numpy as np
import scipy.sparse as sp
import copy
import random
from dgl.data.utils import load_graphs, save_graphs

def random_divide_list(lst, n):
    # 随机排序列表
    random.shuffle(lst)

    # 拷贝列表并计算每等份的大小
    lst_copy = copy.deepcopy(lst)
    size = len(lst_copy) // n
    remainder = len(lst_copy) % n

    result = []
    for i in range(n):
        # 获取列表的一等份
        if i < remainder:
            result.append(lst_copy[i * size:(i + 1) * size + 1])
        else:
            result.append(lst_copy[i * size:(i + 1) * size])

    return result

def normalize_features(mx, norm_row=True):
    """
    Row-normalize sparse matrix
        Code from https://github.com/williamleif/graphsage-simple/
    """

    if norm_row:
        rowsum = np.array(mx.sum(1)) + 0.01
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)

    else:
        column_max = mx.max(dim=0)[0].unsqueeze(0)
        column_min = mx.min(dim=0)[0].unsqueeze(0)
        min_max_column_norm = (mx - column_min) / (column_max - column_min)
        # l2_norm = torch.norm(min_max_column_norm, p=2, dim=-1, keepdim=True)
        mx = min_max_column_norm
    return mx


# def gen_dataset(name):
#     if name == 'yelp':
#         dataset = FraudYelpDataset()
#     elif name == 'amazon':
#         dataset = FraudAmazonDataset()
#     graph = dataset[0]
#     graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
#     data_dir = '../data/{}/{}/gen/'.format(name,name)
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#
#     if name == 'yelp':
#         my_list = list(range(45954))
#     else:
#         my_list = list(range(11944))
#
#
#     num_splits = 5  # 随机等分成3部分
#     random_splits = random_divide_list(my_list,num_splits)
#
#     rate=[0.05,0.015,0.0075,0.005,0.001]
#     for i in range(num_splits):
#
#         sub_graph=dgl.node_subgraph(graph,random_splits[i])
#
#         Generator_noise = np.random.normal(i/5, 1, size=(sub_graph.ndata['feature'].shape[1], 20))
#         Generator_noise = torch.tensor(Generator_noise, dtype=torch.float32)
#         x2 = torch.mm(sub_graph.ndata['feature'], Generator_noise)
#
#         anomaly_index = torch.where(sub_graph.ndata['label'] == 1)[0]
#
#         index0=torch.isin(sub_graph.edges()[0], anomaly_index)
#         index1 = torch.isin(sub_graph.edges()[1], anomaly_index)
#
#         index= (~index0) | (~index1)
#
#         edge_0=sub_graph.edges()[0][index]
#         edge_1=sub_graph.edges()[1][index]
#
#         edge=torch.cat((edge_0.unsqueeze(0),edge_1.unsqueeze(0)))
#
#
#         edge_new=[]
#         anomaly_index=anomaly_index.tolist()
#         for k in range(len(anomaly_index)-1):
#             for m in range(k+1,len(anomaly_index)):
#                 random=np.random.random()
#                 if random<rate[i]:
#                     edge_new.append([anomaly_index[k],anomaly_index[m]])
#                     edge_new.append([anomaly_index[m], anomaly_index[k]])
#
#         edge_new=torch.cat((edge,torch.tensor(edge_new).T),dim=1)
#
#         sub_graph_new=dgl.graph((edge_new[0,0:],edge_new[1,0:]))
#         sub_graph_new.ndata['feature']=torch.cat([sub_graph.ndata['feature'], x2.detach()], dim=1)
#         sub_graph_new.ndata['label'] = sub_graph.ndata['label']
#
#
#
#
#         dgl.save_graphs(data_dir + "my_{}_{}_graph.bin".format(name,i),  sub_graph_new)
#
#



def gen_dataset(name):
    if name == 'yelp':
        dataset = FraudYelpDataset()
        graph = dataset[0]
        graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
    elif name == 'amazon':
        dataset = FraudAmazonDataset()
        graph = dataset[0]
        print('graph:',graph)
        graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
    elif name == 'tfinance':
        dataset, label_dict = load_graphs('../data/tfinance/tfinance1')
        graph = dataset[0]
        graph.ndata['label'] = dataset[0].ndata['label'].argmax(1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
    elif name == 'twibot-20':
        dataset = dgl.load_graphs('../data/twibot-20/twibot-20_graph.bin')
        
        # print('dataset:',dataset)
        graph = dataset[0][0]
        print('graph:',graph)
        
        print(type(graph))
        graph = dgl.to_homogeneous(graph)
        
        train_mask = np.zeros(229580, dtype=np.bool_)
        val_mask = np.zeros(229580, dtype=np.bool_)
        test_mask = np.zeros(229580, dtype=np.bool_)
        # 定义索引范围
        train_idx = np.arange(0, 8278)
        val_idx = np.arange(8278, 10643)
        test_idx = np.arange(10643, 11826)

        # 设置掩码
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        graph.ndata["train_mask"] = torch.tensor(train_mask)
        graph.ndata["val_mask"] = torch.tensor(val_mask)
        graph.ndata["test_mask"] = torch.tensor(test_mask)
        
        text_data = torch.load('../data/twibot-20/tweets_tensor.pt')
        profile_data = torch.load('../data/twibot-20/des_tensor.pt')
        social_data = torch.load('../data/twibot-20/graph_feats2.pt')
        label_data = torch.tensor(torch.load('../data/twibot-20/labels.pt', weights_only=False)).long()
        
        graph.ndata["feature"] = torch.cat([text_data, profile_data, social_data[:229580]], dim=1)
        graph.ndata["label"] = label_data[:229580]
        
    elif name == 'tsocial':
        dataset, label_dict = load_graphs('../data/tsocial/tsocial1')
        graph = dataset[0]
        graph.ndata['feature']=graph.ndata['feature'].float()




    data_dir = '../data/{}/{}/gen/'.format(name,name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if name == 'yelp':
        my_list = list(range(45954))
        num=20
    elif name == 'amazon':
        my_list = list(range(11944))
        num = 20
    elif name == 'twibot-20':
        my_list = list(range(229580))
        num = 5
    elif name == 'tfinance':
        my_list = list(range(39357))
        num = 5
    elif name == 'tsocial':
        my_list = list(range(5781065))
        num = 5


    num_splits = 5  # 随机等分成3部分
    rate=[0.05,0.015,0.0075,0.005,0.001]
    random_splits = random_divide_list(my_list, num_splits)
    for i in range(num_splits):
        sub_graph = dgl.node_subgraph(graph, random_splits[i])
        Generator_noise = np.random.normal(i / 5, 1, size=(sub_graph.ndata['feature'].shape[1], num))
        Generator_noise = torch.tensor(Generator_noise, dtype=torch.float32)
        x2 = torch.mm(sub_graph.ndata['feature'], Generator_noise)

        anomaly_index = torch.where(sub_graph.ndata['label'] == 1)[0]

        index0 = torch.isin(sub_graph.edges()[0], anomaly_index)
        index1 = torch.isin(sub_graph.edges()[1], anomaly_index)

        index = (~index0) | (~index1)

        edge_0 = sub_graph.edges()[0][index]
        edge_1 = sub_graph.edges()[1][index]

        edge = torch.cat((edge_0.unsqueeze(0), edge_1.unsqueeze(0)))

        edge_new = []
        anomaly_index = anomaly_index.tolist()
        for k in range(len(anomaly_index) - 1):
            for m in range(k + 1, len(anomaly_index)):
                random = np.random.random()
                if random < rate[i]:
                    edge_new.append([anomaly_index[k], anomaly_index[m]])
                    edge_new.append([anomaly_index[m], anomaly_index[k]])

        edge_new = torch.cat((edge, torch.tensor(edge_new).T), dim=1)
        sub_graph_new = dgl.graph((edge_new[0, 0:], edge_new[1, 0:]),num_nodes=sub_graph.ndata['feature'].shape[0])
        sub_graph_new.ndata['feature'] = torch.cat([sub_graph.ndata['feature'], x2.detach()], dim=1)
        sub_graph_new.ndata['label'] = sub_graph.ndata['label']
        dgl.save_graphs(data_dir + "my_{}_{}_graph.bin".format(name,i),  sub_graph_new)

if __name__ == '__main__':
    # gen_dataset(name='tfinance' )
    # gen_dataset(name='amazon')
    gen_dataset(name='twibot-20')