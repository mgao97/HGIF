import pandas as pd
import networkx as nx
# from node2vec import Node2Vec
import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import pandas as pd
import easygraph as eg
import numpy as np
import matplotlib.pyplot as plt
import faiss

path = 'Twibot-20/'
# 读取 edge.csv 文件
edge_file = path+'edge.csv'
edges_df = pd.read_csv(edge_file)

# 过滤数据，选择 relation 为 'follow' 和 'friend'
filtered_edges_df = edges_df[edges_df['relation'].isin(['follow', 'friend'])]

filtered_nodes = set(list(filtered_edges_df['source_id'])+list(filtered_edges_df['target_id']))


text_feats = torch.load(path+'tweets_tensor.pt')
# text_feats,text_feats.shape

users = pd.read_csv(path+"user_ids.csv", names=["index", "user_id", "name"], usecols=range(3))
# user.head()

all_nodes = set(list(users['user_id']))

# 筛选以 'u' 开始的 id
all_nodes = {node for node in all_nodes if node.startswith('u')}
print('all_nodes:',len(all_nodes))


# 获取不在 filtered_nodes 中的节点
unfiltered_nodes = all_nodes - filtered_nodes

# 提取 unfiltered_nodes 的文本特征
unfiltered_node_ids = list(unfiltered_nodes)

# 创建一个有向图
G = eg.Graph()

# 逐行添加边到图中
for index, row in filtered_edges_df.iterrows():
    G.add_edge(row['source_id'], row['target_id'])

# print(list(G.nodes.keys())[:10])
print(len(G.nodes),len(G.edges))



# 初始化 dict1
dict1 = {i: [] for i in range(len(filtered_nodes))}

# 创建 node_map
node_map = {nodeid: int(index) for index, nodeid in enumerate(all_nodes)}

# 生成 user_embeddings
user_embeddings = {
    user_id: [text_feats[idx] for idx in tweet_indices]
    for user_id, tweet_indices in dict1.items()
}

# 生成 unfiltered_node_index
unfiltered_node_index = [node_map[u] for u in unfiltered_node_ids]

print('+'*100)

unfiltered_node_feats = np.array([text_feats[node] for node in unfiltered_node_index])

# 初始化 FAISS 索引
dimension = unfiltered_node_feats.shape[1]
index = faiss.IndexFlatL2(dimension)  # 使用 L2 距离
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
index.add(unfiltered_node_feats)

# 初始参数
similarity_threshold = 0.8
k = 3  # 初始 k 值
max_k = 5  # 最大 k 值
min_similarity = 0.5  # 最低相似性阈值

# 记录已添加的边
# existing_edges = set(G.edges)

# 动态调整参数，直到覆盖所有节点
while len(G.nodes) < len(all_nodes):
    # 计算最近邻
    D, I = index.search(unfiltered_node_feats, k)

    # 构建新边
    new_edges = []
    for i in tqdm(range(len(unfiltered_node_ids)), desc=f"Building edges (k={k})"):
        for j in range(1, k):  # 跳过自身（I[i, 0] 是自身）
            if I[i, j] != -1:
                similarity = 1 / (1 + D[i, j])  # 将 L2 距离转换为相似度
                if similarity > similarity_threshold:
                    source = unfiltered_node_ids[i]
                    target = unfiltered_node_ids[I[i, j]]
                    # if (source, target) not in existing_edges:
                    new_edges.append((source, target))
                        # existing_edges.add((source, target))
    
    # 将新边添加到图中
    for edge in new_edges:
        G.add_edge(*edge)
    
    # 检查节点数目是否满足条件
    if len(G.nodes) < len(all_nodes):
        # 调整参数
        if k < max_k:
            k += 1
        else:
            similarity_threshold *= 0.8  # 降低相似性阈值
        print(f"Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
    else:
        break

print(f"Final Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
# Final Nodes: 229580, Edges: 308444



# # skip_gram_params=dict(  # Skip-Gram 参数
# #         window=10,  # 窗口大小
# #         min_count=1,  # 最小词频
# #         batch_words=4  # 每批处理的词数
# #     )

# # node_embeddings,_ = eg.node2vec(G,
# #     dimensions=128,  # 图嵌入的维度
# #     walk_length=80,  # 每次随机游走的长度
# #     num_walks=10,  # 每个节点的随机游走次数
# #     p=0.5,  # 返回超参数，控制随机游走的返回概率
# #     q=4,  # 进出超参数，控制随机游走的向外探索概率
# #     weight_key=None,  # 边的权重键
# #     **skip_gram_params
# # )

# # embeddings_array = {node_id: torch.from_numpy(embedding) for node_id, embedding in node_embeddings.items()}

# # # # 提取节点和嵌入向量
# # # nodes = list(node_embeddings.index_to_key)
# # # embeddings = [node_embeddings[node] for node in nodes]

# # # # 将嵌入向量转换为 NumPy 数组
# # # embeddings_array = np.array(embeddings)

# # embedding_vectors = list(embeddings_array.values())

# # # 确保所有嵌入向量的维度一致
# # # 将列表中的嵌入向量转换为 NumPy 数组
# # embeddings_array = np.stack(embedding_vectors, axis=0)

# # # 转换为 PyTorch tensor
# # embeddings_tensor = torch.tensor(embeddings_array)

# # # 将 NumPy 数组转换为 PyTorch 张量
# # # embeddings_tensor = torch.tensor(embeddings_array)

# # # 保存嵌入到文件
# # output_file = 'graph_feats1.pt'
# # torch.save(embeddings_tensor, output_file)

# # print(f"Node embeddings have been saved to {output_file}")


# import pandas as pd
# import networkx as nx
# import os
# import torch
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from tqdm import tqdm
# import easygraph as eg
# import faiss

# path = 'Twibot-20/'

# # 读取 edge.csv 文件
# edge_file = path + 'edge.csv'
# edges_df = pd.read_csv(edge_file)

# # 过滤数据，选择 relation 为 'follow' 和 'friend'
# filtered_edges_df = edges_df[edges_df['relation'].isin(['follow', 'friend'])]

# filtered_nodes = set(list(filtered_edges_df['source_id']) + list(filtered_edges_df['target_id']))

# text_feats = torch.load(path + 'tweets_tensor.pt')

# users = pd.read_csv(path + 'user_ids.csv')
# all_nodes = set(list(users['user_id']))
# all_nodes = {node for node in all_nodes if node.startswith('u')}
# print('all_nodes:', len(all_nodes))

# unfiltered_nodes = all_nodes - filtered_nodes
# unfiltered_node_ids = list(unfiltered_nodes)

# G = eg.Graph()

# for index, row in filtered_edges_df.iterrows():
#     G.add_edge(row['source_id'], row['target_id'])

# print(len(G.nodes), len(G.edges))

# dict1 = {i: [] for i in range(len(filtered_nodes))}
# node_map = {nodeid: int(index) for index, nodeid in enumerate(all_nodes)}

# user_embeddings = {
#     user_id: [text_feats[idx] for idx in tweet_indices]
#     for user_id, tweet_indices in dict1.items()
# }

# unfiltered_node_index = [node_map[u] for u in unfiltered_node_ids]

# unfiltered_node_feats = np.array([text_feats[node] for node in unfiltered_node_index])

# dimension = unfiltered_node_feats.shape[1]
# index = faiss.IndexFlatL2(dimension)  # 使用 L2 距离
# if faiss.get_num_gpus() > 0:
#     res = faiss.StandardGpuResources()
#     index = faiss.index_cpu_to_gpu(res, 0, index)
# index.add(unfiltered_node_feats)

# similarity_threshold = 0.8
# k = 3
# max_k = 5
# min_similarity = 0.5

# existing_edges = set(G.edges)

# while len(G.nodes) < len(all_nodes):
#     D, I = index.search(unfiltered_node_feats, k)

#     new_edges = []
#     for i in tqdm(range(len(unfiltered_node_ids)), desc=f"Building edges (k={k})"):
#         for j in range(1, k):  # 跳过自身（I[i, 0] 是自身）
#             if I[i, j] != -1:
#                 similarity = 1 / (1 + D[i, j])  # 将 L2 距离转换为相似度
#                 if similarity > similarity_threshold:
#                     source = unfiltered_node_ids[i]
#                     target = unfiltered_node_ids[I[i, j]]
#                     if (source, target) not in existing_edges:
#                         new_edges.append((source, target))
#                         existing_edges.add((source, target))

#     for edge in new_edges:
#         G.add_edge(*edge)

#     if len(G.nodes) < len(all_nodes):
#         if k < max_k:
#             k += 1
#         else:
#             similarity_threshold *= 0.8
#         print(f"Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
#     else:
#         break

# print(f"Final Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")

# 生成 edge_new.csv 文件
# 获取所有边（包括原始边和新增边）
all_edges = list(G.edges())

# 创建新的 DataFrame
edge_new_df = pd.DataFrame(all_edges, columns=['source_id', 'target_id'])

# 为原始边和新增边设置 relation
original_edges = filtered_edges_df[['source_id', 'target_id']]
edge_new_df = edge_new_df.merge(original_edges, on=['source_id', 'target_id'], how='left', indicator=True)
edge_new_df['relation'] = 'similarity'  # 默认为 similarity
edge_new_df.loc[edge_new_df['_merge'] == 'both', 'relation'] = filtered_edges_df['relation']  # 原始边的 relation

# 只保留需要的列
edge_new_df = edge_new_df[['source_id', 'relation', 'target_id']]

# 保存为 CSV 文件
edge_new_df.to_csv(path + 'edge_new.csv', index=False)