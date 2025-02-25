import argparse
import numpy as np
import random
import torch
import dgl
# from logger import Logger
from parse import parser_add_main_args
from model import Model
import os
from data_utils import *
from sklearn.model_selection import train_test_split

from create_synthetic import *

rewrite_print = print
def print(*arg):
    file_path = './log.txt'
    # 打印到控制台
    rewrite_print(*arg)
    # 保存到文件
    rewrite_print(*arg, file=open(file_path, "a", encoding="utf-8"))


def train_pre(args):
    auc_best=0
    for epoch in range(args.train_epochs):
        model.train()
        random.shuffle(all_node)
        num_batches = int(len(all_node) / args.batch_size) + 1
        for batch in range(num_batches):
            optimizer_gnn.zero_grad()
            i_start = batch * args.batch_size
            i_end = min((batch + 1) * args.batch_size, len(all_node))
            batch_nodes = all_node[i_start:i_end]
            weight = (1 - labels[batch_nodes]).sum().item() / labels[batch_nodes].sum().item()
            beta = 1 * args.beta * epoch / args.train_epochs + args.beta * (1 - epoch / args.train_epochs)
            Var, Mean,contrast_loss = model(dataset_tr,batch_nodes, weight)
            outer_loss = Var + beta * Mean+contrast_loss*0.05
            print(' Var + beta * Mean', Var + beta * Mean)
            print(' contrast_loss', contrast_loss)
            optimizer_gnn.zero_grad()
            outer_loss.backward()
            optimizer_gnn.step()

        if epoch % args.display_step == 0:
            with torch.no_grad():
                aucs = evaluate_whole_graph(model,datasets_te)
                if aucs[0]>auc_best:
                    auc_best=aucs[0]
                    print('  Saving model ...')
                    torch.save(model.state_dict(), model_path_saver)
            print(f'Epoch: {epoch:02d}, '
                  f'Mean Loss: {Mean:.4f}, '
                  f'Var Loss: {Var:.4f}, ')
            test_info = ''
            for test_acc in aucs[0:]:
                test_info += f'Test: {100 * test_acc:.2f}% '
            print(test_info)


def test_cond(args):
    for i in range(len(datasets_te)):
        print('i',i)
        print("Model path: {}".format(model_path_saver))
        model.load_state_dict(torch.load(model_path_saver))
        optimizer_contrast = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=args.weight_decay)
        for param in model.linear.parameters():
            param.requires_grad = False
        args.K=2
        g_list = graph_aug1(args, datasets_te[i])
        labels = datasets_te[i].ndata['label']
        sampled_idx_train = [i for i in range(labels.shape[0])]
        for epoch in range(args.test_epochs):
            if epoch % args.display_step == 0:
                with torch.no_grad():
                    aucs,f1 = evaluate_graph(model, datasets_te[i])
                print(f'Epoch: {epoch:02d}, 'f'Train_auc: {100 * aucs:.2f}%, 'f'Train_f1: {100 * f1:.2f}%,')
            random.shuffle(sampled_idx_train)
            num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
            model.train()
            for batch in range(num_batches):
                optimizer_contrast.zero_grad()
                i_start = batch * args.batch_size
                i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
                batch_nodes = sampled_idx_train[i_start:i_end]
                contrast_loss = model.test(g_list,batch_nodes)
                outer_loss = contrast_loss*0.1
                print(' contrast_loss', contrast_loss)
                optimizer_contrast.zero_grad()
                outer_loss.backward()
                optimizer_contrast.step()

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(0)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

def get_dataset(dataset, sub_dataset=None):

    dataset = dgl.load_graphs('../data/{}/{}/gen/my_{}_{}_graph.bin'.format(dataset,dataset,dataset,sub_dataset))

    dataset[0][0].ndata['feature']=torch.tensor(normalize_features(dataset[0][0].ndata['feature'], norm_row=False), dtype=torch.float32)
    return dataset[0][0]





tr_sub, te_subs = [0], list(range(1, 5))
dataset_tr = get_dataset(dataset=args.dataset, sub_dataset=tr_sub[0])
datasets_te = [get_dataset(dataset=args.dataset, sub_dataset=te_subs[i]) for i in range(len(te_subs))]

# logger = Logger(args.runs, args)
print('dataset:', args.dataset)


dir_saver = args.save_dir
model_path_saver = os.path.join(dir_saver, '{}_model.pkl'.format(args.dataset))


labels = dataset_tr.ndata['label']
index = list(range(len(labels)))



all_node=[i for i in range(labels.shape[0])]



model = Model(args, dataset_tr,dataset_tr.num_nodes(), dataset_tr.ndata['label'].max().item() + 1, dataset_tr.ndata['feature'].shape[1], device)
optimizer_gnn = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

train_pre(args)
test_cond(args)





