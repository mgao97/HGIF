import torch

from nets import *
from data_utils import *
from torch.nn import Sequential




class Model(nn.Module):
    def __init__(self, args,dataset, n, c, d, device):
        super(Model, self).__init__()

        self.gnn = BWGNN(in_feats=d,h_feats=args.emb_dim,d=args.C)

        self.device = device
        self.args = args
        self.linear = nn.Linear(args.emb_dim*2, c)
        self.g_list=graph_aug(self.args, dataset)
        self.proj_head1 = Sequential(nn.Linear(args.emb_dim, args.emb_dim))
        self.proj_head2 = Sequential(nn.Linear(args.emb_dim, args.emb_dim))


    def forward(self, data,batch_nodes,weight):
        y =  data.ndata['label'].to(self.device)
        Loss, Log_p = [], 0
        embedding_list=[]
        for k in range(len(self.g_list)):
            embedding_lf,embedding_hf = self.gnn(self.g_list[k].ndata['feature'],self.g_list[k])

            embedding=torch.cat((embedding_lf,embedding_hf),dim=1)

            out = self.linear(embedding)
            loss = self.sup_loss(y[batch_nodes], out[batch_nodes],weight)
            Loss.append(loss.view(-1))
            embedding_list.append((embedding_lf,embedding_hf))
        Loss = torch.cat(Loss, dim=0)
        Var, Mean = torch.var_mean(Loss)

        embedding_lf_0,embedding_hf_0  = self.proj_head1(embedding_list[0][0]), self.proj_head2(embedding_list[0][1])
        embedding_lf_1,embedding_hf_1  = self.proj_head1(embedding_list[2][0]), self.proj_head2(embedding_list[2][1])

        negatives_mask = ~torch.eye(len(batch_nodes) * 2, len(batch_nodes) * 2, dtype=bool).to(self.device)


        negatives_loss0=  Negatives_Loss(embedding_hf_0[batch_nodes], embedding_lf_0[batch_nodes], len(batch_nodes),self.args.temperature)
        negatives_loss1 = Negatives_Loss(embedding_hf_1[batch_nodes], embedding_lf_1[batch_nodes], len(batch_nodes),self.args.temperature)
        negatives_loss2 = Negatives_Loss(embedding_hf_0[batch_nodes], embedding_lf_1[batch_nodes], len(batch_nodes),self.args.temperature)
        negatives_loss3 = Negatives_Loss(embedding_lf_0[batch_nodes], embedding_hf_1[batch_nodes], len(batch_nodes),self.args.temperature)

        negatives_loss=negatives_loss0+negatives_loss1+negatives_loss2+negatives_loss3

        contrast_loss0 = ContrastiveLoss(embedding_lf_0[batch_nodes], embedding_lf_1[batch_nodes],len(batch_nodes),negatives_loss,self.args.temperature, negatives_mask)
        contrast_loss1 = ContrastiveLoss(embedding_hf_0[batch_nodes], embedding_hf_1[batch_nodes],len(batch_nodes),negatives_loss,self.args.temperature, negatives_mask)
        contrast_loss=(contrast_loss0+contrast_loss1)/2

        return Var, Mean,contrast_loss

    def inference(self, data):
        graph = dgl.add_self_loop(data)
        embedding_lf,embedding_hf = self.gnn(graph.ndata['feature'],graph)
        embedding = torch.cat((embedding_lf, embedding_hf), dim=1)
        out = self.linear(embedding)
        return out

    def test(self,g_list,batch_nodes):
        embedding_list = []
        for k in range(len(g_list)):
            embedding_lf, embedding_hf = self.gnn(g_list[k].ndata['feature'],g_list[k])
            embedding_list.append((embedding_lf,embedding_hf))

        embedding_lf_0, embedding_hf_0 = embedding_list[0][0], embedding_list[0][1]
        embedding_lf_1, embedding_hf_1 = embedding_list[1][0], embedding_list[1][1]

        negatives_mask = ~torch.eye(len(batch_nodes) * 2, len(batch_nodes) * 2, dtype=bool).to(self.device)
        negatives_loss0 = Negatives_Loss(embedding_hf_0[batch_nodes], embedding_lf_0[batch_nodes], len(batch_nodes),self.args.temperature)
        negatives_loss1 = Negatives_Loss(embedding_hf_1[batch_nodes], embedding_lf_1[batch_nodes], len(batch_nodes),self.args.temperature)
        negatives_loss2 = Negatives_Loss(embedding_hf_0[batch_nodes], embedding_lf_1[batch_nodes], len(batch_nodes),self.args.temperature)
        negatives_loss3 = Negatives_Loss(embedding_lf_0[batch_nodes], embedding_hf_1[batch_nodes], len(batch_nodes),self.args.temperature)
        negatives_loss = negatives_loss0 + negatives_loss1 + negatives_loss2 + negatives_loss3


        contrast_loss0 = ContrastiveLoss(embedding_lf_0[batch_nodes], embedding_lf_1[batch_nodes], len(batch_nodes), negatives_loss, self.args.temperature, negatives_mask)
        contrast_loss1 = ContrastiveLoss(embedding_hf_0[batch_nodes], embedding_hf_1[batch_nodes], len(batch_nodes), negatives_loss, self.args.temperature, negatives_mask)
        contrast_loss = (contrast_loss0 + contrast_loss1) / 2



        return contrast_loss


    def sup_loss(self, y, pred, weight):
        loss = F.cross_entropy(pred.to(self.device), y.to(self.device), weight=torch.tensor([1., weight]).to(self.device))
        return loss

