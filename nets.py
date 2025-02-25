import dgl
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import dgl.function as fn
from copy import deepcopy
import scipy
from torch.nn import init
import sympy
b_xent = nn.BCEWithLogitsLoss()

device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

class PolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin
        # self.reset_parameters()
        # self.linear2 = nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h

def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas

class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, d=2):
        super(BWGNN, self).__init__()

        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(h_feats, h_feats, self.thetas[i], lin=False))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*(int(len(self.conv)/2)), h_feats)
        self.linear4 = nn.Linear(h_feats*(len(self.conv)-int(len(self.conv)/2)), h_feats)
        self.act = nn.ReLU()
        self.d = d

    def forward(self, in_feat,graph):
        self.g = graph
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final_l = torch.zeros([len(in_feat), 0])
        h_final_h = torch.zeros([len(in_feat), 0])
        for i in range(int(len(self.conv)/2)):
            h0 = self.conv[i](self.g, h)
            h_final_l = torch.cat([h_final_l, h0], -1)

        for i in range(int(len(self.conv) / 2),len(self.conv)):
            h0 = self.conv[i](self.g, h)
            h_final_h = torch.cat([h_final_h, h0], -1)

            # print(h_final.shape)
        h_l = self.linear3(h_final_l)
        h_l = self.act(h_l)

        h_h = self.linear4(h_final_h)
        h_h = self.act(h_h)

        return h_l,h_h













def ContrastiveLoss(z_i, z_j, batch_size, negatives_loss,temperature, negatives_mask):
    representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                            dim=2)  # simi_mat: (2*bs, 2*bs)

    sim_ij = torch.diag(similarity_matrix, batch_size)  # bs
    sim_ji = torch.diag(similarity_matrix, -batch_size)  # bs
    positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

    nominator = torch.exp(positives / temperature).to(device)  # 2*bs
    denominator = negatives_mask * torch.exp(similarity_matrix / temperature).to(device)  # 2*bs, 2*bs

    loss_partial = -torch.log(nominator / (negatives_loss+torch.sum(denominator, dim=1)))  # 2*bs
    loss = torch.sum(loss_partial) / (2 * batch_size)
    return loss


def Negatives_Loss(z_i, z_j, batch_size, temperature):
    representations = torch.cat([z_i, z_j], dim=0)

    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                            dim=2)  # simi_mat: (2*bs, 2*bs)
    sim_ij = torch.diag(similarity_matrix, batch_size)  # bs
    sim_ji = torch.diag(similarity_matrix, -batch_size)  # bs
    negatives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

    nominator = torch.exp(negatives / temperature).to(device)

    return nominator






