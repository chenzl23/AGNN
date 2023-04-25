from texttable import Texttable
import torch
from torch_sparse import SparseTensor, spmm
import numpy as np

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def to_normalized_sparsetensor(edge_index, N, mode='DAD'):
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5) 
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    if mode == 'DA':
        return deg_inv_sqrt.view(-1,1) * deg_inv_sqrt.view(-1,1) * adj
    if mode == 'DAD':
        return deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def spdensemm(sp_mx, mx):
    row, col, value = sp_mx.coo()

    index = torch.stack([row, col], dim=0)
    out = spmm(index, value, sp_mx.size(0), sp_mx.size(1), mx)

    return out

def get_sparse_L(adj):
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    row, col, value = adj.coo()
    value = -value
    adj = adj.set_diag()
    value[row == col] += deg
    adj.set_value(value)

    return adj

def to_sp(adj):
    N = adj.shape[0]
    edge_index = np.argwhere(adj > 0)
    row, col = edge_index
    sp_adj = SparseTensor(row=row, col=col, value=adj.to_sparse().values(), sparse_sizes=(N, N))
    return sp_adj