import torch, os
from torch_geometric.datasets import Planetoid
import torch
import scipy.sparse as sp
import numpy as np
import scipy.io
from torch_geometric.data import Data
from utils import get_sparse_L, to_normalized_sparsetensor, to_sp

"""
Here we don't do any normalization for input node features
    graph.adj: SparseTensor (use matmul) 
"""
def load_data(args):
    data_name = args.dataset_name
    if data_name in ['citeseer']:
        return load_ptg_data(args)
    else:
        return load_classic_data(args)

'''
Load data from torch_geometric api
'''
def load_ptg_data(args):
    # pytorch geometric data
    DATA_ROOT = 'dataset'
    if not os.path.exists(DATA_ROOT):
        os.mkdir(DATA_ROOT)
    dataset = Planetoid(os.path.join(DATA_ROOT, args.dataset_name), args.dataset_name, num_train_per_class=args.num_train_per_class, num_val=args.num_val, num_test=args.num_test)
    graph = dataset[0]

    # masks
    graph.valid_mask = graph.val_mask
    N = graph.num_nodes

    edges = graph.edge_index
    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0, :], edges[1, :])),
                        shape=(N, N),
                        dtype=np.float32).toarray()

    adj = torch.from_numpy(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_ori = adj
    adj = adj + torch.eye(adj.shape[0], adj.shape[0])
    edge_index = np.argwhere(adj > 0)

    adj = to_normalized_sparsetensor(edge_index, adj.shape[0])
    graph.adj = adj

    L = get_sparse_L(to_sp(adj_ori))
    graph.L = L 

    if args.feature_normalize == 1:
        print("Feature Normalized.")
        graph.x = normalize(graph.x)
    
    return graph


def load_classic_data(args):
    data_path = os.path.join("./dataset", args.dataset_name)

    features, adj, edge_index, gnd, L = loadMatData(data_path, args.feature_normalize)

    train_mask, valid_mask, test_mask = generate_permutation(gnd.numpy(), args)

    data = Data(x=features, edge_index = edge_index, y=gnd, train_mask = train_mask, valid_mask = valid_mask, test_mask = test_mask, adj = adj, L = L)

    return data

def count_each_class_num(gnd):
    '''
    Count the number of samples in each class
    '''
    count_dict = {}
    for label in gnd:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict

def generate_permutation(gnd, args):
    '''
    Generate permutation for training, validating and testing data.
    '''
    N = gnd.shape[0]
    each_class_num = count_each_class_num(gnd)
    training_each_class_num = {} ## number of labeled samples for each class

    for label in each_class_num.keys():
        training_each_class_num[label] = args.num_train_per_class
        valid_num = args.num_val
        test_num = args.num_test

    # index of labeled and unlabeled samples
    train_mask = torch.from_numpy(np.full((N), False))
    valid_mask = torch.from_numpy(np.full((N), False))
    test_mask = torch.from_numpy(np.full((N), False))

    # shuffle the data
    data_idx = np.random.permutation(range(N))

    # Get training data
    for idx in data_idx:
        label = gnd[idx]
        if (training_each_class_num[label] > 0):
            training_each_class_num[label] -= 1
            train_mask[idx] = True
    for idx in data_idx:
        if train_mask[idx] == True:
            continue
        if (valid_num > 0):
            valid_num  -= 1
            valid_mask[idx] = True
        elif (test_num > 0):
            test_num -= 1
            test_mask[idx] = True
    return train_mask, valid_mask, test_mask


def loadMatData(data_path, feature_normalize):
    '''
    return features: Tensor
    edges: Sparse Tensor
    edge_weights: Sparse Tensor
    gnd: Tensor
    '''
    data = scipy.io.loadmat(data_path) 
    features = data['X'] #.dtype = 'float32'
    features = torch.from_numpy(features).float()

    gnd = data['Y']
    gnd = gnd.flatten()
    if np.min(gnd) == 1:
        gnd = gnd - 1
    gnd = torch.from_numpy(gnd)


    
    adj = data['adj']
    adj = torch.from_numpy(adj)
    # build symmetric adjacency matrix
    adj = adj + adj.t().multiply(adj.t() > adj) - adj.multiply(adj.t() > adj)
    adj_ori = adj
    adj = adj + torch.eye(adj.shape[0], adj.shape[0])
    edge_index = np.argwhere(adj > 0)
    adj = to_normalized_sparsetensor(edge_index, adj.shape[0])
    
    L = get_sparse_L(to_sp(adj_ori))

    if feature_normalize == 1:
        print("Feature Normalized.")
        features = normalize(features)

    return features, adj, edge_index, gnd, L


    

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    if isinstance(mx, np.ndarray):
        return torch.from_numpy(mx)
    else:
        return mx
