import torch.nn as nn
import torch.nn.functional as F
from GCNConv import GraphConvolution
import torch
from utils import spdensemm


class Net(nn.Module):
    def __init__(self, args, dim_in, dim_out, N, device):
        super(Net,self).__init__()
        self.args = args

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.device = device
        
        self.setup_layers(args)

        self.relu = nn.ReLU(inplace=True)

        self.dropout_rate = args.dropout


    
    def setup_layers(self, args):
        """
        Creating the layes based on the args.
        """
        self.args.layers = [args.hidden_dim for i in range(args.layer_num//2 - 1)]
        self.args.layers = [self.dim_in] + self.args.layers + [self.dim_out]
        # topology layer
        self.layers_tp = nn.ModuleList()
        self.pai_1 = nn.ParameterList()
        self.pai_2 = nn.ParameterList()
        self.weight_1 = nn.ModuleList()
        self.weight_2 = nn.ModuleList()
        for i, _ in enumerate(self.args.layers[:-1]):
            self.layers_tp.append(GraphConvolution(self.args.layers[i], self.args.layers[i+1]))
        for i, _ in enumerate(self.args.layers[:-1]):
            self.pai_1.append(nn.Parameter(torch.randn(self.args.layers[i+1], self.args.layers[i+1])))
            self.pai_2.append(nn.Parameter(torch.randn(self.dim_in, self.args.layers[i+1])))
            self.weight_1.append(nn.Linear(self.args.layers[i+1], self.dim_out, bias=True))
            self.weight_2.append(nn.Linear(self.args.layers[i+1], self.dim_out, bias=True))
        self.theta1 = torch.tensor(args.theta1, dtype=torch.float32)
        self.theta2 = torch.tensor(args.theta2, dtype=torch.float32)
        self.lamda = torch.tensor(args.lamda, dtype=torch.float32)


    def activation(self, x):
        w1 = (2 * self.theta2 - self.theta1) / self.theta2
        w2 = w1 - 1
        new_x = w1 * (F.relu(x - self.theta1) - F.relu(- x - self.theta1)) - w2 * (F.relu(x - self.theta2) - F.relu(- x - self.theta2))
        return new_x

    def forward(self, data):
        A = data.adj

        Z_tp_list = list()
        Z_embed_list = list()

        L = data.L

        Z_tp = data.x
        X = data.x
        for i in range(len(self.layers_tp)):
            Z_tp = self.relu(self.layers_tp[i](Z_tp, A))
            Z_tp = F.dropout(Z_tp, p=self.dropout_rate, training=self.training)
            Z_tp_list.append(torch.tanh(self.weight_1[i](Z_tp)))

            Z_embed = Z_tp
            Z_embed = Z_embed.matmul(self.pai_1[i]) + X.matmul(self.pai_2[i]) - self.lamda * spdensemm(L, Z_embed)
            Z_embed = self.activation(Z_embed)
            Z_embed = F.dropout(Z_embed, p=self.dropout_rate, training=self.training)
            Z_embed_list.append(torch.tanh(self.weight_2[i](Z_embed)))
            Z_tp = Z_embed

        

        return Z_tp_list, Z_embed_list

