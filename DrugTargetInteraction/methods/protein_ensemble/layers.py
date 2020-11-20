import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
#from torch_geometric.nn.conv import MessagePassing
from message_passing import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add

class CPANConv(MessagePassing):
    
    def __init__(self, config):

        super(CPANConv, self).__init__('add')

        self.in_channels = config.nhid1
        self.out_channels = config.nhid2
        self.heads = config.heads
        self.negative_slope = config.alpha
        self.dropout = config.dropout
        self.mod = config.mod

        self.weight = Parameter(
            torch.Tensor(config.nhid1, config.heads * config.nhid2))
        self.att = Parameter(torch.Tensor(1, config.heads, 2 * config.nhid2))
        self.bias = Parameter(torch.Tensor(config.heads * config.nhid2))
        
        '''
        self.ln1 = Sequential(Linear(config.nhid2, config.nhid2), 
                              Dropout(config.dropout), ReLU())
        self.bn1 = torch.nn.BatchNorm1d(config.nhid2)
        self.ln2 = Sequential(Linear(config.nhid2, config.nhid2), 
                              Dropout(config.dropout), ReLU())
        self.bn2 = torch.nn.BatchNorm1d(config.nhid2)
        '''
        self.fc_x = Linear(config.nhid1, config.heads * config.nhid2, bias=True)
        
        self.mlp = Sequential(Linear(config.nhid2, config.nhid2), 
                              Dropout(config.dropout), ReLU(),
                              BatchNorm1d(config.nhid2),
                              Linear(config.nhid2, config.nhid2), 
                              Dropout(config.dropout), ReLU(),
                              BatchNorm1d(config.nhid2))
        
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        #print('edge_index', edge_index)
        
        #print('x.size', x.size())
        #print('weight.size', self.weight.size())
        
        #x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        x = self.fc_x(x).view(-1, self.heads, self.out_channels)
        #print('x.size', x.size())
        
        output = self.propagate(edge_index, x=x, num_nodes=x.size(0))
        
        output = self.mlp(output)
        
        return output

    def message(self, x_i, x_j, edge_index, num_nodes):
        # Compute attention coefficients.
        
        #print('x_i.size', x_i.size())
        #print('att.size', self.att.size())
        
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1) 
              
        #print('alpha.size', alpha.size())
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], num_nodes)
        
        '''
        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
        '''
        
        if self.mod == "origin":
            alpha = alpha
        elif self.mod == "additive":
            alpha = torch.where(alpha > 0, alpha+1, alpha)
        elif self.mod == "scaled":
            ones = alpha.new_ones(edge_index[0].size())
            add_degree = scatter_add(ones, edge_index[0], dim_size=num_nodes) 
            degree = add_degree[edge_index[0]].unsqueeze(-1)
            alpha = alpha * degree
        else:
            print('Wrong mod! Use Origin')
            alpha = alpha
        
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.out_channels)

        #if self.bias is not None:
            #aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
