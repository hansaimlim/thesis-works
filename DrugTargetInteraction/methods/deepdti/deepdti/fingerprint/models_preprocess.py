from __future__ import print_function
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GraphDegreeConv(nn.Module):

    def __init__(self, node_size, edge_size, output_size, degree_list,
            ntype, etype, batch_normalize=True):
        super(GraphDegreeConv, self).__init__()
        self.ntype = ntype
        self.etype = etype
        self.node_size = node_size
        self.edge_size = edge_size
        self.output_size = output_size

        self.batch_normalize = batch_normalize
        if self.batch_normalize:
            self.normalize = nn.BatchNorm1d(output_size, affine=False)

        self.bias = nn.Parameter(torch.zeros(1, output_size))
        self.linear = nn.Linear(node_size, output_size, bias=False)
        self.degree_list = degree_list
        self.degree_layer_list = nn.ModuleList()
        for degree in degree_list:
            self.degree_layer_list.append(nn.Linear(node_size + edge_size, output_size, bias=False))

    def forward(self, atom_repr, bond_repr, batch_atom_degree_idxs, batch_bond_degree_idxs):
        logging.debug("Convolutional layer: {0}".format(self.linear))

        degree_activation_list = []
        for d_idx, degree_layer in enumerate(self.degree_layer_list):
            degree = self.degree_list[d_idx]
            atom_neighbor_list = batch_atom_degree_idxs[degree]
            bond_neighbor_list = batch_bond_degree_idxs[degree]
            if degree == 0 and atom_neighbor_list:
                zero = Variable(torch.zeros(len(atom_neighbor_list), self.output_size))
                if torch.cuda.is_available():
                    zero = zero.cuda()
                degree_activation_list.append(zero)
            else:
                if atom_neighbor_list:
                    # (#nodes, #degree+1, node_size)
                    atom_neighbor_repr = atom_repr[atom_neighbor_list, ...]
                    atom_neighbor_repr = torch.sum(atom_neighbor_repr,dim=1,keepdim=False)
                    # (#nodes, #degree, edge_size)
                    bond_neighbor_repr = bond_repr[bond_neighbor_list, ...] # need to be pooled by degree
                    bond_neighbor_repr = torch.sum(bond_neighbor_repr,dim=1,keepdim=False)
                    # (#nodes, node_size + edge_size)
                    stacked = torch.cat([atom_neighbor_repr, bond_neighbor_repr], dim=1)
                    #summed = torch.sum(stacked, dim=1, keepdim=False)
                    # (#nodes, output_size)
                    degree_activation = degree_layer(stacked)
                    degree_activation_list.append(degree_activation)

        neighbor_repr = torch.cat(degree_activation_list, dim=0)
        self_repr = self.linear(atom_repr)
        # size = (#nodes, #output_size)

        activations = self_repr + neighbor_repr + self.bias.expand_as(self_repr)
        if self.batch_normalize:
            activations = self.normalize(activations)
        return F.relu(activations)

class NeuralFingerprint(nn.Module):

    def __init__(self, node_size, edge_size, conv_layer_sizes, output_size, type_map,
            degree_list, molecule_batch_dict, batch_normalize=True):
        """
        Args:
            node_size (int): dimension of node representations
            edge_size (int): dimension of edge representations
            conv_layer_sizes (list of int): the lengths of the output vectors
                of convolutional layers
            output_size (int): length of the finger print vector
            type_map (dict string:string): type of the batch nodes, vertex nodes,
                and edge nodes
            degree_list (list of int): a list of degrees for different
                convolutional parameters
            molecule_batch_dict[ikey]={'smiles':str(smi),
                'neighbor_by_degree':neighbor_by_degree,
                'atoms':atoms,'bonds':bonds}
            batch_normalize (bool): enable batch normalization (default True)
        """
        super(NeuralFingerprint, self).__init__()
        self.num_layers = len(conv_layer_sizes)
        self.output_size = output_size
        self.batch_type = type_map['batch'] # molecule
        self.ntype = type_map['node'] # atom
        self.etype = type_map['edge'] # bond
        self.degree_list = degree_list
        self.molecule_batch_dict=molecule_batch_dict

        self.conv_layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        layers_sizes = [node_size] + conv_layer_sizes
        for input_size in layers_sizes:
            self.out_layers.append(nn.Linear(input_size, output_size))
        for prev_size, next_size in zip(layers_sizes[:-1], layers_sizes[1:]):
            self.conv_layers.append(
                GraphDegreeConv(prev_size, edge_size, next_size, degree_list,
                                self.ntype, self.etype, batch_normalize=batch_normalize))

    def forward(self, batch_ikeys):
        """
        Args: 
            batch_ikeys: list of inchikeys for minibatch
        Returns:
            fingerprint: A tensor variable with shape (batch_size, output_size)
        """
        molecule_batch_dict=self.molecule_batch_dict
        batch_size = len(batch_ikeys) # num_molecules in a batch
        fingerprint_batch = Variable(torch.zeros(batch_size, self.output_size))
        if torch.cuda.is_available(): 
            fingerprint_batch = fingerprint_batch.cuda()
        atom_repr = [] # num_atoms in a batch
        bond_repr = [] # num_bomds in a batch
        atom_batch = {m:[] for m in range(len(batch_ikeys))} # dict of atom-molecule indicator, which molecule an atom belongs to
        bond_batch = {m:[] for m in range(len(batch_ikeys))} # dict of bond-molecule indicator, which molecule a bond belongs to
        batch_atom_degree_idxs = {d: [] for d in self.degree_list} #index indicator for batch repr, what are indices for each degree
        batch_bond_degree_idxs = {d: [] for d in self.degree_list} #index indicator for batch repr, what are indices for each degree
        batch_atom_idx=0
        batch_bond_idx=0
        for idx,ikey in enumerate(batch_ikeys):
            moldict=molecule_batch_dict[ikey]
            atoms=moldict['atoms']
            bonds=moldict['bonds']
            neighbor_by_degree=moldict['neighbor_by_degree']
            for d in self.degree_list:
                dn=neighbor_by_degree[d]
                atom_i_ = torch.LongTensor([batch_atom_idx+ai for ai in torch.LongTensor(dn['atom']).flatten()])
                bond_i_ = torch.LongTensor([batch_bond_idx+bi for bi in torch.LongTensor(dn['bond']).flatten()])
                batch_atom_degree_idxs[d].expand(atom_i_)
                batch_bond_degree_idxs[d].expand(bond_i_)
            for aid in range(len(atoms)):
                atom_repr.append(atoms[aid])
                atom_batch[idx].append(batch_atom_idx) #molecule idx -> list of atom idx in the batch
                batch_atom_idx+=1
            for bid in range(len(bonds)):
                bond_repr.append(bonds[bid])
                bond_batch[idx].append(batch_bond_idx)
                batch_bond_idx+=1
            
        atom_repr = Variable(torch.FloatTensor(atom_repr))
        bond_repr = Variable(torch.FloatTensor(bond_repr))
        if torch.cuda.is_available():
            atom_repr = atom_repr.cuda()
            bond_repr = bond_repr.cuda()
        logging.debug("Bond representation: {0}:{1}, Layer: {2}".format(bond_repr.size(), type(bond_repr.data), linear))
        def fingerprint_update(linear, node_repr,atom_batch):
            logging.debug("Updating fingerprint...")
            logging.debug("Atom representation: {0}:{1}, Layer: {2}".format(atom_repr.size(), type(atom_repr.data), linear))
            atom_activations = F.softmax(linear(node_repr),dim=1)
            logging.debug("atom size: {0}".format(atom_activations.size()))
            update = torch.cat([torch.sum(atom_activations[atom_idx, ...], dim=0, keepdim=True) for atom_idx in atom_batch], dim=0)
            return update

        for layer_idx in xrange(self.num_layers):
        # (#nodes, #output_size)
            logging.debug("Degree convolution: layer:{}, {}".format(layer_idx, self.out_layers[layer_idx]))
            fingerprint += fingerprint_update(self.out_layers[layer_idx], atom_repr, atom_batch)
            logging.debug("Fingerprint updated. layer:{}".format(layer_idx))
            atom_repr = self.conv_layers[layer_idx](atom_repr, bond_repr, batch_atom_degree_idxs, batch_bond_degree_idxs)
            logging.debug("Atom representation updated. layer:{}".format(layer_idx))
        fingerprint += fingerprint_update(self.out_layers[-1], atom_repr, atom_batch)
        logging.debug("Fingerprint updated. last layer.")
        fingerprint_batch = fingerprint_batch.unsqueeze(1)
        logging.debug("Fingerprint shape: {}".format(fingerprint_batch.size()))
        return fingerprint_batch