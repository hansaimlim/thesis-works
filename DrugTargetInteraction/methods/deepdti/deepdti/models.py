from __future__ import print_function
import logging
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils 

class PSSMEmbedding(nn.Module):

    def __init__(self, hidden_size):
        super(PSSMEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(1,5), stride=1, padding=(0,2))
        self.conv1_down = nn.Conv2d(4, 4, kernel_size=(1,2), stride=(1,2), padding=0)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(1,5), stride=1, padding=(0,2))
        self.conv2_down = nn.Conv2d(8, 8, kernel_size=(1,2), stride=(1,2), padding=0)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(1,5), stride=1, padding=(0,2))
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=(5,5), stride=1, padding=(2,2))
        self.conv4_down = nn.Conv2d(32, 32, kernel_size=(4,1), stride=(4,1), padding=0) #(-1,32,175,5)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=(5,5), stride=1, padding=(2,2))
        self.conv5_down = nn.Conv2d(32, 32, kernel_size=(5,1), stride=(5,1), padding=0) #(-1,32,35,5)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=(5,5), stride=1, padding=(2,2))
        self.bn6 = nn.BatchNorm2d(32)
    def forward(self, x):
        x = x.view(-1, 1, 700, 20)
        x = F.relu(self.bn1(self.conv1_down(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2_down(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4_down(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5_down(self.conv5(x))))
        x = F.relu(self.bn6(self.conv6(x)))
        x = x.view(-1, x.shape[1],x.shape[2]*x.shape[3]) #(-1, 32, 175)
        return x

class PSSMEmbedding2(nn.Module):

    def __init__(self, hidden_size):
        super(PSSMEmbedding2, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3,5), stride=1, padding=(1,2))
        self.conv1_down = nn.Conv2d(4, 4, kernel_size=(1,2), stride=(1,2), padding=0)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(3,5), stride=1, padding=(1,2))
        self.conv2_down = nn.Conv2d(8, 8, kernel_size=(1,2), stride=(1,2), padding=0)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(3,5), stride=1, padding=(1,2))
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=(5,5), stride=1, padding=(2,2))
        self.conv4_down = nn.Conv2d(32, 32, kernel_size=(4,1), stride=(4,1), padding=0) #(-1,32,175,5)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=(5,5), stride=1, padding=(2,2))
        self.conv5_down = nn.Conv2d(32, 32, kernel_size=(5,1), stride=(5,1), padding=0) #(-1,32,35,5)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=(5,5), stride=1, padding=(2,2))
        self.bn6 = nn.BatchNorm2d(32)
    def forward(self, x):
        x = x.view(-1, 1, 700, 20)
        x = F.relu(self.bn1(self.conv1_down(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2_down(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4_down(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5_down(self.conv5(x))))
        x = F.relu(self.bn6(self.conv6(x)))
        x = x.view(-1, x.shape[1],x.shape[2]*x.shape[3]) #(-1, 32, 175)
        return x

class EmbeddingTransform(nn.Module):

    def __init__(self, input_size, hidden_size, out_size,
                 dropout_p=0.2):
        super(EmbeddingTransform, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, embedding):
        logging.debug("Non linear transformation...")
        embedding = self.dropout(embedding)
        hidden = self.transform(embedding)
        logging.debug("Done")
        return hidden

class DeepDTI(nn.Module):
    def __init__(self, hidden_size, input_dropout_p):
        super(DeepDTI, self).__init__()
        self.hidden_size = hidden_size
        self.pad = dict()
        from fingerprint.models import NeuralFingerprint
        import fingerprint.features as fp_feature
        type_map = dict(batch='molecule', node='atom', edge='bond')
        conv_layer_sizes = [20, 20, 20, 20]
        output_size = hidden_size
        degrees = [0, 1, 2, 3, 4, 5]
        fp_model = NeuralFingerprint(
            fp_feature.num_atom_features(),
            fp_feature.num_bond_features(),
            conv_layer_sizes,
            output_size,
            type_map,
            degrees)
        self.add_module('chemical', fp_model)
        self.add_module('protein', PSSMEmbedding(hidden_size))

        for param in self.parameters():
            param.data.uniform_(-0.08, 0.08)

    def forward(self, type_, batch_input, **kwargs):
        batch_embedding = getattr(self, type_)(batch_input, **kwargs)
        return batch_embedding

class AttentivePooling(nn.Module):
    """ Attentive pooling network according to https://arxiv.org/pdf/1602.03609.pdf """
    def __init__(self, hidden_size):
        super(AttentivePooling, self).__init__()
        self.hidden_size = hidden_size
        self.param = nn.Parameter(torch.zeros(hidden_size, 175)) #AP params for chem-PSSM

    def forward(self, first, second):
        """ Calculate attentive pooling attention weighted representation and
        attention scores for the two inputs.

        Args:
            first: output from one source with size (batch_size, length_1, hidden_size)
            second: outputs from other sources with size (batch_size, length_2, hidden_size)

        Returns:
            (rep_1, attn_1): attention weighted representations and attention scores
            for the first input
            (rep_2, attn_2): attention weighted representations and attention scores
            for the second input
        """
        logging.debug("AttentivePooling params: {0}, {1}".format(first.size(), second.size()))
        
        param = self.param.expand(first.size(0), self.hidden_size, 175)
        score_m = torch.bmm(first, param)
        score_m = torch.tanh(torch.bmm(score_m, second.transpose(1, 2)))

        attn_first = torch.max(score_m, 2, keepdim=False)[0]
        attn_second = torch.max(score_m, 1, keepdim=False)[0]

        w_first = F.softmax(attn_first,dim=1)
        w_second = F.softmax(attn_second,dim=1)

        logging.debug("AttentivePooling weights: {0}, {1}".format(w_first.size(), w_second.size()))

        rep_first = torch.bmm(w_first.unsqueeze(1), first).squeeze(1)
        rep_second = torch.bmm(w_second.unsqueeze(1), second).squeeze(1)

        return ((rep_first, w_first), (rep_second, w_second))

class Predict(nn.Module):
    """ Prepare a similarity prediction model for each distinct pair of
    entity types.
    """
    def __init__(self, hidden_size, dropout_p):
        super(Predict, self).__init__()
        self.hidden_size = hidden_size

        self.add_module('chemical', EmbeddingTransform(hidden_size, hidden_size, hidden_size, dropout_p=dropout_p))
        self.add_module('protein', EmbeddingTransform(175, hidden_size, hidden_size, dropout_p=dropout_p))
        self.add_module(" ".join(('chemical', 'protein')), AttentivePooling(hidden_size))

        for param in self.parameters():
            param.data.uniform_(-0.08, 0.08)

    def forward(self, first, second):
        """ Calculate the 'similarity' between two inputs, where the first input
        is a matrix and the second batched matrices.

        Args:
            first: output from one source with size (length_1, hidden_size)
            second: outputs from other sources with size (batch_size, length_2, hidden_size)

        Returns:
            prob: a `batch_size` vector that contains the probabilities that each
            entity in the second input has association with the first input
        """
        first, second = sorted((first, second), key=lambda x: x['type'])
        attn_model = getattr(self, " ".join((first['type'], second['type'])))
        (rep_first, w_first), (rep_second, w_second) = attn_model(first['embedding'], second['embedding'])

        rep_first = getattr(self, first['type'])(rep_first).unsqueeze(1)
        rep_second = getattr(self, second['type'])(rep_second).unsqueeze(2)
        logging.debug("Transformed representation vectors: {0}, {1}".format(rep_first.size(), rep_second.size()))

        return torch.bmm(rep_first, rep_second).squeeze(), (w_first, w_second)
