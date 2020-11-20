from __future__ import print_function
import sys
import logging
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import numpy as np
from models import DeepDTI, Predict
from utils import get_embedding

class Predictor():
    def __init__(self, model_path, pred_model_path, uniprot2pssm, ikey2smiles, protein_weight):
        hidden_size = 64
        self.model = DeepDTI(hidden_size, input_dropout_p=0.0)
        self.pred_model = Predict(hidden_size, dropout_p=0.0, protein_weight=protein_weight)
        if os.path.exists(model_path) and os.path.exists(pred_model_path):
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.pred_model.load_state_dict(torch.load(pred_model_path))
            except:
                print("Model parameters do not match with provided model.")
                raise
        else:
            raise ValueError("Model not existed")
        self.model.train(False)
        self.pred_model.train(False)
        self.uniprot2pssm=uniprot2pssm
        self.ikey2smiles=ikey2smiles
        if torch.cuda.is_available():
            self.model.cuda()
            self.pred_model.cuda()

    def predict(self, chem_node, prot_node):
        chem_embed = get_embedding(self.model,
                  'chemical', [self.ikey2smiles[chem_node]], volatile=True)
        prot_embed = get_embedding(self.model,
                  'protein', [self.uniprot2pssm[prot_node]], volatile=True)
        prob, (weight_u, weight_v) = self.pred_model(dict(type='chemical', embedding=chem_embed),
                                                 dict(type='protein', embedding=prot_embed))
        explanation = None
        return prob, explanation

if __name__=='__main__':
    None
