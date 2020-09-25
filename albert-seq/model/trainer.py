from __future__ import print_function
import sys
import logging
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import numpy as np
from fingerprint.graph import load_from_mol_tuple
from utils import get_lstm_embedding

def load_data(edges_dict,datatype='train',ikey2mol=None):
    #load training pairs with labels
    #edges: list of tuples (inchikey,uniprotID)
    #labels: list of float activity values for each edge
    count=0
    count_skipped=0
    labels=[]
    edges=[]
    chems=[]
    prots=[]
    for cp in edges_dict.keys():
        chem,prot=cp.strip().split('\t')
        if chem not in ikey2mol:
            count_skipped+=1
            continue
        chems.append(chem)
        prots.append(prot)
        count+=1
        labels.append(edges_dict[cp])
        edges.append((chem,prot))
    chems=list(set(chems))
    prots=list(set(prots))
    logging.info("Total {} chemicals, {} proteins, {} activities loaded for {} data. {} chemicals skipped for non-Mol conversion".format(len(chems),
        len(prots),len(labels),datatype,count_skipped))
    return edges, labels

class Trainer():
    def __init__(self, model=None,
                 epoch=100, batch_size=32, ckpt_dir="./temp/",
                 optimizer='adam',l2=1e-3, lr=1e-5, scheduler='cosineannealing',
                 prediction_mode=None,
                 ikey2smiles=None,
                 protein_embedding_type=None,
                 uniprot2triplets=None,
                 uniprot2pssm=None,
                 uniprot2singletrepr=None,
                 ikey2mol=None,
                 berttokenizer=None):
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.train_epoch = epoch
        self.optimizer = optimizer
        self.l2 = l2
        self.lr = lr
        self.scheduler = scheduler
        if self.scheduler.lower()=='cyclic':
            self.optimizer = 'sgd'
            logging.info("CyclicLR scheduler is used. Optimizer is set to {}".format(self.optimizer.upper()))
        self.model = model
        self.prediction_mode = prediction_mode
        if self.prediction_mode is None:
            raise AttributeError("Prediction mode must be specified (binary or continuous)")
        self.prottype = protein_embedding_type
        if self.prottype is None:
            raise AttributeError("Protein embedding type must be specified (PSSM, LSTM, or ALBERT)")
        self.uniprot2triplets = uniprot2triplets
        self.uniprot2pssm = uniprot2pssm
        self.uniprot2singletrepr = uniprot2singletrepr
        self.ikey2smiles = ikey2smiles
        self.ikey2mol = ikey2mol
        self.berttokenizer = berttokenizer
        if self.model is None:
            raise AttributeError("model not provided")
        if self.uniprot2triplets is None:
            raise AttributeError("dict uniprot2triplets not provided")
        if self.ikey2mol is None:
            raise AttributeError("dict ikey2mol not provided")
        if self.berttokenizer is None:
            raise AttributeError("Bert tokenizer not provided")

    def train(self, edges, train_evaluator, dev_edges, dev_evaluator, force_debug=False):
        uniprot2triplets=self.uniprot2triplets
        uniprot2pssm=self.uniprot2pssm
        uniprot2singletrepr=self.uniprot2singletrepr
        ikey2smiles=self.ikey2smiles
        ikey2mol=self.ikey2mol
        berttokenizer=self.berttokenizer 
        def get_repr_from_pairs(pairs):
            chem_repr = [(self.ikey2smiles[pair[0]],self.ikey2mol[pair[0]]) for pair in pairs] #
            if self.prottype.lower() in ['albert','bert','nlp']:
                prot_repr = torch.stack([torch.tensor(
                    berttokenizer.encode(self.uniprot2triplets[pair[1]])) for pair in pairs]) ## tokenize triplets
            elif self.prottype.lower() in ['pssm','cnn']:
                prot_repr = torch.stack([torch.tensor(uniprot2pssm[pair[1]]) for pair in pairs]).unsqueeze(1)
            elif self.prottype.lower() in ['lstm','rnn','ping']:
                prot_repr = get_lstm_embedding([uniprot2singletrepr[pair[1]] for pair in pairs])
            return (chem_repr,prot_repr)

        model=self.model
        parameters = list(self.model.parameters())
        if self.optimizer=='adam':
            optimizer = torch.optim.Adam(parameters,lr=self.lr,weight_decay=self.l2)
            logging.info("Optimizer {}, LR {}, Weight Decay {}".format(self.optimizer, self.lr, self.l2))
        elif self.optimizer=='sgd':
            optimizer = torch.optim.SGD(parameters,lr=self.lr,weight_decay=self.l2)
            logging.info("Optimizer {}, LR {}, Weight Decay {}".format(self.optimizer, self.lr, self.l2))
        if self.scheduler=='cosineannealing':
            tmax=10
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=tmax)
            logging.info("Scheduler {}, T_max {}".format(self.scheduler, tmax))
        elif self.scheduler=='cyclic':
            max_lr=self.lr
            base_lr=self.lr*0.01
            scheduler= torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr)
            logging.info("Scheduler {}, base_lr {:.8f}, max_lr {:.8f} ".format(self.scheduler, base_lr, max_lr))

        stime=time.time()
        train_pairs, train_labels=load_data(edges,datatype='train',ikey2mol=ikey2mol)
        dev_pairs, dev_labels=load_data(dev_edges,datatype='dev',ikey2mol=ikey2mol)
        etime=time.time() - stime
        logging.debug("{:.2f} seconds to load data sets".format(etime))
        best_target_metric=-np.inf
        best_epoch=0
        step = 0
        total_loss = 0
        batch_size = self.batch_size
        if force_debug:
            batch_per_epoch=20
        else:
            batch_per_epoch=int(np.ceil(len(train_labels)/batch_size))
        
        if self.prediction_mode.lower() in ['binary']:
            record_dict = {'epoch':[],
                       'total_loss':[],
                       'train_f1':[],
                       'train_auc':[],
                       'train_aupr':[],
                       'dev_f1':[],
                       'dev_auc':[],
                       'dev_aupr':[],
                      }
            print("Epoch\tData\tF1\tAUC\tAUPR")
        else:
            record_dict = {'epoch':[],
                       'total_loss':[],
                       'train_pearson':[],
                       'train_spearman':[],
                       'dev_pearson':[],
                       'dev_spearman':[]
                      }
            print("Epoch\tData\tPearsonCorr\tSpearmanCorr")
            
        for epoch in range(1, self.train_epoch + 1):
            model.train()
            # Random shuffle all training pairs
            train_data_idxs=list(range(len(train_labels)))
            np.random.shuffle(train_data_idxs)
            epoch_loss_total = 0
            batch_prep_time=0;batch_train_time=0;batch_optim_time=0
            logging.info("Epoch {0} started".format(epoch))
            
            for batch_ in range(batch_per_epoch):

                stime=time.time()
                choices = train_data_idxs[batch_*batch_size:(batch_+1)*batch_size]
                batch_labels = torch.tensor([train_labels[idx] for idx in choices]).cuda()
                batch_train_pairs = [train_pairs[idx] for idx in choices]
                batch_chem_repr,batch_prot_repr = get_repr_from_pairs(batch_train_pairs)
                batch_chem_embed = load_from_mol_tuple(batch_chem_repr)
                if isinstance(batch_chem_embed, Variable) and torch.cuda.is_available():
                    batch_chem_embed = batch_chem_embed.cuda()
                if isinstance(batch_prot_repr, Variable) and torch.cuda.is_available():
                    batch_prot_repr = batch_prot_repr.cuda()
                batch_input = {'protein': batch_prot_repr,'ligand': batch_chem_embed}
                batch_prep_t=time.time() - stime
                batch_prep_time+=batch_prep_t
                stime=time.time()
                batch_logits = model(batch_input)
                batch_train_t=time.time() - stime
                batch_train_time+=batch_train_t
                stime=time.time()
                
                if self.prediction_mode.lower() in ['binary']:
                    loss_fn = torch.nn.CrossEntropyLoss()
                    batch_labels = batch_labels.long()
                else:
                    loss_fn = torch.nn.MSELoss()
                    batch_labels = batch_labels.float().reshape(-1,1)
                loss = loss_fn(batch_logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                batch_optim_t=time.time() - stime
                batch_optim_time+=batch_optim_t
                total_loss+=loss.item()
                epoch_loss_total += loss.item()
            
            logging.info("Epoch {}: Loss {}".format(epoch,loss.item()))
            
            trainmetrics=train_evaluator.eval(model,train_pairs,train_labels,epoch)
            devmetrics=dev_evaluator.eval(model,dev_pairs,dev_labels,epoch)
            record_dict['epoch'].append(epoch)
            record_dict['total_loss'].append(loss.item())
            if self.prediction_mode.lower() in ['binary']:
                record_dict['train_f1'].append(trainmetrics[0])
                record_dict['train_auc'].append(trainmetrics[1])
                record_dict['train_aupr'].append(trainmetrics[2])
                record_dict['dev_f1'].append(devmetrics[0])
                record_dict['dev_auc'].append(devmetrics[1])
                record_dict['dev_aupr'].append(devmetrics[2])
                target_metric = devmetrics[0] #f1 for binary
            else:
                record_dict['train_pearson'].append(trainmetrics[1])
                record_dict['train_spearman'].append(trainmetrics[2])
                record_dict['dev_pearson'].append(devmetrics[1])
                record_dict['dev_spearman'].append(devmetrics[2])
                target_metric = devmetrics[2] #spearman for continuous

            if target_metric > best_target_metric:
                best_target_metric = target_metric #new best spearman correlation
                best_epoch = epoch
                path = os.path.join(self.checkpoint_dir, "epoch_{0}".format(epoch))
                if not os.path.exists(path):
                    os.mkdir(path)
                torch.save(model.state_dict(), os.path.join(path, 'model.dat'))
                logging.info("New best metric {:.6f} at epoch {}".format(target_metric,best_epoch))
        logging.info("Epoch {}: BatchPrepTime {:.1f}, BatchTrainTime {:.1f}, BatchOptimTime {:.1f}".format(epoch,
                    batch_prep_time,batch_train_time,batch_optim_time))
        logging.info("DevMetric {:.6f} at epoch {}. Current best DevMetric {:.6f} at epoch {}".format(
                target_metric,epoch,best_target_metric,best_epoch))
        print("Best DevMetric {:.6f} at epoch {}".format(best_target_metric,best_epoch))
        return record_dict
    
class TrainerForInference():
    '''
    This module does not evaluate performances every epoch
    Instead, it keep trains the model upto the provided epoch
    It is used to train the model before inference, where all available data sets
     are used for training
    '''
    def __init__(self, model=None,
                 epoch=100, batch_size=32, ckpt_dir="./temp/",
                 optimizer='adam',l2=1e-3, lr=1e-5, scheduler='cosineannealing',
                 prediction_mode=None,
                 ikey2smiles=None,
                 protein_embedding_type=None,
                 uniprot2triplets=None,
                 uniprot2pssm=None,
                 uniprot2singletrepr=None,
                 ikey2mol=None,
                 berttokenizer=None):
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.train_epoch = epoch
        self.optimizer = optimizer
        self.l2 = l2
        self.lr = lr
        self.scheduler = scheduler
        if self.scheduler.lower()=='cyclic':
            self.optimizer = 'sgd'
            logging.info("CyclicLR scheduler is used. Optimizer is set to {}".format(self.optimizer.upper()))
        self.model = model
        self.prediction_mode = prediction_mode
        if self.prediction_mode is None:
            raise AttributeError("Prediction mode must be specified (binary or continuous)")
        self.prottype = protein_embedding_type
        if self.prottype is None:
            raise AttributeError("Protein embedding type must be specified (PSSM, LSTM, or ALBERT)")
        self.uniprot2triplets = uniprot2triplets
        self.uniprot2pssm = uniprot2pssm
        self.uniprot2singletrepr = uniprot2singletrepr
        self.ikey2smiles = ikey2smiles
        self.ikey2mol = ikey2mol
        self.berttokenizer = berttokenizer
        if self.model is None:
            raise AttributeError("model not provided")
        if self.uniprot2triplets is None:
            raise AttributeError("dict uniprot2triplets not provided")
        if self.ikey2mol is None:
            raise AttributeError("dict ikey2mol not provided")
        if self.berttokenizer is None:
            raise AttributeError("Bert tokenizer not provided")

    def train(self, edges, train_evaluator, force_debug=False):
        uniprot2triplets=self.uniprot2triplets
        uniprot2pssm=self.uniprot2pssm
        uniprot2singletrepr=self.uniprot2singletrepr
        ikey2smiles=self.ikey2smiles
        ikey2mol=self.ikey2mol
        berttokenizer=self.berttokenizer 
        def get_repr_from_pairs(pairs):
            chem_repr = [(self.ikey2smiles[pair[0]],self.ikey2mol[pair[0]]) for pair in pairs] #
            if self.prottype.lower() in ['albert','bert','nlp']:
                prot_repr = torch.stack([torch.tensor(
                    berttokenizer.encode(self.uniprot2triplets[pair[1]])) for pair in pairs]) ## tokenize triplets
            elif self.prottype.lower() in ['pssm','cnn']:
                prot_repr = torch.stack([torch.tensor(uniprot2pssm[pair[1]]) for pair in pairs]).unsqueeze(1)
            elif self.prottype.lower() in ['lstm','rnn','ping']:
                prot_repr = get_lstm_embedding([uniprot2singletrepr[pair[1]] for pair in pairs])
            return (chem_repr,prot_repr)

        model=self.model
        parameters = list(self.model.parameters())
        if self.optimizer=='adam':
            optimizer = torch.optim.Adam(parameters,lr=self.lr,weight_decay=self.l2)
            logging.info("Optimizer {}, LR {}, Weight Decay {}".format(self.optimizer, self.lr, self.l2))
        elif self.optimizer=='sgd':
            optimizer = torch.optim.SGD(parameters,lr=self.lr,weight_decay=self.l2)
            logging.info("Optimizer {}, LR {}, Weight Decay {}".format(self.optimizer, self.lr, self.l2))
        if self.scheduler=='cosineannealing':
            tmax=10
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=tmax)
            logging.info("Scheduler {}, T_max {}".format(self.scheduler, tmax))
        elif self.scheduler=='cyclic':
            max_lr=self.lr
            base_lr=self.lr*0.01
            scheduler= torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr)
            logging.info("Scheduler {}, base_lr {:.8f}, max_lr {:.8f} ".format(self.scheduler, base_lr, max_lr))

        stime=time.time()
        train_pairs, train_labels=load_data(edges,datatype='train',ikey2mol=ikey2mol)
        etime=time.time() - stime
        step = 0
        total_loss = 0
        batch_size = self.batch_size
        if force_debug:
            batch_per_epoch=20
        else:
            batch_per_epoch=int(np.ceil(len(train_labels)/batch_size))
        
        if self.prediction_mode.lower() in ['binary']:
            record_dict = {'epoch':[],
                       'total_loss':[],
                       'train_f1':[],
                       'train_auc':[],
                       'train_aupr':[]
                      }
            print("Epoch\tData\tF1\tAUC\tAUPR")
        else:
            record_dict = {'epoch':[],
                       'total_loss':[],
                       'train_pearson':[],
                       'train_spearman':[]
                      }
            print("Epoch\tData\tPearsonCorr\tSpearmanCorr")
            
        for epoch in range(1, self.train_epoch + 1):
            model.train()
            # Random shuffle all training pairs
            train_data_idxs=list(range(len(train_labels)))
            np.random.shuffle(train_data_idxs)
            epoch_loss_total = 0
            batch_prep_time=0;batch_train_time=0;batch_optim_time=0
            logging.info("Epoch {0} started".format(epoch))
            
            for batch_ in range(batch_per_epoch):

                stime=time.time()
                choices = train_data_idxs[batch_*batch_size:(batch_+1)*batch_size]
                batch_labels = torch.tensor([train_labels[idx] for idx in choices]).cuda()
                batch_train_pairs = [train_pairs[idx] for idx in choices]
                batch_chem_repr,batch_prot_repr = get_repr_from_pairs(batch_train_pairs)
                batch_chem_embed = load_from_mol_tuple(batch_chem_repr)
                if isinstance(batch_chem_embed, Variable) and torch.cuda.is_available():
                    batch_chem_embed = batch_chem_embed.cuda()
                if isinstance(batch_prot_repr, Variable) and torch.cuda.is_available():
                    batch_prot_repr = batch_prot_repr.cuda()
                batch_input = {'protein': batch_prot_repr,'ligand': batch_chem_embed}
                batch_prep_t=time.time() - stime
                batch_prep_time+=batch_prep_t
                stime=time.time()
                batch_logits = model(batch_input)
                batch_train_t=time.time() - stime
                batch_train_time+=batch_train_t
                stime=time.time()
                
                if self.prediction_mode.lower() in ['binary']:
                    loss_fn = torch.nn.CrossEntropyLoss()
                    batch_labels = batch_labels.long()
                else:
                    loss_fn = torch.nn.MSELoss()
                    batch_labels = batch_labels.float().reshape(-1,1)
                loss = loss_fn(batch_logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                batch_optim_t=time.time() - stime
                batch_optim_time+=batch_optim_t
                total_loss+=loss.item()
                epoch_loss_total += loss.item()
            
            logging.info("Epoch {}: Loss {}".format(epoch,loss.item()))
            
            trainmetrics=train_evaluator.eval(model,train_pairs,train_labels,epoch)
            record_dict['epoch'].append(epoch)
            record_dict['total_loss'].append(loss.item())
            if self.prediction_mode.lower() in ['binary']:
                record_dict['train_f1'].append(trainmetrics[0])
                record_dict['train_auc'].append(trainmetrics[1])
                record_dict['train_aupr'].append(trainmetrics[2])
            else:
                record_dict['train_pearson'].append(trainmetrics[1])
                record_dict['train_spearman'].append(trainmetrics[2])

        path = os.path.join(self.checkpoint_dir, "epoch_{0}".format(epoch))
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(model.state_dict(), os.path.join(path,'model.dat'))
        return record_dict