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
from models import DeepDTI, Predict
from evaluator import Evaluator
from utils import get_embedding
from utils import get_pkd_bin_idxs 

class Trainer():
    def __init__(self, epoch=100, batch_size=256, ckpt_dir="../checkpoints/",
            optimizer='adam',l2=0.001, lr=1e-5, scheduler='cosineannealing',uniprot2pssm={},ikey2smiles={}):
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.train_epoch = epoch
        self.optimizer=optimizer
        self.l2=l2
        self.lr=lr
        self.scheduler=scheduler
        self.uniprot2pssm = uniprot2pssm
        self.ikey2smiles = ikey2smiles

    @staticmethod
    def load_data(edges_dict,datatype='train'):
        #load training pairs with labels
        #edges: list of tuples (inchikey,uniprotID)
        #labels: list of float activity values for each edge
        count=0
        labels=[]
        edges=[]
        chems=[]
        prots=[]
        for cp in edges_dict.keys():
            chem,prot=cp.strip().split('\t')
            chems.append(chem)
            prots.append(prot)
            count+=1
            labels.append(edges_dict[cp])
            edges.append((chem,prot))
        chems=list(set(chems))
        prots=list(set(prots))
        logging.info("Total {} chemicals, {} proteins, {} activities loaded for {} data".format(len(chems),
            len(prots),len(labels),datatype))
        return edges, labels

    def train(self, model, pred_model, edges, train_evaluator, dev_edges, dev_evaluator):
        uniprot2pssm = self.uniprot2pssm
        ikey2smiles = self.ikey2smiles
        self.model = model
        self.pred_model = pred_model
        parameters = list(model.parameters()) + list(pred_model.parameters())
        if self.optimizer=='adam':
            optimizer = torch.optim.Adam(parameters,lr=self.lr,weight_decay=self.l2)
            print("Optimizer {}, LR {}, Weight Decay {}".format(self.optimizer, self.lr, self.l2))
        if self.scheduler=='cosineannealing':
            tmax=10
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=tmax)
            print("Scheduler {}, T_max {}".format(self.scheduler, tmax))

#        print("Model's state_dict:")
#        for param_tensor in model.state_dict():
#            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#        print("Pred model's state_dict:")
#        for param_tensor in pred_model.state_dict():
#            print(param_tensor, "\t", pred_model.state_dict()[param_tensor].size())
#        # Print optimizer's state_dict
#        print("Optimizer's state_dict:")
#        for var_name in optimizer.state_dict():
#            print(var_name, "\t", optimizer.state_dict()[var_name])
#        print("Scheduler's state_dict:")
#        for var_name in scheduler.state_dict():
#            print(var_name, "\t", scheduler.state_dict()[var_name])

        stime=time.time()
        train_pairs, train_labels=self.load_data(edges,datatype='train')
        bin_idxs=get_pkd_bin_idxs(np.array(train_labels,dtype=np.float32))
        for pb in bin_idxs:
            logging.debug("{}th Training bin length {}".format(pb,len(bin_idxs[pb])))

        dev_pairs, dev_labels=self.load_data(dev_edges,datatype='dev')
        dev_bin_idxs=get_pkd_bin_idxs(np.array(dev_labels,dtype=np.float32))                
        for pb in dev_bin_idxs:
            logging.debug("{}th Dev bin length {}".format(pb,len(dev_bin_idxs[pb])))

        etime=time.time() - stime
        logging.debug("{:.2f} seconds to load data sets".format(etime))
        logging.info("Training started...")
        step = 0
        total_loss = 0
        prev_dev_pc=-999
        best_epoch=0
        batch_size = self.batch_size
        batch_per_epoch=int(np.ceil(len(train_labels)/batch_size))
        print("Epoch\tData\tRMSE, RMSEstd\tPC, PCstd,\tSP, SPstd\tF1, F1std\tAUC, AUCstd")
        for epoch in range(1, self.train_epoch + 1):
            logging.info("Epoch {0} started".format(epoch))
            count = 0
            train_loss = 0
            chem_embed_time=0;prot_embed_time=0;batch_prep_time=0;batch_train_time=0;batch_optim_time=0
            for batch_ in range(batch_per_epoch):
                count += 1
                if batch_per_epoch > 300 and count % 100 == 0:
                    logging.info("Epoch {} progress: {}/{}. Train loss {}".format(epoch,
                        count, batch_per_epoch, train_loss/count))

                choices=np.array([],dtype=np.int32)
                for pb in bin_idxs.keys():
                    bin_idx=bin_idxs[pb]
                    if len(bin_idx)>=(batch_size/16):
                        choice=np.random.choice(bin_idx,int(batch_size/16), replace=False)
                    else: #for underrepresented bins
                        bin_idx=bin_idx+bin_idxs[6]+bin_idxs[7]+bin_idxs[8]+bin_idxs[9]+bin_idxs[10]
                        choice=np.random.choice(bin_idx,int(batch_size/16), replace=False)
                    choices=np.concatenate((choices,choice),axis=0)
                logging.debug("{} batch indice selected".format(choices.shape))
                stime=time.time()

                batch_labels = [train_labels[idx] for idx in choices]
                batch_train_pairs = [train_pairs[idx] for idx in choices]
                batch_chem_repr = [ikey2smiles[pair[0]] for pair in batch_train_pairs]
                batch_chem_embed = get_embedding(self.model,'chemical',batch_chem_repr)
                chem_embed_t=time.time() - stime
                chem_embed_time+=chem_embed_t
                logging.debug("{:.2f} seconds for chemical embedding Epoch {}, Batch {}".format(chem_embed_t,epoch,batch_))
                logging.debug("Batch source embedding shape {}".format(batch_chem_embed.shape))
                stime=time.time()
                # target representation
                batch_prot_repr = [uniprot2pssm[pair[1]] for pair in batch_train_pairs]
                batch_prot_embed = get_embedding(self.model,'protein', batch_prot_repr)
                prot_embed_t=time.time() - stime
                prot_embed_time+=prot_embed_t
                stime=time.time()
                logging.debug("{:.2f} seconds for protein embedding Epoch {}, Batch {}".format(prot_embed_t,epoch,batch_))
                logging.debug("Batch target embedding shape {}".format(batch_prot_embed.shape))
                stime=time.time()
                probs, _ = pred_model(dict(type='chemical', embedding=batch_chem_embed),
                            dict(type='protein', embedding=batch_prot_embed))
                
                batch_train_t=time.time() - stime
                batch_train_time+=batch_train_t
                stime=time.time()
                logging.debug("{:.2f} seconds for batch training. Epoch {}, Batch {}".format(batch_train_t,epoch,batch_))
                logging.debug("Done")
                batch_labels=torch.tensor(batch_labels,device=probs.device)
                
                loss_fn = torch.nn.MSELoss()
               # loss_fn = torch.nn.SmoothL1Loss()
                logging.debug("Getting MSE loss...")
                loss = loss_fn(batch_labels, probs)
                optimizer.zero_grad()
                logging.debug("Back propagating...")
                loss.backward()
                optimizer.step()
                batch_optim_t=time.time() - stime
                batch_optim_time+=batch_optim_t
                logging.debug("{:.2f} seconds for batch backprop. Epoch {}, Batch {}".format(batch_optim_t,epoch,batch_))
                logging.debug("Done")
                step += 1
                total_loss+=loss.item()
                train_loss+=loss.item()
                scheduler.step()

            if epoch:
                #evaluate performance every epoch
                train_pc=train_evaluator.eval(model,pred_model,train_pairs,train_labels,bin_idxs,epoch)
                dev_pc=dev_evaluator.eval(model,pred_model,dev_pairs,dev_labels,dev_bin_idxs,epoch)
                if dev_pc > prev_dev_pc:
                    prev_dev_pc = dev_pc #best pearson correlation
                    best_epoch = epoch
                    path = os.path.join(self.checkpoint_dir, "epoch_{0}".format(epoch))
                    if not os.path.exists(path):
                        os.mkdir(path)
                    torch.save(model.state_dict(), os.path.join(path, 'model.dat'))
                    torch.save(pred_model.state_dict(), os.path.join(path, 'pred_model.dat'))
                    logging.info("New best dev P.corr {} at epoch {}".format(dev_pc,best_epoch))
            logging.info("Epoch {}: ChemEmbedTime {:.1f}, ProtEmbedTime {:.1f}, BatchPrepTime {:.1f},\
  BatchTrainTime {:.1f}, BatchOptimTime {:.1f}"\
 .format(epoch,chem_embed_time,prot_embed_time,batch_prep_time,batch_train_time,batch_optim_time))
        print("Best Dev PC {} at epoch {}".format(prev_dev_pc,best_epoch))

