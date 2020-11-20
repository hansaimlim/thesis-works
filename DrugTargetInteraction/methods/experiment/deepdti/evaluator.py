from __future__ import print_function
import sys
import copy
import logging
import itertools
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn import metrics
import numpy as np
from evaluation_metrics_python2 import rmse, pearson, spearman, ci, f1, average_AUC

from utils import get_embedding


class Evaluator():
    def __init__(self, ikey2smiles, uniprot2pssm, molecule_dict,
            datatype='train',nsplit=20,nsample=512):
        self.ikey2smiles=ikey2smiles
        self.uniprot2pssm=uniprot2pssm
        self.moldict=molecule_dict
        self.datatype=datatype
        if (nsplit is None) or (nsample is None):
            logging.info("Evaluator for {} data. Whole samples will be evaluated.".format(datatype))
            self.nsplit=None
            self.nsample=None
        else:
            logging.info("Evaluator for {} data will random-pick {} samples for {} times.".format(datatype,nsample,nsplit))
            self.nsplit=nsplit
            self.nsample=nsample

    def eval(self,model,pred_model,pairs,labels,bin_idxs,epoch):
        #evalute performance at epoch=epoch
        #if nsplit=20, randomly pick nsample from pairs and evaluate, report mean and stdev
        #return one metric to apply early stopping
        datatype=self.datatype
        model.eval()
        pred_model.eval()
        nsplit=self.nsplit
        nsample=self.nsample
        ikey2smiles=self.ikey2smiles
        uniprot2pssm=self.uniprot2pssm
        moldict=self.moldict
        def run_model(model,pred_model,chem_repr,prot_repr,label):
            chem_embed = get_embedding(model,'chemical',chem_repr,volatile=True)
            prot_embed = get_embedding(model,'protein',prot_repr,volatile=True)
            with torch.no_grad():
                probs, _ = pred_model(dict(type='chemical', embedding=chem_embed),
                        dict(type='protein', embedding=prot_embed))
            label=torch.tensor(label,device=probs.device)
            mse,pc,sr,cidx,f1,auc=evaluate_regression_idg_dream(label.cpu().detach().numpy().reshape((-1,)),
                    probs.cpu().detach().numpy().reshape((-1,)))
            return mse,pc,sr,cidx,f1,auc

        since=time.time()
        rmse_parts=[];loss_parts=[];pearson_parts=[];spearman_parts=[];f1_parts=[];auc_parts=[]
        if nsplit is None:
            #evaluate all pairs if small enough
            chem_repr = [pair[0] for pair in pairs]
            prot_repr = [uniprot2pssm[pair[1]] for pair in pairs]
            mse,pc,sr,cidx,f1,auc=run_model(model,pred_model,chem_repr,prot_repr,labels)
            rmse_parts.append(mse)
            pearson_parts.append(pc)
            spearman_parts.append(sr)
            f1_parts.append(f1)
            auc_parts.append(auc)
        else:
            for i in range(nsplit):
                choices=np.array([],dtype=np.int32)
                for pb in bin_idxs.keys():
                    bin_idx=bin_idxs[pb]
                    if len(bin_idx)>=(nsample/16):
                        choice=np.random.choice(bin_idx,int(nsample/16), replace=False)
                    else: #for underrepresented bins
                        bin_idx=bin_idx+bin_idxs[6]+bin_idxs[7]+bin_idxs[8]+bin_idxs[9]+bin_idxs[10]
                    choice=np.random.choice(bin_idx,int(nsample/16), replace=False)
                    choices=np.concatenate((choices,choice),axis=0)
                batch_pairs = [pairs[idx] for idx in choices]
                batch_chem_repr = [pair[0] for pair in batch_pairs]
                batch_prot_repr = [uniprot2pssm[pair[1]] for pair in batch_pairs]
                batch_label = [labels[idx] for idx in choices]
                mse,pc,sr,cidx,f1,auc=run_model(model,pred_model,batch_chem_repr,batch_prot_repr,batch_label)
                rmse_parts.append(mse)
                pearson_parts.append(pc)
                spearman_parts.append(sr)
                f1_parts.append(f1)
                auc_parts.append(auc)
        rmse_mean,rmse_std=np.nanmean(rmse_parts),np.std(rmse_parts)
        pc_mean,pc_std=np.nanmean(pearson_parts),np.std(pearson_parts)
        sp_mean,sp_std=np.nanmean(spearman_parts),np.std(spearman_parts)
        f1_mean,f1_std=np.nanmean(f1_parts),np.std(f1_parts)
        auc_mean,auc_std=np.nanmean(auc_parts),np.std(auc_parts)
        eval_time=time.time() - since
        logging.info("{:.2f} seconds for {} evaluation. Epoch {}".format(eval_time,datatype,epoch))
        print("{}\t{}\t{:.5f}, {:.5f}\t{:.5f}, {:.5f}\t{:.5f}, {:.5f}\t{:.5f}, {:.5f}\t{:.5f}, {:.5f}".format(epoch,
            datatype,rmse_mean,rmse_std,pc_mean,pc_std,sp_mean,sp_std,f1_mean,f1_std,auc_mean,auc_std))
        return pc_mean

def evaluate_regression_idg_dream(y_true,y_hat):
  #rmse(y,y_hat)
  #pearson(y,y_hat)
  #spearman(y,f)
  #ci(y,f)
  #f1(y,f)
  #average_AUC(y,f)
  root_mse=rmse(y_true,y_hat)
  pc=pearson(y_true,y_hat)
  sr=spearman(y_true,y_hat)
  cidx=ci(y_true,y_hat)
  f1score=f1(y_true,y_hat)
  auc=average_AUC(y_true,y_hat)
  return root_mse,pc,sr,cidx,f1score,auc

def evaluate_binary_predictions(label,predprobs):
  label=np.argmax(np.array(label),axis=1)
  probs=np.array(predprobs)
  predclass=np.argmax(probs,axis=1)
  tn, fp, fn, tp = metrics.confusion_matrix(label,predclass).ravel()
  accuracy=float(tp+tn)/float(tp+tn+fp+fn)
  if tp==0:
    precision=0.0
    recall=0.0
  else:
    precision=float(tp)/float(tp+fp)
    recall=float(tp)/float(tp+fn)
  fpr, tpr, thresholds = metrics.roc_curve(label, probs[:,1], pos_label=1)
  auc=metrics.auc(fpr, tpr)
  prec, reca, thresholds = metrics.precision_recall_curve(label, probs[:,1], pos_label=1)
  aupr=metrics.auc(reca,prec)
  return accuracy,precision,recall,auc,aupr

