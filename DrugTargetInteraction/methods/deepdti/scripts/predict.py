import os
import re
import sys
import csv
import argparse
import logging
import random
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deepdti.models import DeepDTI, Predict
from deepdti.predict import Predictor
from deepdti.utils import load_uniprot2pssm, load_ikey2smiles

parser = argparse.ArgumentParser("Train network embedding.")
parser.add_argument('model_path', help="The file path where model is saved.")
parser.add_argument('pred_model_path', help="The file path where model is saved.")
parser.add_argument('datapath', help='Path to the dataset to predict.')
parser.add_argument('--round', default=1, help="IDG DREAM challenge round (1,2)")
parser.add_argument('--log', default="INFO", help="Logging level.")
parser.add_argument('--hidden_size', default=64, type=int, help="Hidden vector size. (default 64)")
parser.add_argument('--random_seed', default=7706, help="Random seed.")

opt = parser.parse_args()

seed = opt.random_seed
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

def load_data(edges_dict,datatype='round1',label=None):
    #load test pairs
    #if label=None, assume the activities are unknown
    #if label is not None, load known activities to compare with prediction
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
        if label is not None:
    	    labels.append(edges_dict[cp])
        else:
            labels.append(0)
	edges.append((chem,prot))
    chems=list(set(chems))
    prots=list(set(prots))
    logging.info("Total {} chemicals, {} proteins, {} activities loaded for {} data".format(len(chems),
	len(prots),len(labels),datatype))
    return edges, labels
def load_edges_from_file(edgefile, label=None):
    edges={}
    with open(edgefile,'r') as f:
        for line in f:
            line=line.strip().split('\t')
            ikey=line[0]
            uni=line[1]
            edge=ikey+'\t'+uni
            if label is not None:
                val=float(line[2])
                edges[edge]=val
            else:
                edges[edge]=0
    return edges

if __name__ == '__main__':
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=getattr(logging, opt.log.upper()))
    logging.info(opt)
    logging.info("Initializing model...")
    hidden_size = opt.hidden_size
    model = DeepDTI(hidden_size,input_dropout_p=0)
    logging.debug("Done.")
    if torch.cuda.is_available():
        logging.info("Moving model to GPU ...")
        model.cuda()
        logging.debug("Done")

    edges=load_edges_from_file(opt.datapath)
    edges,labels=load_data(edges,datatype='round1')
    uniprot2pssm=load_uniprot2pssm()
    ikey2smiles=load_ikey2smiles()
    inf = Predictor(opt.model_path, opt.pred_model_path, uniprot2pssm, ikey2smiles)

    prot_node_ids=[edge[1] for edge in edges]
    round1_dict={"Q9P2K8":"Q9P2K8(Kin.Dom.2,S808G)","P23458":"P23458(JH1domain-catalytic)",
            "Q9P2K8(Kin.Dom.2,S808G)":"Q9P2K8",
            "Q15418":"Q15418(Kin.Dom.1)","O75582":"O75582(Kin.Dom.1)","Q9UK32":"Q9UK32(Kin.Dom.2)",
            "P29597":"P29597(JH2domain-pseudokinase)"}
    for g in round1_dict.keys():
        round1_dict[round1_dict[g]]=g
    logging.info("Inference started...")

    pair2pred={}
    for idx in range(len(edges)):
        edge=edges[idx]
        chem_node=edge[0]
        prot_node=edge[1]
        pair=chem_node+','+prot_node
        
        tgt_node_id_ori=prot_node
        prob,explanation=inf.predict(chem_node, prot_node)
        prob=prob.item()
        pair2pred[pair]=prob+0.6
        if prot_node in round1_dict:
            prot_node_convert=round1_dict[prot_node] #convert gene id for round1 data
            pair=chem_node+','+prot_node_convert
            pair2pred[pair]=prob+0.6
        if opt.round==1:
            continue
        else:
            print("{}\t{}\t{}\t{}".format(chem_node,prot_node,tgt_node_id_ori,prob))

    if opt.round==1:
        round1_template='../../../inputs/IDG_DREAM/round1/round_1_template.csv'
        with open(round1_template,'r') as f:
            for l in  csv.reader(f, quotechar='"', delimiter=',',
                    quoting=csv.QUOTE_ALL, skipinitialspace=True):
                if l[0]=='Compound_SMILES':
                    #header line
                    line=','.join(l)+',pKd_[M]_pred'
                else:
                    pair=l[1]+','+l[3]
                    pred=pair2pred[pair]
                    if l[-1]=='GCN2(Kin.Dom.2,S808G)': #manually insert quote
                        l[-1]='"GCN2(Kin.Dom.2,S808G)"'
                    line=','.join(l)+','+str(pred)
                print("{}".format(line))



