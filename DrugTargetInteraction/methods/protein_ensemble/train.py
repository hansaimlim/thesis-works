from __future__ import division
from __future__ import print_function
import os
import sys
import glob
import argparse
import logging
import random
import time
import torch
import math
import socket
from datetime import datetime
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils.data as Data
from rdkit.Chem import MolFromSmilesl
from models import MolecularGraphCoupler
from trainer import Trainer
from utils import load_edges_from_file, load_ikey2smiles, load_uniprot2pssm, load_uniprot2singletrepr, save_json, load_json
from evaluator import Evaluator
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, BertForPreTraining
from transformers import AlbertTokenizer, AlbertModel, AlbertForMaskedLM, AlbertConfig
from transformers.modeling_albert import load_tf_weights_in_albert
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#from tensorboardX import SummaryWriter
parser = argparse.ArgumentParser("Train Albert-GraphCNN-TransferEnsemble.")
parser.add_argument('--protein_embedding_type', type=str, default='albert', help="albert, lstm, pssm are available options")
parser.add_argument('--chem_dropout', type=float, default=0.1, help="Dropout prob for chemical fingerprint")
parser.add_argument('--chem_conv_layer_sizes', type=list, default=[20,20,20,20],help='Conv layers for chemicals')
parser.add_argument('--chem_feature_size', type=int, default=128,help='chemical fingerprint dimension')
parser.add_argument('--chem_degrees',type=list, default=[0,1,2,3,4,5],help='Atomic connectivity degrees for chemical molecules')
#### args for AlbertResnet model
parser.add_argument('--albert_layers_frozen',type=int,default=10, help='how many layers of pretrained albert to be frozen')
parser.add_argument('--prot_feature_size',type=int,default=256, help='protein representation dimension')
parser.add_argument('--prot_max_seq_len',type=int,default=256, help='maximum length of a protein sequence including special tokens')
parser.add_argument('--prot_dropout', type=float, default=0.1, help="Dropout prob for protein representation")
#### args for LSTM protein Embedding
parser.add_argument('--lstm_embedding_size',type=int,default=128, help='protein representation dimension for LSTM')
parser.add_argument('--lstm_num_layers',type=int,default=3, help='num LSTM layers')
parser.add_argument('--lstm_hidden_size',type=int,default=64, help='protein representation dimension for LSTM')
parser.add_argument('--lstm_out_size',type=int,default=128, help='protein representation dimension for LSTM')
parser.add_argument('--lstm_input_dropout', type=float, default=0.2, help="Dropout prob for protein representation")
parser.add_argument('--lstm_output_dropout', type=float, default=0.3, help="Dropout prob for protein representation")
#### args for PSSM protein Embedding
#### args for Attentive Pooling
parser.add_argument('--ap_dropout', type=float, default=0.1, help="Dropout prob for chem&prot during attentive pooling")
parser.add_argument('--ap_feature_size',type=int,default=64,help='attentive pooling feature dimension')
#### args for model training and optimization
parser.add_argument('--datapath', default='data/activity/protein_based_split',help='Path to the train/dev dataset.')
parser.add_argument('--prediction_mode', default='binary', type=str, help='set to continuous and provide pretrained checkpoint')
parser.add_argument('--from_pretrained_checkpoint', type=str2bool, nargs='?',const=True,
                    default=False, help="If true, pretrained checkpoints are loaded and resume training")
parser.add_argument('--pretrained_checkpoint_dir', default="temp/", 
        help="Directory where pretrained checkpoints are saved. ignored if --from_pretrained_checkpoint is false")
parser.add_argument('--random_seed', default=705, help="Random seed.")
parser.add_argument('--epoch', default=100, type=int, help='Number of training epoches (default 50)')
parser.add_argument('--batch', default=64, type=int, help="Batch size. (default 64)")
parser.add_argument('--max_eval_steps', default=1000, type=int, help="Max evaluation steps. (nsamples=batch*steps)")
parser.add_argument('--optimizer', type=str, default='adam', help="optimizer")
parser.add_argument('--scheduler', type=str, default='cosineannealing', help="scheduler to adjust learning rate [cyclic or cosineannealing]")
parser.add_argument('--checkpoint_dir',default="temp/", 
        help="Directory to store checkpoints. If starting from pretrained checkpoint, use the same directory as pretrained models.")
parser.add_argument('--lr', type=float, default=2e-5, help="Initial learning rate")
parser.add_argument('--l2', type=float, default=1e-4, help="L2 regularization weight")
parser.add_argument('--num_threads', default=8, type=int, help='Number of threads for torch')
#### args for debugging
parser.add_argument('--log', default="INFO", help="Logging level. Set to DEBUG for more details.")
parser.add_argument('--force_debug', type=str2bool, nargs='?',const=True,
                    default=False, help="Force debug mode for shorter batches")
parser.add_argument('--no_cuda', type=str2bool, nargs='?',const=True, default=False, help='Disables CUDA training.')
opt = parser.parse_args()

seed = opt.random_seed
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    FORMAT = '%(asctime)-15s %(message)s'
    if opt.force_debug:
        print("Force-debug mode...")
        logging.basicConfig(format=FORMAT, level=getattr(logging, 'DEBUG'))
        opt.epoch=2
        opt.max_eval_steps=10
    else:
        logging.basicConfig(format=FORMAT, level=getattr(logging, opt.log.upper()))
    logging.info(opt)
        
    opt.albertdatapath='data/albertdata/'
    opt.albertconfig=os.path.join(opt.albertdatapath,'albertconfig/albert_config_tiny_google.json')
    opt.albertvocab=os.path.join(opt.albertdatapath,'vocab/pfam_vocab_triplets.txt')
    opt.albert_pretrained_checkpoint=os.path.join(opt.albertdatapath,"pretrained_whole_pfam/model.ckpt-1500000")
    if opt.prediction_mode == 'binary':
        opt.traindata = os.path.join(opt.datapath,'kinase_binary_train.tsv') #train data for binary labels
        opt.devdata = os.path.join(opt.datapath,'kinase_binary_dev.tsv') #dev data for binary labels
    else:
        opt.traindata = os.path.join(opt.datapath,'kinase_consolidated_train.tsv') #train data for continuous labels
        opt.devdata = os.path.join(opt.datapath,'kinase_consolidated_dev.tsv') #dev data for continuous labels
    ### arguments that are less frequently modified

    since=time.time()
    if not os.path.exists(opt.checkpoint_dir):
        os.mkdir(opt.checkpoint_dir)
        logging.info("Checkpoint directory {} created".format(opt.checkpoint_dir))
    logging.info("Initializing model...")
    albertconfig = AlbertConfig.from_pretrained(opt.albertconfig)
    berttokenizer = BertTokenizer.from_pretrained(opt.albertvocab)
    model = MolecularGraphCoupler(
                 protein_embedding_type=opt.protein_embedding_type, #could be albert, LSTM, 
                 prediction_mode=opt.prediction_mode, #could be continuous (e.g. pKi, pIC50)
                 #protein features - albert
                 albertconfig=albertconfig,
                 tokenizer=berttokenizer,
                 ckpt_path=opt.albert_pretrained_checkpoint,
                 albert_layers_frozen=opt.albert_layers_frozen,
                 #protein features - LSTM
                 lstm_vocab_size=26,
                 lstm_embedding_size=opt.lstm_embedding_size,
                 lstm_hidden_size=opt.lstm_hidden_size,
                 lstm_num_layers=opt.lstm_num_layers,
                 lstm_out_size=opt.lstm_out_size,
                 lstm_input_dropout_p=opt.lstm_input_dropout,
                 lstm_output_dropout_p=opt.lstm_output_dropout,
                 #chemical features
                 conv_layer_sizes=opt.chem_conv_layer_sizes,
                 output_size=opt.chem_feature_size,
                 degrees=opt.chem_degrees,
                 #attentive pooler features
                 ap_hidden_size=opt.ap_feature_size,
                 ap_dropout=opt.ap_dropout
                )

    logging.info("Loading protein representations...")
    #uniprot2triplets=load_json('data/protein/uniprot2triplets.json') #kinases
    uniprot2triplets=load_json('data/protein/gpcr_uniprot2triplets.json') #gpcrs
    ##for transformers package version 2.0
    for uni in uniprot2triplets.keys():
        triplets = uniprot2triplets[uni].strip().split(' ')
        triplets.pop(0)
        triplets.pop(-1)
        uniprot2triplets[uni] = ' '.join(triplets)
        
    uniprot2pssm = load_uniprot2pssm(max_len=512,padding=0)
    uniprot2singletrepr = load_uniprot2singletrepr(binding_site=False)
    logging.info("Protein representations successfully loaded.\nLoading protein-ligand interactions.")
    
    edges,train_ikeys,train_uniprots = load_edges_from_file(opt.traindata,
                                                                allowed_uniprots=list(uniprot2triplets.keys()),
                                                                sep='\t',
                                                                header=False)
    #use test pairs for evaluation
    dev_edges,dev_ikeys,dev_uniprots = load_edges_from_file(opt.devdata,
                                                            allowed_uniprots=list(uniprot2triplets.keys()),
                                                            sep='\t',
                                                            header=False)
    logging.info("Protein-ligand interactions successfully loaded.")
    torch.set_num_threads(opt.num_threads)
    if opt.from_pretrained_checkpoint:
        model.load_state_dict(torch.load(os.path.join(opt.pretrained_checkpoint_dir,'model.dat')))
        opt.is_pretrained=True
        logging.info("Pretrained checkpoints loaded from {}".format(opt.pretrained_checkpoint_dir))
    else:
        opt.is_pretrained=False
    config_path=opt.checkpoint_dir+'config.json'
    save_json(vars(opt),config_path)
    logging.info("model configurations saved to {}".format(config_path))
    if torch.cuda.is_available():
        logging.info("Moving model to GPU ...")
        model=model.cuda()
        logging.debug("Done")
    else:
        model=model.cpu()
        logging.debug("Running on CPU...")

    ikey2smiles = load_ikey2smiles()
    ikey2mol = {}
    ikeys=list(set(train_ikeys+dev_ikeys))
    for ikey in ikeys: 
        try:
            mol = MolFromSmiles(ikey2smiles[ikey])
            ikey2mol[ikey]=mol
        except:
            continue
    
    trainer = Trainer(model=model,
                      berttokenizer=berttokenizer,
                      epoch=opt.epoch, batch_size=opt.batch, ckpt_dir=opt.checkpoint_dir,
                      optimizer=opt.optimizer,l2=opt.l2, lr=opt.lr, scheduler=opt.scheduler,
                      ikey2smiles=ikey2smiles,ikey2mol=ikey2mol,uniprot2triplets=uniprot2triplets,
                      uniprot2pssm=uniprot2pssm,uniprot2singletrepr=uniprot2singletrepr,
                      prediction_mode=opt.prediction_mode,
                      protein_embedding_type=opt.protein_embedding_type)

    train_evaluator=Evaluator(ikey2smiles=ikey2smiles,
                              ikey2mol=ikey2mol,
                              berttokenizer=berttokenizer,
                              uniprot2triplets=uniprot2triplets,
                              uniprot2pssm=uniprot2pssm,
                              uniprot2singletrepr=uniprot2singletrepr,
                              prediction_mode=opt.prediction_mode,
                              protein_embedding_type=opt.protein_embedding_type,
                              datatype='train',
                              max_steps=opt.max_eval_steps,
                              batch=opt.batch,
                              shuffle=True)
    dev_evaluator=Evaluator(ikey2smiles=ikey2smiles,
                            ikey2mol=ikey2mol,
                            berttokenizer=berttokenizer,
                            uniprot2triplets=uniprot2triplets,
                            uniprot2pssm=uniprot2pssm,
                            uniprot2singletrepr=uniprot2singletrepr,
                            prediction_mode=opt.prediction_mode,
                            protein_embedding_type=opt.protein_embedding_type,
                            datatype='dev',
                            max_steps=opt.max_eval_steps,
                            batch=opt.batch,
                            shuffle=False)
    logging.debug("Train and Dev evaluators initialized.\nStart training...")

    record_dict = trainer.train(edges, train_evaluator, dev_edges, dev_evaluator, force_debug=opt.force_debug)
    record_path=opt.checkpoint_dir+'training_record.json'
    save_json(record_dict,record_path)
    logging.info("Training record saved to {}".format(record_path))
    total_time=(time.time()-since)/60.0 #seconds to minutes
    logging.debug("{:.2f} minutes to complete train/evaluate {} epochs".format(total_time,opt.epoch))
    print("{:.2f} minutes to complete train/evaluate {} epochs".format(total_time,opt.epoch))
    print("Training record saved to {}".format(record_path))
