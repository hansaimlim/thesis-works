import os
import sys
import argparse
import logging
import random
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deepdti.models import DeepDTI, Predict
from deepdti.trainer import Trainer
from deepdti.utils import load_edges_from_file, load_uniprot2pssm, load_ikey2smiles
from deepdti.evaluator import Evaluator
from deepdti.fingerprint.preprocess_molecules import get_molecule_dict
parser = argparse.ArgumentParser("Train DeepDTI.")
parser.add_argument('data_path', help='Path to the train/dev dataset.')
parser.add_argument('--log', default="INFO", help="Logging level. Set to DEBUG for more details.")
parser.add_argument('--random_seed', default=7706, help="Random seed.")

parser.add_argument('--epoch', default=100, type=int, help='Number of training epoches (default 100)')
parser.add_argument('--batch_size', default=256, type=int, help="Batch size. (default 256)")
parser.add_argument('--hidden_size', default=64, type=int, help="Hidden vector size. (default 64)")
parser.add_argument('--num_threads', default=4, type=int, help="Number of threads for parallelism (default 4)")
parser.add_argument('--checkpoint_dir', default="../checkpoints/", help="Directory to store checkpoints.")
parser.add_argument('--lr', type=float, default=1e-5, help="Initial learning rate")
parser.add_argument('--l2', type=float, default=1e-3, help="L2 regularization weight")
parser.add_argument('--optimizer', type=str, default='adam', help="optimizer")
parser.add_argument('--scheduler', type=str, default='cosineannealing', help="scheduler to adjust learning rate")
parser.add_argument('--dropout_chem', type=float, default=0.0, help="Dropout prob for chemical fingerprint")
parser.add_argument('--dropout_prot', type=float, default=0.0, help="Dropout prob for protein PSSM feature Conv")
parser.add_argument('--attn_dropout', type=float, default=0.0, help="Dropout prob for chem&prot during attentive pooling")

opt = parser.parse_args()

seed = opt.random_seed
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=getattr(logging, opt.log.upper()))
    logging.info(opt)
    traindat='trainpairs' #data file used to fit model
    devdat='devpairs' #data file used to evaluate model fit
    print("Plain training mode. {}->training, {}->evaluation\n\
            Input data path {}\n\
            nbatch={}, hidden_size={}, dropout_chem={}, dropout_prot={}, AttentivePooling_dropout={}\n\
            Optimizer={}, Initial Learning rate={}, Scheduler={}, L2-penalty={}\n\
            checkpoint path {}".format(traindat,devdat,opt.data_path,opt.batch_size,
                opt.hidden_size,opt.dropout_chem,opt.dropout_prot,opt.attn_dropout,
                opt.optimizer,opt.lr,opt.scheduler,opt.l2,opt.checkpoint_dir))
    #use dev pairs for training
    edges = load_edges_from_file(os.path.join(opt.data_path, traindat))
    #use test pairs for evaluation
    dev_edges = load_edges_from_file(os.path.join(opt.data_path, devdat))
    logging.info("Initializing model...")
    hidden_size = opt.hidden_size
    since=time.time()
    molecule_dict = get_molecule_dict('../../../inputs/plain/chemicals.tsv')
    preprocess_time=time.time()-since
    logging.debug("{:.2f} seconds to preprocess {} molecules".format(preprocess_time,len(molecule_dict)))
    print("{:.2f} seconds to preprocess {} molecules".format(preprocess_time,len(molecule_dict)))
    model = DeepDTI(hidden_size,molecule_dict,dropout_chem=opt.dropout_chem,dropout_prot=opt.dropout_prot)
    pred_model = Predict(hidden_size,attn_dropout=opt.attn_dropout)
    torch.set_num_threads(opt.num_threads)
    if torch.cuda.is_available():
        logging.info("Moving model to GPU ...")
        model.cuda()
        pred_model.cuda()
        logging.debug("Done")
    uniprot2pssm = load_uniprot2pssm()
    ikey2smiles = load_ikey2smiles()
    trainer = Trainer(
        epoch=opt.epoch,
        batch_size=opt.batch_size,
        ckpt_dir=opt.checkpoint_dir,
        optimizer=opt.optimizer,
        scheduler=opt.scheduler,
        lr=opt.lr,
        l2=opt.l2,
        uniprot2pssm=uniprot2pssm,
        ikey2smiles=ikey2smiles,
        molecule_dict=molecule_dict
        )
    if not os.path.exists(opt.checkpoint_dir):
        os.mkdir(opt.checkpoint_dir)
        logging.info("Checkpoint directory {} created".format(opt.checkpoint_dir))
    train_evaluator=Evaluator(ikey2smiles,uniprot2pssm,molecule_dict,datatype='train',nsplit=10,nsample=512)
    dev_evaluator=Evaluator(ikey2smiles,uniprot2pssm,molecule_dict,datatype='dev',nsplit=10,nsample=512)
    logging.debug("Train and Dev evaluators initialized.")
    trainer.train(
        model,
        pred_model,
        edges,
        train_evaluator,
        dev_edges,
        dev_evaluator
        )
    total_time=(time.time()-since)/60.0 #seconds to minutes
    logging.debug("{:.2f} minutes to complete train/evaluate {} epochs".format(total_time,opt.epoch))
    print("{:.2f} minutes to complete train/evaluate {} epochs".format(total_time,opt.epoch))
