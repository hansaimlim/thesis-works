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
from deepdti.fingerprint.preprocess_molecules import get_molecule_degree_features

parser = argparse.ArgumentParser("Train DeepDTI.")
parser.add_argument('data_path', help='Path to the train/dev dataset.')
parser.add_argument('--log', default="INFO", help="Logging level. Set to DEBUG for more details.")
parser.add_argument('--random_seed', default=7706, help="Random seed.")

parser.add_argument('--epoch', default=100, type=int, help='Number of training epoches (default 100)')
parser.add_argument('--batch_size', default=256, type=int, help="Batch size. (default 256)")
parser.add_argument('--hidden_size', default=64, type=int, help="Hidden vector size. (default 64)")
parser.add_argument('--num_threads', default=4, type=int, help="Number of threads for parallelism (default 4)")
parser.add_argument('--checkpoint_dir', default="../checkpoints/", help="Directory to store checkpoints.")
parser.add_argument('--input_dropout_p', type=float, default=0.3)
parser.add_argument('--dropout_p', type=float, default=0)
parser.add_argument('--protein_weight', type=float, default=1.0, help="Scalar weight to scale protein representations (default 1.0)")

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
    logging.info("Loading DTI edges...Devpairs->training, Testpairs->evaluation")
    print("DevTrain Mode: Devpairs->training, Testpairs->evaluation")

    #use dev pairs for training
    edges = load_edges_from_file(os.path.join(opt.data_path, 'devpairs')) #same as dev for temporary debugging
    #use test pairs for evaluation
    dev_edges = load_edges_from_file(os.path.join(opt.data_path, 'testpairs'))
    logging.info("Initializing model...")
    hidden_size = opt.hidden_size
    since=time.time()
    molecule_dict = get_molecule_degree_features('../../../inputs/plain/chemicals.tsv')
    preprocess_time=time.time()-since
    logging.debug("{:.2f} seconds to preprocess {} molecules".format(preprocess_time,len(molecule_dict)))
    print("{:.2f} seconds to preprocess {} molecules".format(preprocess_time,len(molecule_dict)))
    model = DeepDTI(hidden_size,input_dropout_p=opt.input_dropout_p)
    pred_model = Predict(hidden_size,dropout_p=opt.dropout_p,protein_weight=opt.protein_weight)
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
        uniprot2pssm=uniprot2pssm,
        ikey2smiles=ikey2smiles,
        molecule_dict=molecule_dict
        )
    if not os.path.exists(opt.checkpoint_dir):
        os.mkdir(opt.checkpoint_dir)
        logging.info("Checkpoint directory {} created".format(opt.checkpoint_dir))
    dev_evaluator=Evaluator(ikey2smiles,uniprot2pssm,molecule_dict,datatype='dev',nsplit=10,nsample=512)
    test_evaluator=Evaluator(ikey2smiles,uniprot2pssm,molecule_dict,datatype='test',nsplit=10,nsample=512)
    trainer.train(
        model,
        pred_model,
        edges,
        dev_evaluator,
        dev_edges,
        test_evaluator
        )
