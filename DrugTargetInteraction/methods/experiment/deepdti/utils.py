from __future__ import print_function
import datetime

import torch
from torch.autograd import Variable
import os
import sys
import re
import gzip
import numpy as np


TYPE_MOLECULE = 'graph'
TYPE_SEQUENCE_PSSM = 'sequence-pssm'
SEQ_MAX_LEN = 700

def load_ikey2smiles():
    file_path='../../../data/Integrated/chemicals/integrated_chemicals.tsv.gz'
    ikey2smiles={}
    with gzip.open(file_path,'r') as fin:
        for line in fin:
            line=line.strip().split('\t')
            ikey=line[1]
            smi=line[2]
            ikey2smiles[ikey]=smi
    return ikey2smiles

def load_uniprot2pssm(max_len=SEQ_MAX_LEN,padding=0):
    #maximum sequence length: max_len
    #pssm padded with zeros if len<max_len
    base_path='../../../data/kinome_assay/sequence_feature/'
    pssm_dir=base_path+'kinase_domain_pssm_uniref50/'
    #protfile=base_path+'prot_bsite_sample' #padding test
    protfile=base_path+'kinases_with_bsite'
    uniprot2pssm={}
    pssm_files=os.listdir(pssm_dir)
    manual_dict={'P52333_JH1domain-catalytic':'P52333_Kin.Dom.2-C-terminal.dat',
	    'Q9P2K8_Kin.Dom.2,S808G':'Q9P2K8_S808G_Kin.Dom.2-C-terminal.dat',
	    'P23458' :'P23458_JH2domain-pseudokinase.dat',
	    'P29597' :'P29597_JH2domain-pseudokinase.dat',
	    'O75582' :'O75582_Kin.Dom.1-N-terminal.dat',
	    'Q15418' :'Q15418_Kin.Dom.1-N-terminal.dat',
	    'Q9P2K8' :'Q9P2K8_Kin.Dom.1-N-terminal.dat',
	    'Q9UK32' :'Q9UK32_Kin.Dom.2-C-terminal.dat'}
    with open(protfile,'r') as f:
        for line in f:
            uniprot=line.strip()
            line=line.strip()
            line=line.replace('(','_').replace(')','')
            line=line.replace('-nonphosphorylated','').replace('-phosphorylated','').replace('-autoinhibited','')

            matchkd=re.search(r'Kin\.Dom',line,re.I)
            matchjh=re.search(r'JH\ddomain',line,re.I)
            if line in list(manual_dict.keys()):
                fname=manual_dict[line]
            elif matchkd:
                matchkd=re.search(r'Kin\.Dom\.(\d)',line,re.I)
                if matchkd is None:
                    fname=line+'.dat'
                elif matchkd.group(1)==str(1):
                    fname=line+'-N-terminal.dat'
                elif matchkd.group(1)==str(2):
                    fname=line+'-C-terminal.dat'
            elif matchjh:
                fname=line+'.dat'
            else:
                fname=line+'_Kin.Dom.dat'
            if fname not in pssm_files:
                print("PSSM file {} not found for protein {}".format(fname,line))
            else:
                pssm=[]
                with open(pssm_dir+fname,'r') as f:
                    for line in f:
                        line=line.strip().lstrip().split()
                        if len(line)==0: #empty line
                            continue
                        else:
                            try:
                                resnum=int(line[0])
                            except: #non-pssm field
                                continue
                            res_vector=np.array(line[2:22],dtype=np.float32)
                            pssm.append(res_vector)
                pssm=np.array(pssm,dtype=np.float32)
                pssm=np.pad(pssm,((0,max_len-pssm.shape[0]),(0,0)),'constant',constant_values=padding) #pad to the bottom
                uniprot2pssm[uniprot]=pssm
                #print("PSSM shape {} loaded for {} from file {}".format(uniprot2pssm[uniprot].shape,uniprot,fname))
    return uniprot2pssm

def load_edges_from_file(edgefile):
    edges={}
    with open(edgefile,'r') as f:
        for line in f:
            line=line.strip().split('\t')
            ikey=line[0]
            uni=line[1]
            val=float(line[2])
            edge=ikey+'\t'+uni
            edges[edge]=val
    return edges

def get_pkd_bin_idxs(label):
  #input: list of pkd(or pki) activities
  #output: dict of indices for activity bins
  #purpose: to choose train minibatches across activity range
  pkd_bins={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[]}
  for i in range(len(label)):
    pkd=label[i]
    idx=int(i)
    if (pkd==np.inf) or np.isnan(pkd): #skip inf, nan
      continue
    if pkd<=0: #no pkd should be negative
      continue
    if pkd<=4.0: #concentrated in bin ranges [4.2,4.8] [4.8,5.3] [5.3,6.0] [6.0,7.0]
      pkd_bins[1].append(idx)
    elif pkd<=4.4:
      pkd_bins[2].append(idx)
    elif pkd<=4.8:
      pkd_bins[3].append(idx)
    elif pkd<=5.2:
      pkd_bins[4].append(idx)
    elif pkd<=5.6:
      pkd_bins[5].append(idx)
    elif pkd<=6.0:
      pkd_bins[6].append(idx)
    elif pkd<=6.4:
      pkd_bins[7].append(idx)
    elif pkd<=6.8:
      pkd_bins[8].append(idx)
    elif pkd<=7.2:
      pkd_bins[9].append(idx)
    elif pkd<=7.6:
      pkd_bins[10].append(idx)
    elif pkd<=8.0:
      pkd_bins[11].append(idx)
    elif pkd<=8.4:
      pkd_bins[12].append(idx)
    elif pkd<=8.8:
      pkd_bins[13].append(idx)
    elif pkd<=9.2:
      pkd_bins[14].append(idx)
    elif pkd<=10.0:
      pkd_bins[15].append(idx)
    elif pkd<=16.0:
      pkd_bins[16].append(idx)
    else:
      continue
  return pkd_bins
def get_embedding(model, node_type, batch_repr, volatile=False):

    if node_type == 'protein':
        if volatile:
            with torch.no_grad():
                batch_repr = Variable(torch.FloatTensor(batch_repr))
        else:
            batch_repr = Variable(torch.FloatTensor(batch_repr))
    elif node_type == 'chemical':
        if volatile:
            with torch.no_grad():
                #batch_repr = Variable(torch.FloatTensor(batch_repr))
                batch_repr = batch_repr
        else:
            #batch_repr = Variable(torch.FloatTensor(batch_repr))
            batch_repr = batch_repr
    else:
        raise ValueError("Invalid node type {}. Currently, chemical and protein nodes are supported.".format(node_type))

    if isinstance(batch_repr, Variable) and torch.cuda.is_available():
        batch_repr = batch_repr.cuda()

    batch_embedding = model(node_type, batch_repr)
    return batch_embedding

def load_dict(path):
    """ Load a dictionary and a corresponding reverse dictionary from the given file
    where line number (0-indexed) is key and line string is value. """
    retdict = list()
    rev_retdict = dict()
    with open(path) as fin:
        for idx, line in enumerate(fin):
            text = line.strip()
            retdict.append(text)
            rev_retdict[text] = idx
    return retdict, rev_retdict

def load_repr(path, config, node_list):
    """ Load the representations of each node in the `node_list` given
    the representation type and configurations.

    Args:
        path: Path of the graph data directory
        config: Node configuration JSON object
        node_list: The list of nodes for which to load representations

    Returns:
        repr_info: A dictionary that contains representation information
        node_list: List of nodes with loaded representations, the change
        is in-place though.
    """
    repr_type = config['representation']
    if repr_type == TYPE_MOLECULE:
        return load_molecule_repr(path, config, node_list)
    elif repr_type == TYPE_SEQUENCE_PSSM:
        return load_pssm_repr(path, config, node_list)
    else:
        raise ValueError("{0} Node type not supported!".format(repr_type))

def load_molecule_repr(path, config, node_list):
    import deepnet.fingerprint.features as fp_feature
    graph_vocab_path = os.path.join(path, config['graph_path'])
    graph_list, _ = load_dict(graph_vocab_path)
    for node, graph in zip(node_list, graph_list):
        node.set_data(graph)
    info = dict(embedding_type=TYPE_MOLECULE,
                atom_size=fp_feature.num_atom_features(),
                bond_size=fp_feature.num_bond_features())
    return info, node_list

def load_pssm_repr(path, config, node_list):
    seq_path = os.path.join(path, config['pssm_path'])
    info = dict(embedding_type=TYPE_SEQUENCE_PSSM,
            max_len=SEQ_MAX_LEN)
    return info, node_list

if __name__=='__main__':
    uniprot2pssm=load_uniprot2pssm()
    #O43683
    #O60285
    #O60674(JH1domain-catalytic)
    #O60674(JH2domain-pseudokinase)
    print(uniprot2pssm['O60674(JH1domain-catalytic)'])
    print(uniprot2pssm['O60674(JH2domain-pseudokinase)'])


