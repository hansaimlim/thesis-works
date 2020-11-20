import os
import re
import sys
import json
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Seq import MutableSeq
from Bio.Alphabet import IUPAC
import networkx as nx
##### JSON modules #####
class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)
def save_json(data,filename):
  with open(filename, 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=4, cls=NumpyEncoder)
def load_json(filename):
  with open(filename, 'r') as fp:
    data = json.load(fp)
  return data
##### JSON modules #####
def load_aa_dict(by='sidechain'):
    if by =='sidechain': #class by sidechain property
        return {"A":0,"I":0,"L":0,"M":0,"V":0,"J":0, #aliphatic
            "F":1,"W":1,"Y":1, #aromatic
            "N":2,"C":2,"Q":2,"S":2,"T":2,"U":2, #polar neutral
            "D":3,"E":3, #charged, acidic
            "R":4,"H":4,"K":4, #charged, basic
            "G":5,"P":5,  #unique side chain
            "X":6, "-":6 #all others, including gaps
            }
    else:
        return {"A":0,"R":1,"N":2,"D":3,
            "C":4,"Q":5,"E":6,"G":7,
            "H":8,"I":9,"L":10,"K":11,
            "M":12,"F":13,"P":14,"S":15,
            "T":16,"W":17,"Y":18,"V":19,
            "Z":20,"B":21,"J":22,"X":23,"U":24,"-":25}
def load_triplet_idxs():
    """
    Load 1-based amino acid residue indice for triplets
    Triplets are defined from 3D structure of P11309
    Triplets based on the alignment of kinase binding site sequences
    mutation_1_based_indices
    [(1,2,12),(1,12,13),(2,3,4),(2,3,11),(2,3,12),(2,4,11),(2,11,12),
    (3,4,11),(3,11,12),(5,6,9),(5,9,10),(6,7,8),(6,7,9),(6,8,9),(7,8,9),
    (10,11,16),(11,12,15),(11,15,16),(12,13,14),(12,14,15),(14,15,19),
    (14,19,20),(14,20,21),(14,21,22),(15,16,18),(15,18,19),(26,27,35),
    (26,27,36),(26,35,36),(27,34,35),(27,35,36),(30,31,32),(30,31,33),
    (30,32,33),(31,32,33),(31,32,34),(31,33,34),(32,33,34),(38,39,40),(39,40,41)]
1                     12345678911111111112222222222333333333344
2                              01234567890123456789012345678901
                      
PIM1          ,P11309,PLLGSGGFGSVYSAIKILILERPEPQDDIKDENILIIDFGS
                      PL---------Y----------------------------- (1,2,12)
                      P----------YS---------------------------- (1,12,13)
                      -LLG------------------------------------- (2,3,4)
                      -LL-------V------------------------------ (2,3,11)
                      -LL--------Y----------------------------- (2,3,12)
                      -L-G------V------------------------------ (2,4,11)
                      -L--------VY----------------------------- (2,11,12)
                      --LG------V------------------------------ (3,4,11)
                      --L-------VY----------------------------- (3,11,12)
                      ----SG--G-------------------------------- (5,6,9)
                      ----S---GS------------------------------- (5,9,10)
                      -----GGF--------------------------------- (6,7,8)
                      -----GG-G-------------------------------- (6,7,9)
                      -----G-FG-------------------------------- (6,8,9)
                      ------GFG-------------------------------- (7,8,9)
                      ---------SV----K------------------------- (10,11,16)
                      ----------VY--I-------------------------- (11,12,15)
                      ----------V---IK------------------------- (11,15,16)
                      -----------YSA--------------------------- (12,13,14)
                      -----------Y-AI-------------------------- (12,14,15)
                      -------------AI---I---------------------- (14,15,19)
                      -------------A----IL--------------------- (14,19,20)
                      -------------A-----LE-------------------- (14,20,21)
                      -------------A------ER------------------- (14,21,22)
                      --------------IK-L----------------------- (15,16,18)
                      --------------I--LI---------------------- (15,18,19)
                      -------------------------QD-------L------ (26,27,35)
                      -------------------------QD--------I----- (26,27,36)
                      -------------------------Q--------LI----- (26,35,36)
                      --------------------------D------IL------ (27,34,35)
                      --------------------------D-------LI----- (27,35,36)
                      -----------------------------KDE--------- (30,31,32)
                      -----------------------------KD-N-------- (30,31,33)
                      -----------------------------K-EN-------- (30,32,33)
                      ------------------------------DEN-------- (31,32,33)
                      ------------------------------DE-I------- (31,32,34)
                      ------------------------------D-NI------- (31,33,34)
                      -------------------------------ENI------- (32,33,34)
                      -------------------------------------DFG- (38,39,40)
                      --------------------------------------FGS (39,40,41)
    """
    indices=[(1,2,12),(1,12,13),(2,3,4),(2,3,11),(2,3,12),(2,4,11),(2,11,12),
        (3,4,11),(3,11,12),(5,6,9),(5,9,10),(6,7,8),(6,7,9),(6,8,9),(7,8,9),
        (10,11,16),(11,12,15),(11,15,16),(12,13,14),(12,14,15),(14,15,19),
        (14,19,20),(14,20,21),(14,21,22),(15,16,18),(15,18,19),(26,27,35),
        (26,27,36),(26,35,36),(27,34,35),(27,35,36),(30,31,32),(30,31,33),
        (30,32,33),(31,32,33),(31,32,34),(31,33,34),(32,33,34),(38,39,40),(39,40,41)]
    return indices

def load_kinase_binding_site_pssm(normalize=True):
    """
    Load PSSM values for kinase binding sites
    If normalize=True, global normalization is applied to each values
    Global normalization means to use every single PSSM value equally,
    globalsum=sum(every_single_pssm)
    globalmean=mean(every_single_pssm)

    return: list of dicts of uniprot->pssm_binding_site
    """
    with open('./kinase_bsite_pssm.json') as f:
        pssm=json.load(f)
    all_values=[]
    for uniprot in pssm:
        ps=pssm[uniprot]
        for p in ps:
            for val in p:
                all_values.append(val)
        pssm[uniprot]=np.array(ps,dtype=np.float32)
            
    gmean=np.nanmean(all_values)
    gstd=np.nanstd(all_values)
    if normalize:
        for uniprot in pssm:
            ps=pssm[uniprot]
            ps_normalized=[]
            for p in ps:
                p=(p-gmean)/gstd
                ps_normalized.append(p)
            pssm[uniprot]=np.array(ps_normalized,dtype=np.float32)
    return pssm
def load_aaidx(normalize=True):
    """Amino acid index from multiple references
       each index table is dict("one_letter_amino_acid")->float_value
       (e.g. hpob['K'] -> hydrophobicity value for Lysine)
       each index table contains reference
       (e.g. hpob['reference'] for reference info)
    """
    with open('./aaindex.json') as f:
        aaidx=json.load(f)
    hpob=aaidx["average_hydrophobicity"]
    vdw=aaidx["van_der_Waals_volume"]
    pol=aaidx["polarity"]
    charge=aaidx["net_charge"]
    vol_buried=aaidx["average_buried_volume"]
    asa_tri=aaidx["accessible_surface_area_tripeptide"]
    asa_folded=aaidx["accessible_surface_area_folded"]

    aaidxs=[hpob,vdw,pol,vol_buried,asa_tri,asa_folded]
    if normalize:
        #z-scale normalize: subtract mean, divide by stdev
        #do NOT normalize charge values
        for table in aaidxs:
            vals=[]
            for aa in table:
                if aa=='reference':
                    continue
                else:
                    vals.append(table[aa])
            mean=np.nanmean(vals)
            std=np.std(vals)
            for aa in table:
                if aa=='reference':
                    continue
                else:
                    table[aa]=(table[aa]-mean)/std #z-score
    return hpob,vdw,pol,charge,vol_buried,asa_tri,asa_folded
def load_kinase_binding_site_seq():
    uniprot2seq={}
    with open('./kinase_binding_site_residues.tsv') as f:
        for line in f:
            line=line.strip().split('\t')
            uni=line[0]
            seq=line[1]
            uniprot2seq[uni]=seq
    return uniprot2seq

def build_kinase_graph():
    #load z-scores for each amino acid index
    hydrophobicity,vdw,polarity,charge,vol_buried,asa_tri,asa_folded=load_aaidx(normalize=True)
    aa_label=load_aa_dict(by='sidechain') #aa label by sidechain type
    uniprot2pssm=load_uniprot2pssm() #uniprot->{'pssm':pssm, 'seq':seq} for domains
    uniprot2seq=load_kinase_binding_site_seq()
    uniprot2bindingsitepssm=load_kinase_binding_site_pssm(normalize=True) 
    uniprot2class=load_kinase_ec()
    triplets=load_triplet_idxs() #tuples of 1-based triplet indice


    kl=open('kinase_binding_site_graph_id.tsv','w')
    adjmat=open('kinase_binding_site_A.txt','w')
    gi=open('kinase_binding_site_graph_indicator.txt','w')
    gl=open('kinase_binding_site_graph_labels.txt','w')
    nl=open('kinase_binding_site_node_label.txt','w')
    nr=open('kinase_binding_site_node_attributes.txt','w')    
    nr_pssm=open('kinase_binding_site_node_attributes_pssm.txt','w')    
    node_idx=1 #1-based index
    edge_idx=1
    uniprots=list(uniprot2pssm.keys())
    #without_bsite=['P42345','Q9UBF8','O00750','O75747','Q8NEB9','P42336','P42338','O00329','P48736','Q9Y2I7',
    #'P78356','Q8TBX8','Q99755','O60331','Q9BRS2','Q9BVS4','O14730']
    #for u in without_bsite:
    #    uniprots.remove(u)
    spmat=[] #shortest-path matrix for each graph
    for protidx in range(len(uniprots)): #for each graph
        uniprot=uniprots[protidx]
        protdict=uniprot2pssm[uniprot]
        kl.write("{}\t{}\n".format(protidx+1,uniprot))
        if uniprot=='O75582' or uniprot=='O75582(Kin.Dom.1)':
            uniprot='O75582(Kin.Dom.1-N-terminal)' #force to use the first kinase domain for ambiguous labels
        elif uniprot=='O75582(Kin.Dom.2)':
            uniprot='O75582(Kin.Dom.2-C-terminal)' #use proper domain sequence
        elif uniprot=='O75676(Kin.Dom.1)':
            uniprot='O75676(Kin.Dom.1-N-terminal)'
        elif uniprot=='O75676(Kin.Dom.2)':
            uniprot='O75676(Kin.Dom.2-C-terminal)'
        elif uniprot=='P23458':
            uniprot='P23458(JH1domain-catalytic)'
        elif uniprot=='P29597':
            uniprot='P29597(JH1domain-catalytic)'
        elif uniprot=='P51812(Kin.Dom.1)':
            uniprot='P51812(Kin.Dom.1-N-terminal)'
        elif uniprot=='P51812(Kin.Dom.2)':
            uniprot='P51812(Kin.Dom.2-C-terminal)'
        elif uniprot=='Q15349(Kin.Dom.1)':
            uniprot='Q15349(Kin.Dom.1-N-terminal)'
        elif uniprot=='Q15349(Kin.Dom.2)':
            uniprot='Q15349(Kin.Dom.2-C-terminal)'
        elif uniprot=='Q15418' or uniprot=='Q15418(Kin.Dom.1)':
            uniprot='Q15418(Kin.Dom.1-N-terminal)'
        elif uniprot=='Q15418(Kin.Dom.2)':
            uniprot='Q15418(Kin.Dom.2-C-terminal)'
        elif uniprot=='Q9P2K8(Kin.Dom.2,S808G)' or uniprot=='Q9P2K8':
            uniprot='Q9P2K8(Kin.Dom.2-C-terminal,S808G)'
        elif uniprot=='Q9UK32' or uniprot=='Q9UK32(Kin.Dom.1)':
            uniprot='Q9UK32(Kin.Dom.1-N-terminal)'
        elif uniprot=='Q9UK32(Kin.Dom.2)':
            uniprot='Q9UK32(Kin.Dom.2-C-terminal)'
        pssm=uniprot2bindingsitepssm[uniprot]
        seq=uniprot2seq[uniprot]

        uni_match=re.match(r'([A-Z0-9]{6})(.*)',uniprot,re.M)
        try:
            prot_class=uniprot2class[uni_match.group(1)]
        except:
            prot_class=0
            print("protein {} class not found. set to class 0; unknown class".format(uniprot))
        gl.write("{}\n".format(prot_class))
        mol_nodes=[] #list of node_idx in a graph; idx is global idx
        G=nx.Graph() #protein graph object
        nodes_within_graph=[] #list of node_idx in a graph; restart idx for each new graph
        edges_within_graph=[] #list of (node_i,node_j) in a graph; to skip redundant edges
        for aaidx,aa in enumerate(seq): #for each node
            G.add_node(aaidx+1) #networkx takes 1-based indice
            nodes_within_graph.append(aaidx+1)
            aalab=aa_label[aa]
            hyp=hydrophobicity[aa]
            vdw_=vdw[aa]
            pol=polarity[aa]
            chg=charge[aa]
            vol_bur=vol_buried[aa]
            asatri=asa_tri[aa]
            asafold=asa_folded[aa]
            nr.write("{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n"
                .format(hyp,vdw_,pol,chg,vol_bur,asatri,asafold))
            mol_nodes.append(node_idx)
            gi.write("{}\n".format(protidx+1)) #current node belongs to protidx protein (1-based index)
            nl.write("{}\n".format(aalab)) #amino acid type label
            ps=','.join([str(a1) for a1 in pssm[aaidx,:]]) #pssm for current node in string format
            nr_pssm.write("{}\n".format(ps))
            node_idx+=1 #update to next node idx
        for _,triplet in enumerate(triplets): #indices in triplet are 1-based
            #for each triplet
            for src_i,src in enumerate(triplet):
                for tgt in triplet[src_i+1:]:
                    forward_edge=(src,tgt)
                    reverse_edge=(tgt,src)
                    if forward_edge not in edges_within_graph:
                        adjmat.write("{},{}\n".format(mol_nodes[src-1],mol_nodes[tgt-1])) #forward edge
                        G.add_edge(src,tgt)
                        edges_within_graph.append(forward_edge)
                        edge_idx+=1
                    if reverse_edge not in edges_within_graph:
                        adjmat.write("{},{}\n".format(mol_nodes[tgt-1],mol_nodes[src-1])) #reverse edge
                        G.add_edge(tgt,src)
                        edges_within_graph.append(reverse_edge)
                        edge_idx+=1
        spvec=np.zeros((len(nodes_within_graph),len(nodes_within_graph)),dtype=np.float32)
        for _,nodei in enumerate(nodes_within_graph): #nodes_within_graph are 1-based indice
            for _,nodej in enumerate(nodes_within_graph):
                if nodei>=nodej: #only on upper triangle
                    continue
                else:
                    try:
                        splen=len(nx.shortest_path(G,nodei,nodej))-1 #networkx graph takes 1-based indice
                    except:
                        splen=-1 #disconnected nodes due to spatial alignment
                    spvec[nodei-1,nodej-1]=splen #numpy arrays are 0-based
                    spvec[nodej-1,nodei-1]=splen
        spmat.append(spvec)
    kl.close()
    adjmat.close()
    gi.close()
    gl.close()
    nl.close()
    nr.close()
    nr_pssm.close()
    np.save('kinase_binding_site_spfile.npy',spmat)
    print("Total {} nodes, {} edges for {} graphs.".format(node_idx+1,edge_idx+1,protidx+1))
        
def load_kinase_ec():
    """load kinase E.C. classes
        Kinases belong to E.C.2.7.xxx
        Use xxx to classify kinases
        '0' indicates unknown class
    """
    protfile='../../../../data/Integrated/proteins/kinase_enzyme_classes.tsv'
    uniprot2class={}
    with open(protfile,'r') as f:
        next(f)
        for line in f:
            line=line.strip().split('\t')
            uni=line[0]
            ec=line[-1].strip().split(';')

            matchobj=re.match(r'2\.7\.(\d+)\.(.*)',ec[0],re.M)
            if matchobj:
                subclass=matchobj.group(1)
                if subclass==7: #merge 2.7.7 with 2.7.1
                    subclass=1
                uniprot2class[uni]=subclass
            else:
                uniprot2class[uni]=0
    return uniprot2class

def load_uniprot2pssm():
    
    base_path='../../../../data/kinome_assay/sequence_feature/'
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
	    'Q9UK32' :'Q9UK32_Kin.Dom.2-C-terminal.dat',
        'Q8NI60' :'Q8NI60.dat',
        'Q8IY84' :'Q8IY84.dat',
        'Q96D53' :'Q96D53.dat'}
    with open(protfile,'r') as f:
        protidx=0 #number of protein
        n_edges=0 #number of aa-aa connections
        n_nodes=0 #number of amino acid residues in total
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
                seq=[]
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
                            seq.append(str(line[1]))
                            res_vector=np.array(line[2:22],dtype=np.float32)
                            pssm.append(res_vector)
                pssm=np.array(pssm,dtype=np.float32)
                #pssm=np.pad(pssm,((0,max_len-pssm.shape[0]),(0,0)),'constant',constant_values=padding) #pad to the bottom
                uniprot2pssm[uniprot]={'pssm':pssm,'seq':seq}
                #print("PSSM shape {} loaded for {} from file {}".format(uniprot2pssm[uniprot].shape,uniprot,fname))
    return uniprot2pssm
def parse_pssm(pssm_file,binding_residue_idxs):
    #expect residue idxs start from 1, not 0
    #expect pssm_file in plain text -out_ascii_pssm from psi-blast
    idx2feat={}
    with open(pssm_file,'r') as inf:
        next(inf) #first line is empty
        next(inf) #second line is annotation
        next(inf) #third line is AA header
        for line in inf:
            if line=='':
                continue
            line=line.strip().split()
            try:
                idx=int(line[0])
            except:
                continue
            feat=line[2:22]
            idx2feat[idx]=feat
    features=[]
    for idx in binding_residue_idxs:
        if idx <=0: #gap indicator
            features.append(np.zeros((20,)))
        else:
            features.append(np.array(idx2feat[idx],dtype=np.float32))
    return np.array(features,dtype=np.float32)
def get_binding_site_idx():
    bres_idxs_withgaps=[]
    for seq_record in SeqIO.parse('binding_residues2.txt',"fasta"):
        seqid=seq_record.id
        sequence=seq_record.seq
        for idx, residue in enumerate(sequence):
            if residue=='*':
                bres_idxs_withgaps.append(idx)
    return bres_idxs_withgaps

def get_binding_site_idx_nogap(aligned_seq,bres_idxs_withgaps):
    #N-residue for kinase domain (usually ~250 AAs)
    #aligned sequence for binding residues
    bres_idxs_within_domain=[] #idxs without gaps; positions within the domain
    for _,bidx in enumerate(bres_idxs_withgaps):
        if aligned_seq[bidx]=='-': #gap
            bres_idxs_within_domain.append(-1) #indicator for gap; 0-vector pssm
        else:
            new_idx=bidx-aligned_seq[:bidx].count('-') #idx within seq, assuming no gaps
            bres_idxs_within_domain.append(new_idx)
    return bres_idxs_within_domain

def update_pssm_bsite():
    to_update=['Q8NI60','Q8IY84','Q96D53']
    basepath='../../../../data/kinome_assay/sequence_feature/kinase_domain_pssm_uniref50/'
    uniprot2seq={}
    bsite_idxs_withgaps=get_binding_site_idx()
    for seq_record in SeqIO.parse('aln_fasta2.txt',"fasta"):
        bres_idxs_withgaps=[]
        seqid=seq_record.id.strip().split('|')[1].upper()
        sequence=seq_record.seq
        uniprot2seq[seqid]=sequence
    uniprot2bindingsitepssm=load_kinase_binding_site_pssm(normalize=False)
    for uni in to_update:
        pssmfile=basepath+uni+'.dat'
        seq=uniprot2seq[uni]
        bsite_idxs_without_gaps=get_binding_site_idx_nogap(seq,bsite_idxs_withgaps)
        bsite_idxs_without_gaps_1=[bidx+1 for bidx in bsite_idxs_without_gaps] #0-based to 1-based idx
        pssm=parse_pssm(pssmfile,bsite_idxs_without_gaps_1)
        uniprot2bindingsitepssm[uni]=pssm
    save_json(uniprot2bindingsitepssm,'kinase_bsite_pssm_v2.json')


if __name__=='__main__':
    #update_pssm_bsite() #fill in missing kinase pssms
    build_kinase_graph()
    #uniprot2pssm=load_uniprot2pssm() #uniprot->{'pssm':pssm, 'seq':seq} for domains
    #uniprot2seq=load_kinase_binding_site_seq()
    #uniprot2bindingsitepssm=load_kinase_binding_site_pssm(normalize=True) 
    #uniprot2class=load_kinase_ec()
    #triplets=load_triplet_idxs()
    #for uniprot in uniprot2bindingsitepssm:
    #    if uniprot in uniprot2pssm:
    #        continue
    #    else:
    #        print("Protein id {} not properly mapped. ID not found in kinase_bsite_pssm.json".format(uniprot))
        