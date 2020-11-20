import os
import re
import sys
import json
import numpy as np

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
def build_kinase_graph():
    #load z-scores for each amino acid index
    hydrophobicity,vdw,polarity,charge,vol_buried,asa_tri,asa_folded=load_aaidx(normalize=True)
    uniprot2pssm=load_uniprot2pssm()
    uniprot2class=load_kinase_ec()

    adjmat=open('kinase_A.txt','w')
    gi=open('kinase_graph_indicator.txt','w')
    gl=open('kinase_graph_label.txt','w')
    nl=open('kinase_node_label.txt','w')
    nr=open('kinase_node_attribute.txt','w')    
    nr_pssm=open('kinase_node_attribute_pssm.txt','w')    
    node_idx=0
    edge_idx=0
    uniprots=list(uniprot2pssm.keys())
    for protidx in range(len(uniprots)): #for each graph
        uniprot=uniprots[protidx]
        protdict=uniprot2pssm[uniprot]
        pssm=protdict['pssm']
        seq=protdict['seq']
        uni_match=re.match(r'([A-Z0-9]{6})(.*)',uniprot,re.M)
        try:
            prot_class=uniprot2class[uni_match.group(1)]
        except:
            prot_class=0
            print("protein {} class not found. set to class 0; unknown class".format(uniprot))
        gl.write("{}\n".format(prot_class))
        mol_nodes=[] #list of node_idx in a graph
        for aaidx,aa in enumerate(seq): #for each node
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
            gi.write("{}\n".format(protidx)) #current node belongs to protidx protein
            ps=','.join([str(a1) for a1 in pssm[aaidx,:]]) #pssm for current node in string format
            nr_pssm.write("{}\n".format(ps))
            node_idx+=1 #update to next node idx
        for _,tup in enumerate(list(zip(mol_nodes[:-1],mol_nodes[1:]))):
            #for each edge
            adjmat.write("{},{}\n".format(tup[0],tup[1])) #forward edge
            edge_idx+=1
            adjmat.write("{},{}\n".format(tup[1],tup[0])) #reverse edge
            edge_idx+=1

    adjmat.close()
    gi.close()
    gl.close()
    nl.close()
    nr.close()
    nr_pssm.close()
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
            ec=line[-1]
            matchobj=re.match(r'2\.7\.(\d+)\.(.*)',ec,re.M)
            if matchobj:
                subclass=matchobj.group(1)
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
	    'Q9UK32' :'Q9UK32_Kin.Dom.2-C-terminal.dat'}
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

if __name__=='__main__':
#    uniprot2class=load_kinase_ec()
#    classcount={}
#    for uni in uniprot2class:
#        cla=uniprot2class[uni]
#        if cla in classcount:
#            classcount[cla]+=1
#        else:
#            classcount[cla]=1
#    print("Class: Count")
#    for cla in classcount:
#        print("{}: {}".format(cla,classcount[cla]))
    
    build_kinase_graph()