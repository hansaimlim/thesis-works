import os
import re
import sys
import json
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
def remove_modifications(gene):
  #remove modifications from gene name
  #modifications include phosphorylation and autoinhibition
  gene=str(gene)
  gene=gene.replace('-nonphosphorylated','')
  gene=gene.replace('-phosphorylated','')
  gene=gene.replace('-autoinhibited','')
  return gene
def mutant_to_filename(gene):
  #convert gene names with mutations into filename format
  # e.g. Q15349(Kin.Dom.1-N-terminal)  ->  Q15349_Kin.Dom.1-N-terminal
  #return the input genename if no () found
  gene=str(gene)
  matchobj=re.match(r'(.*)\((.*)\)',gene,re.I|re.M)
  if matchobj:
    g=matchobj.group(1)
    m=matchobj.group(2)
    gene=g+'_'+m
    return gene
  else: #no mutation info in the gene name
    return gene
def add_kindom(gene,kd='Kin.Dom'):
  gene=str(gene)
  gene=gene+'_'+kd
  return gene
def remove_kindom(gene):
  #return gene name with kindom info separated
  #GENE1-Kin.Dom.2-C-terminal -> (GENE1, 2)
  #GENE2-Kin.Dom.1-N-terminal -> (GENE2, 1)
  #GENE5-Kin.Dom -> (GENE5, 3)
  #Gene7 -> (Gene7, 0)
  gene=str(gene)
  kd=re.match(r'(.*)(\W?Kin\.Dom)',gene,re.M|re.I)
  kd1=re.match(r'(.*)(\W?Kin\.Dom\.1-N-terminal)',gene,re.M|re.I)
  kd2=re.match(r'(.*)(\W?Kin\.Dom\.2-C-terminal)',gene,re.M|re.I)
  kd3=re.match(r'(.*)(\W?JH1domain-catalytic)',gene,re.M|re.I)
  kd4=re.match(r'(.*)(\W?JH2domain-pseudokinase)',gene,re.M|re.I)
  if kd1: #Kindom 1
    gene=kd1.group(1).replace(')','').replace('(','').replace('_','')
    k=kd1.group(2)
  elif kd2: #Kindom 2
    gene=kd2.group(1).replace(')','').replace('(','').replace('_','')
    k=kd2.group(2)
  elif kd: #Kindom-single
    gene=kd.group(1).replace(')','').replace('(','').replace('_','')
    k=kd.group(2)
  elif kd3: #Kindom jh1
    gene=kd3.group(1).replace(')','').replace('(','').replace('_','')
    k=kd3.group(2)
  elif kd4: #Kindom jh2
    gene=kd4.group(1).replace(')','').replace('(','').replace('_','')
    k=kd4.group(2)
  else:
    return gene,0 #return original genename, no kindom info found (0)
  return gene,k

uni2idx={}
with open('integrated_protein_list.tsv','r') as f:
  next(f)
  for line in f:
    line=line.strip().split('\t')
    idx=int(line[0])
    uni=str(line[1])
    uniref=str(line[2])
    mod=str(line[3])
    uni2idx[uni]=idx
    if uni=='O60674_V617F_JH2domain-pseudokinase':
      uni2idx['O60674(V617F,JH2domain-pseudokinase)']=idx
    if uni=='Q9P2K8_S808G_Kin.Dom.2-C-terminal':
      uni2idx['Q9P2K8(S808G,Kin.Dom.2-C-terminal)']=idx
    if uni=='Q9P2K8_S808G_Kin.Dom.1-N-terminal':
      uni2idx['Q9P2K8(S808G,Kin.Dom.1-N-terminal)']=idx

    

ikey2idx=load_json('integrated_InChIKey2Index.json')

active='integrated_active.tsv'
inactive='integrated_inactive.tsv'
#with open(active,'r') as f:
with open(inactive,'r') as f:
  for line in f:
    line=line.strip().split('\t')
    ikey=line[0]
    try:
      chemidx=ikey2idx[ikey]
    except:
      chemidx=None
      continue
     # with open('ikey_not_found.txt','a') as out:
     #   out.write("{}\n".format(ikey))
    uni=line[1]
    if uni=='P0DMV9,P0DMV8':
      pidx1=uni2idx['P0DMV8']
      pidx2=uni2idx['P0DMV9']
      print("{},{}".format(chemidx,pidx1))
      print("{},{}".format(chemidx,pidx2))
      continue
    if uni=='O95819,E7ESS2':
      pidx=uni2idx['O95819']
      print("{},{}".format(chemidx,pidx))
      continue
    if uni=='Q0GMK6-P24385' or uni=='Q0GMK6-P30281':
      pidx=uni2idx['Q0GMK6']
      print("{},{}".format(chemidx,pidx))
      continue
    if uni=='Q15661,P20231':
      pidx=uni2idx['Q15661']
      print("{},{}".format(chemidx,pidx))
      continue
    
    uni=remove_modifications(uni)
    uni,kd=remove_kindom(uni)
    uni=mutant_to_filename(uni)
    try:
      protidx=uni2idx[uni]
    except:
      try:
        uni=uni+'_Kin.Dom'
        protidx=uni2idx[uni]
      except:
        with open('prot_not_found.txt','a') as out:
          out.write("{}\t{}\t{}\n".format(uni,line[1],kd))
    print("{},{}".format(chemidx,protidx))
    












