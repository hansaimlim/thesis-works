import os
import sys
import pandas as pd
import numpy as np
from utils import pandas_df_continuous, pandas_df_binary

def get_pubchem_cyp450(dataframe=True):
  #if dataframe=True, a pandas dataframe is returned
  #dataframe has integer indice from 0,
  # column names are ['InChIKey','UniProt','Activity']
  fpath='../../CYP450/PubChem/'
  protmap=fpath+'cyp450_id_mapping.tsv'
  target2uniprot={}
  cid2ikey={}
  with open(protmap,'r') as f:
    for line in f:
      line=line.strip().split('\t')
      target=line[0]
      uni=line[2]
      target2uniprot[target]=uni
  chemmap=fpath+'pubchem_cyp450_cid_to_ikey.txt'
  with open(chemmap,'r') as f:
    for line in f:
      line=line.strip().split('\t')
      cid=int(line[0])
      ikey=str(line[1])
      cid2ikey[cid]=ikey
  targets=['cyp1a2','cyp2c19','cyp2c9','cyp2d6','cyp3a4']
  data=[]
  
  for target in targets:
    activefile=fpath+target+'/active.csv'
    inactivefile=fpath+target+'/inactive.csv'
    uni=target2uniprot[target]
    with open(activefile,'r') as f:
      next(f)
      for line in f:
        line=line.strip().split(',')
        try:
          cid=int(line[2])
        except:
          continue
        ikey=cid2ikey[cid]
        tup=(ikey,uni,'Active')
        data.append(tup)
    with open(inactivefile,'r') as f:
      next(f)
      for line in f:
        line=line.strip().split(',')
        try:
          cid=int(line[2])
        except:
          continue
        ikey=cid2ikey[cid]
        tup=(ikey,uni,'Inactive')
        data.append(tup)
  if dataframe:
    data=pandas_df_binary(data)
  return data

if __name__=='__main__':
  binary_cyp450_data=get_pubchem_cyp450(dataframe=True)
  print(binary_cyp450_data)
