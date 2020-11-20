import os
import sys
from utils import pandas_df_binary
def get_drugbank(dataframe=True):
  #if dataframe=True, a pandas dataframe is returned
  #dataframe has integer indice from 0,
  # column names are ['InChIKey','UniProt','Activity']
  fpath='../../DrugBank/'
  assay=fpath+'drugbank_5.0.10_drug_all_targets.txt'
  data=[]
  with open(assay,'r') as f:
    for line in f:
      line=line.strip().split('\t')
      ikey=line[0]
      uni=line[1]
      tup=(ikey,uni,'Active')
      data.append(tup)
  if dataframe:
    data=pandas_df_binary(data)
  return data

if __name__=='__main__':
  drugbank=get_drugbank(dataframe=True)
  print(drugbank)
