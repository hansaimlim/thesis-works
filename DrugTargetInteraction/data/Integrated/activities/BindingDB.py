import os
import sys
import pandas as pd
import numpy as np
from utils import pandas_df_continuous

def get_bindingdb_by_assay_type(assay_type='pKd',dataframe=True):
  bindingdb_path='../../BindingDB/'
  pic50=bindingdb_path+'BindingDB_pIC50.tsv'
  pkd=bindingdb_path+'BindingDB_pKd.tsv'
  pki=bindingdb_path+'BindingDB_pKi.tsv'
  if (assay_type=='pIC50') or (assay_type=='pic50'):
    infile=pic50
    atype='pIC50'
  elif (assay_type=='pKd') or (assay_type=='pkd'):
    infile=pkd
    atype='pKd'
  elif (assay_type=='pKi') or (assay_type=='pki'):
    infile=pki
    atype='pKi'
  else:
    print("Error in parsing BindingDB data. Choose a proper assay type (pIC50, pKd, or pKi)")
    sys.exit()
  data=[]
  with open(infile,'r') as f:
    for line in f:
      line=line.strip().split('\t')
      ikey=str(line[0])
      uni=str(line[1])
      rel=line[2]
      val=float(line[3])
      tup=(ikey,uni,atype,rel,val)
      data.append(tup)
  if dataframe:
    data=pandas_df_continuous(data)
  return data

if __name__=='__main__':
  pkd_data=get_bindingdb_by_assay_type(assay_type='pIC50')
  print(pkd_data)
