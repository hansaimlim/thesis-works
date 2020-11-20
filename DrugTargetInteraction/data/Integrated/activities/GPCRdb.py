import os
import re
import sys
import csv
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
from Idmap import load_chembl2uniprot, load_chembl2ikey
from utils import decompress_gzip, compress_gzip, pandas_df_continuous

def get_gpcrdb(assay_type='pKd',dataframe=True):
  #if dataframe=True, a pandas dataframe is returned
  #dataframe has integer indice from 0,
  # column names are ['InChIKey','UniProt','Activity_type','Relation','Activity_value']
  fpath='../../GPCRdb/'
  gpcrfile=fpath+'GPCR_assays.csv.gz'
  if Path(gpcrfile).is_file(): #decompress if compressed
    gpcrfile=decompress_gzip(gpcrfile)
  else:
    gpcrfile=fpath+'GPCR_assays.csv' #already decompressed
    
  chembl2uniprot=load_chembl2uniprot()
  chembl2ikey=load_chembl2ikey()
  
  data=[]
  with open(gpcrfile,'r') as f:
    #activity types : ['AC50', 'Potency', 'IC50', 'EC50', 'Kd', 'Ki']
    #activity units : ['nM']
    next(f)
    for l in  csv.reader(f, quotechar='"', delimiter=',',quoting=csv.QUOTE_ALL, skipinitialspace=True):
      smi=l[8]
      c_chembl=l[14] #chembl ID for chemical molecule
      rel=l[25];atype=l[26];unit=l[27];val=l[28]
      atype='p'+atype #e.g. IC50 -> pIC50
      if atype.lower() != assay_type.lower():
        continue
      p_chembl=l[29]
      try:
        ikey=chembl2ikey[c_chembl]
      except:
        continue
      try:
        uni=chembl2uniprot[p_chembl]
      except:
        continue
      try:
        val=np.float(val)
      except:
        continue
      val=-np.log10(val)+9.0 #all activities are in nM
      if np.isinf(val) or np.isnan(val):
        continue #skip inf or NaN
      if rel!='=': #need to flip sign for -log conversion unless '='
        lt=re.search(r'<',rel)
        gt=re.search(r'>',rel)
        if lt:
          if gt: #relation shouldn't contain both > and <
            continue
          rel=rel.replace('<','>')
        elif gt:
          rel=rel.replace('>','<')
        else:
          continue
      tup=(ikey,uni,atype,rel,val)
      data.append(tup)
      
  if dataframe:
    data=pandas_df_continuous(data)
  return data

if __name__=='__main__':
  gpcr_data=get_gpcrdb(dataframe=True)
  print(gpcr_data)
