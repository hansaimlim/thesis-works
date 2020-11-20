import os
import re
import sys
import csv
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
from Idmap import get_protein_idmap, load_chemname2ikey, load_chembl2ikey
from utils import decompress_gzip, compress_gzip, pandas_df_continuous, pandas_df_binary

def get_jcim_pki(dataframe=True):
  jcim='../../kinome_assay/other_published/JCIM_activity_pKi.tsv'
  gene2uniprot,uniprot2genes,mutgene2mutuniprot,mutgene2filename=get_protein_idmap()
  chemname2ikey=load_chemname2ikey()
  data=[]
  with open(jcim,'r') as f:
    next(f)
    for line in f:
      line_ori=line
      line=line.strip().split('\t')
      org=line[0]
      gene=str(line[1]).replace(' ','') #genes in JCIM may contain white spaces
      comp=str(line[2]) #compound name
      pki=float(line[3]) #pKi in Ki (M)
      try:
        uni=gene2uniprot[gene]
      except:
      #  with open('./extra_genes_jcim.txt','a') as out:
      #    out.write("{}\n".format(gene))
        continue
      try:
        ikey=chemname2ikey[comp]
      except:
        with open('./extra_chemicals_jcim.txt','a') as out:
          out.write("{}\n".format(comp))
        continue
      tup=(ikey,uni,'pKi','=',pki)
      data.append(tup)
  if dataframe:
    data=pandas_df_continuous(data)
  return data

def get_pkis_pki(dataframe=True):
  pkis='../../kinome_assay/other_published/pkis_pki.tsv'
  gene2uniprot,uniprot2genes,mutgene2mutuniprot,mutgene2filename=get_protein_idmap()
  chembl2ikey=load_chembl2ikey()
  data=[]
  with open(pkis,'r') as f:
    for line in f:
      line=line.strip().split('\t')
      chembl=line[0]
      gene=line[1]
      pki=float(line[2])
      try:
        ikey=chembl2ikey[chembl]
      except:
        #there may be outdated ChEMBL molecules removed from ChEMBL DB
        continue
      try:
        uni=gene2uniprot[gene]
      except:
        #target genes not properly mapped to any protein
        #with open('extra_genes_pkis.txt','a') as out:
        #  out.write("{}\n".format(gene))
        continue
      tup=(ikey,uni,'pKi','=',pki)
      data.append(tup)
  if dataframe:
    data=pandas_df_continuous(data)
  return data

def get_plos_pki(dataframe=True):
  pkis='../../kinome_assay/other_published/plos_pki.tsv'
  gene2uniprot,uniprot2genes,mutgene2mutuniprot,mutgene2filename=get_protein_idmap()
  data=[]
  with open(pkis,'r') as f:
    for line in f:
      line=line.strip().split('\t')
      ikey=line[0]
      gene=line[1]
      modobj=re.match(r'(.*)-(.*ted)$',gene,re.M|re.I)
      if modobj:
        gene=modobj.group(1)
        mod=modobj.group(2)
      else:
        mod=None
      pki=float(line[2])
      try:
        uni=gene2uniprot[gene]
      except:
        #target genes not properly mapped to any protein
        #with open('extra_genes_plos.txt','a') as out:
        #  out.write("{}\n".format(gene))
        continue
      if mod is not None:
        uni=uni+'-'+mod
      tup=(ikey,uni,'pKi','=',pki)
      data.append(tup)
  if dataframe:
    data=pandas_df_continuous(data)
  return data
def get_science_pki(dataframe=True):
  pkis='../../kinome_assay/other_published/science_pki.tsv'
  gene2uniprot,uniprot2genes,mutgene2mutuniprot,mutgene2filename=get_protein_idmap()
  data=[]
  with open(pkis,'r') as f:
    for line in f:
      line=line.strip().split('\t')
      ikey=line[0]
      gene=line[1]
      modobj=re.match(r'(.*)-(.*ted)$',gene,re.M|re.I)
      if modobj:
        gene=modobj.group(1)
        mod=modobj.group(2)
      else:
        mod=None
      pki=float(line[2])
      try:
        uni=gene2uniprot[gene]
      except:
        #target genes not properly mapped to any protein
        #with open('extra_genes_science.txt','a') as out:
        #  out.write("{}\n".format(gene))
        continue
      if mod is not None:
        uni=uni+'-'+mod
      tup=(ikey,uni,'pKi','=',pki)
      data.append(tup)
  if dataframe:
    data=pandas_df_continuous(data)
  return data
def get_kinomescan_null_inactive(assay_type='pKd', dataframe=True):
  #if dataframe=True, a pandas dataframe is returned
  #dataframe has integer indice from 0,
  # column names are ['InChIKey','UniProt','Activity']
  gene2uniprot,uniprot2genes,mutgene2mutuniprot,mutgene2filename=get_protein_idmap()
  chemname2ikey=load_chemname2ikey()
  kinomepath='../../kinome_assay/LINCS/'
  if assay_type=='pKd' or assay_type=='pkd':
    assay_type='pKd'
    activity=kinomepath+'LINCS_kinomescan_kd_inactive_null.tsv' # assay inactive, without numeric value reported
  elif assay_type=='pi' or assay_type=='pI':
    assay_type='pPI' #Percent Inhibition Standardized = compound_concentration_nM*(100-%activity)/%activity
    activity=kinomepath+'LINCS_kinomescan_pi_inactive_null.tsv' # assay inactive, without numeric value reported
  else:
    print("Choose activity. pKd or pI.")
    sys.exit()
  data=[]
  with open(activity,'r') as f:
    for l in f:
      l=l.strip().split('\t')
      drug=l[0]
      gene=l[1]
      modobj=re.match(r'(.*)-(.*ted)$',gene,re.M|re.I)
      if modobj:
        gene=modobj.group(1)
        mod=modobj.group(2)
      else:
        mod=None
      ikey=chemname2ikey[drug]
      try:
        uni=gene2uniprot[gene]
      except:
       # with open('./extra_genes_in_kinomescan.txt','a') as out: #collect unmapped genes
       #   out.write("{}\n".format(gene))
        continue
      if mod is not None:
        uni=uni+'-'+mod
      tup=(ikey,uni,'Inactive')
      data.append(tup)
  if dataframe:
    data=pandas_df_binary(data)
  return data

def get_kinomescan(assay_type='pKd', dataframe=True):
  #if dataframe=True, a pandas dataframe is returned
  #dataframe has integer indice from 0,
  # column names are ['InChIKey','UniProt','Activity_type','Relation','Activity_value']
  gene2uniprot,uniprot2genes,mutgene2mutuniprot,mutgene2filename=get_protein_idmap()
  chemname2ikey=load_chemname2ikey()
  kinomepath='../../kinome_assay/LINCS/'
  if assay_type=='pKd' or assay_type=='pkd':
    assay_type='pKd'
    activity=kinomepath+'LINCS_kinomescan_kd_nM.tsv' #assay with numeric activity value in Kd
    null_activity=kinomepath+'LINCS_kinomescan_kd_inactive_null.tsv' # assay inactive, without numeric value reported
  elif assay_type=='pi' or assay_type=='pI':
    assay_type='pPI' #Percent Inhibition Standardized = compound_concentration_nM*(100-%activity)/%activity
    activity=kinomepath+'LINCS_kinomescan_pi_nM.tsv' #assay with numeric activity value in PI
    null_activity=kinomepath+'LINCS_kinomescan_pi_inactive_null.tsv' # assay inactive, without numeric value reported
  else:
    print("Choose activity. pKd or pI.")
    sys.exit()
    
  data=[]
  with open(activity,'r') as f:
    for l in f:
      l=l.strip().split('\t')
      drug=l[0]
      gene=l[1]
      modobj=re.match(r'(.*)-(.*ted)$',gene,re.M|re.I)
      if modobj:
        gene=modobj.group(1)
        mod=modobj.group(2)
      else:
        mod=None
      val=float(l[2])
      if val<=0: #nonsense data, Kd must be positive
        continue
      val=-np.log10(val)+9.0 #all activities are in nM
      if np.isinf(val) or np.isnan(val):
        continue #skip inf or NaN
      ikey=chemname2ikey[drug]
      try:
        uni=gene2uniprot[gene]
      except:
       # with open('./extra_genes_in_kinomescan.txt','a') as out: #collect unmapped genes
       #   out.write("{}\n".format(gene))
        continue
      if mod is not None:
        uni=uni+'-'+mod
      tup=(ikey,uni,assay_type,'=',val)
      data.append(tup)
  if dataframe:
    data=pandas_df_continuous(data)
  return data

if __name__=='__main__':
#  ks_data=get_kinomescan(assay_type='pKd',dataframe=True)
#  print(ks_data)
#  ks_data=get_kinomescan(assay_type='pi',dataframe=True)
#  print(ks_data)
#  ks_data=get_kinomescan_null_inactive(assay_type='pKd', dataframe=True)
#  print(ks_data)
#  ks_data=get_kinomescan_null_inactive(assay_type='pi', dataframe=True)
#  print(ks_data)
#   jcim=get_jcim_pki(dataframe=True)
#   print(jcim)
#  pkis=get_pkis_pki(dataframe=True)
#  print(pkis)
#  plos=get_plos_pki(dataframe=True)
#  print(plos)
  sci=get_science_pki(dataframe=True)
  print(sci)
