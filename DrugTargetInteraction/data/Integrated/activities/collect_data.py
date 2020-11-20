import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from BindingDB import get_bindingdb_by_assay_type
from ChEMBL  import get_chembl_by_assay_type, get_chembl_cyp450_by_assay_type
from GPCRdb import get_gpcrdb
from KinomeAssay import get_jcim_pki, get_pkis_pki, get_plos_pki, get_science_pki, get_kinomescan_null_inactive, get_kinomescan
from PubChemCYP450 import get_pubchem_cyp450
from DrugBank import get_drugbank
from utils import compress_gzip, decompress_gzip, query_yes_no

def collect_pkd():
  kinomescan=get_kinomescan(assay_type='pKd', dataframe=True)
  chemblcyp450=get_chembl_cyp450_by_assay_type(assay_type='pKd',dataframe=True)
  chembl=get_chembl_by_assay_type(assay_type='pKd',dataframe=True)
  bindingdb=get_bindingdb_by_assay_type(assay_type='pKd',dataframe=True)
  gpcrdb=get_gpcrdb(assay_type='pKd',dataframe=True)
  pkd_df=pd.concat([kinomescan,chemblcyp450,chembl,bindingdb,gpcrdb])
  return pkd_df.reset_index(drop=True)
def collect_pki():
  sci=get_science_pki(dataframe=True)
  plos=get_plos_pki(dataframe=True)
  pkis=get_pkis_pki(dataframe=True)
  jcim=get_jcim_pki(dataframe=True)
  gpcrdb=get_gpcrdb(assay_type='pKi',dataframe=True)
  chembl=get_chembl_by_assay_type(assay_type='pKi',dataframe=True)
  chemblcyp450=get_chembl_cyp450_by_assay_type(assay_type='pki',dataframe=True)
  bindingdb=get_bindingdb_by_assay_type(assay_type='pKi',dataframe=True)
  pki_df=pd.concat([sci,plos,pkis,jcim,gpcrdb,chembl,chemblcyp450,bindingdb])
  return pki_df.reset_index(drop=True)
def collect_pic50():
  gpcrdb=get_gpcrdb(assay_type='pic50',dataframe=True)
  chembl=get_chembl_by_assay_type(assay_type='pic50',dataframe=True)
  chemblcyp450=get_chembl_cyp450_by_assay_type(assay_type='pic50',dataframe=True)
  bindingdb=get_bindingdb_by_assay_type(assay_type='pic50',dataframe=True)
  pic50_df=pd.concat([gpcrdb,chembl,chemblcyp450,bindingdb])
  return pic50_df.reset_index(drop=True)
def collect_binary():
  kinomescan1=get_kinomescan_null_inactive(assay_type='pKd', dataframe=True) #binary from pKd assay
  kinomescan2=get_kinomescan_null_inactive(assay_type='pI', dataframe=True) #binary from %inhibit assay
  pubchemcyp450=get_pubchem_cyp450(dataframe=True)
  drugbank=get_drugbank(dataframe=True)
  binary_df=pd.concat([kinomescan1,kinomescan2,pubchemcyp450,drugbank])
  binary_df=binary_df.drop_duplicates(subset=['InChIKey','UniProt','Activity'],keep='first')
  return binary_df.reset_index(drop=True)

def collect_kinases():
  kinomescan_pkd=get_kinomescan(assay_type='pKd', dataframe=True)
  sci=get_science_pki(dataframe=True)
  plos=get_plos_pki(dataframe=True)
  pkis=get_pkis_pki(dataframe=True)
  jcim=get_jcim_pki(dataframe=True)
  chembl1=get_chembl_by_assay_type(assay_type='pKd',dataframe=True)
  bindingdb1=get_bindingdb_by_assay_type(assay_type='pKd',dataframe=True)
  chembl2=get_chembl_by_assay_type(assay_type='pKi',dataframe=True)
  bindingdb2=get_bindingdb_by_assay_type(assay_type='pKi',dataframe=True)
  chembl3=get_chembl_by_assay_type(assay_type='pic50',dataframe=True)
  bindingdb3=get_bindingdb_by_assay_type(assay_type='pic50',dataframe=True)
  df=pd.concat([jcim,chembl1,chembl2,chembl3,bindingdb1,bindingdb2,bindingdb3])
  uniprots=[]
  with open('../proteins/kinase_enzyme_classes.tsv','r') as f:
    for line in f:
      line=line.strip().split('\t')
      uniprot=line[0]
      uniprots.append(uniprot)
  df=df[df['UniProt'].isin(uniprots)]
  df=pd.concat([df,kinomescan_pkd,sci,plos,pkis])
  return df.reset_index(drop=True)

def collect_by_uniprot_file(uniprot_file):
  uniprots=[]
  #with open('../proteins/disease_related_genes_id_mapping.tsv','r') as f:
  with open(uniprot_file,'r') as f:
    for line in f:
      line=line.strip().split('\t')
      uniprot=line[0]
      uniprots.append(uniprot)
  gpcrdb1=get_gpcrdb(assay_type='pKd',dataframe=True)
  gpcrdb2=get_gpcrdb(assay_type='pic50',dataframe=True)
  gpcrdb3=get_gpcrdb(assay_type='pKi',dataframe=True)
  chembl1=get_chembl_by_assay_type(assay_type='pic50',dataframe=True)
  chembl2=get_chembl_by_assay_type(assay_type='pki',dataframe=True)
  chembl3=get_chembl_by_assay_type(assay_type='pkd',dataframe=True)
  bindingdb1=get_bindingdb_by_assay_type(assay_type='pic50',dataframe=True)
  bindingdb2=get_bindingdb_by_assay_type(assay_type='pki',dataframe=True)
  bindingdb3=get_bindingdb_by_assay_type(assay_type='pkd',dataframe=True)
  drugbank=get_drugbank(dataframe=True)
  df=pd.concat([chembl1,chembl2,chembl3,bindingdb1,bindingdb2,bindingdb3,gpcrdb1,gpcrdb2,gpcrdb3])
  df=df[df['UniProt'].isin(uniprots)]
  df=df.reset_index(drop=True)
  df_bin=drugbank[drugbank['UniProt'].isin(uniprots)].reset_index(drop=True)
  return df,df_bin
def collect_cyp450():
  cyp450_uniprots=[]
  with open("../proteins/cytochrome_p450_id_mapping.tsv",'r') as f:
    for line in f:
      line=line.strip().split('\t')
      uniprot=line[0]
      cyp450_uniprots.append(uniprot)
  chembl1=get_chembl_by_assay_type(assay_type='pic50',dataframe=True)
  chembl2=get_chembl_by_assay_type(assay_type='pki',dataframe=True)
  chembl3=get_chembl_by_assay_type(assay_type='pkd',dataframe=True)
  chemblcyp450=get_chembl_cyp450_by_assay_type(assay_type='pic50',dataframe=True)
  bindingdb1=get_bindingdb_by_assay_type(assay_type='pic50',dataframe=True)
  bindingdb2=get_bindingdb_by_assay_type(assay_type='pki',dataframe=True)
  bindingdb3=get_bindingdb_by_assay_type(assay_type='pkd',dataframe=True)
  drugbank=get_drugbank(dataframe=True)
  pubchemcyp450=get_pubchem_cyp450(dataframe=True)
  cyp_df=pd.concat([chembl1,chembl2,chembl3,chemblcyp450,bindingdb1,bindingdb2,bindingdb3])
  cyp_df_bin=pd.concat([pubchemcyp450,drugbank])
  cyp_df_cont=cyp_df[cyp_df['UniProt'].isin(cyp450_uniprots)]
  cyp_df_cont=cyp_df_cont.reset_index(drop=True)
  cyp_df_bin=cyp_df_bin[cyp_df_bin['UniProt'].isin(cyp450_uniprots)]
  cyp_df_bin=cyp_df_bin.drop_duplicates(subset=['InChIKey','UniProt','Activity'],keep='first')
  cyp_df_bin=cyp_df_bin.reset_index(drop=True)
  return cyp_df_cont,cyp_df_bin


def write_out(args,df):
  if args.outfmt.lower()=="tsv":
    df.to_csv(args.outfile, sep='\t', encoding='utf-8')
  elif args.outfmt.lower()=="csv":
    df.to_csv(args.outfile, sep=',', encoding='utf-8')
#  elif args.outfmt.lower()=="json":
#    df=df.to_json()
#    save_json(df,args.outfile)
  elif args.outfmt.lower() in ['hdf','h5']:
    df.to_hdf(args.outfile,'table', mode='w')
  else:
    print("{} is invalid option for output format. 'tsv' is recommended.".format(args.outfmt))
    print("Dataframe is exported to {}".format(outfile))
    df.to_csv(args.outfile, sep='\t', encoding='utf-8')

def average_df(df,keep=False):
  #does not apply if relation is NOT '='
  #For all activity with relation '=',
  # average activity if multiple records exist
  ### df : dataframe to average
  ### keep : want to keep the activities with '>' or '<' relation?
  equality=df.loc[df['Relation'] == '=']
  edf=equality.groupby(['InChIKey', 'UniProt']).mean()
  if keep:
    inequality=df.loc[df['Relation'] != '=']
  return edf

def main(args):
  if args.pandas:
    dataframe=True
  else:
    dataframe=False
  outfile=args.outfile
  if Path(outfile).is_file(): #outfile already exist. Ask for overwrite
    ans=query_yes_no('File {} exists.\nOverwrite?'.format(outfile), default='yes')
    if not ans:
      print("Please check the file {}".format(outfile))
      sys.exit()

  if (args.target.lower() =='kinase') or (args.target.lower() == 'kinases'): #kinase targets
    df=collect_kinases()
    write_out(args,df)
    sys.exit()
  elif (args.target.lower() =='gpcr') or (args.target.lower() == 'gpcrs'): #G protein coupled receptors
    df_cont,df_bin=collect_by_uniprot_file('../proteins/gpcr_id_mapping.tsv')
    if args.assay_type.lower()=='binary':
      write_out(args,df_bin)
    elif args.assay_type.lower()=='continuous':
      write_out(args,df_cont)
    else:
      raise ValueError("assay_type {} is not supported for disease-related genes dataset. Choose from ('continuous','binary')".format(args.assay_type))
    sys.exit()
  elif (args.target.lower() =='cyp450') or (args.target.lower() == 'cyp450s'): #cytochrome P450
    cyp_cont,cyp_bin=collect_cyp450() #cannot use collect_by_uniprot_file function; source databases are different
    if args.assay_type.lower()=='continuous':
      write_out(args,cyp_cont)
    elif args.assay_type.lower()=='binary':
      write_out(args,cyp_bin)
    else:
      raise ValueError("assay_type {} is not supported for CYP450 dataset. Choose from ('continuous','binary')".format(args.assay_type))
    sys.exit()
  elif (args.target.lower() =='nr') or (args.target.lower() == 'nrs'): #nuclear receptors
    df_cont,df_bin=collect_by_uniprot_file('../proteins/nuclear_receptor_id_mapping.tsv')
    if args.assay_type.lower()=='binary':
      write_out(args,df_bin)
    elif args.assay_type.lower()=='continuous':
      write_out(args,df_cont)
    else:
      raise ValueError("assay_type {} is not supported for nuclear receptor dataset. Choose from ('continuous','binary')".format(args.assay_type))
    sys.exit()
  elif (args.target.lower() =='tf') or (args.target.lower() == 'tfs'): #transcription factors
    df_cont,df_bin=collect_by_uniprot_file('../proteins/transcription_factor_id_mapping.tsv')
    if args.assay_type.lower()=='binary':
      write_out(args,df_bin)
    elif args.assay_type.lower()=='continuous':
      write_out(args,df_cont)
    else:
      raise ValueError("assay_type {} is not supported for transcription factor dataset. Choose from ('continuous','binary')".format(args.assay_type))
    sys.exit()
  elif (args.target.lower() =='cancer') or (args.target.lower() == 'cancers'): #cancer-related genes
    df_cont,df_bin=collect_by_uniprot_file('../proteins/cancer_related_genes_id_mapping.tsv')
    if args.assay_type.lower()=='binary':
      write_out(args,df_bin)
    elif args.assay_type.lower()=='continuous':
      write_out(args,df_cont)
    else:
      raise ValueError("assay_type {} is not supported for cancer-related genes dataset. Choose from ('continuous','binary')".format(args.assay_type))
    sys.exit()
  elif (args.target.lower() =='disease') or (args.target.lower() == 'diseases'):  #disease related genes
    df_cont,df_bin=collect_by_uniprot_file('../proteins/disease_related_genes_id_mapping.tsv')
    if args.assay_type.lower()=='binary':
      write_out(args,df_bin)
    elif args.assay_type.lower()=='continuous':
      write_out(args,df_cont)
    else:
      raise ValueError("assay_type {} is not supported for disease-related genes dataset. Choose from ('continuous','binary')".format(args.assay_type))
    sys.exit()
  elif (args.target.lower() =='cardio') or (args.target.lower() == 'cardiovascular'): #cardiovascular disease related genes
    df_cont,df_bin=collect_by_uniprot_file('../proteins/cardiovascular_disease_candidate_id_mapping.tsv')
    if args.assay_type.lower()=='binary':
      write_out(args,df_bin)
    elif args.assay_type.lower()=='continuous':
      write_out(args,df_cont)
    else:
      raise ValueError("assay_type {} is not supported for FDA-approved target dataset. Choose from ('continuous','binary')".format(args.assay_type))
    sys.exit()
  elif (args.target.lower() =='fda') or (args.target.lower() == 'approved'): #FDA approved drug targets
    df_cont,df_bin=collect_by_uniprot_file('../proteins/fda_approved_target_id_mapping.tsv')
    if args.assay_type.lower()=='binary':
      write_out(args,df_bin)
    elif args.assay_type.lower()=='continuous':
      write_out(args,df_cont)
    else:
      raise ValueError("assay_type {} is not supported for FDA-approved target dataset. Choose from ('continuous','binary')".format(args.assay_type))
    sys.exit()
  elif (args.target.lower() =='potential') or (args.target.lower() == 'pdt'): #potential drug targets
    df_cont,df_bin=collect_by_uniprot_file('../proteins/potential_drug_target_id_mapping.tsv')
    if args.assay_type.lower()=='binary':
      write_out(args,df_bin)
    elif args.assay_type.lower()=='continuous':
      write_out(args,df_cont)
    else:
      raise ValueError("assay_type {} is not supported for potential drug target dataset. Choose from ('continuous','binary')".format(args.assay_type))
    sys.exit()
  elif (args.target.lower() =='all') or (args.target.lower() == 'any'): #all targets
    #continue to next block
    None
  else:
    print("{} target feature is an invalid option".format(args.target))
    sys.exit()

  if args.assay_type.lower() == 'pkd':
    df=collect_pkd()
  elif args.assay_type.lower() =='pki':
    df=collect_pki()
  elif args.assay_type.lower() =='pic50':
    df=collect_pic50()
  elif args.assay_type.lower() =='binary':
    df=collect_binary()
  elif args.assay_type.lower() =='continuous':
    pkd_df=collect_pkd()
    pki_df=collect_pki()
    pic50_df=collect_pic50()
    df=pd.concat([pkd_df,pki_df,pic50_df])
    df=df.reset_index(drop=True)
  else:
    print("{} is invalid assay type. Please choose {pki, pkd, pic50, continuous, binary}".format(args.assay_type))
  
  write_out(args,df)
    
if __name__=='__main__':
  parser = argparse.ArgumentParser("Extract chemical-protein associations")
  parser.add_argument('--assay_type', default="pKd", help="Assay activity type (pKd, pKi, pIC50, continuous, binary)")
  parser.add_argument('--pandas', default=1, help="Whether to get data as Pandas Dataframe (1 for yes, 0 for no)")
  parser.add_argument('--target', default="all", help='Target protein class (all, kinase, GPCR, cyp450, ...)')
  parser.add_argument('--outfmt', default="tsv", help='Output file format. (tsv is preferred over csv due to gene mutants having commas)\
    \navailable formats are {tsv, csv, json, hdf}.')
  parser.add_argument('--outfile', default="outfile.tsv", help="Output file name where dataframe is exported.")
  args = parser.parse_args()
  main(args)
  
