import os
import sys
import re
from pathlib import Path
import subprocess
import json
from utils import compress_gzip, decompress_gzip

def get_protein_idmap():
  protpath='../proteins/'
  uniprot=protpath+'uniprot_id_mapping.tsv'
  gene2uniprot={} #genename -> reference uniprot accession
  uniprot2genes={} #reference uniprot accession -> list of genenames
  mutgene2mutuniprot={} #mutant genename -> mutant uniprot accession; ABL1(E255K) -> P00519(E255K)
  mutgene2filename={} #mutant genename -> name for individual file; ABL1(E255K) -> P00519_E255K
  with open(uniprot,'r') as f:
    next(f)
    for line in f:
      line=line.strip().split('\t')
      uni=line[0]
      g1=line[3].strip()
      g2=line[4].strip()
      g3=line[5].strip()
      genes=[]
      if len(g1)>0:
        g1=g1.split(' ')
        genes+=g1
      if len(g2)>0:
        g2=g2.split(' ')
        genes+=g2
      if len(g3)>0:
        g3=g3.split(' ')
        genes+=g3
      genes=list(set(genes))
      uniprot2genes[uni]=genes
      for g in genes:
        g=g.replace(';','') #remove semicolons if found
        gene2uniprot[g]=uni
  with open(protpath+'mutant_id_conversion.tsv','r') as f:
    for line in f:
      line=line.strip().split('\t')
      mutgene=line[0]
      mutuniprot=line[1]
      filename=line[2]
      mutgene2mutuniprot[mutgene]=mutuniprot
      gene2uniprot[mutgene]=mutuniprot
      uniprot2genes[mutuniprot]=[mutgene]
      mutgene2filename[mutgene]=filename

  return gene2uniprot,uniprot2genes,mutgene2mutuniprot,mutgene2filename

def load_chembl2ikey():
  chemmap='../../ChEMBL24/ChEMBL24_all_compounds.csv.gz'
  if Path(chemmap).is_file(): #decompress if compressed
    chemmap=decompress_gzip(chemmap)
  else:
    chemmap='../../ChEMBL24/ChEMBL24_all_compounds.csv' #already decompressed
  chembl2ikey={}
  with open(chemmap,'r') as f:
    next(f)
    for line in f:
      line=line.strip().split(',')
      chembl=str(line[0])
      molregno=line[1]
      ikey=str(line[2]).lstrip()
      chembl2ikey[chembl]=ikey
  return chembl2ikey
def load_chemname2ikey():
  #return a dict{Chemical_name->InChIKey}
  #contains KinomeScan chemical names
  #contains JCIM dataset chemical names
  chemmap='../../kinome_assay/LINCS/list/HMS-LINCS_KinomeScan_chemicals.txt'
  chemname2ikey={}
  with open(chemmap,'r') as f:
    next(f)
    for line in f:
      line=line.strip().split('\t')
      name=line[0]
      ikey=line[1]
      smi=line[2]
      chemname2ikey[name]=ikey
  chemmap='../../kinome_assay/other_published/jcim_compounds.tsv'
  with open(chemmap,'r') as f:
    for line in f:
      line=line.strip().split('\t')
      name=line[0]
      ikey=line[1]
      smi=line[2]
      chemname2ikey[name]=ikey
  return chemname2ikey
def load_chembl2uniprot():
  protmap='../../ChEMBL24/ChEMBL24_all_targets.tsv'
  chembl2uniprot={}
  with open(protmap,'r') as f:
    next(f)
    for line in f:
      line=line.strip().split('\t')
      uni=str(line[0])
      gene=str(line[1])
      chembl=str(line[2])
      chembl2uniprot[chembl]=uni
  return chembl2uniprot
