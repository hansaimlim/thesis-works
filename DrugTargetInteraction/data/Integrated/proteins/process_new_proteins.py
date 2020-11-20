import os
import re
import sys
from pathlib import Path
import subprocess
from multiprocessing import Pool
from multiprocessing import freeze_support, cpu_count
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC


#fastalist=os.listdir('uniprot_fasta_human/')
#pssmlist=os.listdir('pssm_human_uniref50/')

#for fa in fastalist:
#  ps=str(fa).replace('.fas','.dat')
#  if ps in pssmlist:
#    continue
#  else:
#    print(fa)

def merge_uniprot_idmap(original,new):
  #combine two uniprot_id_mappings
  uni_exist=[]
  merged_idmap='merged_uniprot_id_mapping.tsv' #create new file for backup safety
  out=open(merged_idmap,'w')
  with open(original,'r') as f:
    for line in f:
      line_ori=line.strip()
      out.write("{}\n".format(line_ori))
      line=line.strip().split('\t')
      uni=line[0]
      uni_exist.append(uni)
  with open(new,'r') as f:
    next(f)
    for line in f:
      line_ori=line.strip()
      line=line.strip().split('\t')
      uni=line[0]
      if uni in uni_exist:
        continue
      else:
        out.write("{}\n".format(line_ori)) #new protein
  print("Merged ID map in {}".format(merged_idmap))
  out.close()
  return merged_idmap
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

def get_uniprot_no_pssm():
  uniprot_map='uniprot_id_mapping.tsv'
  pssm_dir='pssm_uniref50/'
  pssmlist=os.listdir(pssm_dir)
  with open(uniprot_map,'r') as f:
    next(f)
    for line in f:
      line=line.strip().split('\t')
      uni=line[0]
      ps=uni+'.dat'
      if ps in pssmlist:
        continue
      else:
        print(uni)


def main():
  pssm_dir='pssm_uniref50/'
  pssmlist=os.listdir(pssm_dir)

  kindom_fasta_list=os.listdir('kinase_domain_fasta/')
  genelist='uniq_genes'
  emgenelist='extra_genes_wt'
  with open(emgenelist,'r') as f:
    for line in f:
      uni=line.strip()
      uni=remove_modifications(uni)
      uni=mutant_to_filename(uni)
      uni=uni.strip().split('_')
      uni=uni[0]
      ps=uni+'.dat'
      if ps in pssmlist:
        continue
      else:
        print(uni)
        continue
        if uni+'.fas' in kindom_fasta_list:
          #print("{} found in kinase domain fasta directory".format(uni))
          continue
        else:
          print("{}".format(uni))

if __name__=='__main__':
  get_uniprot_no_pssm()
  sys.exit()
  main()
