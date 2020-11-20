import os
import re
import sys
import numpy as np
from pathlib import Path
import subprocess
from multiprocessing import Pool
from multiprocessing import freeze_support, cpu_count
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
def run_psiblast(fasta_in,pssm_out,ncpu,db='nr',n_iter=3):
  #db could be ['uniref50','nr']
  if ncpu is None:
    ncpu=1
  elif ncpu < 1:
    ncpu=1
  psiblastcmd="psiblast -query {} -db {} -out_ascii_pssm {} -num_threads {} -num_iterations {} -comp_based_stats 1"\
  .format(fasta_in,db,pssm_out,ncpu,n_iter)
  output = subprocess.check_output(['bash','-c',psiblastcmd])
  print("psiblast search for {} complete!".format(fasta_in))

def get_pssm_from_fasta(whole_fasta,fasta_dir,pssm_dir,njobs=4,ncpu=4,db='nr',n_iter=3):
  pssm_files=[]
  inputs=[]
  for seq_record in SeqIO.parse(whole_fasta,"fasta"):
    seqids=str(seq_record.id).strip().split('|')
    seqid=seqids[0] #uniprot id (with mutation info) 
    sequence=seq_record.seq
    if Path(pssm_dir).is_dir():
      #fasta file exist
      None
    else:
      print("PSSM directory {} does not exist".format(pssm_dir))
      sys.exit()
    if Path(fasta_dir).is_dir():
      None
    else:
      print("FASTA directory {} does not exist".format(fasta_dir))
      sys.exit()
    fasta_out=os.path.join(fasta_dir,seqid+".fas")
    pssm_out=os.path.join(pssm_dir,seqid+".dat")
    SeqIO.write(seq_record,fasta_out,"fasta") #write fasta for individual protein
    inp=(fasta_out,pssm_out,ncpu,db,n_iter)
    inputs.append(inp)
    pssm_files.append(pssm_out)
  with Pool(njobs) as pool: #use multicore
    try:
#      print("{} total processes. Run {} processes at a time...".format(len(inputs),njobs))
      pool.starmap(run_psiblast,inputs)
    except BaseException as e:
      print("Error occurred:\n"+str(e)+"\n") 
  return pssm_files

if __name__ == '__main__':
#  import psutil
### Limit CPU usage
#  psutil.cpu_count()
#  p = psutil.Process()
#  p.cpu_affinity()  # get
#  p.cpu_affinity([11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])
### Limit CPU usage

#  njobs=6 #n jobs run at a time
#  ncpus=4 #each jobs use ncpus -num_threads option in psiblast
  whole_fasta='./kinase_domains_extra.fas' #single file containing multiple sequences in FASTA format
#  whole_fasta='./fasta_sample.fas'
#  bsite_fasta='./kinase_binding_sites.fas'
  fasta_dir='./temp/' #output directory for individual fasta files
#  pssm_dir_uniref50='./separate_pssm_uniref50/'
  pssm_dir_nr='./kinase_domain_pssm_nr/' #output directory for PSSM against NR database
  pssm_files=get_pssm_from_fasta(whole_fasta,fasta_dir,pssm_dir_nr,njobs=3,ncpu=9,db='nr',n_iter=3)
