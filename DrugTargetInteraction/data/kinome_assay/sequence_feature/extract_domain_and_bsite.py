import os
import re
import sys
import json
import subprocess
import numpy as np
import pickle
from pathlib import Path
#import random
#import string
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Seq import MutableSeq
from Bio.Alphabet import IUPAC
from Bio.ExPASy import ScanProsite

def get_kindom_pfam(kinase_fasta,domain_fasta_path):
  list_of_pfam_kindom=['PS50011']
 # list_of_pfam_kindom=['PS00239','PS00240','PS00378'',PS51748','PS00790','PS00791','PS01042','PS50032','PS50109','PS51158',\
 # 'PS51285','PS51455','PS51545','PS50011']
  fasta_files=[]
  for seq_record in SeqIO.parse(kinase_fasta,"fasta"):
    if seq_record.id=='alignment':
      continue
    seqids=str(seq_record.id).strip().split('|')
    seq=str(seq_record.seq)
    uniprot=seqids[1]
    
    if uniprot in ['P23458','O60674','P29597']: #JAK1 or JAK2 or TYK2 kinase, C-term kinase is catalytic, the other is pseudokinase
      if uniprot=='P23458':
        catalytic=seq[874:1153] #875-1153
        pseudokinase=seq[582:855] #583-855
        catrange="875:1153"
        pserange="585:855"
      elif uniprot == 'O60674':
        catalytic=seq[848:1126] #849-1126
        pseudokinase=seq[544:809] #545-809
        catrange="849:1126"
        pserange="545:809"
      elif uniprot == 'P29597':
        catalytic=seq[896:1176] #897-1176
        pseudokinase=seq[588:875] #589-875
        catrange="897:1176"
        pserange="589:875"
      uni_cat=uniprot+'_JH1domain-catalytic'
      uni_pse=uniprot+'_JH2domain-pseudokinase'
      seqrecord_cat=SeqRecord(Seq(catalytic,IUPAC.protein),uni_cat+'|'+catrange,'','')
      seqrecord_pse=SeqRecord(Seq(pseudokinase,IUPAC.protein),uni_pse+'|'+pserange,'','')
      uni_cat_file=os.path.join(domain_fasta_path,uni_cat+'.fas')
      uni_pse_file=os.path.join(domain_fasta_path,uni_pse+'.fas')
      with open(uni_cat_file,'w') as out: 
        SeqIO.write(seqrecord_cat,out,"fasta") #catalytic domain
      with open(uni_pse_file,'w') as out: 
        SeqIO.write(seqrecord_pse,out,"fasta") #catalytic domain
      fasta_files.append(uni_cat_file)
      fasta_files.append(uni_pse_file)
      continue

    handle = ScanProsite.scan(seq=seq)
    results = ScanProsite.read(handle)
    n_kindom=0;n_term_kindom_start=99999
    for result in results:
      sig=result['signature_ac']
      if sig in list_of_pfam_kindom:
        start=result['start']
        stop=result['stop']
        if start<n_term_kindom_start:
          n_term_kindom_start=start #lower number start = N-term kindom start position
        n_kindom+=1

    for result in results:
      sig=result['signature_ac']
      if n_kindom==1:
        if sig in list_of_pfam_kindom:
          start=result['start']
          stop=result['stop']
          seqrange=str(start)+':'+str(stop)
         # print(result)
          subseq=seq[start-1:stop]
         # print(subseq)
          seq_record = SeqRecord(Seq(subseq,IUPAC.protein),uniprot+'_Kin.Dom|'+seqrange,'','')
          fasta_out=os.path.join(domain_fasta_path,uniprot+'.fas')
          with open(fasta_out,'w') as out: 
            SeqIO.write(seq_record,out,"fasta") #catalytic domain
          fasta_files.append(fasta_out)
      elif n_kindom==2: #two kinase domains in one protein
        if sig in list_of_pfam_kindom:
          start=result['start']
          stop=result['stop']
          kindom=seq[start-1:stop]
          seqrange=str(start)+':'+str(stop)
          if start==n_term_kindom_start:
            #N-term kindom, Kin.Dom.1
            outfile=os.path.join(domain_fasta_path,uniprot+'_Kin.Dom.1-N-terminal.fas')
            seq_record = SeqRecord(Seq(kindom,IUPAC.protein),uniprot+'_Kin.Dom.1-N-terminal|'+seqrange,'','')
          else:
            #C-term kindom, Kin.Dom.2
            outfile=os.path.join(domain_fasta_path,uniprot+'_Kin.Dom.2-C-terminal.fas')
            seq_record = SeqRecord(Seq(kindom,IUPAC.protein),uniprot+'_Kin.Dom.2-C-terminal|'+seqrange,'','')
          with open(outfile,'w') as out: 
            SeqIO.write(seq_record,out,"fasta")
          fasta_files.append(outfile)
  return fasta_files

def run_interpro_cdd(kinase_fasta,json_output):
  #run interpro standalone package to obtain active site residues via CDD annotation
  interprocommand="interproscan.sh -appl CDD -t p -i {} -f json -o {}".format(kinase_fasta,json_output)
  output=subprocess.check_output(['bash','-c',interprocommand])
  return json_output
def run_interpro_prosite(kinase_fasta,json_output):
  #run interpro standalone package to obtain active site residues via CDD annotation
  interprocommand="interproscan.sh -appl ProSitePatterns -t p -i {} -f json -o {}".format(kinase_fasta,json_output)
  output=subprocess.check_output(['bash','-c',interprocommand])
  return json_output

def get_active_site_residues_from_cdd_json(json_file):
  site_idxs=[]
  with open(json_file) as j:
    data=json.load(j)
    data=data['results'][0]['matches'][0]['locations'][0]['sites']
    for site in data:
      if site['description'] in ['active site','ATP binding site']:
        #collect active site and ATP binding site only
        None
      else: #other sites skipped (e.g. peptide binding site)
        continue
      locs=site['siteLocations']
      for loc in locs:
        if loc['start']!=loc['end']:
          continue
        else:
          site_idxs.append(loc['start'])
  site_idxs=list(set(site_idxs))
  site_idxs.sort()
  return site_idxs

if __name__ == '__main__':
  kinase_fasta='./kinomescan_full_sequence.fasta'
  #kinase_fasta='./testfiles/sample_sequence.fasta' #small test file

  kinases_wob='./kinases_without_known_binding_residues.fas' #collect proteins without b-site in a separate fasta
  domain_fasta_path='./kinase_domain_fasta/'
  fasta_files=get_kindom_pfam(kinase_fasta,domain_fasta_path)
  print("Kinase domain search complete")

#  max_site_len=0
#  kinase_wob_count=0
#  for fasta in fasta_files:
#    json_out=str(fasta).replace('.fas','.json')
#    json_out=run_interpro_cdd(fasta,json_out)
#    try:
#      site_idxs=get_active_site_residues_from_cdd_json(json_out)
#    except:
#      site_idxs=[]
#    if len(site_idxs)==0: #no annotated or predicted binding residues. Collect fasta and use MSA in later steps
#      for seq_record in SeqIO.parse(fasta,"fasta"):
#        with open(kinases_wob,'a') as out: 
#          SeqIO.write(seq_record,out,"fasta")
#      kinase_wob_count+=1
#    else:
#      print("{}\t{}".format(fasta,len(site_idxs)))
#      if len(site_idxs)>max_site_len:
#        max_site_len=len(site_idxs)
#  print("longest binding site length {}".format(max_site_len))
#  print("{} kinases without known binding sites.\nWritten in file {}".format(kinase_wob_count,kinases_wob))     
