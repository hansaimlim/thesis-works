import os
import sys
import json
import random
import string
import subprocess
from pathlib import Path
import numpy as np
#from rdkit import Chem

##### JSON modules #####
class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)
def save_json(data,filename):
  with open(filename, 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=4, cls=NumpyEncoder)
def load_json(filename):
  with open(filename, 'r') as fp:
    data = json.load(fp)
  return data
##### JSON modules #####
##### GZIP modules #####
def decompress_gzip(zip_file,ext='gz'):
  command="gunzip -f {}".format(zip_file)
  output = subprocess.check_output(['bash','-c',command])
  unzip_file=zip_file.replace(str('.'+ext),'')
  return unzip_file
def compress_gzip(unzip_file,ext='gz',level=9):
  command="gzip -{} {}".format(level,unzip_file)
  output = subprocess.check_output(['bash','-c',command])
  zip_file = unzip_file+"."+ext
  return zip_file
##### GZIP modules #####
##### MadFast modules #####
def run_madfast(query_file,target_file,cutoff):
  cutoff=float(cutoff)
  #random named dissimilarity matrix to prevent collision
  tanimoto_filename=target_file+''.join(random.choice(string.ascii_uppercase + string.digits+ string.ascii_lowercase) for _ in range(20))
  try:
    madfast_command=os.path.join('searchStorage.sh')+" -context createSimpleEcfp6Context -qmf {} -qidname\
 -tmf {} -tidname -mode MOSTSIMILARS -maxdissim {} -out {}".format(query_file,target_file,(1.0-cutoff),tanimoto_filename)
    output=subprocess.check_output(['bash','-c',madfast_command])
    return tanimoto_filename
  except BaseException as e:
    print('Error occurred during searchStorage: '+str(e)+"\n")
    with open('./madfast_cutoff_error_command.log','a') as errorlog:
      errorlog.write("{}\n".format(madfast_command))
def parse_sim(datfile, outfile, ut=True):
  #parse dissimilarity list
  #assume all listed pairs are filtered by cutoff, no cutoff needed here
  #ut: upper-triangle only; set to True to reduce output file size
  outfile=str(outfile)
  fout=open(outfile,"a+")
  linenum=0
  with open(datfile,"r") as inf:
    next(inf)
    for line in inf:
      if line=='': #skip empty lines
        continue
      line=line.strip().split('\t')
      idx1=int(line[0])
      idx2=int(line[1])
      if ut:
        if idx2<=idx1: #exclude diagonal or lower-triangle
          continue
      dissim=float(line[2])
      sim=float(1.0-dissim)
      fout.write("{0:},{1:},{2:.6f}\n".format(idx1,idx2,sim))
  fout.close()
##### MadFast modules #####
def delete_file(filename):
  command="rm {}".format(filename)
  output=subprocess.check_output(['bash','-c',command])

def load_current_chemicals(path='./'):
  ikey2idx=load_ikey2idx(path)
  idx2ikeys=load_idx2ikeys(path)
  idx2smiles=load_idx2smiles(path)
  return ikey2idx,idx2ikeys,idx2smiles

def load_ikey2idx(path='./'):
  ikey2idx_file='integrated_InChIKey2Index.json.gz'
  if Path(ikey2idx_file).is_file():
    ikey2idx_file=decompress_gzip(ikey2idx_file)
  else:
    ikey2idx_file='integrated_InChIKey2Index.json'
  ikey2idx=load_json(ikey2idx_file)
  return ikey2idx
def load_idx2ikeys(path='./'):
  idx2ikeys_file='integrated_Index2InChIKeys.json.gz'
  if Path(idx2ikeys_file).is_file():
    idx2ikeys_file=decompress_gzip(idx2ikeys_file)
  else:
    idx2ikeys_file='integrated_Index2InChIKeys.json'
  idx2ikeys=load_json(idx2ikeys_file)
  return idx2ikeys
def load_idx2smiles(path='./'):
  idx2smiles_file='integrated_Index2SMILES.json.gz'
  if Path(idx2smiles_file).is_file():
    idx2smiles_file=decompress_gzip(idx2smiles_file)
  else:
    idx2smiles_file='integrated_Index2SMILES.json'
  idx2smiles=load_json(idx2smiles_file)
  return idx2smiles

def find_new_chemicals(new_chemical_file):
  #Assume 'new_chemical_file' contains InChIKey and SMILES separated by a tab character
  #need to modify the code in this function if input file is differently formatted
  ikey2idx=load_ikey2idx('./') #load inchikey-index dict
  idx2ikeys=load_idx2ikeys('./')
  maxidx=int(np.max(np.array(list(idx2ikeys.keys()),dtype=np.float32)))
  print("Max chem Index {}".format(maxidx))
  new_chems=[]
  with open(new_chemical_file,'r') as f:
    appeared_newchems=[]
    for line in f:
      line=line.strip().split('\t')
      ikey=str(line[0])
      smi=str(line[1])
      try:
        idx=ikey2idx[ikey]
      except:
        if ikey in appeared_newchems: #skip redundant chemicals
          continue
        maxidx+=1
        tup=(maxidx,ikey,smi) #(chemIndex,inchikey,smiles)
        new_chems.append(tup)
        appeared_newchems.append(ikey)
  return new_chems

def printout_new_chemicals(new_chem_list):
  #prepare input file for MadFast calculation
  #new_chem_list is a list of tuples, tup(chemIndex,InChIKey,SMILES)
  outfile=''.join(random.choice(string.ascii_uppercase + string.digits+ string.ascii_lowercase) for _ in range(20))
  with open(outfile,'w') as out: #print out new chemicals for madfast calculation
    for chem in new_chem_list:
      idx=int(chem[0])
      ikey=str(chem[1])
      smi=str(chem[2])
      out.write("{}\t{}\n".format(smi,idx))
  return outfile

def finalize_update(new_chems,oldchem_smiles,oldchem_tsv,new_sim_file):

  if Path(oldchem_tsv).is_file(): #decompress if compressed
    oldchem_tsv=decompress_gzip(oldchem_tsv)
  else:
    oldchem_tsv='integrated_chemicals.tsv' #list of Index\tInChIKey\tSMILES

  print("Updating chemical lists...")
  out1=open(oldchem_smiles,'a') #list of SMILES\tIndex
  out2=open(oldchem_tsv,'a') #list of Index\tInChIKey\tSMILES
  #save updated chemical lists in json
  ikey2idx=load_ikey2idx()
  idx2ikeys=load_idx2ikeys()
  idx2smiles=load_idx2smiles()
  for chem in new_chems: 
    #updating chemical information dictionaries
    #need to save in json to finalize update
    idx=int(chem[0])
    ikey=str(chem[1])
    smi=str(chem[2])
    ikey2idx[ikey]=idx
    idx2ikeys[str(idx)]=[ikey] #idx converted to string; json keys must be strings to avoid conflict
    idx2smiles[str(idx)]=smi #idx converted to string
    out1.write("{}\t{}\n".format(smi,idx))
    out2.write("{}\t{}\t{}\n".format(idx,ikey,smi))
    #update and save smiles list
  out1.close()
  out2.close()
  oldchem_smiles=compress_gzip(oldchem_smiles,ext='gz',level=9)
  oldchem_tsv=compress_gzip(oldchem_tsv,ext='gz',level=9)
  print(oldchem_smiles)
  print(oldchem_tsv)
  ikey2idx_file='integrated_InChIKey2Index.json'
  idx2ikeys_file='integrated_Index2InChIKeys.json'
  idx2smiles_file='integrated_Index2SMILES.json'
  save_json(ikey2idx,ikey2idx_file)
  save_json(idx2ikeys,idx2ikeys_file)
  save_json(idx2smiles,idx2smiles_file)
  ikey2idx_file=compress_gzip(ikey2idx_file,ext='gz',level=9)
  idx2ikeys_file=compress_gzip(idx2ikeys_file,ext='gz',level=9)
  idx2smiles_file=compress_gzip(idx2smiles_file,ext='gz',level=9)
  print(ikey2idx_file)
  print(idx2ikeys_file)
  print(idx2smiles_file)
  print("Chemical lists updated.")

  old_chemsim='integrated_chem_chem_sim_threshold03_reduced.csv.gz' #list of chem-chem sim already calculated
  #reduced version contains upper-triangle only
  if Path(old_chemsim).is_file(): #decompress if compressed
    old_chemsim=decompress_gzip(old_chemsim)
  else:
    old_chemsim='integrated_chem_chem_sim_threshold03_reduced.csv' #list of chem-chem sim already calculated

  with open(old_chemsim,'a') as out: #append new chemical similarities to old version
    with open(new_sim_file,'r') as f:
      for line in f:
        line=line.strip()
        out.write("{}\n".format(line))
  new_chemsim=compress_gzip(old_chemsim,ext='gz',level=9)
  print(new_chemsim)
  print("Similarity file updated.")
  
def main(new_chemical_file):
  cutoff=0.3 #minimum similarity to record
  similarity_outfile=''.join(random.choice(string.ascii_uppercase + string.digits+ string.ascii_lowercase) for _ in range(16))
  #get new chemicals
  new_chems=find_new_chemicals(new_chemical_file)
  num_new_chems=len(new_chems)
  if len(new_chems)==0:
    print("Nothing to update in {}".format(new_chemical_file))
    sys.exit()
  #prepare MadFast input for new chemicals
  newchem_smiles=printout_new_chemicals(new_chems)
  
  print("New chemicals are in {}".format(newchem_smiles))
 
  oldchem_smiles='integrated_chemicals.smi.gz' #list of SMILES\tIndex ; gzip compressed
  if Path(oldchem_smiles).is_file(): #decompress if compressed
    oldchem_smiles=decompress_gzip(oldchem_smiles)
  else:
    oldchem_smiles='integrated_chemicals.smi' #list of SMILES\tIndex   
  
  #calculate similarity for new-all, new-new
  madfast_out=run_madfast(newchem_smiles,newchem_smiles,cutoff) #new-new calculation
  parse_sim(madfast_out, similarity_outfile, ut=True) #record only upper-triangle
  delete_file(madfast_out)
  madfast_out=run_madfast(oldchem_smiles,newchem_smiles,cutoff) #old-new calculation; make sure the order, old-new, NOT new-old
  parse_sim(madfast_out, similarity_outfile, ut=True) #record only upper-triangle
  delete_file(madfast_out)
  finalize_update(new_chems,oldchem_smiles,'integrated_chemicals.tsv.gz',similarity_outfile) #finalize update
  delete_file(newchem_smiles)
  print("{} new chemicals found and updated".format(num_new_chems))
  print("New similarity info are in {}.\nCheckout the file and delete it as necessary.".format(similarity_outfile))
  print("Update complete. Make sure to update github repo.")

if __name__=='__main__':
  #new_chemical_file='/data/saturn/a/hlim/others/Qiao/synergy/chemicals.tsv'
  new_chemical_file='extra_chem.txt'
  main(new_chemical_file)
