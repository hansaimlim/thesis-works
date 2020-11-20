import os
import sys
import csv
import glob
import numpy as np
#Small Molecule HMS LINCS ID,Small Molecule Name,Protein HMS LINCS ID,Protein Name,Kd,Conc unit
#Small Molecule HMS LINCS ID,Small Molecule Name,Protein HMS LINCS ID,Protein Name,% Control,Assay compound conc,Conc unit
def parse_csv(infile,chem2ikey,pair2kd,pair2pi,pair2piconc):
  with open(infile,'r') as inf:
    header=inf.readline()
    header=header.strip().split(',')
    atype=header[4] #assay type
    
    for l in  csv.reader(inf, quotechar='"', delimiter=',',quoting=csv.QUOTE_ALL, skipinitialspace=True):
      chemname=l[1]
      prot_discoverx=str(l[3]).replace(' ','') #remove empty space in protein id
      pair=chemname+'\t'+prot_discoverx
      activity=l[4]
      try:
        activity=np.float(activity)
      except:
        continue
      if atype=='Kd':
        pair2kd[pair]=activity
      elif atype=="% Control":
        unit=l[-1] #nM or uM
        conc=np.float(l[-2]) #compound concentration
        if unit=='nM':
          conc=conc*np.float(1e-9) #nM to M
        elif unit=='uM':
          conc=conc*np.float(1e-6) #uM to M
        else:
          print("Error. Check the unit {} for pair {}".format(unit,pair))
          sys.exit()
        pair2pi[pair]=activity
        pair2piconc[pair]=conc
        
  return pair2kd,pair2pi,pair2piconc

def get_chem2ikey():
  chem2ikey={}
  chemfile='/raid/home/hansaimlim/dataset/kinome_assay/LINCS/HMS-LINCS_KinomeScan_chemicals.txt'
  with open(chemfile,'r') as inf:
    for line in inf:
      line=line.strip().split('\t')
      chemname=str(line[0])
      ikey=str(line[1])
      smi=str(line[2])
      chem2ikey[chemname]=ikey
  return chem2ikey
def get_ikey2smi():
  ikey2smi={}
  chemfile='/raid/home/hansaimlim/dataset/kinome_assay/LINCS/HMS-LINCS_KinomeScan_chemicals.txt'
  with open(chemfile,'r') as inf:
    for line in inf:
      line=line.strip().split('\t')
      chemname=str(line[0])
      ikey=str(line[1])
      smi=str(line[2])
      ikey2smi[ikey]=smi
  return ikey2smi

def main():
  chem2ikey=get_chem2ikey()
  assays=glob.glob("/raid/home/hansaimlim/dataset/kinome_assay/LINCS/csv/train_and_dev/*.csv")
  pair2kd={}
  pair2pi={}
  pair2piconc={} #compound conc for percent inhibition assay
  for assay in assays:
    pair2kd,pair2pi,pair2piconc=parse_csv(assay,chem2ikey,pair2kd,pair2pi,pair2piconc)
  pairs=list(set(list(pair2kd.keys())+list(pair2pi.keys())))
  for pair in pairs:
    try:
      kd=pair2kd[pair]
    except:
      kd=None
    try:
      pi=pair2pi[pair]
    except:
      pi=None
    try:
      kd=float(kd)
      pi=float(pi)
      piconc=pair2piconc[pair]
      print("{}\t{}\t{}\t{}".format(pair,kd,pi,piconc))
    except:
      continue
if __name__=='__main__':
  main()
