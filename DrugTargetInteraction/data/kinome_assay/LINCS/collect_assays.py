import os
import sys
import csv
import glob
import numpy as np
#Small Molecule HMS LINCS ID,Small Molecule Name,Protein HMS LINCS ID,Protein Name,Kd,Conc unit
#Small Molecule HMS LINCS ID,Small Molecule Name,Protein HMS LINCS ID,Protein Name,% Control,Assay compound conc,Conc unit
def parse_csv(infile,chem2ikey,pair2kd,pair2kdunit,pair2pi,pair2piconc):
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
        activity=-1 #mark for inactive pairs
      if atype=='Kd':
        pair2kd[pair]=activity
        pair2kdunit[pair]=l[-1]
      elif atype=="% Control":
        unit=l[-1] #nM or uM
        conc=np.float(l[-2]) #compound concentration
        if unit=='nM':
          conc=conc
        elif unit=='uM':
          conc=conc*1000.0 #to nM
        else:
          print("Error. Check the unit {} for pair {}".format(unit,pair))
          sys.exit()
        pair2pi[pair]=activity
        pair2piconc[pair]=conc
        
  return pair2kd,pair2kdunit,pair2pi,pair2piconc

def get_chem2ikey():
  chem2ikey={}
  chemfile='/raid/home/hansaimlim/DrugTargetInteraction/data/kinome_assay/LINCS/list/HMS-LINCS_KinomeScan_chemicals.txt'
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
  chemfile='/raid/home/hansaimlim/DrugTargetInteraction/data/kinome_assay/LINCS/list/HMS-LINCS_KinomeScan_chemicals.txt'
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
  assays=glob.glob("/raid/home/hansaimlim/DrugTargetInteraction/data/kinome_assay/LINCS/assays/*.csv")
  pair2kd={}
  pair2kdunit={}
  pair2pi={}
  pair2piconc={} #compound conc for percent inhibition assay
  for assay in assays:
    pair2kd,pair2kdunit,pair2pi,pair2piconc=parse_csv(assay,chem2ikey,pair2kd,pair2kdunit,pair2pi,pair2piconc)
  pairs=list(set(list(pair2kd.keys())+list(pair2pi.keys())))
  for pair in pairs:
    try:
      kd=pair2kd[pair]
      if kd==-1: #inactive; no measurement
        with open('./LINCS_kinomescan_kd_inactive_null.tsv','a') as out:
          out.write("{}\n".format(pair))
      else:
        kd=np.float(kd)
        unit=pair2kdunit[pair]
        if unit=='uM':
          kd=kd*1000.0 #to nM
        elif unit=='nM':
          kd=kd
        else:
          print("Units other than uM or nM found. {} {} for pair {}".format(kd,unit,pair))
          sys.exit()
        with open('./LINCS_kinomescan_kd_nM.tsv','a') as out:
          out.write("{}\t{}\n".format(pair,kd))
    except:
      kd=None
    try:
      pi=pair2pi[pair]
      if pi==-1:
        with open('./LINCS_kinomescan_pi_inactive_null.tsv','a') as out:
          out.write("{}\n".format(pair))
      else:
        pi=np.float(pi)
        piconc=pair2piconc[pair] #compound concentration in nM
        with open('./LINCS_kinomescan_pi_nM.tsv','a') as out:
          out.write("{}\t{}\t{}\n".format(pair,pi,piconc))
    except:
      pi=None
if __name__=='__main__':
  main()
