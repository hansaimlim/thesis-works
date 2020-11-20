import os
import re
import sys
import csv
import numpy as np

def parse_assays(assay,outfile,atype='kd'):
  
  with open(assay,'r') as inf:
    next(inf)
    count=0
    for line in inf:
      line=line.split('\t')
      ikey=line[0]
      uni=line[1]
      ki=line[2].strip().lstrip()
      ic50=line[3].strip().lstrip()
      kd=line[4].strip().lstrip()
      if atype=='kd':
        if kd=='' or kd==0.0:
          continue
      if atype=='ki':
        if ki=='' or ki==0.0:
          continue
      if atype=='ic50':
        if ic50=='' or ic50==0.0:
          continue
#      print(ikey,uni,ki,ic50,kd)
#      count+=1
#      if count==10:
#        sys.exit()
      if ki!='':
        match_ki=re.search( r'([><= ]+)([0-9.eE+\-]+)$', ki, re.M)
        if match_ki:
          rel=match_ki.group(1).strip().lstrip()
          ki=float(match_ki.group(2))
          if rel=='>':
            rel='<'
          elif rel=='<':
            rel='>'
          elif rel=='>>':
            rel='<'
          elif rel=='<<':
            rel='>'
          else:
            rel='='
        else:
          rel='='
        ki=float(ki)
        if ki==0.0:
          continue
        try:
          pkd=-np.log10(ki)+9.0
          with open(outfile,'a') as out:
            out.write("{}\t{}\t{}\t{}\n".format(ikey,uni,rel,pkd))
        except:
          continue
      if kd!='':
        match_kd=re.search( r'([><= ]+)([0-9.eE+\-]+)$', kd, re.M)
        if match_kd:
          rel=match_kd.group(1).strip().lstrip()
          kd=float(match_kd.group(2))
          if rel=='>':
            rel='<'
          elif rel=='<':
            rel='>'
          elif rel=='>>':
            rel='<'
          elif rel=='<<':
            rel='>'
          else:
            rel='='
        else:
          rel='='
        kd=float(kd)
        if kd==0.0:
          continue
        try:
          pkd=-np.log10(kd)+9.0
          with open(outfile,'a') as out:
            out.write("{}\t{}\t{}\t{}\n".format(ikey,uni,rel,pkd))
        except:
          continue
      if ic50!='':
        match_ic50=re.search( r'([><= ]+)([0-9.eE+\-]+)$', ic50, re.M)
        if match_ic50:
          rel=match_ic50.group(1).strip().lstrip()
          ic50=float(match_ic50.group(2))
          if rel=='>':
            rel='<'
          elif rel=='<':
            rel='>'
          elif rel=='>>':
            rel='<'
          elif rel=='<<':
            rel='>'
          else:
            rel='='
        else:
          rel='='
        ic50=float(ic50)
        if ic50==0.0:
          continue
        try:
          pkd=-np.log10(ic50)+9.0
          with open(outfile,'a') as out:
            out.write("{}\t{}\t{}\t{}\n".format(ikey,uni,rel,pkd))
        except:
          continue

pdsp='BindingDB_PDSPKi_activity.tsv'
#PDSP_Ki data
#InChIKey        UniProt Ki(nM)  IC50(nM)        Kd(nM)  EC50(nM)
#MLDQSYUQSLUEPG-UHFFFAOYSA-N     P43140   1.86
pubchem='BindingDB_PubChem_activity.tsv'
#InChIKey        UniProt Ki(nM)  IC50(nM)        Kd(nM)  EC50(nM)
#MWQOUSHKYPZGHQ-UHFFFAOYSA-N     Q01196           15300
bindingdb='BindingDB_BindingDB_Inhibition_activity.tsv'
#InChIKey        UniProt Ki(nM)  IC50(nM)        Kd(nM)  EC50(nM)
#MVQUQGLRQPMJPU-UHFFFAOYSA-N     P12931           8.7
uspatent='BindingDB_USPatent_activity.tsv'

outfile_kd='./BindingDB_pKd.tsv'
outfile_ic50='./BindingDB_pIC50.tsv'
outfile_ki='./BindingDB_pKi.tsv'
parse_assays(pdsp,outfile_kd,atype='kd')
parse_assays(pdsp,outfile_ic50,atype='ic50')
parse_assays(pdsp,outfile_ki,atype='ki')
parse_assays(pubchem,outfile_kd,atype='kd')
parse_assays(pubchem,outfile_ic50,atype='ic50')
parse_assays(pubchem,outfile_ki,atype='ki')
parse_assays(bindingdb,outfile_kd,atype='kd')
parse_assays(bindingdb,outfile_ic50,atype='ic50')
parse_assays(bindingdb,outfile_ki,atype='ki')
parse_assays(uspatent,outfile_kd,atype='kd')
parse_assays(uspatent,outfile_ic50,atype='ic50')
parse_assays(uspatent,outfile_ki,atype='ki')

sys.exit()
pair2activity={}
with open(outfile,'r') as f:
  for line in f:
    line=line.strip().split('\t')
    ikey=line[0]
    uni=line[1]
    rel=line[2]
    val=float(line[3])
    if val<0:
      continue
    pair=ikey+','+uni
    if rel!='=':
      continue
    try:
      activity_exist=pair2activity[pair]
      activity_exist=np.append(activity_exist,val)
      pair2activity[pair]=activity_exist
    except:
      pair2activity[pair]=np.array([val],dtype=np.float32)

with open('./bindingdb_pdspki_pubchem_uspatent_standardized_pkd.csv','w') as out:
  for pair in pair2activity.keys():
    mean_activity=np.nanmean(pair2activity[pair])
    out.write("{},{}\n".format(pair,mean_activity))
