import os
import sys
import csv
import numpy as np

def standardize_activity(valtype,rel,val,unit):
  #assume activity thresholds for:
  #pKd=7, pKi=5.3, pIC50=5 (values in Molarity before log)
  #return standard type and value (pKd, pKi, pIC50, pEC50, pAC50)
  
  #log values must be in NA unit. otherwise discard the data
  kdtypes=['KD','Kd','KDAPP','LOG KD','-LOG KD','LOG 1/KD'] #units: NM, NMOL/L, MM, or NA
  kitypes=['KI','Ki','LOG KI','LOGKI','PKI(UM)','-LOG KI'] #units: NM, NMOL/L, 
  ictypes=['IC50','-LOG IC50','LOG IC50','logIC50','LOGIC50','LOG(10^6/IC50)',"LOG(10'3/IC50)",'LOG(1/IC50)'] #units: same above
  ectypes=['EC50','AC50','LOG EC50','-LOG EC50'] #units: NM, NMOL/L, or NA
  bindtypes=['BINDING','BINDING ACTIVITY'] #units: %, NM, or NA
  acttypes=['POTENCY','ACTIVITY','INHIBITION','Inhibition','Activity','REMAINING ACTIVITY','INHIBITORY POTENCY']
  #units: NM, %, NA, UM/L,
  try:
    val=float(val)
  except:
    return None,None

  fliprelation=False
  if valtype in kdtypes:
    if valtype in ['KD','Kd','KDAPP']:
      if (unit=='NM') or (unit=='NMOL/L'):
        s_val=-np.log10(val)+float(9.0) #pKd=9.0 if Kd=1 nM
        fliprelation=True
      elif unit=='MM':
        s_val=-np.log10(val)+float(3.0)
        fliprelation=True
      else:
        s_val=None;s_type=None
    elif valtype=='LOG KD':
      s_val=-float(val)
      fliprelation=True
    elif (valtype=='-LOG KD') or (valtype=='LOG 1/KD'):
      s_val=float(val)
    else:
      s_unit=None
      s_val=None
  elif valtype in kitypes:
    #get pKi
    if valtype in ['KI','Ki']:
      if (unit=='NM') or (unit=='NMOL/L'):
        s_val=-np.log10(val)+float(9.0)
        fliprelation=True
      else:
        s_val=None;s_type=None
    elif valtype in ['LOG KI','LOGKI']:
      s_val=-val
      fliprelation=True
    elif valtype=='-LOG KI':
      s_val=val
    elif valtype=='PKI(UM)':
      s_val=float(val)+float(6.0)

  elif valtype in ictypes:
    #get pIC50
    if valtype in ['-LOG IC50','PIC50','LOG(1/IC50)']:
      s_val=val
    elif valtype in ['LOG IC50','logIC50','LOGIC50']:
      s_val=-val
      fliprelation=True
    else:
      s_val=None
      s_rel=None
#  elif valtype in ectypes:
    #get pEC50
  else:
    s_rel=None
    s_val=None
    return s_rel,s_val
  if fliprelation: #inequality sign flipped
    if rel=='=':
      s_rel=rel
    elif rel=='>=':
      s_rel='<='
    elif rel=='>':
      s_rel='<'
    elif rel=='<=':
      s_rel='>='
    elif rel=='<':
      s_rel='>'
    else:
      s_rel='=' #some unknown relation?
  else:
    s_rel=rel
  return s_rel,s_val

def standardize_activity_file(infile,outfile):
  with open(infile,'r') as inf:
    for line in inf:
      line=line.strip().split('\t')
      #chembl,ikey,chemname,mutationinfo,targetid,targetprefname,acttype,actrel,actval,actunit,\
      #inhibitor_type,compound_concentration,compound_concentration_unit,\
      #substrate_type,substrate_relation,substrate_value,substrate_units =
      try:
        chembl,ikey,chemname,geneinfo,targetid,targetprefname,acttype,actrel,actval,actunit =\
      line[0],line[1],line[2],line[3],line[4],line[5],line[6],line[7],line[8],line[9]
      except:
        continue
      s_rel,s_val=standardize_activity(acttype,actrel,actval,actunit)
      if (s_val is None) or (s_val <= 0.0):
        continue
      else:
        with open(outfile,'a') as out:
          out.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(chembl,ikey,geneinfo,targetid,s_rel,s_val))

files=['DTC_activity.tsv', 'DTC_binding.tsv', 'DTC_ec50.tsv', 'DTC_ic50.tsv', 'DTC_kd.tsv', 'DTC_ki.tsv']
standardize_activity_file(files[5],'DTC_pKi.tsv')
standardize_activity_file(files[4],'DTC_pKd.tsv')
standardize_activity_file(files[3],'DTC_pIC50.tsv')

