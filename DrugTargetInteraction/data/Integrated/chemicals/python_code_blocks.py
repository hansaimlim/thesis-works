import os
import sys
import numpy as np
import urllib3
import json
from rdkit import Chem
urllib3.disable_warnings()

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


def smi2ikey(smi):
  """
  Convert SMILES to InChIKey
  RDKit is required
  """
  #smi='CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O' #naproxen
  #inchikey = 'CMWTZPSULFXXJA-VIFPVBQESA-N'
  try:
    mol=Chem.rdmolfiles.MolFromSmiles(smi)
    ikey=Chem.inchi.MolToInchiKey(mol)
    return ikey
  except:
    return None

def cidsearch_pubchem(cid):
  """
  Obtain InChIKey and SMILES by PubChem CID
  urllib3 module is required
  make sure to delay a few seconds every 5 requests
  """
  cid=str(cid)
  target_url='https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/'+cid+'/property/InChIKey,CanonicalSMILES/json'
  http = urllib3.PoolManager()
  response = http.request('GET',target_url)
  data = response.data.decode('utf-8')
  if data is None:
    return None,None
  else:
    #{'PropertyTable': {'Properties': [{'CID': 2244, 'CanonicalSMILES': 'CC(=O)OC1=CC=CC=C1C(=O)O', 'InChIKey': 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N'}]}}
    data=json.loads(data)
    try:
      data=data['PropertyTable']['Properties'][0]
      cid=data['CID']
      smi=data['CanonicalSMILES']
      ikey=data['InChIKey']
      return ikey,smi
    except:
      return None,None

def ikeysearch_pubchem(ikey):
  """
  Search for SMILES by InChIKey using PubChem PUGREST
  urllib3 module is required
  make sure to delay a few seconds every 5 requests
  inchikey may take multiple inchikeys joined by commas
  e.g ikey='JFTHHZLVWUKENE-UHFFFAOYSA-P,BSYNRYMUTXBXSQ-UHFFFAOYSA-N' for two inchikeys in one request
  Limit to ~100 InChIKeys per request
  """
  ikey=str(ikey)
  target_url='https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/'+ikey+'/property/InChIKey,CanonicalSMILES,IsomericSMILES/json'
  http = urllib3.PoolManager()
  response = http.request('GET',target_url)
  data = response.data.decode('utf-8')
  if data is None:
    return None
  else:
    return json.loads(data)

def ikey2smi_cactus(ikey):
  """
  search for SMILES by InChIKey through NIH-CACTUS
  returns the first SMILES in case multiple zwitterionic form exists
  """
  #ikey='JFTHHZLVWUKENE-UHFFFAOYSA-P'
  #smi_ori='C(CCCCCNc1cc[n+](Cc2ccccc2)c3ccccc13)CCCCNc4cc[n+](Cc5ccccc5)c6ccccc46'
  target_url = 'http://cactus.nci.nih.gov/chemical/structure/InChIKey='+ikey+'/smiles?structure_index=0'
  http = urllib3.PoolManager()
  response = http.request('GET', target_url)
  data = response.data.decode('utf-8')
  return data

