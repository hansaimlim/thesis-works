#!/usr/bin/python
import MySQLdb as db
import sys
import pickle

def load_pickle(filename):
  with open(filename,'rb') as handle:
    data=pickle.load(handle)
  return data
def save_pickle(data,filename):
  with open(filename,'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
def get_compound_by_molregno(molregno):
  val=(str(molregno),)
  qry="SELECT moldict.chembl_id, cstr.standard_inchi_key, cstr.canonical_smiles FROM chembl_24.compound_structures cstr\
       INNER JOIN chembl_24.molecule_dictionary moldict ON cstr.molregno = moldict.molregno\
       WHERE cstr.molregno=%s"
  cur.execute(qry,val)
  structure=cur.fetchone()
  qry="SELECT mw_freebase, alogp, hba, hbd, psa, rtb, ro3_pass, num_ro5_violations, acd_most_apka, acd_most_bpka, acd_logp, acd_logd, molecular_species, aromatic_rings,\
       heavy_atoms, qed_weighted, hba_lipinski, hbd_lipinski, num_lipinski_ro5_violations FROM compound_properties WHERE molregno=%s"
  cur.execute(qry,val)
  properties=cur.fetchone()
  return (structure,properties)

def get_all_molregno():
  qry="SELECT distinct(molregno) FROM chembl_24.compound_records"
  cur.execute(qry)
  result=cur.fetchall()
  return result

  
passwd=raw_input("MySQL password?")
con=db.connect(host='127.0.0.1',user='hlim',passwd=passwd,db='chembl_24')
cur=con.cursor()

#compound_file='./ChEMBL23_compounds_list.csv'
#chem_prot_index='./ChEMBL23_chem_prot.csv'
print("Start parsing ChEMBL23 database...")
molregnos=get_all_molregno()

ikey2chembl={}
chembl2ikey={}
mol2chembl={}
chembl2mol={}
count=1

outfile=open('./ChEMBL24_all_compounds.csv','w')
outfile.write("Index, ChEMBL_ID, Molregno, InChIKey, SMILES, MolWeight, AlogP, HBAcc, HBDonor, PSArea, RotaBonds, Ro3Pass, Ro5Viol, aPKA, bPKA, logP, logD, MolType, Aromatic, Hatoms, QED\n")
for mol in molregnos:
  mol=str(mol[0])
  structure, prop = get_compound_by_molregno(mol)
  try:
    chembl,ikey,smile=str(structure[0]),str(structure[1]),str(structure[2])
  except:
    continue
  ikey2chembl[ikey]=chembl
  chembl2ikey[chembl]=ikey
  mol2chembl[mol]=chembl
  chembl2mol[chembl]=mol
  
  try:
    mw,alogp,hba,hbd,psa,rtb,ro3pass,ro5viol,apka,bpka,logp,logd,moltype,aromatic,hatoms,qed=prop[0],prop[1],prop[2],prop[3],prop[4],\
    prop[5],prop[6],prop[7],prop[8],prop[9],prop[10],prop[11],prop[12],prop[13],prop[14],prop[15]
    outfile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(count,chembl,mol,ikey,\
    smile,mw,alogp,hba,hbd,psa,rtb,ro3pass,ro5viol,apka,bpka,logp,logd,moltype,aromatic,hatoms,qed))
  except:
    outfile.write("{}, {}, {}, {}, {},\n".format(count,chembl,mol,ikey,smile))
  count+=1
  
save_pickle(ikey2chembl,'./ikey2chembl.pickle')
save_pickle(chembl2ikey,'./chembl2ikey.pickle')
save_pickle(mol2chembl,'./mol2chembl.pickle')
save_pickle(chembl2mol,'./chembl2mol.pickle')
