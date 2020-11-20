import os
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
def get_all_targets():
  qry="SELECT td.chembl_id, cseq.accession, csyn.component_synonym, td.pref_name, td.tax_id, td.organism FROM target_dictionary td\
       INNER JOIN target_components tc ON td.tid=tc.tid \
       INNER JOIN component_sequences cseq ON tc.component_id=cseq.component_id\
       INNER JOIN component_synonyms csyn ON tc.component_id=csyn.component_id\
       WHERE td.target_type='SINGLE PROTEIN' AND csyn.syn_type='GENE_SYMBOL'"
  cur.execute(qry)
  result=cur.fetchall()
  return result
def get_all_variants():
  qry="SELECT vseq.accession, vseq.mutation, csyn.component_synonym, vseq.organism FROM variant_sequences vseq\
       INNER JOIN component_sequences cseq ON vseq.accession=cseq.accession\
       INNER JOIN component_synonyms csyn ON cseq.component_id=csyn.component_id\
       WHERE csyn.syn_type='GENE_SYMBOL'"
  cur.execute(qry)
  result=cur.fetchall()
  return result

passwd=raw_input("MySQL password?")
con=db.connect(host='127.0.0.1',user='hlim',passwd=passwd,db='chembl_24')
cur=con.cursor()

print("Start parsing ChEMBL24 database...")

targets=get_all_targets()
outfile=open('./ChEMBL24_all_targets.csv','w')
outfile.write("UniProt\tGeneSymbol\tChEMBL\tPrefName\tTaxID\tOrganism\n")
for target in targets:
  chembl=target[0]
  acc=target[1]
  synonym=target[2]
  pref=target[3]
  taxid=target[4]
  org=target[5]
  outfile.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(acc,synonym,chembl,pref,taxid,org))
outfile.close()
#save_pickle(ikey2chembl,'./ikey2chembl.pickle')
variants=get_all_variants()
outfile=open('./ChEMBL24_all_variants.tsv','w')
outfile.write("UniProt\tMutation\tOrganism\n")
for var in variants:
  acc=var[0]
  mut=var[1]
  syn=var[2]
  org=var[3]
  outfile.write("{}\t{}\t{}\t{}\n".format(acc,mut,syn,org))

outfile.close()
sys.exit()
