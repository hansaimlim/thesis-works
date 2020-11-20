#!/usr/bin/python
import MySQLdb as db
import sys

def get_compound(molregno):
    val=(str(molregno),)
    qry="SELECT cstr.standard_inchi_key, cstr.canonical_smiles FROM chembl_24.compound_structures cstr\
         WHERE cstr.molregno=%s"
    cur.execute(qry,val)
    result=cur.fetchone()
    return result
def get_target(tid):
    val=(str(tid),)
    qry="SELECT cseq.accession, cseq.sequence, cseq.description, csyn.component_synonym FROM component_sequences cseq\
         INNER JOIN target_components tcomp ON cseq.component_id=tcomp.component_id\
         INNER JOIN component_synonyms csyn ON tcomp.component_id=csyn.component_id\
         WHERE tcomp.tid=%s AND cseq.component_type='PROTEIN'"
    cur.execute(qry,val)
    result=cur.fetchone()
    return result
def split_seq(seq, num):
    return [ seq[start:start+num] for start in range(0, len(seq), num) ]
def get_cheminfo(molregno):
    val=(str(molregno),)
    qry="SELECT chembl_id, pref_name, max_phase, therapeutic_flag, availability_type from molecule_dictionary\
         WHERE molregno=%s"
    cur.execute(qry,val)
    result=cur.fetchone()
    return result


passwd=raw_input("MySQL password?")
con=db.connect(host='127.0.0.1',user='hlim',passwd=passwd,db='chembl_24');
cur=con.cursor()

chemInfo_file='./ChEMBL24_chemInfo.tsv'
protInfo_file='./ChEMBL24_protInfo.tsv'
new_cheminfo='./ChEMBL24_chemInfo_detailed.tsv'
print "Start parsing ChEMBL24 database..."

S=open(new_cheminfo,"w")
S.write("ChemIndex\tChEMBL_id\tMolregno\tInChIKey\tCanonical_SMILES\n")
with open(chemInfo_file,"r") as inf:
    next(inf)
    for line in inf:
        line=line.strip().split("\t")
        chemidx=str(line[0])
        molregno=str(line[1])
        ikey=str(line[2])
        smi=str(line[3])
        record=get_cheminfo(molregno)
        chembl_id=str(record[0])
        S.write(chemidx+"\t"+chembl_id+"\t"+molregno+"\t"+ikey+"\t"+smi+"\n")
