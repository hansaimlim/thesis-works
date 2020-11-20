#!/usr/bin/python
import MySQLdb as db
import sys
import numpy as np
def get_ic50_activities():
    #collect activity information from ChEMBL db
    #standard_units = nM only
    qry="SELECT cstr.standard_inchi_key, ay.variant_id, ay.tid, act.standard_value, act.standard_relation, act.standard_units\
         FROM chembl_24.activities act INNER JOIN chembl_24.assays ay ON act.assay_id=ay.assay_id\
         INNER JOIN compound_structures cstr ON cstr.molregno=act.molregno\
         WHERE act.potential_duplicate = 0 AND act.standard_type = 'IC50'\
         AND ay.confidence_score>=8 AND act.standard_units='nM'"
    cur.execute(qry)
    result=cur.fetchall()
    return result
def get_pic50_activities():
    #collect activity information from ChEMBL db
    #standard_units = NULL only
    qry="SELECT cstr.standard_inchi_key, ay.variant_id, ay.tid, act.standard_value, act.standard_relation, act.standard_units\
         FROM chembl_24.activities act INNER JOIN chembl_24.assays ay ON act.assay_id=ay.assay_id\
         INNER JOIN compound_structures cstr ON cstr.molregno=act.molregno\
         WHERE act.potential_duplicate = 0 AND act.standard_type = 'pIC50'\
         AND ay.confidence_score>=8 AND act.standard_units is NULL"
    cur.execute(qry)
    result=cur.fetchall()
    return result
def get_kd_activities():
    #collect activity information from ChEMBL db
    #standard_units = nM only
    qry="SELECT cstr.standard_inchi_key, ay.variant_id, ay.tid, act.standard_value, act.standard_relation, act.standard_units\
         FROM chembl_24.activities act INNER JOIN chembl_24.assays ay ON act.assay_id=ay.assay_id\
         INNER JOIN compound_structures cstr ON cstr.molregno=act.molregno\
         WHERE act.potential_duplicate = 0 AND act.standard_type = 'Kd'\
         AND ay.confidence_score>=8 AND act.standard_units = 'nM'"
    cur.execute(qry)
    result=cur.fetchall()
    return result
def get_pkd_activities():
    #collect activity information from ChEMBL db
    #standard_units = either '-' or NULL; no need to restrict
    qry="SELECT cstr.standard_inchi_key, ay.variant_id, ay.tid, act.standard_value, act.standard_relation, act.standard_units\
         FROM chembl_24.activities act INNER JOIN chembl_24.assays ay ON act.assay_id=ay.assay_id\
         INNER JOIN compound_structures cstr ON cstr.molregno=act.molregno\
         WHERE act.potential_duplicate = 0 AND act.standard_type = 'pKD'\
         AND ay.confidence_score>=8"
    cur.execute(qry)
    result=cur.fetchall()
    return result
def get_ki_activities():
    #standard_units = 
    #collect activity information from ChEMBL db
    qry="SELECT cstr.standard_inchi_key, ay.variant_id, ay.tid, act.standard_value, act.standard_relation, act.standard_units\
         FROM chembl_24.activities act INNER JOIN chembl_24.assays ay ON act.assay_id=ay.assay_id\
         INNER JOIN compound_structures cstr ON cstr.molregno=act.molregno\
         WHERE act.potential_duplicate = 0 AND act.standard_type = 'Ki'\
         AND ay.confidence_score>=8 AND act.standard_units = 'nM'"
    cur.execute(qry)
    result=cur.fetchall()
    return result
def get_pki_activities():
    #standard_units = NULL only
    #collect activity information from ChEMBL db
    qry="SELECT cstr.standard_inchi_key, ay.variant_id, ay.tid, act.standard_value, act.standard_relation, act.standard_units\
         FROM chembl_24.activities act INNER JOIN chembl_24.assays ay ON act.assay_id=ay.assay_id\
         INNER JOIN compound_structures cstr ON cstr.molregno=act.molregno\
         WHERE act.potential_duplicate = 0 AND act.standard_type = 'pKi'\
         AND ay.confidence_score>=8 AND standard_units is NULL"
    cur.execute(qry)
    result=cur.fetchall()
    return result
def get_compound(molregno):
    val=(str(molregno),)
    qry="SELECT cstr.standard_inchi_key, cstr.canonical_smiles, moldict.chembl_id FROM chembl_24.compound_structures cstr\
         INNER JOIN molecule_dictionary moldict ON cstr.molregno = moldict.molregno\
         WHERE cstr.molregno=%s"
    cur.execute(qry,val)
    result=cur.fetchone()
    return result
def get_target_accession_if_mutation(tid,mutation_id=None):
  if mutation_id:
    qry="SELECT varseq.accession, varseq.mutation FROM variant_sequences varseq WHERE variant_id = %s"
    val=(mutation_id,)
  else:
    qry="SELECT cseq.accession FROM component_sequences cseq\
         INNER JOIN target_components tcomp ON cseq.component_id=tcomp.component_id\
         WHERE tcomp.tid=%s"
    val=(tid,)
  cur.execute(qry,val)
  result=cur.fetchone()
  return result

def get_target(tid,mutation_id=None):
    if mutation_id:
      val=(mutation_id,)
      qry="SELECT varseq.accession, varseq.sequence, varseq.mutation, varseq.isoform FROM variant_sequences varseq\
           WHERE variant_id = %s" 
    else:
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

def output_activity(activities,outfile,neglog=False):
  #expect nM unit with neglog=True
  #or expect unitless -log(activity); activity in Molar unit
  with open(outfile,'a') as out:
    for activity in activities:
      ikey=activity[0]
      try:
        ikey=str(ikey)
      except:
        print("Skipping.........1")
        continue
      variant_id=activity[1] #could be none if wild-type
      tid=activity[2]
      accessions=get_target_accession_if_mutation(tid,mutation_id=variant_id)
      if accessions is None:
        print("Skipping.........2")
        continue
      elif len(accessions)==2:
        if variant_id is None:
          print("Skipping.........3")
          continue
        else:
          accession=str(accessions[0])+"("+str(accessions[1])+")"
      else:
        accession=str(accessions[0])
      value=activity[3]
      rel=activity[4]
      unit=activity[5]
      if value is None:
        print("Skipping.........4")
        continue
      if neglog:
        #take negative log of the value
        value=-np.log10(float(value))+float(9.0)
        if rel is None:
          rel='NA'
        elif rel=='>':
          rel='<'
        elif rel=='>>':
          rel='<<'
        elif rel=='>=':
          rel='<='
        elif rel=='<':
          rel='>'
        elif rel=='<<':
          rel='>>'
        elif rel=='<=':
          rel='>='
        elif rel=='=':
          rel='='
        elif rel=='~':
          rel='='
        else:
          print("Skipping.........6")
          continue
      out.write("{}\t{}\t{}\t{}\n".format(ikey,accession,rel,value))
passwd=raw_input("MySQL password?")
con=db.connect(host='127.0.0.1',user='hlim',passwd=passwd,db='chembl_24')
cur=con.cursor()

pic50_file='./ChEMBL24_pIC50.tsv'
pki_file='./ChEMBL24_pKi.tsv'
pkd_file='./ChEMBL24_pKd.tsv'
#chem_prot_index='./ChEMBL23_chem_prot.csv'
chemInfo_file='./ChEMBL24_chemInfo.tsv'
protInfo_file='./ChEMBL24_protInfo.tsv'
prot_fasta_file='./ChEMBL24_prot_seq.fas'
print("Start parsing ChEMBL24 database...")


ic50_activities=get_ic50_activities()
print("parsing {} IC50 activity data".format(len(ic50_activities)))
output_activity(ic50_activities,pic50_file,neglog=True);ic50_activities=None
pic50_activities=get_pic50_activities()
print("parsing {} pIC50 activity data".format(len(pic50_activities)))
output_activity(pic50_activities,pic50_file,neglog=False);pic50_activities=None
kd_activities=get_kd_activities()
print("parsing {} kd activity data".format(len(kd_activities)))
output_activity(kd_activities,pkd_file,neglog=True);kd_activities=None
pkd_activities=get_pkd_activities()
print("parsing {} pkd activity data".format(len(pkd_activities)))
output_activity(pkd_activities,pkd_file,neglog=False);pkd_activities=None

ki_activities=get_ki_activities()
print("parsing {} ki activity data".format(len(ki_activities)))
output_activity(ki_activities,pki_file,neglog=True);ki_activities=None
pki_activities=get_pki_activities()
print("parsing {} pki activity data".format(len(pki_activities)))
output_activity(pki_activities,pki_file,neglog=False);pki_activities=None

