import os
import re
import sys
import csv

def get_idmap():
  uniprot='uniprot_id_mapping.tsv'
  gene2uniprot={} #genename -> reference uniprot accession
  uniprot2genes={} #reference uniprot accession -> list of genenames
  mutgene2mutuniprot={} #mutant genename -> mutant uniprot accession; ABL1(E255K) -> P00519(E255K)
  mutgene2filename={} #mutant genename -> name for individual file; ABL1(E255K) -> P00519_E255K
  with open(uniprot,'r') as f:
    next(f)
    for line in f:
      line=line.strip().split('\t')
      uni=line[0]
      g1=line[3].strip()
      g2=line[4].strip()
      g3=line[5].strip()
      genes=[]
      if len(g1)>0:
        g1=g1.split(' ')
        genes+=g1
      if len(g2)>0:
        g2=g2.split(' ')
        genes+=g2
      if len(g3)>0:
        g3=g3.split(' ')
        genes+=g3
      genes=list(set(genes))
      uniprot2genes[uni]=genes
      for g in genes:
        gene2uniprot[g]=uni

  with open('mutant_id_conversion.tsv','r') as f:
    for line in f:
      line=line.strip().split('\t')
      mutgene=line[0]
      mutuniprot=line[1]
      filename=line[2]
      mutgene2mutuniprot[mutgene]=mutuniprot
      gene2uniprot[mutgene]=mutuniprot
      uniprot2genes[mutuniprot]=[mutgene]
      mutgene2filename[mutgene]=filename

  return gene2uniprot,uniprot2genes,mutgene2mutuniprot,mutgene2filename

if __name__=='__main__':
  complexes=['ABL1_BCR','ALK_EML4','ALK_NPM1',\
             'CSNK2A1_CSNK2B_CSNK2A2','MAP3K7_TAB1',\
             'MTOR_FKBP1A','PIK3CA_PIK3R1','PIK3CB_PIK3R1',\
             'PIK3CD_PIK3R1','PRKAA1_PRKAB1_PRKAG1','PRKAA1_PRKAB2_PRKAG1',\
             'PRKAA2_PRKAB1_PRKAG1','STK11_CAB39_STRADA',\
             'CDK1_CCNA2','CDK1_CCNB1','CDK1_CCNE1',\
             'CDK2_CCNA2','CDK2_CCNE1','CDK3_CCNE1',\
             'CDK4_CCND1','CDK4_CCND3','CDK5_CDK5R1_p25',\
             'CDK5_CDK5R1_p35','CDK6_CCND1','CDK6_CCND3',\
             'CDK7_CCNH_MNAT1','CDK8_CCNC','CDK9_CCNK',\
             'CDK9_CCNT1']
  gene2uniprot,uniprot2genes,mutgene2mutuniprot,mutgene2filename=get_idmap()
  with open('jcim_target','r') as f:
    for line in f:
      line=line.strip()
      if line in complexes:
        continue
      try:
        uni=gene2uniprot[line]
        print("{}\t{}".format(line,uni))
      except:
        try:
          uni=mutgene2mutuniprot[line]
          print("{}\t{}".format(line,uni))
        except:
          print("{} Not found".format(line))
