import os
import sys
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Seq import MutableSeq
from Bio.Alphabet import IUPAC

def load_uniprot2sequence(fasta):
  uniprot2seq={}
  for seq_record in SeqIO.parse(fasta,"fasta"):
    if seq_record.id=='alignment':
      continue
    seqids=str(seq_record.id).strip().split('|')
    seq=str(seq_record.seq)
    uni=seqids[0].replace('>','')
    uniprot2seq[uni]=seq
  return uniprot2seq

def get_mutation_info(gene):
  #assume mutant info is delimited by _ from the gene ID
  #assume multiple mutations are delimited by , (e.g. Y29T,K50I)
  gene=str(gene)
  ge=gene.strip().split('_')
  if len(ge)==1:
    #no mutation
    return gene,None
  g=ge[0]
  mut=ge[1].strip().split(',')
  return g,mut #list of mutations

def identify_mutation(mut):
  #e.g. mut='Y29T'
  #returns {'type':'point','pos':29,'ori':'Y','mut':'T'}
  #e.g. mut='L747-T751del'
  #returns {'type':'del','pos_start':747,'pos_end':751,'ori_start':'L','ori_end':'T'}
  delobj=re.match(r'(.*)-(.*)del',mut,re.M|re.I)
  insobj=re.match(r'(.*)ins',mut,re.M|re.I)
  pointobj=re.match(r'([A-Z])([0-9]+)([A-Z])',mut,re.M|re.I)
  if delobj:
    mtype='del'
    start=delobj.group(1)
    startobj=re.match(r'([A-Z])([0-9]+)',start,re.M|re.I)
    ori_start=startobj.group(1)
    pos_start=int(startobj.group(2))
    end=delobj.group(2)
    endobj=re.match(r'([A-Z])([0-9]+)',end,re.M|re.I)
    ori_end=endobj.group(1)
    pos_end=int(endobj.group(2))
    d={'type':mtype,'pos_start':pos_start,'pos_end':pos_end,'ori_start':ori_start,'ori_end':ori_end}
  elif insobj:
    mtype='ins'
    d={'type':mtype}
  elif pointobj:
    ori=pointobj.group(1)
    pos=int(pointobj.group(2))
    mut=pointobj.group(3)
    mtype='point'
    d={'type':mtype,'pos':pos,'ori':ori,'mut':mut}
  else:
    print("No mutation info identified from {}".format(mut))
    d=None
  return d

outpath='extra_fasta/'
whole_fasta='blastdb/whole_fasta.fas'
mutant_list='genes_not_found' #only point mutations at this point
uniprot2seq=load_uniprot2sequence(whole_fasta)
with open(mutant_list,'r') as f:
  count=0
  for line in f:
    gene=line.strip()
    gene_ori=gene
    gene,mut=get_mutation_info(gene)
    if mut is None:
      #no mutation
      continue
    try:
      seq=uniprot2seq[gene]
    except:
      print("{}".format(gene))
      continue
    for m in mut:
      #returns {'type':'point','pos':29,'ori':'Y','mut':'T'}
      md=identify_mutation(m)
      if md['type']=='point':
        if seq[md['pos']-1]==md['ori']: #residue matches
          seq=seq[:md['pos']-1]+md['mut']+seq[md['pos']:] #replace residue
        else:
          print("Residue mismatch in {}. {}th residue {} found, not {} in {}".format(gene,md['pos'],seq[md['pos']-1],md['ori'],m))
          continue
      else:
        print("Non-point mutation found {} in {}".format(m,gene))
        continue
    mutstring=','.join(mut)
    outfile=outpath+gene_ori+'.fas'
    seq_record = SeqRecord(Seq(seq,IUPAC.protein),gene_ori,'','')
    with open(outfile,'w') as out:
      SeqIO.write(seq_record,out,"fasta")
     
#    print(gene,mut)
#    print(uniprot2seq[gene]) #original sequence
#    print(seq) #mutant sequence
#    count+=1
#    if count==10:
#      sys.exit()



