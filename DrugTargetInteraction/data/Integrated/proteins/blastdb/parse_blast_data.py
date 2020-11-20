import os
import sys
import numpy as np

data='blast_1e8.dat'
self_score={}
with open(data,'r') as f:
  for line in f:
    line=line.strip().split('\t')
    g1=line[0].strip().split('|')[0]
    g2=line[1].strip().split('|')[0]
    score=float(line[-1])
    if g1==g2:
      self_score[g1]=score

with open(data,'r') as f:
  for line in f:
    line=line.strip().split('\t')
    g1=line[0].strip().split('|')[0]
    g2=line[1].strip().split('|')[0]
    score=float(line[-1])/float(self_score[g1])
    if g1==g2:
      continue
    print("{}\t{}\t{:.6f}".format(g1,g2,score))

