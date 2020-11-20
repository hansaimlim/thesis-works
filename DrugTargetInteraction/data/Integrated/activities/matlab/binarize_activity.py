import os
import sys
import numpy as np

activity_path='../tsv/'
binary=activity_path+'integrated_binary_activity.tsv'
cont=activity_path+'integrated_continuous_activity.tsv'
pic50=activity_path+'integrated_pic50.tsv'
pkd=activity_path+'integrated_pkd.tsv'
pki=activity_path+'integrated_pki.tsv'

pair2continuous={}
pair2binary={}

with open(binary,'r') as f:
  next(f)
  for line in f:
    line=line.strip().split('\t')
    row=line[0]
    ikey=line[1]
    uni=line[2]
    pair=ikey+'\t'+uni
    activity=line[3]
    if activity=='Active':
      act=1
    elif activity=='Inactive':
      act=0
    else:
      continue
    try:
      act=pair2binary[pair]
      act.append(activity)
      pair2binary[pair]=act
    except:
      pair2binary[pair]=[activity]
    
contfiles=[pic50,pkd,pki]
#first clear multi-record pairs
for c in contfiles:
  #record continuous activity if relation is '='
  #record binary activity if relation is inequality

  ###conditions are stricter for records to be active
  #if activity > x where x >= 7.0 -> record active
  #if activity > x where x < 7.0 -> record inactive
  #if activity >> x where x >=6.3 -> record active
  #if activity >> x where x <6.3 -> record inactive
  #if activity < x where x < 8.0 -> record inactive
  #if activity < x where x >= 8.0 -> record active
  #if activity << x where x < 9.3 -> record inactive
  #if activity << x where x >= 9.3 -> record active
  with open(c,'r') as f:
    next(f)
    for line in f:
      line=line.strip().split('\t')
      row=line[0]
      ikey=line[1]
      uni=line[2]
      assay_type=line[3]
      rel=line[4]
      try:
        val=float(line[5])
      except:
#        with open('error_log.txt','a') as out:
#          out.write("row {} in file {}. Line={}\n".format(row,c,line))
        continue
      pair=ikey+'\t'+uni
      if rel=='=': #continuous record
        try:
          act=pair2continuous[pair]
          act.append(val)
          pair2continuous[pair]=act
        except:
          pair2continuous[pair]=[val]
      else:
        #inequality. binary record
        if rel=='<' or rel=='<=':
        #if activity < x where x < 8.3 -> record inactive
        #if activity < x where x >= 8.3 -> record active
          if val >=8.3:
            #binary active
            activity=1
          else:
            #binary inactive
            activity=0
          try:
            act=pair2binary[pair]
            act.append(activity)
            pair2binary[pair]=act
          except:
            pair2binary[pair]=[activity]
        elif rel=='<<':
        #if activity << x where x < 9.3 -> record inactive
        #if activity << x where x >= 9.3 -> record active
          if val >=9.3:
            #binary active
            activity=1
          else:
            #binary inactive
            activity=0
          try:
            act=pair2binary[pair]
            act.append(activity)
            pair2binary[pair]=act
          except:
            pair2binary[pair]=[activity]

        elif rel=='>' or rel=='>=':
        #if activity > x where x >= 7.0 -> record active
        #if activity > x where x < 7.0 -> record inactive
          if val >=7.0:
            #binary active
            activity=1
          else:
            #binary inactive
            activity=0
          try:
            act=pair2binary[pair]
            act.append(activity)
            pair2binary[pair]=act
          except:
            pair2binary[pair]=[activity]

        elif rel=='>>':
        #if activity >> x where x >=6.3 -> record active
        #if activity >> x where x <6.3 -> record inactive
          if val >=6.3:
            #binary active
            activity=1
          else:
            #binary inactive
            activity=0
          try:
            act=pair2binary[pair]
            act.append(activity)
            pair2binary[pair]=act
          except:
            pair2binary[pair]=[activity]

        else:
          continue



for pair in pair2continuous:
  score=0
  cont_activity=pair2continuous[pair]
  try:
    binary_activity=pair2binary[pair]
  except:
    binary_activity=[]
  
  #temporarily skip single-record pairs
#  if len(cont_activity)+len(binary_activity)==1:
#    continue  
  #temporarily skip single-record pairs

  for c in cont_activity:
    #strong potency gets more score than moderate ones
    if c>=10.0:
      score+=3.0
    elif c>=9.0:
      score+=2.0
    elif c>=8.0:
      score+=1.5
    elif c>=7.0:
      score+=1
    elif c>=6.0:
      score-=1
    elif c>=5.0:
      score-=1.5
    elif c>=4.0:
      score-=2.0
    else:
      score-=3.0
  for b in binary_activity:
    #binary activities takes 1 points each
    if b==1.0:
      score+=1
    else:
      score-=1
  if score>=0:
    decision=1
    with open('integrated_active.tsv','a') as out:
      out.write("{}\n".format(pair))
  else:
    decision=0
    with open('integrated_inactive.tsv','a') as out:
      out.write("{}\n".format(pair))
#  print("{}\t{},{}\t{},{}".format(pair,cont_activity,binary_activity,score,decision))
#  print("{}\t{}".format(pair,decision))
        
      


