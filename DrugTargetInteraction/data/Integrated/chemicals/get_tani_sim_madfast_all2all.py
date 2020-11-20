import os
import sys
import math
import time
import random
import string
import subprocess
from multiprocessing import Pool
from multiprocessing import freeze_support, cpu_count
import psutil
global madfast_path_global
madfast_path_global='/data/saturn/a/hlim/ChemAxon/madfast-cli-0.2.3/bin/'  #Change this to your madfast bin directory

def get_tani_sim_wholematrix(whole_smiles, threshold, outfile,nproc=9):
    #split smi file into pieces of equal lines
    # bash command split large_smi_file -l lines
    lines=2000
    threshold=float(threshold)
    outfile=str(outfile)
    split_smi_files=split_file(whole_smiles,lines)
    filecount=len(split_smi_files)
   
    inputs=[]
    for qfile in split_smi_files: #query-to-large file
      inp=(qfile,whole_smiles,threshold,outfile)
      inputs.append(inp)
    with Pool(nproc) as pool: #use multicore
      try:
        R=pool.starmap(get_madfast_result,inputs)
        print("{} total processes. Run {} processes at a time...".format(len(inputs),nproc))
      except BaseException as e:
        print("Error occurred:\n"+str(e)+"\n")
    print("{} total processes. Processed {} jobs at a time...".format(len(inputs),nproc))
    return split_smi_files

def get_madfast_result(qfile,large_smi_file,threshold,outfile):
  print("MadFast search for {} started...".format(qfile))
  OF=run_madfast(qfile,large_smi_file,threshold)
  print("Dissimilarity matrix {} created".format(OF))
  parse_sim(OF, ikey2idx, outfile)
  print("Similarity information appended to {}".format(outfile))
  delete_file(OF)
  print("Dissimilarity matrix {} deleted".format(OF))
 
def split_file(infile, lines):
    #split file by given lines and return file names
    files=[] #contains filenames
    wccommand="wc -l %s"%str(infile)
    output=subprocess.check_output(['bash','-c',wccommand])
    output=output.decode('utf8')
    total_lines=output.strip().split(' ')[0]
    print(total_lines)
    filecount=int(math.ceil(float(total_lines)/float(lines)))
    splitcommand="split %s -d -a 4 -l %d" %(str(infile),int(lines)) #file name will be x0000, x0001, x0002...
    output = subprocess.check_output(['bash','-c',splitcommand])
    print("File splitted into %s pieces."%str(filecount))
    suffix=int(0)
    for i in range(0,filecount):
        str_suffix=str(format(suffix,'04d'))
        fname="x"+str_suffix
        files.append(fname)
        suffix+=1
    return files

def delete_file(filename):
    delcommand="rm %s"%str(filename)
    output=subprocess.check_output(['bash','-c',delcommand])
    
def run_madfast(query_file,target_file,cutoff):
    cutoff=float(cutoff)
    #random named dissimilarity matrix to prevent collision
    tanimoto_filename=query_file+''.join(random.choice(string.ascii_uppercase + string.digits+ string.ascii_lowercase) for _ in range(20))
    try:
        madfast_command=os.path.join(madfast_path_global,'searchStorage.sh')+" -context createSimpleEcfp4Context -qmf {} -qidname\
 -tmf {} -tidname -mode MOSTSIMILARS -maxdissim {} -out {}".format(query_file,target_file,(1.0-cutoff),tanimoto_filename)
        output=subprocess.check_output(['bash','-c',madfast_command])
        return tanimoto_filename
    except BaseException as e:
        print('Error occurred during searchStorage: '+str(e)+"\n")
        with open('./madfast_cutoff_error_command.log','a') as errorlog:
          errorlog.write("{}\n".format(madfast_command))

def parse_sim(datfile, ikey2idx, outfile):
    #parse dissimilarity list
    #assume all listed pairs are filtered by cutoff, no cutoff needed here
    outfile=str(outfile)
    fout=open(outfile,"a+")
    linenum=0
    with open(datfile,"r") as inf:
      next(inf)
      for line in inf:
        if line=='': #skip empty lines
          continue
        line=line.strip().split('\t')
        idx1=line[0]
        idx2=line[1]
        dissim=float(line[2])
        sim=float(1.0-dissim)
        fout.write("{0:},{1:},{2:.10f}\n".format(idx1,idx2,sim))
    fout.close()
if __name__ == '__main__':
  psutil.cpu_count()
  p = psutil.Process()
  p.cpu_affinity()  # get
  p.cpu_affinity([i for i in range(5,47)])  # set; from now on, process will run on CPUs in the range
  threshold=0.3 #minimum similarity to record
  outfile='./integrated_chem_chem_sim_threshold03.csv' #output file for chemical-chemical similarity
  whole_smiles='./integrated_chemicals.smi' #each line is "SMILES\tID" for each chemical
  stime=time.time()
  split_smi_files=get_tani_sim_wholematrix(whole_smiles, threshold, outfile, nproc=20) #nproc=N to run N jobs at the same time
  for qfile in split_smi_files:
    delete_file(qfile)
    print("{} deleted".format(qfile))
  etime=float(time.time()-stime)/3600.0
  
  print("Total {:.2f} hours".format(etime)) #measure total time
