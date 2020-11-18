import os
import sys
import numpy as np
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description="Convert multple FASTA files into one training/test set for masked lm")
    parser.add_argument('--input_dir',type=str,required=True)
    parser.add_argument('--output_dir',type=str,required=True)
    parser.add_argument('--test_prob',type=float,default=0.05,help='fraction of data for test set')
    return parser.parse_args()

def main(args):
    input_files = os.listdir(args.input_dir)
    out_train = open(os.path.join(args.output_dir,'train.txt'),'w')
    out_test = open(os.path.join(args.output_dir,'test.txt'),'w')
    out_ids = open(os.path.join(args.output_dir,'train_seqid.txt'),'w')
    for input_file in input_files:

        with open(os.path.join(args.input_dir,input_file),'r') as inf:
            seq = ''
            seqid = ''
            for line in inf:
                if line.startswith(">"):
                    if seq:
                        seq = ' '.join([r for r in seq])
                        seq += "\n" #append newline
                        if np.random.rand() < args.test_prob:
                            #sequences appear in test set
                            #they can also present in training set as it is for masked language modeling task
                            out_test.write(seq)
                        out_train.write(seq)
                        out_ids.write(seqid.replace('>','')+"\n")
                        #flush previous sequence
                        seq = ''
                        seqid = ''
                    seqid = line.strip().split('|')[0]
                else:
                    seq+=line.strip()

    out_train.close()
    out_test.close()
    out_ids.close()

if __name__ == '__main__':
    args=parse_args()
    main(args)