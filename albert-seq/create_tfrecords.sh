#!/usr/bin/env bash
OUTPUT_DIR=/data/saturn/a/hlim/Pfam/pli_bert/pfam_triplet_corpora/testout/ #where to output tfrecords?
filelist=( /data/saturn/a/hlim/Pfam/pli_bert/pfam_triplet_corpora/* ) #where are the corpora?
num_files=${#filelist[@]} #17747

vocabfile=/data/saturn/a/hlim/Pfam/pli_bert/data/vocab/pfam_vocab_triplets.txt #triplet vocabs differ from singlet vocabs
max_iter=493
chunk_size=36 #how many files to process at a time : (chunk_size * max_iter) > num_files

#num_files=34 #for debugging
#max_iter=3 #for debugging
#chunk_size=10 #for debugging


#Albert-specific settings
dupefactor=20 # coverage = maskedprob*dupefactor
maxseqlen=256
maxpredperseq=40
maskedprob=0.15

for i in $( seq 0 $max_iter )
do
	if [ $i -eq $max_iter ]; then
		i1=$(( i*chunk_size))
		i2=$(( num_files - 1 ))
	else
		i1=$(( i*chunk_size ))
		i2=$(( i*chunk_size + chunk_size -1 ))
	fi

	for j in $( seq $i1 $i2)
	do
		infile=${filelist[$j]}
		base=${infile##*/}
		if [ $j -eq $i2 ]; then
		    python -m albert.create_pretraining_data \
			    --do_whole_word_mask=True \
			    --non_chinese=True \
			    --input_file=$infile \
			    --output_file=${OUTPUT_DIR}${base}.tfrecord \
			    --vocab_file=${vocabfile} \
			    --do_lower_case=True \
			    --dupe_factor=${dupefactor} \
			    --random_seed=${RANDOM} \
			    --max_seq_length=${maxseqlen} --max_predictions_per_seq=${maxpredperseq} --masked_lm_prob=${maskedprob} &&
			sleep 15s
		else
		    python -m albert.create_pretraining_data \
			    --do_whole_word_mask=True \
			    --non_chinese=True \
			    --input_file=$infile \
			    --output_file=${OUTPUT_DIR}${base}.tfrecord \
			    --vocab_file=${vocabfile} \
			    --do_lower_case=True \
			    --dupe_factor=${dupefactor} \
			    --random_seed=${RANDOM} \
			    --max_seq_length=${maxseqlen} --max_predictions_per_seq=${maxpredperseq} --masked_lm_prob=${maskedprob} &
		fi
	done
done
