#!/usr/bin/bash

GPU_ID=6
config=/workspace/albert/albert_config/albert_tiny_singlet.json
vocab=/workspace/pli_bert/data/vocab/pfam_vocab.txt
tfrecord=( `ls /workspace/pli_bert/pfam_triplet_corpora/pfam_clustered_singlets_tfrecords_v2/*` )
output_dir=/workspace/pli_bert/pfam_triplet_corpora/albert_pretrained_singlet/

learning_rate_early=0.00176 #default albert lr is 0.00176
learning_rate_mature=0.000176 #default albert lr is 0.00176
eval_steps=50000
checkpoint_steps=20000
max_predictions_per_seq=30

declare -i ckpt=10000
declare -i next_ckpt=20000
declare -i increment=10000

warmup_steps=10000
warmup_iter=40 #how many times to warm up
train_iter=160 #training sessions after warm-up

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m albert.run_pretraining_albertseq \
    --input_file=${tfrecord} \
    --output_dir=${output_dir} \
    --albert_config_file=${config} \
    --do_train=True \
    --do_eval=True \
    --train_batch_size=256 \
    --eval_batch_size=256 \
    --max_seq_length=256 \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --optimizer='lamb' \
    --learning_rate=${learning_rate_early} \
    --num_train_steps=${increment} \
    --num_warmup_steps=${warmup_steps} \
    --max_eval_steps=${eval_steps} \
    --save_checkpoints_steps=${checkpoint_steps}

for i in $(seq 0 $warmup_iter)
do
	init_ckpt=${output_dir}model.ckpt-${ckpt}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python -m albert.run_pretraining_albertseq \
	    --input_file=${tfrecord} \
	    --output_dir=${output_dir} \
	    --albert_config_file=${config} \
	    --do_train=True \
	    --do_eval=True \
	    --train_batch_size=256 \
	    --eval_batch_size=256 \
	    --max_seq_length=256 \
            --max_predictions_per_seq=${max_predictions_per_seq} \
	    --optimizer='lamb' \
	    --init_checkpoint=${init_ckpt} \
	    --learning_rate=${learning_rate_early} \
	    --num_train_steps=${next_ckpt} \
	    --num_warmup_steps=${warmup_steps} \
	    --max_eval_steps=${eval_steps} \
	    --save_checkpoints_steps=${checkpoint_steps}
	ckpt=$ckpt+$increment
	next_ckpt=$next_ckpt+$increment
done


for i in $(seq 0 $train_iter)
do
	init_ckpt=${output_dir}model.ckpt-${ckpt}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python -m albert.run_pretraining_albertseq \
	    --input_file=${tfrecord} \
	    --output_dir=${output_dir} \
	    --albert_config_file=${config} \
	    --do_train=True \
	    --do_eval=True \
	    --train_batch_size=256 \
	    --eval_batch_size=256 \
	    --max_seq_length=256 \
            --max_predictions_per_seq=${max_predictions_per_seq} \
	    --optimizer='lamb' \
	    --learning_rate=${learning_rate_mature} \
            --init_checkpoint=${init_ckpt} \
	    --num_train_steps=${next_ckpt} \
	    --max_eval_steps=${eval_steps} \
	    --save_checkpoints_steps=${checkpoint_steps}
	ckpt=$ckpt+$increment
	next_ckpt=$next_ckpt+$increment
done

