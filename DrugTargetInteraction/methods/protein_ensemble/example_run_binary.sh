
### 
CUDA_VISIBLE_DEVICES=1 python train.py \
    --protein_embedding_type=albert \ #protein representation based on ALBERT
    --prediction_mode=binary \ #activity mode is binary
    --from_pretrained_checkpoint=False \ #True if training from a previous checkpoint
    --pretrained_checkpoint_dir=temp/ \ #provide if --from_pretrained_checkpoint is True
    --batch=128 \ #mini-batch size. ALBERT-based model takes largest amount of GPU memory
    --epoch=100 \ #max training epoch
    --lr=2e-5 \ #learning rate
    --log=info \ #--log=debug for detailed information
    --force_debug=False \ #--force_debug=True for short debugging mode (forced to max 3 epochs with smaller amount of data)
    --no_cuda=False \ #set to true for CPU mode
    --checkpoint_dir=albert_binary_default/ > albert_binary_default/performance_log.txt
    #--checkpoint_dir is where the checkpoints are saved (with config and performance log)

CUDA_VISIBLE_DEVICES=0 python train.py \
    --protein_embedding_type=pssm \ #protein representation based on PSSM-ResNet
    --prediction_mode=binary \
    --from_pretrained_checkpoint=False \
    --pretrained_checkpoint_dir=temp/ \
    --batch=1024 \
    --epoch=100 \
    --lr=2e-4 \
    --log=info \
    --force_debug=False \
    --no_cuda=False \
    --checkpoint_dir=pssm_binary_lr2e4/ > pssm_binary_lr2e4/performance_log.txt 

CUDA_VISIBLE_DEVICES=2 python train.py \
    --protein_embedding_type=lstm \ #protein representation based on LSTM
    --prediction_mode=binary \
    --from_pretrained_checkpoint=False \
    --pretrained_checkpoint_dir=temp/ \
    --batch=512 \
    --epoch=100 \
    --lr=2e-5 \
    --log=info \
    --force_debug=False \
    --no_cuda=False \
    --checkpoint_dir=lstm_binary_default/ > lstm_binary_default/performance_log.txt
