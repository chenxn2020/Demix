DATA_DIR=dataset

MODEL_NAME=DistMult
DATASET_NAME=WN18RR
Loss_Name=BCE_Loss
LTModel_NAME=MixupLitModel
DATA_PATH=$DATA_DIR/$DATASET_NAME
TRAIN_BS=512
NUM_NUG=128
DIM=1000
MARGIN=200.0
LEARNING_RATE=0.002
MAX_EPOCHES=200
EVAL_BS=16
CHECK_PER_EPOCH=5
EARLY_STOP=10
GPU=2

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --max_epochs $MAX_EPOCHES \
    --emb_dim $DIM \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --lr $LEARNING_RATE \
    --check_per_epoch $CHECK_PER_EPOCH \
    --early_stop_patience $EARLY_STOP \
    --margin $MARGIN \
    --loss_name $Loss_Name \
    --litmodel_name $LTModel_NAME \
    --calc_filter \
    --use_wandb \
    --mix_epoch 8 \
    --pos_threshold 1 \
    --lr_begin 60 \
    --lr_step 40 \
    --lr_change 0.1 \
    --adv_temp 1 \
    --use_weight 1 \
    --num_neg $NUM_NUG \
    --regularization 0.000005 \
    --mix_neg 50 \
    --cnd_size 70 \







