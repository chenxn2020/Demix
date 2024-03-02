DATA_DIR=dataset

MODEL_NAME=TransE
DATASET_NAME=WN18RR
Loss_Name=BCE_Loss
LTModel_NAME=MixupLitModel
DATA_PATH=$DATA_DIR/$DATASET_NAME
TRAIN_BS=512
DIM=500
MARGIN=9.0
LEARNING_RATE=15e-5
MAX_EPOCHES=500
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
    --lr_begin 160 \
    --lr_step 150 \
    --lr_change 0.5 \
    --adv_temp 0.5 \
    --use_weight 1 \
    --mix_neg 50 \
    --num_neg 128 \
    --cnd_size 70 \





