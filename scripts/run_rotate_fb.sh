DATA_DIR=dataset

MODEL_NAME=RotatE
DATASET_NAME=FB15K237
Loss_Name=BCE_Loss
LTModel_NAME=MixupLitModel
DATA_PATH=$DATA_DIR/$DATASET_NAME
TRAIN_BS=512
DIM=1000
MARGIN=9.0
LEARNING_RATE=1e-4
MAX_EPOCHES=120
EVAL_BS=16
CHECK_PER_EPOCH=5
EARLY_STOP=4
GPU=7


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
    --use_wandb \
    --calc_filter \
    --pos_threshold 3 \
    --mix_epoch 8 \
    --lr_begin 20 \
    --lr_step 30 \
    --lr_change 0.1 \
    --use_weight 1 \
    --adv_temp 1 \
    --mix_neg 50 \
    --num_neg 128 \
    --cnd_size 70 \





