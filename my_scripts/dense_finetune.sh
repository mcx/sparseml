#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
NPROC=$(($(echo $CUDA_VISIBLE_DEVICES | grep -o "," | wc -l)+1))

ROOT=$HOME/work/llama2.cnn_dailymail.train/src/neuralmagic/sparseml
source $ROOT/my_scripts/start_here.sh

DATASET=cnn_dailymail
DATASET_CONFIG_NAME="3.0.0"
WORKERS=16
MAX_LEN=1024  # testing
#DATASET=open_platypus

SRC_MODEL_NAME=Llama-2-7b-hf
SRC_MODEL=/network/tuan/models/llama/$SRC_MODEL_NAME

RECIPE_DIR=$ROOT/my_recipes
FSDP_CONFIG=$RECIPE_DIR/fsdp.yaml

LR=5e-5
WARM=0.1
EPOCHS=1

ID=$RANDOM

DST_MODEL_DIR=$HOME/models/llama2/cnn_dailymail/dense_finetuned
DST_MODEL_NAME=$SRC_MODEL_NAME@$DATASET@LR$LR@WARM$WARM@EP$EPOCHS@ID$ID
DST_MODEL=$DST_MODEL_DIR/$DST_MODEL_NAME

accelerate launch \
    --config_file $FSDP_CONFIG \
    --no_python sparseml.transformers.text_generation.train \
    --model_name $SRC_MODEL \
    --dataset_name $DATASET \
    --dataset_config_name $DATASET_CONFIG_NAME \
    --preprocessing_num_workers $WORKERS \
    --max_seq_length $MAX_LEN \
    --do_eval \
    --learning_rate $LR \
    --warmup_ratio $WARM \
    --output_dir $DST_MODEL \
    --num_train_epochs $EPOCHS \
    --report_to wandb \
    --run_name $DST_MODEL_NAME


#     --splits "train"  <== Error. Why do we need this?
