#!/usr/bin/env bash

CONFIG=$1
GPUS=0

env CUDA_VISIBLE_DEVICES=$GPUS \
python \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
