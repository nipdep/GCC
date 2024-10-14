#!/bin/bash
load_path=$1
gpu=$2
dataset=$3
ARGS=${@:4}

declare -A epochs=(["usa_airport"]=30 ["h-index"]=5 ["imdb-binary"]=30 ["imdb-multi"]=30 ["collab"]=30 ["rdt-b"]=100 ["rdt-5k"]=100)

python ctrain.py \
  --exp CTraining \
  --model-path saved \
  --tb-path tensorboard \
  --tb-freq 5 \
  --ctrain \
  --dataset $dataset \
  --epochs ${epochs[$dataset]} \
  --resume "$load_path/current.pth" \
  --gpu $gpu \
  $ARGS

