#!/usr/bin/env bash

# Experiment 7 Work
# Exp List
#1. Shape (no embed init)
#2. Shape (glove init)
#3. Color (no embed init)
#4. Color (glove init)
#5. <S> shape color </s> (no embed init)
#6. <S> shape color </s> (glove init)
#7. <S> full caps </s> (no embed init)
#8. <S> full caps </s> (glove init)

# Experiment1
DATA_DIR="${PWD}/data/simple_1"
LOG_DIR="${PWD}/models/exp7"
CNN_CKPT="${PWD}/data/od_model/factor"
PARSE_TYPE="shape"
EXP_TAG="shape_no_init"
python3 train_network.py --data_dir=$DATA_DIR \
                         --log_dir=$LOG_DIR \
                         --cnn_ckpt=$CNN_CKPT \
                         --parse_type=$PARSE_TYPE \
                         --exp_tag=$EXP_TAG

# Experiment2
DATA_DIR="${PWD}/data/simple_1"
LOG_DIR="${PWD}/models/exp7"
CNN_CKPT="${PWD}/data/od_model/factor"
PARSE_TYPE="shape"
GLOVE_DIR="${PWD}/data/glove.6B"
GLOVE_DIM=50
EXP_TAG="shape_glove"
python3 train_network.py --data_dir=$DATA_DIR \
                         --log_dir=$LOG_DIR \
                         --cnn_ckpt=$CNN_CKPT \
                         --parse_type=$PARSE_TYPE \
                         --glove_dir=$GLOVE_DIR \
                         --exp_tag=$EXP_TAG

# Experiment3
DATA_DIR="${PWD}/data/simple_1"
LOG_DIR="${PWD}/models/exp7"
CNN_CKPT="${PWD}/data/od_model/factor"
PARSE_TYPE="color"
EXP_TAG="color_no_init"
python3 train_network.py --data_dir=$DATA_DIR \
                         --log_dir=$LOG_DIR \
                         --cnn_ckpt=$CNN_CKPT \
                         --parse_type=$PARSE_TYPE \
                         --exp_tag=$EXP_TAG

# Experiment4
DATA_DIR="${PWD}/data/simple_1"
LOG_DIR="${PWD}/models/exp7"
CNN_CKPT="${PWD}/data/od_model/factor"
PARSE_TYPE="color"
GLOVE_DIR="${PWD}/data/glove.6B"
GLOVE_DIM=50
EXP_TAG="color_glove"
python3 train_network.py --data_dir=$DATA_DIR \
                         --log_dir=$LOG_DIR \
                         --cnn_ckpt=$CNN_CKPT \
                         --parse_type=$PARSE_TYPE \
                         --glove_dir=$GLOVE_DIR \
                         --exp_tag=$EXP_TAG
