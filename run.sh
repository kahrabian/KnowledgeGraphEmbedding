#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models

#The first four parameters must be provided
MODE=${1}
MODEL=${2}
DATASET=${3}
GPU_DEVICE=${4}
SAVE_ID=${5}

FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"

#Only used in training
BATCH_SIZE=${6}
NEGATIVE_SAMPLE_SIZE=${7}
HIDDEN_DIM=${8}
GAMMA=${9}
EPSILON=${10}
ALPHA=${11}
LEARNING_RATE=${12}
MAX_STEPS=${13}
TEST_BATCH_SIZE=${14}

EVAL_MODE=${15}
TIME_HIDDEN_DIM=${16}
NEGATIVE_TIM_SAMPLE_SIZE=${17}

RELATIVE_HIDDEN_DIM=${18}

if [ $MODE == "train" ]
then

echo "Start Training......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_train \
    --cpu_num 2 \
    --do_valid \
    --do_test \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -e $EPSILON -a $ALPHA \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    --eval_mode $EVAL_MODE --time_hidden_dim $TIME_HIDDEN_DIM \
    -nt $NEGATIVE_TIM_SAMPLE_SIZE --relative_hidden_dim $RELATIVE_HIDDEN_DIM \
    ${19} ${20} ${21} ${22} ${23} ${24} ${25}

elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_valid --cuda -init $SAVE
    
elif [ $MODE == "test" ]
then

echo "Start Evaluation on Test Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

else
   echo "Unknown MODE" $MODE
fi