#!/bin/sh
#SBATCH --account=def-jinguo
#SBATCH --job-name=kge-derotate
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kian.ahrabian@mail.mcgill.ca

module load python/3.7.4 gcc/7.3.0 cuda/10.0.130 cudnn/7.6 openmpi/3.1.2 nccl/2.3.5

source ${VENVDIR}/gg/bin/activate

python -u -c 'import torch; print(torch.__version__)'

DATASET=GitGraph_TI_0.01
SAVE=models/${DATASET}_${SLURM_JOB_ID}

python -u codes/run.py \
    --dataset data/$DATASET \
    --model RotatE \
    --static_dim 384 \
    --absolute_dim 128 \
    --relative_dim 1 \
    --gamma 6.0 \
    --epsilon 10.0 \
    --alpha 0.5 \
    --learning_rate 0.00003 \
    --negative_sample_size 256 \
    --negative_time_sample_size 8 \
    --negative_max_time_gap 259200 \
    --batch_size 64 \
    --test_batch_size 1 \
    --max_steps 200000 \
    --warm_up_steps 100000 \
    --save_path $SAVE \
    --mode head \
    --valid_steps 40000 \
    --log_steps 100 \
    --test_log_steps 1000 \
    --do_train --do_valid --do_test \
    --negative_adversarial_sampling --negative_type_sampling --heuristic_evaluation --type_evaluation \
