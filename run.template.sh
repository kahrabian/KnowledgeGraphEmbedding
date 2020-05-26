#!/bin/sh
#SBATCH --account=def-jinguo
#SBATCH --job-name=kge-derotate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kian.ahrabian@mail.mcgill.ca

module load python/3.7.4 gcc/7.3.0 cuda/10.0.130 cudnn/7.6 openmpi/3.1.2 nccl/2.3.5

source ${VENVDIR}/gg/bin/activate

python -u -c 'import torch; print(torch.__version__)'

MODEL=RotatE
DATASET=GitGraph_TI_0.01
ID=${MODEL}_${DATASET}_${SLURM_JOB_ID}

python -u codes/run.py \
    --id ${ID} \
    --dataset data/${DATASET} \
    --model ${MODEL} \
    --static_dim 128 \
    --absolute_dim 256 \
    --relative_dim 128 \
    --dropout 0.2 \
    --gamma 6.0 \
    --alpha 0.5 \
    --lmbda 0.0 \
    --learning_rate 0.00003 \
    --learning_rate_steps 100000 \
    --weight_decay 0.0 \
    --criterion NS \
    --negative_sample_size 256 \
    --negative_time_sample_size 8 \
    --negative_max_time_gap 259200 \
    --batch_size 64 \
    --test_batch_size 1 \
    --max_steps 200000 \
    --save_path models/${SAVE} \
    --metric MRR \
    --mode head \
    --valid_steps 40000 \
    --valid_approximation 0 \
    --log_steps 100 \
    --test_log_steps 1000 \
    --log_dir runs/${LOG} \
    --timezone "America/Montreal" \
    --do_train --do_valid --do_eval --do_test \
    --negative_adversarial_sampling --negative_type_sampling \
    --heuristic_evaluation --type_evaluation
