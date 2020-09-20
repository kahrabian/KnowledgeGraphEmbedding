#!/bin/bash
#SBATCH --account=def-jinguo
#SBATCH --job-name=k-hop
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1:00:00
#SBATCH --array=0-31
#SBATCH --output=./logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kian.ahrabian@mail.mcgill.ca

source ${VENVDIR}/gg/bin/activate

export TOTAL_TASKS=32
export K=2
export DATASET=icse_DS
python scripts/extract_khop.py
