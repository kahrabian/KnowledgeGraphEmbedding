#!/bin/bash

SD=(128 128 128 96 96 96 64 64 64)
AD=(0   0   0   16 16 16 32 32 32)
RD=(0   32  64  0  32 64 0  32 64)

for DS in ${@}; do
    for i in $(seq 0 8); do
        cp run.template.sh run.${DS}.${i}.sh
        sed -i "s/DS/${DS}/" run.${DS}.${i}.sh
        sed -i "s/SD/${SD[i]}/" run.${DS}.${i}.sh
        sed -i "s/AD/${AD[i]}/" run.${DS}.${i}.sh
        sed -i "s/RD/${RD[i]}/" run.${DS}.${i}.sh
        sbatch run.${DS}.${i}.sh
        rm run.${DS}.${i}.sh
    done
done
