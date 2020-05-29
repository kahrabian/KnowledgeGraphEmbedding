#!/bin/bash

SD=(128 96 64 32 64 32 32 96 64 32)
AD=(0   32 64 96 32 64 32 0  0  0 )
RD=(0   0  0  0  32 32 64 32 64 96)

for DS in ${@}; do
    for i in $(seq 0 10); do
        cp run.template.sh run.${DS}.${i}.sh
        sed -i "s/DS/${DS}/" run.${DS}.${i}.sh
        sed -i "s/SD/${SD[i]}/" run.${DS}.${i}.sh
        sed -i "s/AD/${AD[i]}/" run.${DS}.${i}.sh
        sed -i "s/RD/${RD[i]}/" run.${DS}.${i}.sh
        sbatch run.${DS}.${i}.sh
        rm run.${DS}.${i}.sh
    done
done
