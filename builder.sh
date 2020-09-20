#!/bin/bash

for i in {1..11}; do
    cp build.template.sh build.${i}.sh
    sed -i "s/DS/${i}/" build.${i}.sh
    sbatch build.${i}.sh
    rm build.${i}.sh
done
