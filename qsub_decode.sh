#!/bin/sh

TEMPLATES=simple

for XVAL in 0 1 2 3 4 5; do
    for RATIO in 0.25 0.5 0.75; do
        for SIZE in 5 10 20 40; do
            EXPERIMENT="rotowire_annotated_${TEMPLATES}_r${RATIO}_ctx${SIZE}_x${XVAL}"

            qsub -q 'gpu*@!(dll9|dll10)' -pe smp 8 -l "mem_free=8G,gpu=1,gpu_ram=11G,hostname=*" -cwd -pty yes -j y -b y -o out -e out -N "d_${EXPERIMENT}" ./decode.py --experiment ${EXPERIMENT} --ctx ${SIZE} --templates $TEMPLATES
        done
    done
done