#!/bin/sh

EXPERIMENT="rotowire_annotated"

for SIZE in 5 10 20 40
do
    qsub -q 'gpu*@!(dll9|dll10)' -pe smp 8 -l "mem_free=8G,gpu=2,gpu_ram=11G,hostname=*" -cwd -pty yes -j y -b y -o out -e out -N "d_${EXPERIMENT}_ctx${SIZE}" ./decode.py --experiment ${EXPERIMENT}_ctx${SIZE} --ctx ${SIZE}
done