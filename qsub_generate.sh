#!/bin/bash

# 3396 train
# 726 dev
# 727 test

for MODIFICATION_RATE in 0.25 0.5 0.75; do
    STEP=800
    START=0
    END=$(($START+$STEP))
    TOTAL=3396

    SPLIT="train"
    TEMPLATES="compact"

    EXPERIMENT="rotowire_${TEMPLATES}_r${MODIFICATION_RATE}"
    while [ $START -le $TOTAL ]
    do
       qsub -q 'gpu*@!(dll9|dll10)' -pe smp 8 -l "mem_free=12G,gpu=1,gpu_ram=11G,hostname=*" -cwd -pty yes -j y -b y -o out -e out -N "g_${EXPERIMENT}_${START}_${END}" ./generate.py --output "$EXPERIMENT" --split $SPLIT --start $START --end $END --templates "$TEMPLATES" --modification_rate $MODIFICATION_RATE
       START=$(($START+$STEP))
       END=$(($END+$STEP))
    done
done

# qsub -q 'gpu*@!(dll9|dll10)' -pe smp 8 -l "mem_free=12G,gpu=1,gpu_ram=11G,hostname=*" -cwd -pty yes -j y -b y -o out -e out -N "g_annotated"  ./generate.py --output annotated --annotations gsml.csv --games games.csv