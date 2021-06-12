#!/bin/bash

# 3396 train
# 726 dev
# 727 test

EXPERIMENT=rotowire

STEP=150
START=0
END=$(($START+$STEP))
TOTAL=3396

SPLIT="train"

while [ $START -le $TOTAL ]
do
   qsub -q 'gpu*@!(dll9|dll10)' -pe smp 8 -l "mem_free=12G,gpu=1,gpu_ram=11G,hostname=*" -cwd -pty yes -j y -b y -o out -e out -N "g_${EXPERIMENT}_${START}_${END}" ./generate.py --output $EXPERIMENT --split $SPLIT --start $START --end $END
   START=$(($START+$STEP))
   END=$(($END+$STEP))
done
