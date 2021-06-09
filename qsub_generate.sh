#!/bin/bash

# 3396 train
# 726 dev
# 727 test

EXPERIMENT=rotowire_dctx

STEP=50
START=0
END=$(($START+$STEP))
TOTAL=249

SPLIT="dev"

while [ $START -le $TOTAL ]
do
   qsub -q 'gpu*' -pe smp 8 -l "mem_free=16G,gpu=1,gpu_ram=11G,hostname=*" -cwd -pty yes -j y -b y -o out -e out -N "g_${EXPERIMENT}_${START}_${END}" ./generate.py --output $EXPERIMENT --split $SPLIT --start $START --end $END
   START=$(($START+$STEP))
   END=$(($END+$STEP))
done

