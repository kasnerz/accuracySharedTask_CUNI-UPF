#!/bin/bash

# script for generating final submissions

#--------------------------------
# simple - best precision
#--------------------------------
TEMPLATES=simple
CTX=20
RATIO=0.25
EXPERIMENT="rotowire_${TEMPLATES}_r${RATIO}_ctx${CTX}"

# # generate
# qsub -q 'gpu*@!(dll9|dll10)' -pe smp 8 -l "mem_free=8G,gpu=1,gpu_ram=11G,hostname=*" -cwd -pty yes -j y -b y -o out -e out -N "d_${EXPERIMENT}" \
#    ./decode.py --experiment ${EXPERIMENT} --ctx ${CTX} --templates ${TEMPLATES} --input_file test/games.csv --out_fname submit_${EXPERIMENT}.csv

# cp <----- generate first
cp experiments/${EXPERIMENT}/submit_${EXPERIMENT}.csv ./submit/${TEMPLATES}_best_precision.csv



#--------------------------------
# compact - best precision
#--------------------------------
TEMPLATES=compact
CTX=5
RATIO=0.25
EXPERIMENT="rotowire_${TEMPLATES}_r${RATIO}_ctx${CTX}"

# # generate
# qsub -q 'gpu*@!(dll9|dll10)' -pe smp 8 -l "mem_free=8G,gpu=1,gpu_ram=11G,hostname=*" -cwd -pty yes -j y -b y -o out -e out -N "d_${EXPERIMENT}" \
#    ./decode.py --experiment ${EXPERIMENT} --ctx ${CTX} --templates ${TEMPLATES} --input_file test/games.csv --out_fname submit_${EXPERIMENT}.csv

# cp <----- generate first
cp experiments/${EXPERIMENT}/submit_${EXPERIMENT}.csv ./submit/${TEMPLATES}_best_precision.csv



#--------------------------------
# simple - best recall
#--------------------------------
TEMPLATES=simple
CTX=40
RATIO=0.5
EXPERIMENT="rotowire_${TEMPLATES}_r${RATIO}_ctx${CTX}"

# train
# ./qsub_train.sh --dataset "annotated_${TEMPLATES}_ctx${CTX}" --experiment rotowire_annotated_${TEMPLATES}_r${RATIO}_ctx${CTX} --max_epochs 7 --model_path experiments/rotowire_${TEMPLATES}_r${RATIO}_ctx${CTX}/model.ckpt

# # generate <----- run training first
EXPERIMENT="rotowire_annotated_${TEMPLATES}_r${RATIO}_ctx${CTX}"
# qsub -q 'gpu*@!(dll9|dll10)' -pe smp 8 -l "mem_free=8G,gpu=1,gpu_ram=11G,hostname=*" -cwd -pty yes -j y -b y -o out -e out -N "d_${EXPERIMENT}" \
#    ./decode.py --experiment ${EXPERIMENT} --ctx ${CTX} --templates ${TEMPLATES} --input_file test/games.csv --out_fname submit_${EXPERIMENT}.csv

# cp <----- generate first
cp experiments/${EXPERIMENT}/submit_${EXPERIMENT}.csv ./submit/${TEMPLATES}_best_recall.csv


#--------------------------------
# compact - best recall
#--------------------------------
TEMPLATES=compact
CTX=10
RATIO=0.75
EXPERIMENT="rotowire_${TEMPLATES}_r${RATIO}_ctx${CTX}"

# train
# ./qsub_train.sh --dataset "annotated_${TEMPLATES}_ctx${CTX}" --experiment rotowire_annotated_${TEMPLATES}_r${RATIO}_ctx${CTX} --max_epochs 7 --model_path experiments/rotowire_${TEMPLATES}_r${RATIO}_ctx${CTX}/model.ckpt
# # generate <----- run training first
EXPERIMENT="rotowire_annotated_${TEMPLATES}_r${RATIO}_ctx${CTX}"
# qsub -q 'gpu*@!(dll9|dll10)' -pe smp 8 -l "mem_free=8G,gpu=1,gpu_ram=11G,hostname=*" -cwd -pty yes -j y -b y -o out -e out -N "d_${EXPERIMENT}" \
#    ./decode.py --experiment ${EXPERIMENT} --ctx ${CTX} --templates ${TEMPLATES} --input_file test/games.csv --out_fname submit_${EXPERIMENT}.csv

# cp <----- generate first
cp experiments/${EXPERIMENT}/submit_${EXPERIMENT}.csv ./submit/${TEMPLATES}_best_recall.csv




#--------------------------------
# simple - best F1
#--------------------------------
TEMPLATES=simple
CTX=40
RATIO=0.25
EXPERIMENT="rotowire_${TEMPLATES}_r${RATIO}_ctx${CTX}"

# train
# ./qsub_train.sh --dataset "annotated_${TEMPLATES}_ctx${CTX}" --experiment rotowire_annotated_${TEMPLATES}_r${RATIO}_ctx${CTX} --max_epochs 7 --model_path experiments/rotowire_${TEMPLATES}_r${RATIO}_ctx${CTX}/model.ckpt

# # generate <----- run training first
EXPERIMENT="rotowire_annotated_${TEMPLATES}_r${RATIO}_ctx${CTX}"
# qsub -q 'gpu*@!(dll9|dll10)' -pe smp 8 -l "mem_free=8G,gpu=1,gpu_ram=11G,hostname=*" -cwd -pty yes -j y -b y -o out -e out -N "d_${EXPERIMENT}" \
#    ./decode.py --experiment ${EXPERIMENT} --ctx ${CTX} --templates ${TEMPLATES} --input_file test/games.csv --out_fname submit_${EXPERIMENT}.csv

# cp <----- generate first
cp experiments/${EXPERIMENT}/submit_${EXPERIMENT}.csv ./submit/${TEMPLATES}_best_f1.csv


#--------------------------------
# compact - best F1
#--------------------------------
TEMPLATES=compact
CTX=40
RATIO=0.25
EXPERIMENT="rotowire_${TEMPLATES}_r${RATIO}_ctx${CTX}"

# train
# ./qsub_train.sh --dataset "annotated_${TEMPLATES}_ctx${CTX}" --experiment rotowire_annotated_${TEMPLATES}_r${RATIO}_ctx${CTX} --max_epochs 7 --model_path experiments/rotowire_${TEMPLATES}_r${RATIO}_ctx${CTX}/model.ckpt
# # generate <----- run training first
EXPERIMENT="rotowire_annotated_${TEMPLATES}_r${RATIO}_ctx${CTX}"
# qsub -q 'gpu*@!(dll9|dll10)' -pe smp 8 -l "mem_free=8G,gpu=1,gpu_ram=11G,hostname=*" -cwd -pty yes -j y -b y -o out -e out -N "d_${EXPERIMENT}" \
#    ./decode.py --experiment ${EXPERIMENT} --ctx ${CTX} --templates ${TEMPLATES} --input_file test/games.csv --out_fname submit_${EXPERIMENT}.csv

# cp <----- generate first
cp experiments/${EXPERIMENT}/submit_${EXPERIMENT}.csv ./submit/${TEMPLATES}_best_f1.csv




#--------------------------------
# postprocessing
#--------------------------------
# merge subsequent errors for better precision
# for FILE in experiments/*/out.csv; do echo $FILE; ./postprocess_submission.py $FILE; done
for FILE in submit/*.csv; do 
    echo $FILE
    ./postprocess_submission.py $FILE
done

#--------------------------------
# consistency checks
#--------------------------------
for FILE in submit/*.csv; do 
    echo $FILE
    ./evaluate.py --gsml $FILE --submitted $FILE --token_lookup=test/token_lookup.yaml
done

#--------------------------------
# visualizations
#--------------------------------
for FILE in submit/*.csv; do 
    echo $FILE
    OUT_HTML=`basename $FILE .csv`
    ./utils/gsml_to_html.py test/texts $FILE $FILE test/games.csv > submit/html/$OUT_HTML.html
done