#!/bin/bash

for TEMPLATES in simple compact; do
    for RATIO in 0.25 0.5 0.75; do 
        for SIZE in 5 10 20 40; do 
            EXP=rotowire_annotated_${TEMPLATES}_r${RATIO}_ctx${SIZE}
            echo $EXP
            ./evaluate.py --gsml gsml.csv --submitted experiments/${EXP}/out.csv --token_lookup token_lookup.yaml --n_games_only 10

            echo "-----------------"

       done
    done
done
