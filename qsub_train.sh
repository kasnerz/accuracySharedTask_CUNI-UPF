#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -a|--acc_grad) ACC_GRAD="$2"; shift ;;
        -b|--batch_size) BATCH_SIZE="$2"; shift ;;
        -c|--checkpoint_suffix) CHECKPOINT_SUFFIX="$2"; shift ;;
        -d|--dataset) DATASET="$2";  shift ;;
        -e|--experiment) EXPERIMENT="$2"; shift ;;
        -l|--learning_rate) LEARNING_RATE="$2"; shift ;;
        -m|--model_name) MODEL_NAME="$2"; shift ;;
        -n|--no_val) NO_VAL=1 ;;
        -o|--model_path) MODEL_PATH="$2"; shift ;;
        -p|--max_epochs) EPOCHS="$2"; shift ;;
        -r|--ref_file) REF_FILE="$2"; shift ;;
        -s|--steps) STEPS="$2"; shift ;;
        *) echo "Train: Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

GPUS=1
GPU_RAM=16
MAX_THREADS=8
# MAX_LENGTH=512

EXPERIMENT_NAME=${EXPERIMENT/\//_}

CMD="./train.py "
CMD+="--dataset ${DATASET} "
CMD+="--experiment ${EXPERIMENT} "
CMD+="--gpus ${GPUS} "

# if [[ -n "${STEPS}" ]]; then
#     CMD+="--max_steps ${STEPS} "
# fi

if [[ -n "${EPOCHS}" ]]; then
    CMD+="--max_epochs ${EPOCHS} "
fi

# start training with an existing model
if [[ -n "${MODEL_PATH}" ]]; then
    CMD+="--model_path $MODEL_PATH "
fi

qsub -q 'gpu*' -pe smp "${MAX_THREADS}" -l "mem_free=16G,gpu=${GPUS},gpu_ram=${GPU_RAM}G,hostname=*" -cwd -pty yes -j y -b y -o out -e out -N t_${EXPERIMENT_NAME}${CHECKPOINT_SUFFIX} $CMD
