#!/bin/bash

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model whisper \
    --trainer simple \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --data synthetic_whisper \
    --data_configs.synthetic_whisper.data_path '${COMP597_JOB_STUDENT_STORAGE_DIR}/synthetic_whisper_data.pt' \
    --trainer_stats resource_util \
    --trainer_stats_configs.resource_util.output_dir 'logs' \
    --trainer_stats_configs.resource_util.output_file 'resource_util.csv'
