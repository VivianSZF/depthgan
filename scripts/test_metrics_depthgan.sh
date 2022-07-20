#!/bin/bash

# Help information.
if [[ $# -lt 4 || ${*: -1} == "-h" || ${*: -1} == "--help" ]]; then
    echo "This script tests metrics defined in \`./metrics/\`."
    echo
    echo "Usage: $0 GPUS DATASET MODEL METRICS"
    echo
    echo "Note: More than one metric should be separated by comma." \
         "Also, all metrics assume using all samples from the real dataset" \
         "and 50000 fake samples for GAN-related metrics."
    echo
    echo "Example: $0 1 ~/data/ffhq1024.zip ~/checkpoints/ffhq1024.pth ~/data/val_data.json" \
         "fid_rgb,fid_depth,snapshot,rotation,video"
    echo
    exit 0
fi

# Get an available port for launching distributed training.
# Credit to https://superuser.com/a/1293762.
export DEFAULT_FREE_PORT
DEFAULT_FREE_PORT=$(comm -23 <(seq 49152 65535 | sort) \
                    <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) \
                    | shuf | head -n 1)

GPUS=$1
DATASET=$2
MODEL=$3
TEST_ANNO_PATH=$4
PORT=${PORT:-$DEFAULT_FREE_PORT}

# Parse metrics to test.
METRICS=$5
TEST_FID_DEPTH="false"
TEST_FID_RGB="false"
TEST_SNAPSHOT="false"
TEST_ROTATION="false"
SAVE_VIDEO="false"
if [[ ${METRICS} == "all" ]]; then
    TEST_FID_DEPTH="true"
    TEST_FID_RGB="true"
    TEST_SNAPSHOT="true"
    TEST_ROTATION="true"
    SAVE_VIDEO="true"
else
    array=(${METRICS//,/ })
    for var in ${array[@]}; do
        if [[ ${var} == "fid_depth" ]]; then
            TEST_FID_DEPTH="true"
        fi
        if [[ ${var} == "fid_rgb" ]]; then
            TEST_FID_RGB="true"
        fi
        if [[ ${var} == "snapshot" ]]; then
            TEST_SNAPSHOT="true"
        fi
        if [[ ${var} == "rotation" ]]; then
            TEST_ROTATION="true"
        fi
        if [[ ${var} == "video" ]]; then
            SAVE_VIDEO="true"
        fi
    done
fi

# Detect `python3` command.
# This workaround addresses a common issue:
#   `python` points to python2, which is deprecated.
export PYTHONS
export RVAL

PYTHONS=$(compgen -c | grep "^python3$")

# `$?` is a built-in variable in bash, which is the exit status of the most
# recently-executed command; by convention, 0 means success and anything else
# indicates failure.
RVAL=$?

if [ $RVAL -eq 0 ]; then  # if `python3` exist
    PYTHON="python3"
else
    PYTHON="python"
fi

${PYTHON} -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --master_port=${PORT} \
    ./test_metrics_depthgan.py \
        --launcher="pytorch" \
        --backend="nccl" \
        --dataset ${DATASET} \
        --annotation_path ${TEST_ANNO_PATH} \
        --model ${MODEL} \
        --real_num -1 \
        --fake_num 5 \
        --trunc_rgb 1 \
        --trunc_depth 1 \
        --test_fid_depth ${TEST_FID_DEPTH} \
        --test_fid_rgb ${TEST_FID_RGB} \
        --test_snapshot ${TEST_SNAPSHOT} \
        --test_rotation ${TEST_ROTATION} \
        --video_save ${SAVE_VIDEO} \
        ${@:6}
