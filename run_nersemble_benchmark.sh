#!/bin/bash

ROOT_PATH="../workshop_dataset"

PARTICIPANT_IDS=(393 404 461 477 486)

CONFIG_PATH="./config/fateavatar.yaml"
NAME="FateAvatar"
SUBMIT_DIR="./nersemble_workspace/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ZIP_NAME="mono_flame_avatar_submission_${TIMESTAMP}.zip"

for ID in "${PARTICIPANT_IDS[@]}"; do
    WORKSPACE="./nersemble_workspace/${ID}"

    echo "Running participant ${ID}..."

    python ./benchmark/train_nersemble_benchmark_mono.py \
        --root_path "${ROOT_PATH}" \
        --workspace "${WORKSPACE}" \
        --name "${NAME}" \
        --config "${CONFIG_PATH}" \
        --participant_id "${ID}"

    python ./benchmark/run_nersemble_benchmark.py \
        --root_path "${ROOT_PATH}" \
        --workspace "${WORKSPACE}" \
        --name "${NAME}" \
        --config "${CONFIG_PATH}" \
        --submit_dir "${SUBMIT_DIR}" \
        --participant_id "${ID}"
done

echo "Zipping submission directory..."

cd "${SUBMIT_DIR}"
zip -r "../${ZIP_NAME}" ./*

echo "Saved zip to $(realpath ../${ZIP_NAME})"