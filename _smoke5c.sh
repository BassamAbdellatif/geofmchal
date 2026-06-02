#!/bin/bash
# Phase 5C smoke tests: one per new code path. 2 train batches + full val.
# NOTE: spec wrote `--use-gradnorm False`, but the flag is BooleanOptionalAction
# (no value) -> corrected to `--no-use-gradnorm`. --cache-dir added for speed.
set -u
cd /mnt/head/users/bassam/src/geofmchal
CACHE=/home/bassam/nvme_cache/cache7a
COMMON="--model-type dual_enc_dec_fusion --pixel-inputs alpha_earth,tessera --patch-inputs terramind_s1,terramind_s2 --batch-size 4 --epochs 1 --max-batches 2 --num-workers 4 --cache-dir $CACHE"

run_smoke () {
  name=$1; shift
  log=/tmp/${name}.log
  echo "==================== $name ===================="
  PYTHONUNBUFFERED=1 ./run_env.sh train.py $COMMON --experiment-name "$name" "$@" > "$log" 2>&1
  rc=$?
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)
  echo "rc=$rc   gpu_now=$gpu"
  grep -aoE "height_bridge=[a-z]+|static task weights[^|]*|matched=[0-9]+ +train=[0-9]+ +val=[0-9]+|loss=[0-9.]+|nan|NaN|Traceback|Error|Epoch 1/1 \| .*proxy [0-9.]+[^|]*|7A DONE[^=]*=[0-9.]+" "$log" | tail -10
  echo
}

run_smoke smoke_static    --no-use-gradnorm --static-weights "0.65,0.64,1.70"
run_smoke smoke_no_bridge --no-height-bridge
run_smoke smoke_no_binary --no-binary-head
echo "==================== ALL SMOKES DONE ===================="
