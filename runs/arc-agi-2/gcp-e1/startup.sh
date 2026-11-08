#!/bin/bash
# E1 GCE startup script (1x L4 VM; the original internal spec is not published). Idempotent: every boot restores the
# environment (from gs cache when present), pulls each run's partial store, resumes per task, syncs to GCS, and powers
# off when all 9 runs (3 seeds x F / HA / HB) have 30 done markers. No credentials: VM default service account only.
set -uo pipefail
B=${ARC_E1_BUCKET:-gs://YOUR-BUCKET/arc}  # set to your own GCS bucket/prefix
W=/opt/arc
mkdir -p $W/out && cd $W
exec >> $W/startup.log 2>&1
GS="gcloud storage"
up_log() { $GS cp -q $W/startup.log "$B/out/e1/logs/startup-$(hostname).log" || true; }
fail() { echo "$1 $(date -u +%FT%TZ)"; up_log; poweroff; exit 1; }  # never leave a broken VM billing until max-run
echo "BOOT $(date -u +%FT%TZ) $(hostname)"

echo probe | $GS cp - "$B/out/e1/logs/probe-$(hostname)" || fail BUCKET_WRITE_FAIL
for i in $(seq 1 120); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || fail NO_GPU

$GS cp -q $B/in/e1-inputs-v1.tgz $W/ && tar xzf $W/e1-inputs-v1.tgz -C $W || fail INPUT_FETCH_FAIL
(cd $W/in && sha256sum -c --quiet SHA256SUMS) || fail INPUT_SHA_FAIL

V=$W/venv311
M=$W/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1
mkdir -p $W/input/arc-prize-2026-arc-agi-2 $W/work $M
cp $W/in/arc-agi_test_challenges.json $W/input/arc-prize-2026-arc-agi-2/
export UV_PYTHON_INSTALL_DIR=$W/pyinst UV_CACHE_DIR=$W/uvcache

CACHED=0
if [ ! -x $V/bin/python ] && $GS cp -q $B/cache/env-v1.tgz $W/ 2>/dev/null; then
  tar xzf $W/env-v1.tgz -C $W && rm -f $W/env-v1.tgz && CACHED=1
fi
if [ ! -f $M/config.json ] && $GS cp -q $B/cache/model-v1.tar $W/ 2>/dev/null; then
  tar xf $W/model-v1.tar -C $M && rm -f $W/model-v1.tar
fi
if [ ! -x $V/bin/python ]; then
  # The DLVM image has no pip for /usr/bin/python3 (first VM 10-11): install uv standalone, as Gemma's startup does.
  # uv 0.13.0 = the version that fetched CPython 3.11.17 on VM 10-12 (0.12.5 only knows <=3.11.16); 3.11.17 = Colab run-2.
  curl -LsSf https://astral.sh/uv/0.13.0/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh || fail UV_INSTALL_FAIL
  UV=/usr/local/bin/uv
  $UV python install 3.11.17 || fail PY_INSTALL_FAIL
  $UV venv --python 3.11.17 $V || fail VENV_FAIL
  # Exact final environment of the Colab run-2 calibration (colab-cal-v1/boot_run2.log; torchao is removed below, as
  # there). Second VM 10-12: --exclude-newer dropped every download.pytorch.org file (no upload dates) -> no torch.
  cat > $W/env-lock.txt <<'LOCK'
accelerate==1.15.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
anyio==4.15.1
attrs==26.1.0
bitsandbytes==0.50.2
certifi==2026.7.22
charset-normalizer==3.5.2
cut-cross-entropy==25.1.1
datasets==3.6.0
diffusers==0.39.0
dill==0.3.8
docstring-parser==0.18.0
filelock==3.32.3
frozenlist==1.8.0
fsspec==2025.3.0
h11==0.16.0
hf-transfer==0.1.9
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==0.36.2
idna==3.20
importlib-metadata==9.0.1
jinja2==3.1.6
markupsafe==3.0.3
mpmath==1.3.0
msgspec==0.22.0
multidict==6.9.1
multiprocess==0.70.16
networkx==3.6.1
numpy==2.2.6
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.3
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvtx-cu12==12.8.90
packaging==26.3
pandas==3.0.6
peft==0.21.2
pillow==12.3.0
propcache==0.5.4
protobuf==7.36.2
psutil==7.2.2
pyarrow==25.0.1
python-dateutil==2.9.0.post0
pyyaml==6.0.3
regex==2026.9.29
requests==2.34.2
safetensors==0.8.0
sentencepiece==0.2.2
setuptools==78.1.0
six==1.17.0
sympy==1.14.0
tokenizers==0.21.4
torch==2.8.0+cu128
torchao==0.18.0
torchvision==0.23.0+cu128
tqdm==4.70.1
transformers==4.55.4
triton==3.4.0
trl==0.22.2
typeguard==4.6.0
typing-extensions==4.16.0
tyro==1.0.16
unsloth-zoo==2025.9.9
unsloth==2025.9.7
urllib3==2.8.0
wheel==0.48.0
xformers==0.0.32.post2
xxhash==4.0.1
yarl==1.25.1
zipp==4.1.1
LOCK
  $UV pip install --python $V/bin/python --only-binary :all: --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cu128 -r $W/env-lock.txt || fail ENV_INSTALL_FAIL
  $UV pip uninstall --python $V/bin/python torchao
  $UV pip install --python $V/bin/python --no-deps \
      https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp311-cp311-linux_x86_64.whl \
      || fail FLASH_ATTN_FAIL
  $UV pip freeze --python $V/bin/python
  (cd $W && $V/bin/python $W/in/patch_cell.py > /dev/null) || fail PATCH_CELL_FAIL  # writes $W/qwen3.patch
  command -v patch >/dev/null || { apt-get update -q && apt-get install -y -q patch; } || fail PATCH_TOOL_FAIL
  patch --binary $V/lib/python3.11/site-packages/unsloth/models/qwen3.py $W/qwen3.patch || fail PATCH_FAIL
fi
if [ ! -f $M/config.json ]; then
  curl -fsS -L -o $W/model.tar.gz https://www.kaggle.com/api/v1/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1/download \
      || fail MODEL_DOWNLOAD_FAIL  # public anonymous GET (HEAD returns 404; GET verified 10-12)
  tar xzf $W/model.tar.gz -C $M && rm -f $W/model.tar.gz || fail MODEL_UNTAR_FAIL
  tar cf $W/model-v1.tar -C $M . && $GS cp -q $W/model-v1.tar $B/cache/model-v1.tar && rm -f $W/model-v1.tar  # bf16 weights: gzip gains nothing
fi
(cd $W/work && $V/bin/python -c "import torch; torch.set_default_device('cpu'); from unsloth import FastLanguageModel; import flash_attn, xformers; print('IMPORT_OK', flash_attn.__version__, xformers.__version__, torch.__version__)") \
  || fail IMPORT_FAIL
if [ $CACHED = 0 ] && ! $GS ls $B/cache/env-v1.tgz >/dev/null 2>&1; then
  tar czf $W/env-v1.tgz -C $W venv311 pyinst && $GS cp -q $W/env-v1.tgz $B/cache/env-v1.tgz && rm -f $W/env-v1.tgz
fi
echo "SETUP_DONE $(date -u +%FT%TZ) cached=$CACHED"; up_log

export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 UNSLOTH_DISABLE_STATISTICS=1 KAGGLE_IS_COMPETITION_RERUN=1 \
       ARC_V6_INPUT_ROOT=$W/input ARC_V6_WORK_DIR=$W/work ARC_V11_N_TRAIN_AUG=8 ARC_V11_N_EVAL_AUG=1
N=$(python3 -c "import json; print(len(json.load(open('$W/in/keys30.json'))))")

run_one() {  # name src pass offset cap
  local name=$1 dir=$W/out/$1
  mkdir -p $dir/done
  $GS rsync -q --recursive $B/out/e1/$name $dir 2>/dev/null || true
  local before=$(ls $dir/done | wc -l)
  for attempt in 1 2; do
    [ "$(ls $dir/done | wc -l)" -ge "$N" ] && { echo "RUN_COMPLETE $name"; return 0; }
    echo "RUN_START $name attempt $attempt $(date -u +%FT%TZ)"
    (cd $W/work && $V/bin/python $W/in/driver_e1.py --src $W/in/$2 --pass_label $3 --seed_offset $4 --cap $5 \
        --keys $W/in/keys30.json --run_dir $dir >> $dir/log.txt 2>&1) &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do sleep 120; $GS rsync -q --recursive --delete-unmatched-destination-objects $dir $B/out/e1/$name || true; done
    wait $pid; echo "RUN_EXIT $name rc=$? done=$(ls $dir/done | wc -l)/$N"
    $GS rsync -q --recursive --delete-unmatched-destination-objects $dir $B/out/e1/$name || true
  done
  if [ "$(ls $dir/done | wc -l)" -le "$before" ]; then  # no task finished in two attempts: stop paying for an idle GPU
    echo "RUN_STUCK $name $(date -u +%FT%TZ)"; up_log; poweroff; exit 1
  fi
  up_log
}

for s in 1 2 3; do
  off=$(( (s - 1) * 1000 ))
  run_one F-s$s  v8/src  A $off 1200
  run_one HA-s$s v11/src A $off 800
  run_one HB-s$s v11/src B $off 800
done

ALL=1
for r in F HA HB; do for s in 1 2 3; do
  [ "$(ls $W/out/$r-s$s/done 2>/dev/null | wc -l)" -ge "$N" ] || ALL=0
done; done
echo "ALL_RUNS_COMPLETE=$ALL $(date -u +%FT%TZ)"; up_log
[ $ALL = 1 ] && date -u +%FT%TZ > $W/DONE && $GS cp -q $W/DONE $B/out/e1/DONE
poweroff  # incomplete runs resume on the next VM the orchestrator launches
