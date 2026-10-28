#!/bin/sh
set -eu
CUDA_VISIBLE_DEVICES=0 OPENBLAS_CORETYPE=Prescott OMP_NUM_THREADS=4 /home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python -u /home/liuyu-nfs/Projects/go2-arx5/PI0278/R2/code/infer.py --point COVER934096 < /home/liuyu-nfs/Projects/go2-arx5/PI0278/R2/handoff/COVER934096/INPUT_REQUEST.json
