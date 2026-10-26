CUDA_VISIBLE_DEVICES=2 \
PYTHON="/share/project/wuhaiming/spaces/CC-LLM/scripts/train/env/bin/python" \
MODEL="/share/project/wuhaiming/spaces/CC-LLM/outputs/experiments/rnd2/cpt/Qwen3-1.7B-Base-PGCA-CPT/joint/v0-20261023-114945/checkpoint-12500" \
LORA="/share/project/wuhaiming/spaces/CC-LLM/outputs/experiments/rnd2/finetune/Qwen3-1.7B-Base-PGCA-CPT-ACT/v0-20261025-232029/checkpoint-1000" \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-ACT \
RUN_NAME=rnd2 \
DATASET=c3 \
FEWSHOT=0 \
BATCH_SIZE=32 \
STAGE=all \
bash scripts/evaluation/ancient/run.sh

CUDA_VISIBLE_DEVICES=0 \
PYTHON="/share/project/wuhaiming/spaces/CC-LLM/scripts/train/env/bin/python" \
MODEL="/share/project/wuhaiming/spaces/CC-LLM/outputs/experiments/rnd2/cpt/Qwen3-1.7B-Base-PGCA-CPT/joint/v0-20261023-114945/checkpoint-12500" \
LORA="/share/project/wuhaiming/spaces/CC-LLM/outputs/experiments/rnd2/finetune/Qwen3-1.7B-Base-PGCA-CPT-CSC/v0-20261025-173119/checkpoint-11000" \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-CSC \
RUN_NAME=rnd2 \
BATCH_SIZE=32 \
STAGE=all \
bash scripts/evaluation/csc/run.sh

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 \
MODEL=/share/project/wuhaiming/spaces/CC-LLM/outputs/experiments/rnd2/cpt/Qwen3-1.7B-Base-PGCA-CPT/joint/v0-20261023-114945/checkpoint-12500 \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT \
BENCHMARK=mmlu \
RUN_NAME=full bash scripts/evaluation/lmeh/run.sh
MODEL=/share/project/wuhaiming/data/models/Qwen3-1.7B-Base

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=5 \
MODEL=/share/project/wuhaiming/spaces/CC-LLM/outputs/experiments/rnd2/cpt/Qwen3-1.7B-Base-PGCA-CPT/joint/v0-20261023-114945/checkpoint-12500 \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT \
BENCHMARK=arc \
RUN_NAME=full bash scripts/evaluation/lmeh/run.sh

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 \
MODEL=/share/project/wuhaiming/spaces/CC-LLM/outputs/experiments/rnd2/cpt/Qwen3-1.7B-Base-PGCA-CPT/joint/v0-20261023-114945/checkpoint-12500 \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT \
BENCHMARK=squadv2 \
RUN_NAME=full bash scripts/evaluation/lmeh/run.sh

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=7 \
MODEL=/share/project/wuhaiming/spaces/CC-LLM/outputs/experiments/rnd2/cpt/Qwen3-1.7B-Base-PGCA-CPT/joint/v0-20261023-114945/checkpoint-12500 \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT \
BENCHMARK=gsm8k \
RUN_NAME=full bash scripts/evaluation/lmeh/run.sh

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=2 \
MODEL=/share/project/wuhaiming/spaces/CC-LLM/outputs/experiments/rnd2/cpt/Qwen3-1.7B-Base-PGCA-CPT/joint/v0-20261023-114945/checkpoint-12500 \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT \
BENCHMARK=cmmlu \
RUN_NAME=full bash scripts/evaluation/lmeh/run.sh

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=3 \
MODEL=/share/project/wuhaiming/spaces/CC-LLM/outputs/experiments/rnd2/cpt/Qwen3-1.7B-Base-PGCA-CPT/joint/v0-20261023-114945/checkpoint-12500 \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT \
BENCHMARK=clue \
RUN_NAME=full bash scripts/evaluation/lmeh/run.sh
