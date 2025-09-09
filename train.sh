# Replace train.py with grpo.py or dpo.py
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py