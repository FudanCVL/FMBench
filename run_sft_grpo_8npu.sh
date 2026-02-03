# 4卡 NPU 训练（最常用）
# accelerate launch --config_file accelerate_configs/multi_npu.yaml train_grpo_batch.py

# 8卡 NPU 训练
accelerate launch --config_file accelerate_configs/multi_npu_8cards.yaml train_sft_grpo_batch.py --model_path openPangu-Embedded-1B-V1.1 --lora_path sft-output-1b/checkpoint-200
# accelerate launch --config_file accelerate_configs/multi_npu_8cards.yaml train_sft_grpo_batch.py --model_path openPangu-Embedded-7B-V1.1 --lora_path sft-output-7b/checkpoint-200
# accelerate launch --config_file accelerate_configs/multi_npu_8cards.yaml train_grpo_batch.py --model_path openPangu-Embedded-1B-V1.1
# accelerate launch --config_file accelerate_configs/multi_npu_8cards.yaml train_grpo_batch.py --model_path openPangu-Embedded-7B-V1.1

# 单卡 NPU 训练
# accelerate launch --config_file accelerate_configs/single_npu.yaml train_grpo_batch.py

# 使用绝对路径
# accelerate launch --config_file /opt/fmbench_wyt/accelerate_configs/multi_npu.yaml train_grpo_batch.py