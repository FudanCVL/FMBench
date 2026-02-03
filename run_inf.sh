export CUDA_VISIBLE_DEVICES=1
python run_inf.py \
  --data_path data_json/fm_test.json \
  --lora_path sft-grpo-output-1b/checkpoint-100 \
  --model_path openPangu-Embedded-1B-V1.1 \
  --output_path preds/1b_sft_grpo.jsonl \
  --max_new_tokens 1024

# python run_inf.py \--data_path data_json/fm_test.json \
#   --lora_path grpo-output-1b/checkpoint-100 \
#   --model_path openPangu-Embedded-1B-V1.1 \
#   --output_path preds/1b_grpo.jsonl \
#   --max_new_tokens 1024

# python run_inf.py \--data_path data_json/fm_test.json \
#   --lora_path sft-output-1b/checkpoint-200 \
#   --model_path openPangu-Embedded-1B-V1.1 \
#   --output_path preds/1b_sft.jsonl \
#   --max_new_tokens 1024

# python run_inf.py \--data_path data_json/fm_test.json \
#   --model_path openPangu-Embedded-1B-V1.1 \
#   --output_path preds/1b_original.jsonl \
#   --max_new_tokens 1024

# python run_inf.py \--data_path data_json/fm_test.json \
#   --lora_path sft-grpo-output-7b/checkpoint-100 \
#   --model_path openPangu-Embedded-7B-V1.1 \
#   --output_path preds/7b_sft_grpo.jsonl \
#   --max_new_tokens 1024

# python run_inf.py \--data_path data_json/fm_test.json \
#   --lora_path grpo-output-7b/checkpoint-100 \
#   --model_path openPangu-Embedded-7B-V1.1 \
#   --output_path preds/7b_grpo.jsonl \
#   --max_new_tokens 1024

# python run_inf.py \--data_path data_json/fm_test.json \
#   --lora_path sft-output-7b/checkpoint-200 \
#   --model_path openPangu-Embedded-7B-V1.1 \
#   --output_path preds/7b_sft.jsonl \
#   --max_new_tokens 1024

# python run_inf.py \--data_path data_json/fm_test.json \
#   --model_path openPangu-Embedded-7B-V1.1 \
#   --output_path preds/7b_original.jsonl \
#   --max_new_tokens 1024




