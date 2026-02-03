import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType
from fmbdata import init_fmb_dataset
from helper import init_model_and_tokenizer

def main():
    # ====== 1. 命令行参数解析 ======
    parser = argparse.ArgumentParser(description="SFT Training for OpenPangu with FMBench")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the local model directory (e.g., /opt/pangu/openPangu-Embedded-7B-V1.1)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data_json/fm_train.json",
        help="Path to the training data JSON file"
    )
    args = parser.parse_args()

    model_local_path = args.model_path
    data_path = args.data_path

    # ====== 2. 加载数据集 ======
    fmb_dataset = init_fmb_dataset(data_path)

    # ====== 3. 初始化模型和 tokenizer ======
    tokenizer, model = init_model_and_tokenizer(model_local_path)

    # ====== 4. （可选）统计序列长度分布 ======
    lengths = []
    for x in fmb_dataset:
        messages = x["prompt"] + x["completion"]
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        # 注意：apply_chat_template 已返回 token IDs（如果 tokenize=True），无需再 encode
        lengths.append(len(token_ids))
    
    print("Sequence length percentiles:", np.percentile(lengths, [50, 90, 95, 98, 100]))

    # ====== 5. LoRA 配置 ======
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        inference_mode=False,
    )

    # ====== 6. 训练输出目录 ======
    if '1B' in model_local_path:
        output_dir = "./sft-output-1b"
    elif '7B' in model_local_path:
        output_dir = "./sft-output-7b"
    else:
        output_dir = "./sft-output"

    # ====== 7. 训练配置 ======
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=400,
        save_total_limit=2,
        max_length=2048,
        packing=False,
        completion_only_loss=True,
        save_safetensors=False
    )

    # ====== 8. 初始化 Trainer ======
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=fmb_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,  # tokenizer 用于数据处理
    )

    # ====== 9. 开始训练 ======
    trainer.train()
    print("Training completed.")

if __name__ == "__main__":
    main()