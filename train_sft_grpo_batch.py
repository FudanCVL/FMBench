"""
使用 accelerate 框架进行多卡训练的 GRPO 训练脚本

使用方法:

1. 单卡训练（与原始脚本相同）:
   python train_grpo_batch.py

2. 多卡训练（使用配置文件，推荐）:
   # 使用预定义的配置文件
   accelerate launch --config_file accelerate_configs/multi_npu.yaml train_grpo_batch.py
   
   # 或使用绝对路径
   accelerate launch --config_file /opt/fmbench_wyt/accelerate_configs/multi_npu.yaml train_grpo_batch.py
   
   # 可用的配置文件:
   # - accelerate_configs/single_npu.yaml      (单卡 NPU)
   # - accelerate_configs/multi_npu.yaml      (4卡 NPU)
   # - accelerate_configs/multi_npu_8cards.yaml (8卡 NPU)
   # - accelerate_configs/multi_gpu.yaml       (4卡 GPU)

3. 多卡训练（使用默认配置）:
   # 如果已经通过 accelerate config 配置了默认设置
   accelerate launch train_grpo_batch.py

4. 多卡训练（命令行参数）:
   # 直接在命令行指定参数
   accelerate launch --num_processes=4 --mixed_precision=bf16 train_grpo_batch.py

5. 创建自定义配置文件:
   # 交互式创建配置文件
   accelerate config
   
   # 或手动编辑配置文件，参考 accelerate_configs/ 目录下的示例

配置文件位置:
- 默认配置文件: ~/.cache/huggingface/accelerate/default_config.yaml
- 项目配置文件: accelerate_configs/*.yaml

注意:
- 使用 accelerate 可以自动管理设备分配和分布式训练
- SummaryModel 在每个进程都初始化，确保 reward 计算的一致性
- reward 函数会自动在正确的设备上执行
- 每个进程会自动分配到不同的设备（通过 accelerate 管理）
- 详细配置说明请参考 accelerate_configs/README.md
"""

import torch
from trl import GRPOTrainer, GRPOConfig
from helper import init_model_and_tokenizer, build_model_with_lora
from fmbdata import init_fmb_dataset
import re

# metric.1: structure
# metric.2: semantic

from bert_score import score
from transformers import AutoModelForMaskedLM

import argparse
from collections import Counter
from framework.io import read_jsonl, load_yaml
from framework.validator import validate_markdown

from helper import SummaryModel
from accelerate import Accelerator

def parse_arguments():
    parser = argparse.ArgumentParser(description='GRPO Training Script')
    parser.add_argument('--model_path', type=str, default="/opt/pangu/openPangu-Embedded-7B-V1.1", 
                       help='Path to the model to be used for training')
    parser.add_argument('--lora_path', type=str, default="/opt/fmbench_wyt/sft-output-7b/checkpoint-200",
                       help='Path to the LoRA weights (optional)')
    return parser.parse_args()

# 解析命令行参数
args = parse_arguments()

# 初始化 Accelerator
# 使用 accelerate 进行多卡训练，支持自动设备管理和分布式训练
# 运行方式: accelerate launch train_grpo_batch.py
# 或者: python -m accelerate.commands.launch train_grpo_batch.py
accelerator = Accelerator()

# 获取当前进程的设备
device = accelerator.device

bert_model_path = "local_models/allenai/longformer-base-4096"
# model_local_path = "/opt/pangu/openPangu-Embedded-1B-V1.1" 
model_local_path = args.model_path  # 使用命令行参数

# SummaryModel 初始化
# 注意：在多卡训练时，每个进程都需要能够计算 reward
# 但 SummaryModel 可能较大，可以根据实际情况选择是否在每个进程都初始化
# 如果资源充足，可以在每个进程都初始化；如果资源有限，可以只在主进程初始化
# 这里选择在每个进程都初始化，确保 reward 计算的一致性
summary_model = SummaryModel(model_local_path=model_local_path)

CFG = load_yaml("config/v1.yaml")

# 如何获取结构上的相似度?
# 让一个 frozen-llm或者一个rule-base, 得到输出的结构, 以及spec的结构,
# 把这些结构转换为文本, 然后计算结构上的相似度
# reader-critic GRPO. 
# 那如果直接判断是否相似呢? 
# 可以判断格式相似度, 也可以总结为文本, 然后计算文本相似度. 

def Bertscore_evaluate(cands, refs, local_model_path, device):
    #cands = ["Wave function collapse in quantum mechanics is a fundamental concept."]
    #refs = ["The weather is really nice today—perfect for going out for a walk."]
    P, R, F1 = score(
        cands, refs,
        model_type=local_model_path,  # 看看怎么传入本地绝对路径
        num_layers=5,  # Longformer 通常用 layer 5
        device=device, 
        verbose=True
    )
    #print(f"BERTScore F1: {F1.mean().item():.4f}")
    return F1

# cands = ["What is the capital of China"]
# refs = ["What is the capital of America"]
# local_model_path = "/opt/pangu/fmbench/local_models/allenai/longformer-base-4096"
# F1 = Bertscore_evaluate(cands,refs,local_model_path)

def reward_func_semantic(completions, **kwargs):
    """Reward function that checks if the answer matches the ground truth."""
    # Regular expression to capture content inside \boxed{} or <answer> tags
    # print(kwargs.keys())
    ground_truth = kwargs['ground_truth']

    def extract_answer(completion):
        if isinstance(completion, list):
            content = completion[0]["content"]
        else:
            content = completion
        # print(f"content: {content}")
        # print(f"content: {completion[0].keys()}")
        return content
    
    contents = [extract_answer(completion) for completion in completions]
    ground_truth = [extract_answer(ground_truth) for ground_truth in ground_truth]
    # calculate bertscore

    # print('>>> contents:', contents)
    # print('>>> ground_truth:', ground_truth)
    # input('-wait.1')    
    F1 = Bertscore_evaluate(contents, ground_truth, bert_model_path, device)
    # print(f"F1: {F1}")
    
    # input()
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return F1  # [F1]
    # return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]

def reward_func_structure_1(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # ctr = Counter()
    spec_list = kwargs["spec"]
    total = 0
    passed = 0

    rewards = []
    for completion, spec in zip(completions, spec_list):
        content = completion[0]["content"] if isinstance(completion, list) else completion
        # print('>>> completion:', content)
        # print('>>> spec:', spec)
        res = validate_markdown(content, spec, CFG)

        if not res["hard_fail"]:
            rewards.append(1.0)
        else:
            fails = len(res["hard_fail_reasons"])
            reward = 1.0 - (max(fails, 10) / 10.0)

            rewards.append(reward)
    return rewards
    # return [1.0]



def reward_func_structure_2(completions, **kwargs):
    # rewards = []
    gt_list, pred_list= [], []
    ground_truth = kwargs['ground_truth']
    
    for completion, ground_truth in zip(completions, ground_truth):
        content = completion[0]["content"] if isinstance(completion, list) else completion
        gt_content = ground_truth[0]["content"] if isinstance(ground_truth, list) else ground_truth
        # print('>>> content:', type(content), len(content))
        # print('>>> ground_truth:', type(gt_content), len(gt_content))
        summary = summary_model.get_summary(content)
        ground_truth_summary = summary_model.get_summary(gt_content)
        # print('>>> summary:', summary)
        # print('>>> ground_truth_summary:', ground_truth_summary)
        # print('>>> summary:', summary[:20])
        # print('>>> ground_truth_summary:', ground_truth_summary[:20])
        pred_list.append(summary)
        gt_list.append(ground_truth_summary)
        # get bert score
    
    F1 = Bertscore_evaluate(pred_list, gt_list, bert_model_path, device)
    return F1


if __name__ == "__main__":
    # 在主进程打印信息
    if accelerator.is_main_process:
        print(f"Using device: {device}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Model path: {model_local_path}")
        print(f"LoRA path: {args.lora_path}")
    
    fmb_dataset = init_fmb_dataset(data_path = 'data_json/fm_train.json')

    model_local_path = model_local_path
    lora_path = args.lora_path  # 使用命令行参数中的 LoRA 路径
    # lora_path = None
    
    # 注意：init_model_and_tokenizer 返回 (tokenizer, model)，不是 (model, tokenizer)
    # 对于多卡训练，模型会在 Trainer 初始化时自动分配到正确的设备
    tokenizer, model = init_model_and_tokenizer(model_local_path, lora_path=lora_path)
    # tokenizer, model = init_model_and_tokenizer(model_local_path, lora_path)

    # TODO: 如果模型没有加载lora, 则这里给加上lora, 并且需要冻结基础模型参数, 
    if lora_path is None:
        model = build_model_with_lora(model)
    
    # 如果是多卡训练，需要将模型移动到当前进程的设备上
    # Trainer 会自动处理设备分配，但我们需要确保模型在正确的设备上
    if accelerator.num_processes > 1:
        # 多卡训练时，将模型移动到当前进程的设备
        model = model.to(device)
    
    # 确保只训练 LoRA 参数
    # 如果模型已经是 PeftModel，默认只有 LoRA 参数是可训练的
    # 但我们需要确保基础模型参数被冻结
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # 设置模型为训练模式
    model.train()
    
    # 打印可训练参数数量（只在主进程打印）
    if accelerator.is_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    if '1B' in model_local_path:
        output_dir = "./sft-grpo-output-1b"
    elif '7B' in model_local_path:
        output_dir = "./sft-grpo-output-7b"
    else:
        output_dir = "./sft-grpo-output"
        
    training_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        # gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        save_safetensors=False,
        num_generations=4,
        # 启用分布式训练相关配置
        # 注意：如果使用 accelerate launch，Trainer 会自动检测并使用 DDP
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,  # 在多卡训练时可能需要调整
        # 禁用 gradient checkpointing 以避免与 DDP 的冲突（如果需要可以启用）
        # gradient_checkpointing=False,
    )

    # 创建 trainer
    # GRPOTrainer 继承自 Transformers Trainer，会自动检测分布式环境
    # 如果使用 accelerate launch，Trainer 会自动使用 DDP
    # 注意：不要手动使用 accelerator.prepare() 包装模型，让 Trainer 自己处理
    trainer = GRPOTrainer(
        model=model,
        # reward_funcs=[reward_func_semantic],
        reward_funcs=[reward_func_structure_2, reward_func_semantic],
        # reward_funcs=[reward_func_structure, reward_func_semantic],
        train_dataset=fmb_dataset,
        args=training_args,
        processing_class=tokenizer,
    )
    
    # 开始训练
    # Trainer 会自动处理分布式训练（如果检测到多进程环境）
    # 使用 accelerate launch 时，Trainer 会自动使用 DDP，不需要手动 prepare
    trainer.train()
    
    # 训练完成后，只在主进程保存最终模型
    if accelerator.is_main_process:
        trainer.save_model("./sft-grpo-output-7b/final_model")
        print("Training completed and model saved!")

