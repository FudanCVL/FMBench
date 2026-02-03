"""
评估脚本：计算预测结果的平均 bertscore (semantic reward) 和 reward_func_structure_2

使用方法:
    python run_eval.py --pred_file /opt/fmbench_wyt/preds/1b_sft.jsonl
"""

import torch
import argparse
from framework.io import read_jsonl
from helper import SummaryModel
from bert_score import score
from tqdm import tqdm



# 配置路径
bert_model_path = "local_models/allenai/longformer-base-4096"
model_local_path = "openPangu-Embedded-7B-V1.1"

# 初始化设备
# 优先使用 NPU，然后是 CUDA，最后是 CPU
if hasattr(torch, 'npu') and torch.npu.is_available():
    device = torch.device("npu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 初始化 SummaryModel（用于 reward_func_structure_2）
summary_model = SummaryModel(model_local_path=model_local_path)

def Bertscore_evaluate(cands, refs, local_model_path, device):
    """计算 BERTScore F1 分数"""
    P, R, F1 = score(
        cands, refs,
        model_type=local_model_path,
        num_layers=5,  # Longformer 通常用 layer 5
        device=device, 
        verbose=True
    )
    return F1

def reward_func_semantic(completions, **kwargs):
    """Reward function that checks if the answer matches the ground truth."""
    ground_truth = kwargs['ground_truth']

    def extract_answer(completion):
        if isinstance(completion, list):
            content = completion[0]["content"]
        else:
            content = completion
        return content
    
    contents = [extract_answer(completion) for completion in completions]
    ground_truth = [extract_answer(ground_truth) for ground_truth in ground_truth]
    
    # calculate bertscore
    F1 = Bertscore_evaluate(contents, ground_truth, bert_model_path, device)
    
    return F1

def reward_func_structure_2(completions, **kwargs):
    """Reward function that calculates structure similarity using summary model."""
    gt_list, pred_list = [], []
    ground_truth = kwargs['ground_truth']
    
    for completion, ground_truth in tqdm(zip(completions, ground_truth), 
                                         total=len(completions), 
                                         desc="Generating summaries"):
        content = completion[0]["content"] if isinstance(completion, list) else completion
        gt_content = ground_truth[0]["content"] if isinstance(ground_truth, list) else ground_truth
        
        summary = summary_model.get_summary(content)
        ground_truth_summary = summary_model.get_summary(gt_content)
        
        pred_list.append(summary)
        gt_list.append(ground_truth_summary)
    
    F1 = Bertscore_evaluate(pred_list, gt_list, bert_model_path, device)
    return F1

def evaluate_predictions(pred_file):
    """评估预测文件，计算平均 bertscore 和 reward_func_structure_2"""
    print(f"Reading predictions from: {pred_file}")
    print(f"Using device: {device}")
    
    # 读取预测文件
    predictions = []
    ground_truths = []
    
    print("Reading prediction file...")
    for item in tqdm(read_jsonl(pred_file), desc="Loading predictions"):
        predictions.append(item.get("prediction", ""))
        ground_truths.append(item.get("target_text", ""))
    
    total_samples = len(predictions)
    print(f"Total samples: {total_samples}")
    
    if total_samples == 0:
        print("No samples found in the prediction file!")
        return
    
    # 准备数据格式（转换为 reward 函数期望的格式）
    # reward 函数期望 completions 和 ground_truth 可以是字符串或列表格式
    completions = predictions
    ground_truths = ground_truths
    
    # 计算 semantic reward (bertscore)
    print("\n" + "="*50)
    print("Calculating semantic reward (BERTScore)...")
    print("="*50)
    # semantic_rewards = reward_func_semantic(completions, ground_truth=ground_truth)
    # avg_semantic = semantic_rewards.mean().item()
    semantic_rewards = []

    # 成对计算semantic, 不要一次性计算
    avg_semantic = 0
    for completion, ground_truth in tqdm(zip(completions, ground_truths), 
                                         total=len(completions), 
                                         desc="Calculating semantic reward"):
        semantic_reward = reward_func_semantic([completion], ground_truth=[ground_truth])[0]
        avg_semantic += semantic_reward
        semantic_rewards.append(semantic_reward)
    avg_semantic /= len(completions)

    
    print('> semantic bert score f1:', avg_semantic)

    
    # 计算 structure reward (reward_func_structure_2)
    print("\n" + "="*50)
    print("Calculating structure reward (reward_func_structure_2)...")
    print("="*50)
    structure_rewards = reward_func_structure_2(completions, ground_truth=ground_truths)
    avg_structure = structure_rewards.mean().item()
    
    # 打印结果
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Total samples: {total_samples}")
    # print(f"Average Semantic Reward (BERTScore F1): {avg_semantic:.4f}")
    print(f"Average Structure Reward (reward_func_structure_2):")
    print(avg_structure)
    print('> semantic bert score f1:', avg_semantic)
    print("="*50)
    
    return {
        "total_samples": total_samples,
        "avg_semantic_reward": avg_semantic,
        "avg_structure_reward": avg_structure,
        "semantic_rewards": semantic_rewards,
        "structure_rewards": structure_rewards
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions and calculate average rewards")
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="Path to the prediction JSONL file (e.g., /opt/fmbench_wyt/preds/1b_sft.jsonl)"
    )
    
    args = parser.parse_args()
    
    eval_out = evaluate_predictions(args.pred_file)
    print(eval_out)

    # 保存为log. 
    with open(f"{args.pred_file}_eval_log.txt", "w") as f:
        f.write(f"{args.pred_file}: {eval_out}\n")