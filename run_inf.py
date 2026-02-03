#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from fmbdata import init_fmb_dataset

from peft import PeftModel


def load_model_and_tokenizer(model_path: str, lora_path: Optional[str] = None):
    #load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False, 
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        local_files_only=True
    )

    print('>>> model:', model.device)
    # input("Press Enter to continue...")

    # 加载 LoRA（可选）
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path, device_map={"": "cpu"})

        # 如果你想合并权重可用：model = model.merge_and_unload()
        # 但在某些 NPU/模型实现上可能不如直接 adapter 稳定，就先不默认合并。
    model = model.to(torch.device("npu:0"))
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> str:
    # 使用 chat_template 构造输入
    # add_generation_prompt=True 让模型以 assistant 口吻开始生成
    # print('>>> messages:', messages)
    input_ids = tokenizer.apply_chat_template(
        messages,
        # tokenize=False,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    # print('>>> input_ids:', input_ids)
    # input("Press Enter to continue...")

    device = "npu"
    input_ids = input_ids.to(device)

    # print('>>> ', model.device, input_ids.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][input_ids.shape[-1]:]
    pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return pred.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/opt/pangu/openPangu-Embedded-7B-V1.1")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA adapter 路径，比如 ./sft-output/checkpoint-xxx")
    parser.add_argument("--data_path", type=str, default="/opt/pangu/fmbench/data_json/administrative_document_test.json")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=-1, help="只跑前 N 条，-1 表示全量")
    parser.add_argument("--print_every", type=int, default=1, help="每隔多少条在终端打印一次(1表示每条都打印)")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.lora_path)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    fout = open(args.output_path, "w", encoding="utf-8")

    fmb_dataset = init_fmb_dataset(args.data_path)


    for i, sample in enumerate(tqdm(fmb_dataset, desc="Infer")):
        print('>>> sample:', sample.keys())

        # input("Press Enter to continue...")
        target_text = sample["completion"][0]["content"]
        messages = sample["prompt"]
        # messages = ensure_user_messages(sample.copy())[]
        messages[0]['content'] = messages[0]['content'] + ' /no_think'
        # print('>>> messages:', messages)
        # input("Press Enter to continue...")

        pred = generate_one(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True
        )

        # 主要输出文本（终端）
        if args.print_every > 0 and (i % args.print_every == 0):
            print("\n" + "=" * 80)
            print(f"[idx={i}] PREDICTION:\n{pred}")
            print("-" * 80)
            print(f"[idx={i}] TARGET_TEXT:\n{target_text}")

        rec = {
            "idx": i,
            "prediction": pred,
            "target_text": target_text,
            "prompt": messages,  # 方便你回溯
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    fout.close()
    print(f"\nDone. Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
