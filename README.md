# FMBench: Adaptive Large Language Model Output Formatting

This repository contains the **SFT+GRPO finetuning** implementation for [**openPangu**](https://huggingface.co/collections/FreedomIntelligence/openpangu) with the **FMBench** dataset.

## ⚙️ Installation

**Step 1: Install the environment**

```shell
conda env create -f environment_pangu_grpo_root.yml
```

**Step 2: Replace the installed transformers package with our custom source code.**

## 🤗 Setup

**Step1**: Download the longformer weight checkpoint archive at [Modelscopes](https://www.modelscope.cn/datasets/Cook1e/OpenPangu/tree/master/longformer) and extract it to the project directory.

**Step2**: Download the pangu weight checkpoint archive at [Modelscopes](https://www.modelscope.cn/datasets/Cook1e/OpenPangu/tree/master/checkpoints) and extract it to the project directory.

## 📌 Getting Started

**Note**: The scripts in this repository use the 1B model as an example. To run the 7B model, please update the command-line arguments accordingly.

### Supervised Fine-tuning

``` bash run_sft.sh ```

### SFT+GRPO Fine-tuning 

``` bash run_sft_grpo_8npu.sh ```

### GRPO Fine-tuning

``` bash run_grpo.sh ```

### Run Inference & Evaluation

```shell
bash run_inf.sh
bash run_eval.sh 
```



 