# Accelerate 多卡训练快速开始指南

## 快速启动

### 1. 使用预定义配置文件（推荐）

```bash
# 4卡 NPU 训练
accelerate launch --config_file accelerate_configs/multi_npu.yaml train_grpo_batch.py

# 8卡 NPU 训练
accelerate launch --config_file accelerate_configs/multi_npu_8cards.yaml train_grpo_batch.py

# 单卡 NPU 训练
accelerate launch --config_file accelerate_configs/single_npu.yaml train_grpo_batch.py
```

### 2. 使用默认配置

如果已经通过 `accelerate config` 配置了默认设置：

```bash
accelerate launch train_grpo_batch.py
```

### 3. 命令行参数方式

```bash
# 直接指定参数
accelerate launch --num_processes=4 --mixed_precision=bf16 train_grpo_batch.py
```

## 配置文件位置

- **项目配置文件**: `accelerate_configs/*.yaml`
- **默认配置文件**: `~/.cache/huggingface/accelerate/default_config.yaml`

## 创建自定义配置

### 方法 1: 交互式创建

```bash
accelerate config
```

按照提示选择：
- 分布式类型（单卡/多卡）
- 设备类型（NPU/GPU）
- 混合精度设置
- 进程数量等

### 方法 2: 复制并修改现有配置

```bash
# 复制现有配置文件
cp accelerate_configs/multi_npu.yaml accelerate_configs/my_custom_config.yaml

# 编辑配置文件
vim accelerate_configs/my_custom_config.yaml
```

主要修改项：
- `num_processes`: 进程数量（卡数）
- `npu_ids`: NPU ID 列表（如 `[0, 1, 2, 3]`）
- `mixed_precision`: 混合精度（`'no'`/`'fp16'`/`'bf16'`）

## 验证配置

```bash
# 检查配置是否正确
accelerate env --config_file accelerate_configs/multi_npu.yaml

# 查看当前 accelerate 环境
accelerate env
```

## 常见问题

### Q: 如何指定使用特定的 NPU？

A: 在配置文件中修改 `npu_ids` 字段：
```yaml
npu_ids: [0, 2, 4, 6]  # 只使用 0, 2, 4, 6 号 NPU
```

### Q: 如何禁用混合精度？

A: 在配置文件中设置：
```yaml
mixed_precision: 'no'
```

### Q: 如何查看训练日志？

A: 训练日志会显示每个进程的信息，主进程（rank 0）会显示完整的训练信息。

### Q: 配置文件中的参数含义？

A: 详细说明请参考 `accelerate_configs/README.md`

## 示例：完整训练命令

```bash
# 使用 4 卡 NPU，bf16 混合精度
cd /opt/fmbench_wyt
accelerate launch --config_file accelerate_configs/multi_npu.yaml train_grpo_batch.py

# 使用 8 卡 NPU，bf16 混合精度
accelerate launch --config_file accelerate_configs/multi_npu_8cards.yaml train_grpo_batch.py

# 单卡训练（不使用 accelerate，直接运行）
python train_grpo_batch.py
```


