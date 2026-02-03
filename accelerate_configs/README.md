# Accelerate 配置文件说明

本目录包含用于 GRPO 训练的 accelerate 配置文件。

## 配置文件说明

### 1. single_npu.yaml
单卡 NPU 训练配置
- `num_processes: 1` - 使用 1 个进程
- `distributed_type: NO` - 单卡训练，不使用分布式
- `mixed_precision: 'no'` - 不使用混合精度
- **注意**: 设备通过代码中的 `device_map` 或环境变量指定

### 2. multi_npu.yaml
4 卡 NPU 训练配置
- `num_processes: 4` - 使用 4 个进程
- `distributed_type: MULTI_GPU` - 使用多设备分布式（NPU 兼容模式）
- `mixed_precision: 'bf16'` - 使用 bfloat16 混合精度
- **注意**: 设备通过代码中的设备映射或环境变量自动分配

### 3. multi_npu_8cards.yaml
8 卡 NPU 训练配置
- `num_processes: 8` - 使用 8 个进程
- `distributed_type: MULTI_GPU` - 使用多设备分布式（NPU 兼容模式）
- `mixed_precision: 'bf16'` - 使用 bfloat16 混合精度
- **注意**: 设备通过代码中的设备映射或环境变量自动分配

### 4. multi_gpu.yaml
4 卡 GPU 训练配置（如果使用 GPU）
- `num_processes: 4` - 使用 4 个进程
- `use_npu: false` - 不使用 NPU
- `gpu_ids: all` - 使用所有 GPU
- `mixed_precision: 'bf16'` - 使用 bfloat16 混合精度

## 使用方法

### 方法 1: 使用配置文件启动训练

```bash
# 使用指定的配置文件
accelerate launch --config_file accelerate_configs/multi_npu.yaml train_grpo_batch.py

# 或者使用绝对路径
accelerate launch --config_file /opt/fmbench_wyt/accelerate_configs/multi_npu.yaml train_grpo_batch.py
```

### 方法 2: 使用默认配置文件

如果已经通过 `accelerate config` 配置了默认设置，可以直接运行：

```bash
accelerate launch train_grpo_batch.py
```

默认配置文件位置：`~/.cache/huggingface/accelerate/default_config.yaml`

### 方法 3: 命令行参数覆盖

可以在命令行中覆盖配置文件中的某些参数：

```bash
# 覆盖进程数
accelerate launch --config_file accelerate_configs/multi_npu.yaml --num_processes=8 train_grpo_batch.py

# 覆盖混合精度设置
accelerate launch --config_file accelerate_configs/multi_npu.yaml --mixed_precision=bf16 train_grpo_batch.py
```

## 配置文件参数说明

- `compute_environment`: 计算环境类型（LOCAL_MACHINE 表示本地机器）
- `distributed_type`: 分布式类型（NO/MULTI_GPU）
  - **注意**: 标准 accelerate 可能不支持 `MULTI_NPU`，对于 NPU 设备，使用 `MULTI_GPU` 类型
- `num_processes`: 进程数量（通常等于卡数）
- `num_machines`: 机器数量（单机训练为 1）
- `machine_rank`: 当前机器的 rank（单机训练为 0）
- `mixed_precision`: 混合精度类型（'no'/'fp16'/'bf16'）
- `use_cpu`: 是否使用 CPU
- `gpu_ids`: 指定使用的设备 ID（'all' 或列表）
  - **注意**: 对于 NPU，设备分配可能通过环境变量控制（如 `DEVICE_ID`）

## NPU 使用注意事项

**重要**: 标准 accelerate 库可能不完全支持 NPU 的原生配置。如果遇到配置问题：

1. **使用 MULTI_GPU 类型**: 对于 NPU，配置文件使用 `MULTI_GPU` 类型，但实际设备通过 PyTorch 的设备映射控制

2. **环境变量**: 可能需要通过环境变量指定 NPU 设备：
   ```bash
   export DEVICE_ID=0,1,2,3  # 指定使用的 NPU ID
   ```

3. **设备映射**: 在代码中（如 `helper.py`），确保使用正确的设备映射：
   ```python
   device_map="npu"  # 或具体的设备 ID
   ```

4. **验证配置**: 运行前先验证配置：
   ```bash
   accelerate env --config_file accelerate_configs/multi_npu.yaml
   ```

## 自定义配置

可以根据实际硬件环境修改配置文件：

1. 修改 `num_processes` 为实际的卡数
2. 修改 `gpu_ids` 指定使用的设备（对于 NPU，可能需要通过环境变量或代码中的设备映射控制）
3. 根据模型大小和显存情况调整 `mixed_precision`
4. 如果是多机训练，需要修改 `num_machines` 和 `machine_rank`

### 如何指定使用特定的 NPU？

由于标准 accelerate 可能不支持 `npu_ids` 配置项，可以通过以下方式：

1. **环境变量方式**（推荐）:
   ```bash
   export DEVICE_ID=0,2,4,6  # 指定使用的 NPU ID
   accelerate launch --config_file accelerate_configs/multi_npu.yaml train_grpo_batch.py
   ```

2. **代码中指定**: 在 `helper.py` 的 `init_model_and_tokenizer` 函数中，通过 `device_map` 参数指定：
   ```python
   device_map="npu:0,npu:2,npu:4,npu:6"
   ```

3. **修改配置文件**: 如果 accelerate 版本支持，可以尝试在配置文件中使用 `gpu_ids`:
   ```yaml
   gpu_ids: [0, 2, 4, 6]
   ```

## 验证配置

可以使用以下命令验证配置是否正确：

```bash
accelerate env --config_file accelerate_configs/multi_npu.yaml
```

这会显示当前环境信息和配置是否匹配。

