# Amazon Beauty MLM模型训练指南 (Google Colab)

本指南提供在Google Colab上训练Amazon Beauty MLM模型的完整步骤。

## 1. 环境设置

```python
# 克隆仓库
!git clone https://github.com/YOUR_REPO/HyperTreeRecomSys.git
%cd HyperTreeRecomSys

# 安装依赖
!pip install -q transformers tqdm numpy torch wandb
```

## 2. 上传数据

如果您已经有处理好的数据，可以通过以下方式上传：

```python
from google.colab import files
import os

# 创建必要的目录
!mkdir -p data/beauty_tokenizer
!mkdir -p data/mlm_data

# 上传tokenizer文件
uploaded = files.upload()  # 上传amazon_special_tokens.pkl
!mv amazon_special_tokens.pkl data/beauty_tokenizer/

# 如果有其他tokenizer文件，请继续上传
# ...

# 上传MLM训练数据
uploaded = files.upload()  # 上传train_mlm.txt
!mv train_mlm.txt data/mlm_data/

uploaded = files.upload()  # 上传val_mlm.txt
!mv val_mlm.txt data/mlm_data/
```

或者，您可以从头开始生成数据：

```python
# 重新生成tokenizer和MLM训练数据
!python -m src.data --max_samples 10000 --ensure_all_tokens
```

## 3. 初始化Weights & Biases (可选但推荐)

```python
import wandb
wandb.login()
```

## 4. 训练模型

以下是训练MLM模型的命令，包含了我们的所有增强功能：

```python
!python -m src.mlm_model \
  --mode train \
  --train_file data/mlm_data/train_mlm.txt \
  --val_file data/mlm_data/val_mlm.txt \
  --tokenizer_path data/beauty_tokenizer \
  --output_dir models/mlm \
  --batch_size 32 \
  --max_length 256 \
  --mlm_probability 0.15 \
  --num_epochs 5 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --seed 42 \
  --freeze_pretrained \
  --use_wandb \
  --wandb_project "amazon-beauty-mlm" \
  --wandb_run_name "mlm-training-run-1"
```

### 参数说明

- `--freeze_pretrained`：冻结原始预训练参数，只训练新增token的embedding
- `--max_length`：序列最大长度，建议值为256（涵盖大多数样本）或512（更长但训练更慢）
- `--batch_size`：根据GPU内存大小调整，Colab可使用16-32
- `--num_epochs`：训练轮数，建议3-5轮

## 5. 中断后继续训练

如果训练中断，可以从上次的checkpoint继续：

```python
!python -m src.mlm_model \
  --mode train \
  --train_file data/mlm_data/train_mlm.txt \
  --val_file data/mlm_data/val_mlm.txt \
  --tokenizer_path data/beauty_tokenizer \
  --output_dir models/mlm \
  --resume_from models/mlm/checkpoint_epoch_2.pth \
  --batch_size 32 \
  --max_length 256 \
  --mlm_probability 0.15 \
  --num_epochs 5 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --seed 42 \
  --freeze_pretrained \
  --use_wandb \
  --wandb_project "amazon-beauty-mlm" \
  --wandb_run_name "mlm-training-continued"
```

## 6. 训练监控 (可选)

```python
# 实时查看训练图表
from IPython.display import IFrame
IFrame(src=f"https://wandb.ai/YOUR_USERNAME/amazon-beauty-mlm/runs/YOUR_RUN_ID", width=1080, height=720)
```

## 7. 测试训练好的模型

```python
!python -m src.mlm_model \
  --mode test \
  --tokenizer_path data/beauty_tokenizer \
  --model_path models/mlm/best_model.pth
```

## 注意事项

1. **硬件资源**：建议使用GPU运行时环境
2. **存储**：确保Colab有足够磁盘空间（至少10GB）
3. **训练时间**：完整训练可能需要数小时，请使用带持久性的Colab Pro
4. **保存模型**：定期将训练好的模型下载到本地

## 常见问题

### 如果遇到OOM (内存不足)
- 减小`batch_size`
- 减小`max_length`
- 使用梯度累积（添加`--gradient_accumulation_steps 2`）

### 如果训练损失不下降
- 调整`learning_rate`（尝试1e-4或1e-5）
- 检查数据质量，确保训练样本覆盖所有token类型 