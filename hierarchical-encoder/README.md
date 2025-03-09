# Amazon Beauty 层次编码器

基于HierarchyTransformers实现的Amazon Beauty产品层次编码器。

## 简介

这个项目是对原有MLM模型的改进，使用超球面空间（Poincaré球）来编码产品和类别的层次结构。模型基于HiT（Hierarchy Transformer）框架，通过联合优化超球面聚类损失和超球面向心损失，学习产品和类别的层次结构。

## 方法

1. **实体表示**：使用产品和类别的文本名称及描述作为模型输入，而不是使用ID，保留语义信息。
2. **层次关系**：构建类别-类别和类别-产品的层次关系，形成完整的层次结构。
3. **超球面嵌入**：使用预训练语言模型编码器和Poincaré球，在超球面空间中表示层次关系。
4. **层次损失**：
   - 超球面聚类损失：将相关实体在超球面空间中聚集在一起，将无关实体分开。
   - 超球面向心损失：确保父实体位于超球面空间中靠近原点的位置。

## 目录结构

```
hierarchical-encoder/
├── src/
│   ├── data.py                 # 数据处理
│   ├── hierarchy_model.py      # 层次编码器模型
│   ├── losses.py               # 损失函数
│   ├── train.py                # 训练脚本
│   └── evaluate.py             # 评估脚本
├── run_hit_train.sh            # 训练运行脚本
└── README.md                   # 项目说明
```

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install geoopt  # 超球面几何计算库
```

### 2. 准备数据

模型使用Amazon Beauty元数据文件构建层次结构和训练数据。数据处理流程如下：

- 从元数据中提取产品类别层次结构
- 为产品构建文本描述
- 构建训练三元组（产品，父类别，非相关类别）
- 区分直接关系和间接关系用于多跳推理和混合跳预测任务

### 3. 训练模型

```bash
./run_hit_train.sh
```

或者手动运行：

```bash
python -m src.train \
  --meta_file data/meta_Beauty_2014.json.gz \
  --output_dir models/hit \
  --model_name "bert-base-uncased" \
  --batch_size 256 \
  --num_epochs 20
```

### 4. 评估模型

```bash
python -m src.evaluate \
  --meta_file data/meta_Beauty_2014.json.gz \
  --model_path models/hit \
  --task mixedhop \
  --negative_type random
```

## 参考

本项目基于HierarchyTransformers框架实现，参考论文《Language Models as Hierarchy Encoders》。

```
@inproceedings{NEURIPS2024_1a970a3e,
 author = {He, Yuan and Yuan, Moy and Chen, Jiaoyan and Horrocks, Ian},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Language Models as Hierarchy Encoders},
 year = {2024}
}
``` 