#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the hierarchical encoder model.
This script reuses code from HierarchyTransformers for training a model on Amazon Beauty dataset.
"""

import os
import sys
import logging
import torch
import argparse
import yaml
from datetime import datetime
from tqdm import tqdm

# Configure logging before any imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("开始初始化训练脚本...")

# 获取项目根目录路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(ROOT_DIR)

# Import data loading function
logger.info("开始导入本地模块...")
try:
    from load import load_processed_datasets  # 从load.py导入
    logger.info("成功导入load模块")
except ImportError as e:
    logger.error(f"导入load模块失败: {e}")
    sys.exit(1)

# 导入HierarchyTransformers相关模块
logger.info("开始导入HierarchyTransformers模块...")
try:
    from hierarchy_transformers.models import HierarchyTransformer
    from hierarchy_transformers.losses import HierarchyTransformerLoss
    from hierarchy_transformers.evaluation import HierarchyTransformerEvaluator
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from hierarchy_transformers.models.hierarchy_transformer.hit_trainer import HierarchyTransformerTrainer
    logger.info("成功导入所有外部模块")
except ImportError as e:
    logger.error(f"导入HierarchyTransformers模块失败: {e}")
    logger.error("请确保已安装必要的依赖包，并且HierarchyTransformers位于正确的路径下")
    sys.exit(1)


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"已设置随机种子: {seed}")

def parse_args():
    """Parse command line arguments."""
    logger.info("解析命令行参数...")
    parser = argparse.ArgumentParser(description="Train a hierarchical encoder model on Amazon Beauty dataset")
    
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, default="datasets/amazon_beauty_dataset/processed", 
                        help="Directory containing the processed dataset files")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L12-v2",
                        help="Pre-trained model name or path")
    parser.add_argument("--output_dir", type=str, default="output/amazon-hierarchy",
                        help="Directory to save the model and results")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=20, 
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=128,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Loss function arguments
    parser.add_argument("--clustering_loss_weight", type=float, default=1.0,
                        help="Weight for clustering loss")
    parser.add_argument("--clustering_loss_margin", type=float, default=3.0,
                        help="Margin for clustering loss")
    parser.add_argument("--centripetal_loss_weight", type=float, default=1.0,
                        help="Weight for centripetal loss")
    parser.add_argument("--centripetal_loss_margin", type=float, default=0.5,
                        help="Margin for centripetal loss")
    
    args = parser.parse_args()
    logger.info(f"参数解析完成: {args}")
    return args

def save_config(args, output_dir):
    """Save training configuration to a YAML file."""
    logger.info(f"保存训练配置到 {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    config = vars(args)
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"配置已保存到 {config_path}")

def prepare_dataset_for_training(data_dir):
    """Prepare dataset for training with HierarchyTransformer."""
    logger.info(f"开始从 {data_dir} 加载数据集...")
    
    # 修正数据集路径，确保使用processed子目录
    # 如果data_dir不是以"processed"结尾，则添加"processed"子目录
    if not data_dir.endswith('processed'):
        if data_dir.endswith('mixed'):
            # 将mixed替换为processed
            processed_dir = data_dir.replace('mixed', 'processed')
        else:
            # 添加processed子目录
            processed_dir = os.path.join(data_dir, 'processed')
        logger.info(f"修正数据集路径为: {processed_dir}")
    else:
        processed_dir = data_dir
    
    # 检查processed目录是否存在
    if not os.path.exists(processed_dir):
        logger.error(f"错误: 处理后的数据集目录不存在: {processed_dir}")
        logger.error("请确保已运行数据处理脚本生成处理后的数据集")
        sys.exit(1)
    
    # 使用load_processed_datasets函数加载数据集
    triplet_datasets = load_processed_datasets(processed_dir)
    if triplet_datasets is None:
        logger.error(f"加载数据集失败，请确认数据集目录正确: {processed_dir}")
        sys.exit(1)
    
    logger.info(f"成功加载数据集: 训练集 {len(triplet_datasets['train'])} 样本, "
                f"验证集 {len(triplet_datasets['val'])} 样本, "
                f"测试集 {len(triplet_datasets['test'])} 样本")
    
    # Convert datasets to format compatible with HierarchyTransformerLoss
    # First, extract pairs for evaluation
    logger.info("准备评估数据对...")
    pair_datasets = {}
    for split, dataset in triplet_datasets.items():
        logger.info(f"为 {split} 集创建评估数据对...")
        pairs = []
        for example in dataset:
            # Positive pair (child, parent) with label 1
            pairs.append({
                'child': example['child'],
                'parent': example['parent'],
                'label': 1
            })
            # Negative pair (child, negative) with label 0
            pairs.append({
                'child': example['child'],
                'parent': example['negative'],
                'label': 0
            })
        pair_datasets[split] = pairs
        logger.info(f"{split} 集数据对创建完成: {len(pairs)} 个样本")
    
    logger.info(f"数据集准备完成")
    
    return triplet_datasets, pair_datasets

def main():
    """Main training function."""
    logger.info("开始训练流程...")
    
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    logger.info(f"创建输出目录: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    save_config(args, args.output_dir)
    
    # Load datasets
    triplet_datasets, pair_datasets = prepare_dataset_for_training(args.data_dir)
    
    # Load pre-trained model
    logger.info(f"开始加载预训练模型: {args.model_name}")
    model = HierarchyTransformer.from_pretrained(model_name_or_path=args.model_name)
    logger.info(f"预训练模型加载完成")
    
    # Set up the loss function
    logger.info("设置损失函数...")
    hit_loss = HierarchyTransformerLoss(
        model=model,
        clustering_loss_weight=args.clustering_loss_weight,
        clustering_loss_margin=args.clustering_loss_margin,
        centripetal_loss_weight=args.centripetal_loss_weight,
        centripetal_loss_margin=args.centripetal_loss_margin,
    )
    logger.info(f"损失函数配置: {hit_loss.get_config_dict()}")
    
    # Define validation evaluator
    logger.info("设置验证评估器...")
    val_evaluator = HierarchyTransformerEvaluator(
        child_entities=[example['child'] for example in pair_datasets['val']],
        parent_entities=[example['parent'] for example in pair_datasets['val']],
        labels=[example['label'] for example in pair_datasets['val']],
        batch_size=args.eval_batch_size,
        truth_label=1,
    )
    logger.info("验证评估器设置完成")
    
    # Define training arguments
    logger.info("设置训练参数...")
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=100,
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
    )
    logger.info("训练参数设置完成")
    
    # Create the trainer
    logger.info("创建训练器...")
    trainer = HierarchyTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=triplet_datasets['train'],
        eval_dataset=triplet_datasets['val'],
        loss=hit_loss,
        evaluator=val_evaluator,
    )
    logger.info("训练器创建完成")
    
    # Start training
    logger.info("开始训练过程...")
    trainer.train()
    logger.info("训练完成")
    
    # Evaluate on test set
    logger.info("在测试集上进行评估...")
    val_results = val_evaluator.results
    best_val = val_results.loc[val_results["f1"].idxmax()]
    best_val_centri_weight = float(best_val["centri_weight"])
    best_val_threshold = float(best_val["threshold"])
    logger.info(f"验证集最佳参数: centri_weight={best_val_centri_weight}, threshold={best_val_threshold}")
    
    test_evaluator = HierarchyTransformerEvaluator(
        child_entities=[example['child'] for example in pair_datasets['test']],
        parent_entities=[example['parent'] for example in pair_datasets['test']],
        labels=[example['label'] for example in pair_datasets['test']],
        batch_size=args.eval_batch_size,
        truth_label=1,
    )
    
    test_evaluator(
        model=model,
        output_path=os.path.join(args.output_dir, "eval"),
        best_centri_weight=best_val_centri_weight,
        best_threshold=best_val_threshold,
    )
    logger.info("测试集评估完成")
    
    # Save the final model
    final_output_dir = os.path.join(args.output_dir, "final")
    logger.info(f"保存最终模型到 {final_output_dir}")
    model.save(final_output_dir)
    logger.info(f"模型保存完成")
    logger.info("训练流程全部完成")

if __name__ == "__main__":
    main() 