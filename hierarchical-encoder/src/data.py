#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Amazon产品层次结构数据处理模块

基于评论筛选提取Amazon产品层次结构并构建数据集
"""

import os
import sys
import json
import gzip
import random
import logging
import ast
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# 导入HierarchyTransformers相关模块
try:
    from hierarchy_transformers.datasets.construct import HierarchyDatasetConstructor
    from hierarchy_transformers.datasets.load import load_zenodo_dataset, zenodo_example_to_triplets
except ImportError as e:
    print(f"导入HierarchyTransformers模块失败: {str(e)}")
    raise

def load_entity_lexicon(file_path: str) -> Dict[str, str]:
    """Load entity lexicon from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        entity_lexicon = json.load(f)
    logger.info(f"Loaded {len(entity_lexicon)} entities from lexicon")
    return entity_lexicon

def load_local_dataset(file_path: str, entity_lexicon: Dict[str, str]) -> List[Dict[str, str]]:
    """Load dataset from JSONL file and convert IDs to descriptions"""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                example = json.loads(line.strip())
                child_id = example.get('child')
                parent_id = example.get('parent')
                
                # Skip if child or parent ID not found in lexicon
                if not child_id or not parent_id:
                    continue
                if child_id not in entity_lexicon or parent_id not in entity_lexicon:
                    continue
                    
                # Get descriptions
                child_desc = entity_lexicon[child_id]
                parent_desc = entity_lexicon[parent_id]
                
                # Handle hard negatives if present
                if 'hard_negatives' in example and example['hard_negatives']:
                    for neg_id in example['hard_negatives']:
                        if neg_id in entity_lexicon:
                            examples.append({
                                'child': child_desc,
                                'parent': parent_desc,
                                'negative': entity_lexicon[neg_id]
                            })
                # Handle random negatives
                elif 'random_negatives' in example and example['random_negatives']:
                    for neg_id in example['random_negatives']:
                        if neg_id in entity_lexicon:
                            examples.append({
                                'child': child_desc,
                                'parent': parent_desc,
                                'negative': entity_lexicon[neg_id]
                            })
                            
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse line: {line.strip()}")
                continue
                
    logger.info(f"Loaded {len(examples)} examples from {file_path}")
    return examples

class AmazonDataset(Dataset):
    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.examples[idx]

def prepare_amazon_hit_data_with_reviews(
    data_dir: str,
    use_hard_negatives: bool = True
) -> Tuple[Dataset, Dataset]:
    """
    Prepare Amazon dataset with product reviews
    
    Args:
        data_dir: Directory containing the dataset files
        use_hard_negatives: Whether to use hard negative samples
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load entity lexicon
    entity_lexicon_path = os.path.join(data_dir, 'entity_lexicon.json')
    entity_lexicon = load_entity_lexicon(entity_lexicon_path)
    
    # Load train and validation datasets
    train_path = os.path.join(data_dir, 'train.jsonl')
    val_path = os.path.join(data_dir, 'val.jsonl')
    
    train_examples = load_local_dataset(train_path, entity_lexicon)
    val_examples = load_local_dataset(val_path, entity_lexicon)
    
    train_dataset = AmazonDataset(train_examples)
    val_dataset = AmazonDataset(val_examples)
    
    return train_dataset, val_dataset

def save_processed_datasets(datasets, output_dir):
    """
    将处理后的Dataset对象保存到磁盘
    
    Args:
        datasets: 包含train, val, test的数据集字典
        output_dir: 输出目录
    """
    # 确保使用绝对路径
    output_dir = os.path.abspath(output_dir)
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    logger.info(f"保存处理后的数据集到: {processed_dir}")
    
    for split, dataset in datasets.items():
        split_dir = os.path.join(processed_dir, split)
        # 确保每个分割的目录存在
        os.makedirs(split_dir, exist_ok=True)
        try:
            # 使用绝对路径保存数据集
            dataset.save_to_disk(split_dir)
            logger.info(f"成功保存{split}数据集到: {split_dir}")
        except Exception as e:
            logger.error(f"保存{split}数据集时出错: {str(e)}")
            # 打印更详细的错误信息
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"所有处理后的数据集已保存到: {processed_dir}")
    return processed_dir

def run_test():
    """运行测试"""
    # 默认文件路径
    meta_file_path = "../data/meta_Beauty_2014.json.gz"
    reviews_file_path = "../data/reviews_Beauty_5.json.gz"
    output_dir = "datasets/amazon_beauty_dataset"
    
    # 转换为绝对路径
    meta_file_path = os.path.abspath(meta_file_path)
    reviews_file_path = os.path.abspath(reviews_file_path)
    output_dir = os.path.abspath(output_dir)
    
    # 检查文件是否存在
    if not os.path.exists(meta_file_path) or not os.path.exists(reviews_file_path):
        logger.error("测试数据文件不存在")
        logger.error(f"meta_file_path: {meta_file_path}")
        logger.error(f"reviews_file_path: {reviews_file_path}")
        return
    
    logger.info(f"使用数据文件:")
    logger.info(f"- 元数据: {meta_file_path}")
    logger.info(f"- 评论: {reviews_file_path}")
    logger.info(f"- 输出目录: {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据集
    try:
        logger.info("开始准备数据集...")
        datasets = prepare_amazon_hit_data_with_reviews(
            data_dir=output_dir,
            use_hard_negatives=True
        )
        
        if datasets is None:
            logger.error("数据集准备失败")
            return
            
        # 保存处理后的数据集
        processed_dir = save_processed_datasets(datasets, output_dir)
            
        # 显示成功结果
        logger.info("=" * 50)
        logger.info("测试成功完成")
        logger.info(f"训练集: {len(datasets[0])} 样本")
        logger.info(f"验证集: {len(datasets[1])} 样本")
        logger.info(f"处理后的数据集保存在: {processed_dir}")
        logger.info("=" * 50)
        
        # 展示几个样本的结构
        if len(datasets[0]) > 0:
            logger.info("样本格式:")
            example = datasets[0][0]
            for key, value in example.items():
                if isinstance(value, (list, dict)) or (hasattr(value, 'shape') and len(value.shape) > 0):
                    shape_info = f"[shape: {getattr(value, 'shape', len(value))}]"
                    logger.info(f"- {key}: {type(value).__name__} {shape_info}")
                else:
                    logger.info(f"- {key}: {value[:100]}..." if isinstance(value, str) and len(value) > 100 else f"- {key}: {value}")
            
            # 展示更多样本
            logger.info("\n更多样本示例:")
            for i in range(min(5, len(datasets[0]))):
                example = datasets[0][i]
                logger.info(f"\n样本 {i+1}:")
                logger.info(f"- child: {example['child']}")
                logger.info(f"- parent: {example['parent']}")
                logger.info(f"- negative: {example['negative']}")
    except Exception as e:
        logger.error(f"测试过程出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 设置日志级别为DEBUG以获取更多信息
        import logging
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        
        logger.info("=" * 60)
        logger.info("开始数据处理模块测试")
        logger.info("=" * 60)
        
        run_test()
        
        logger.info("=" * 60)
        logger.info("测试结束")
        logger.info("=" * 60)
    else:
        print("可以通过运行 'python -m src.data test' 来测试数据集构建功能") 