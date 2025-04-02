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
from huggingface_hub import login, HfApi

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
    converted_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            try:
                example = json.loads(line.strip())
                child_id = example.get('child')
                parent_id = example.get('parent')
                
                # Skip if child or parent ID not found in lexicon
                if not child_id or not parent_id:
                    logger.warning(f"Line {line_idx+1}: Missing child or parent ID")
                    continue
                if child_id not in entity_lexicon:
                    logger.warning(f"Line {line_idx+1}: Child ID {child_id} not found in lexicon")
                    continue
                if parent_id not in entity_lexicon:
                    logger.warning(f"Line {line_idx+1}: Parent ID {parent_id} not found in lexicon")
                    continue
                    
                # Convert IDs to descriptions
                child_desc = entity_lexicon[child_id]
                parent_desc = entity_lexicon[parent_id]
                
                # Handle hard negatives if present
                if 'hard_negatives' in example and example['hard_negatives']:
                    for neg_id in example['hard_negatives']:
                        if neg_id in entity_lexicon:
                            # Create example with descriptions
                            examples.append({
                                'child': child_desc,
                                'parent': parent_desc,
                                'negative': entity_lexicon[neg_id],
                                # Store original IDs for debugging
                                'original_child_id': child_id,
                                'original_parent_id': parent_id,
                                'original_negative_id': neg_id
                            })
                            converted_count += 1
                        else:
                            logger.warning(f"Line {line_idx+1}: Hard negative ID {neg_id} not found in lexicon")
                
                # Handle random negatives
                elif 'random_negatives' in example and example['random_negatives']:
                    for neg_id in example['random_negatives']:
                        if neg_id in entity_lexicon:
                            # Create example with descriptions
                            examples.append({
                                'child': child_desc,
                                'parent': parent_desc,
                                'negative': entity_lexicon[neg_id],
                                # Store original IDs for debugging
                                'original_child_id': child_id,
                                'original_parent_id': parent_id,
                                'original_negative_id': neg_id
                            })
                            converted_count += 1
                        else:
                            logger.warning(f"Line {line_idx+1}: Random negative ID {neg_id} not found in lexicon")
                            
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse line {line_idx+1}: {line.strip()}")
                continue
    
    # Print conversion summary
    logger.info(f"Loaded and converted {len(examples)} examples from {file_path}")
    logger.info(f"Successfully converted {converted_count} examples to use text descriptions")
    
    # Show a sample of converted data
    if examples:
        sample = examples[0]
        logger.info("Sample converted example:")
        logger.info(f"  Original: {sample['original_child_id']} -> {sample['original_parent_id']}")
        logger.info(f"  Converted: '{sample['child']}' -> '{sample['parent']}'")
    
    return examples

class AmazonDataset(Dataset):
    def __init__(self, examples: List[Dict[str, str]]):
        # Remove original IDs to avoid them being used in training
        self.examples = []
        for example in examples:
            # Create a new dict with only the required fields
            self.examples.append({
                'child': example['child'],
                'parent': example['parent'],
                'negative': example['negative']
            })
        
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
    
    # Load train and validation datasets from mixed directory
    mixed_dir = os.path.join(data_dir, 'mixed')
    train_path = os.path.join(mixed_dir, 'train.jsonl')
    val_path = os.path.join(mixed_dir, 'val.jsonl')
    
    train_examples = load_local_dataset(train_path, entity_lexicon)
    val_examples = load_local_dataset(val_path, entity_lexicon)
    
    train_dataset = AmazonDataset(train_examples)
    val_dataset = AmazonDataset(val_examples)
    
    return train_dataset, val_dataset

def save_processed_datasets(datasets: Tuple[Dataset, Dataset], output_dir: str):
    """
    将处理后的Dataset对象保存到磁盘
    
    Args:
        datasets: 包含(train_dataset, val_dataset)的元组
        output_dir: 输出目录
    """
    # 确保使用绝对路径
    output_dir = os.path.abspath(output_dir)
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    logger.info(f"保存处理后的数据集到: {processed_dir}")
    
    # 解包数据集元组
    train_dataset, val_dataset = datasets
    dataset_dict = {
        'train': train_dataset,
        'val': val_dataset,
        'test': val_dataset  # 使用验证集作为测试集
    }
    
    for split, dataset in dataset_dict.items():
        split_dir = os.path.join(processed_dir, split)
        # 确保每个分割的目录存在
        os.makedirs(split_dir, exist_ok=True)
        try:
            # 记录数据集信息
            logger.info(f"准备保存{split}数据集, 包含 {len(dataset)} 个样本")
            if len(dataset) > 0:
                logger.info(f"样本示例: {dataset[0]}")
            
            # 使用绝对路径保存数据集
            from datasets import Dataset as HFDataset
            # 转换为Hugging Face的Dataset格式
            hf_dataset = HFDataset.from_list([dict(ex) for ex in dataset])
            hf_dataset.save_to_disk(split_dir)
            logger.info(f"成功保存{split}数据集到: {split_dir}")
        except Exception as e:
            logger.error(f"保存{split}数据集时出错: {str(e)}")
            # 打印更详细的错误信息
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"所有处理后的数据集已保存到: {processed_dir}")
    return processed_dir

def upload_to_huggingface(
    processed_dir: str,
    repo_name: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload Amazon product hierarchy dataset"
) -> str:
    """
    将处理好的数据集上传到Hugging Face Hub
    
    Args:
        processed_dir: 处理后的数据集目录
        repo_name: Hugging Face Hub上的仓库名称，格式为'username/repo_name'
        token: Hugging Face API令牌，如果未提供则尝试使用环境变量或已缓存的令牌
        private: 是否创建私有仓库
        commit_message: 提交信息
        
    Returns:
        上传后的仓库URL
    """
    logger.info(f"准备上传数据集到Hugging Face Hub: {repo_name}")
    
    # 确保processed_dir是绝对路径
    processed_dir = os.path.abspath(processed_dir)
    
    # 检查数据集目录是否存在
    if not os.path.exists(processed_dir):
        logger.error(f"处理后的数据集目录不存在: {processed_dir}")
        return None
    
    # 登录Hugging Face
    try:
        if token:
            login(token=token)
            logger.info("已使用提供的令牌登录Hugging Face")
        else:
            # 尝试使用环境变量或已缓存的令牌登录
            login()
            logger.info("已使用环境变量或缓存的令牌登录Hugging Face")
    except Exception as e:
        logger.error(f"登录Hugging Face时出错: {str(e)}")
        logger.error("请确保已设置HUGGINGFACE_TOKEN环境变量或提供了有效的令牌")
        return None
    
    # 创建数据集文档内容
    readme_content = f"""
# Amazon Product Hierarchy Dataset

This dataset contains Amazon product hierarchical relationships for training hierarchy-aware transformers.

## Dataset Structure

The dataset contains three splits:
- Train: {len(os.listdir(os.path.join(processed_dir, 'train')))} examples
- Validation: {len(os.listdir(os.path.join(processed_dir, 'val')))} examples
- Test: {len(os.listdir(os.path.join(processed_dir, 'test')))} examples

Each example contains:
- `child`: Text description of a product
- `parent`: Text description of the product's category
- `negative`: Text description of a negative (non-parent) category

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_name}")

# Example usage
example = dataset['train'][0]
print(f"Child: {{example['child']}}")
print(f"Parent: {{example['parent']}}")
print(f"Negative: {{example['negative']}}")
```

## Dataset Creation

This dataset was created by processing Amazon Beauty product metadata and reviews, extracting product categories and their hierarchical relationships.
"""
    
    # 使用dataset库正确加载和上传数据集
    try:
        from datasets import load_from_disk, DatasetDict
        import tempfile
        
        logger.info("正在加载本地数据集...")
        
        # 逐个加载每个分割并创建DatasetDict
        datasets_dict = {}
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(processed_dir, split)
            if os.path.exists(split_dir):
                logger.info(f"加载{split}数据集...")
                datasets_dict[split] = load_from_disk(split_dir)
                logger.info(f"成功加载{split}数据集，包含{len(datasets_dict[split])}个样本")
            else:
                logger.warning(f"找不到{split}数据集目录: {split_dir}")
        
        if not datasets_dict:
            logger.error("没有成功加载任何数据集")
            return None
        
        # 将'val'重命名为'validation'以符合Hugging Face惯例
        if 'val' in datasets_dict and 'validation' not in datasets_dict:
            datasets_dict['validation'] = datasets_dict.pop('val')
            logger.info("将'val'重命名为'validation'以符合Hugging Face惯例")
        
        # 创建DatasetDict
        dataset_dict = DatasetDict(datasets_dict)
        logger.info(f"已创建DatasetDict，包含以下分割: {list(dataset_dict.keys())}")
        
        # 设置上传参数
        logger.info(f"开始上传数据集到: {repo_name}")
        dataset_dict.push_to_hub(
            repo_id=repo_name,
            token=token,
            private=private,
            commit_message=commit_message
        )
        
        # 单独上传README文件
        try:
            logger.info("正在创建并上传README文件...")
            api = HfApi()
            
            # 在本地创建临时README文件
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
                f.write(readme_content)
                readme_path = f.name
            
            # 上传README文件
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_name,
                repo_type="dataset",
                token=token,
                commit_message="Add dataset documentation"
            )
            
            # 删除临时文件
            os.unlink(readme_path)
            
            logger.info("README文件上传成功")
        except Exception as e:
            logger.warning(f"README文件上传失败: {str(e)}")
            logger.warning("数据集已上传，但可能缺少文档")
        
        repo_url = f"https://huggingface.co/datasets/{repo_name}"
        logger.info(f"数据集上传成功! 仓库URL: {repo_url}")
        return repo_url
    except Exception as e:
        logger.error(f"上传数据集时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_test():
    """运行测试"""
    try:
        # 设置数据目录，使用项目中已有的数据
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "datasets", "amazon_beauty_dataset")
        
        logger.info(f"使用数据目录: {data_dir}")
        
        # 检查必要文件是否存在
        entity_lexicon_path = os.path.join(data_dir, 'entity_lexicon.json')
        mixed_dir = os.path.join(data_dir, 'mixed')
        train_path = os.path.join(mixed_dir, 'train.jsonl')
        val_path = os.path.join(mixed_dir, 'val.jsonl')
        
        required_files = [entity_lexicon_path, train_path, val_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            logger.error("以下必要文件不存在:")
            for file in missing_files:
                logger.error(f"- {file}")
            return
            
        logger.info("所有必要文件都存在，开始处理数据...")
        
        # 先备份现有的processed目录（如果存在）
        processed_dir = os.path.join(data_dir, "processed")
        if os.path.exists(processed_dir):
            import shutil
            from datetime import datetime
            backup_dir = f"{processed_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"备份现有处理后的数据到: {backup_dir}")
            shutil.move(processed_dir, backup_dir)
        
        # 准备数据集
        logger.info("开始准备数据集...")
        datasets = prepare_amazon_hit_data_with_reviews(
            data_dir=data_dir,
            use_hard_negatives=True
        )
        
        if datasets is None:
            logger.error("数据集准备失败")
            return
            
        # 保存处理后的数据集
        processed_dir = save_processed_datasets(datasets, data_dir)
            
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
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
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
        elif sys.argv[1] == "upload" and len(sys.argv) >= 3:
            # 上传数据集到Hugging Face Hub
            repo_name = sys.argv[2]
            token = sys.argv[3] if len(sys.argv) >= 4 else None
            private = True if len(sys.argv) >= 5 and sys.argv[4].lower() == "private" else False
            
            logging.getLogger(__name__).setLevel(logging.INFO)
            
            logger.info("=" * 60)
            logger.info(f"开始上传数据集到Hugging Face Hub: {repo_name}")
            logger.info("=" * 60)
            
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "datasets", "amazon_beauty_dataset")
            processed_dir = os.path.join(data_dir, "processed")
            
            if not os.path.exists(processed_dir):
                logger.error(f"处理后的数据集目录不存在: {processed_dir}")
                logger.error("请先运行 'python -m src.data test' 生成处理后的数据集")
                sys.exit(1)
            
            repo_url = upload_to_huggingface(
                processed_dir=processed_dir,
                repo_name=repo_name,
                token=token,
                private=private
            )
            
            if repo_url:
                logger.info("=" * 60)
                logger.info(f"数据集已成功上传到: {repo_url}")
                logger.info("=" * 60)
            else:
                logger.error("=" * 60)
                logger.error("数据集上传失败")
                logger.error("=" * 60)
                sys.exit(1)
        else:
            print("可用命令:")
            print("  1. python -m src.data test - 测试数据集构建功能")
            print("  2. python -m src.data upload <repo_name> [token] [private] - 上传数据集到Hugging Face Hub")
            print("     例如: python -m src.data upload username/amazon-hierarchy-dataset")
    else:
        print("可用命令:")
        print("  1. python -m src.data test - 测试数据集构建功能")
        print("  2. python -m src.data upload <repo_name> [token] [private] - 上传数据集到Hugging Face Hub")
        print("     例如: python -m src.data upload username/amazon-hierarchy-dataset") 