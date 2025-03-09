#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Amazon数据处理模块 - 层次编码器版本

此模块处理Amazon数据集，构建层次结构和训练数据。
使用HierarchyTransformers方法进行层次编码训练。
"""

import os
import json
import gzip
import random
import logging
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sys

# 添加external路径以引用HierarchyTransformers
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'external'))

from HierarchyTransformers.src.hierarchy_transformers.datasets.construct import HierarchyDatasetConstructor

logger = logging.getLogger(__name__)

class AmazonTaxonomy:
    """
    Amazon产品层次结构分类法
    
    用于与HierarchyTransformers兼容的简单分类法类
    """
    
    def __init__(self):
        """初始化空分类法"""
        self.nodes = set()  # 所有实体（产品和类别）
        self.edges = set()  # 边集合 (parent, child)
        self.node_attrs = {}  # 节点属性 {node_id: {attr: value}}
    
    def add_node(self, node_id, **attrs):
        """添加节点"""
        self.nodes.add(node_id)
        if node_id not in self.node_attrs:
            self.node_attrs[node_id] = {}
        self.node_attrs[node_id].update(attrs)
    
    def add_edge(self, parent, child):
        """添加边（parent -> child）"""
        self.edges.add((parent, child))
        
        # 确保节点存在
        self.nodes.add(parent)
        self.nodes.add(child)
        
        if parent not in self.node_attrs:
            self.node_attrs[parent] = {}
            
        if child not in self.node_attrs:
            self.node_attrs[child] = {}
    
    def get_children(self, node_id):
        """获取节点的所有子节点"""
        return [child for parent, child in self.edges if parent == node_id]
    
    def get_parents(self, node_id, transitive=False):
        """
        获取节点的所有父节点
        
        Args:
            node_id: 节点ID
            transitive: 是否包括间接父节点
        """
        direct_parents = [parent for parent, child in self.edges if child == node_id]
        
        if not transitive:
            return direct_parents
        
        # 计算传递闭包（所有祖先）
        all_parents = set(direct_parents)
        frontier = set(direct_parents)
        
        while frontier:
            next_frontier = set()
            for node in frontier:
                parents = [parent for parent, child in self.edges if child == node]
                next_frontier.update([p for p in parents if p not in all_parents])
                all_parents.update(parents)
            frontier = next_frontier
        
        return list(all_parents)
    
    def get_node_attributes(self, node_id):
        """获取节点属性"""
        return self.node_attrs.get(node_id, {})


class AmazonHierarchyDataset(Dataset):
    """
    Amazon层次结构数据集
    
    处理类别和产品的层次关系，构建(e, e+, e-)三元组训练数据
    """
    
    def __init__(self, examples, tokenizer, max_length=128):
        """
        初始化数据集
        
        Args:
            examples: 示例列表，每个示例包含 {child, parent, [random|hard]_negatives}
            tokenizer: 用于编码文本的tokenizer
            max_length: 最大序列长度
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        Args:
            idx: 索引
            
        Returns:
            encoded_dict: 编码后的字典，包含:
                - e_input_ids: 实体输入ID
                - e_attention_mask: 实体注意力掩码
                - e_pos_input_ids: 正样本输入ID
                - e_pos_attention_mask: 正样本注意力掩码
                - e_neg_input_ids: 负样本输入ID
                - e_neg_attention_mask: 负样本注意力掩码
        """
        example = self.examples[idx]
        
        # 获取文本
        child_text = example['child']
        parent_text = example['parent']
        negative_text = example['negative']
        
        # 编码实体文本
        child_encoded = self.tokenizer(
            child_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 编码正样本文本
        parent_encoded = self.tokenizer(
            parent_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 编码负样本文本
        negative_encoded = self.tokenizer(
            negative_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移除批次维度
        encoded_dict = {
            'e_input_ids': child_encoded['input_ids'].squeeze(0),
            'e_attention_mask': child_encoded['attention_mask'].squeeze(0),
            'e_pos_input_ids': parent_encoded['input_ids'].squeeze(0),
            'e_pos_attention_mask': parent_encoded['attention_mask'].squeeze(0),
            'e_neg_input_ids': negative_encoded['input_ids'].squeeze(0),
            'e_neg_attention_mask': negative_encoded['attention_mask'].squeeze(0),
        }
        
        return encoded_dict


def extract_amazon_hierarchy(meta_file_path):
    """
    从Amazon元数据中提取层次结构
    
    Args:
        meta_file_path: 元数据文件路径
        
    Returns:
        taxonomy: Amazon分类法对象
    """
    taxonomy = AmazonTaxonomy()
    categories_set = set()  # 所有类别集合
    products = set()  # 产品ID集合
    
    logger.info("从{0}中提取层次结构...".format(meta_file_path))
    
    # 读取元数据文件
    with gzip.open(meta_file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="提取层次结构"):
            # 解析JSON数据
            try:
                item = json.loads(line.strip())
                
                # 检查是否有类别信息
                if 'categories' in item and item['categories']:
                    # 获取产品ID和标题
                    product_id = item['asin']
                    product_title = item.get('title', "Product {0}".format(product_id))
                    
                    # 添加产品节点
                    taxonomy.add_node(product_id, name=product_title)
                    products.add(product_id)
                    
                    # 处理每条类别路径
                    for path in item['categories']:
                        if not path:
                            continue
                            
                        # 添加所有类别到集合
                        for category in path:
                            taxonomy.add_node(category, name=category)
                            categories_set.add(category)
                        
                        # 构建层次关系：产品 -> 最细粒度类别
                        # 最细粒度类别是路径中的最后一个类别
                        finest_category = path[-1]
                        taxonomy.add_edge(finest_category, product_id)
                        
                        # 构建类别之间的层次关系
                        for i in range(len(path)-1):
                            parent = path[i]
                            child = path[i+1]
                            taxonomy.add_edge(parent, child)
                            
            except json.JSONDecodeError:
                continue
    
    logger.info("层次结构提取完成: {0}个实体, {1}个层次关系".format(len(taxonomy.nodes), len(taxonomy.edges)))
    return taxonomy


def build_entity_descriptions(meta_file_path, taxonomy):
    """
    构建实体描述
    
    Args:
        meta_file_path: 元数据文件路径
        taxonomy: 分类法对象
        
    Returns:
        None (直接修改taxonomy的node_attrs)
    """
    logger.info("构建实体描述...")
    
    # 类别已经在提取层次结构时添加了name属性
    
    # 为产品添加更详细的描述
    with gzip.open(meta_file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="构建实体描述"):
            try:
                item = json.loads(line.strip())
                
                # 检查是否为所需产品
                if 'asin' in item and item['asin'] in taxonomy.nodes:
                    product_id = item['asin']
                    
                    # 提取产品相关信息
                    title = item.get('title', '')
                    brand = item.get('brand', '')
                    categories = []
                    
                    # 获取产品所属类别
                    if 'categories' in item and item['categories']:
                        for path in item['categories']:
                            if path:
                                categories.append(path[-1])  # 最细粒度类别
                    
                    # 选择第一个类别（如果有）
                    category = categories[0] if categories else ''
                    
                    # 提取产品主要特性（如果有）
                    feature = ''
                    if 'description' in item and item['description']:
                        feature = item['description'][:100]  # 取描述的前100个字符
                    elif 'feature' in item and item['feature']:
                        if isinstance(item['feature'], list) and len(item['feature']) > 0:
                            feature = item['feature'][0]  # 取第一个特性
                        elif isinstance(item['feature'], str):
                            feature = item['feature']
                    
                    # 构建产品描述文本
                    # 格式: "{标题}: A {类别} by {品牌} that {特性}."
                    description = "{0}".format(title)
                    
                    if category:
                        description += ": A {0}".format(category)
                        
                    if brand:
                        description += " by {0}".format(brand)
                        
                    if feature:
                        description += " that {0}".format(feature)
                    
                    # 更新节点属性
                    taxonomy.node_attrs[product_id]['description'] = description
                    
            except json.JSONDecodeError:
                continue
    
    # 确保所有节点都有描述
    for node in taxonomy.nodes:
        if 'description' not in taxonomy.node_attrs[node]:
            taxonomy.node_attrs[node]['description'] = taxonomy.node_attrs[node].get('name', "Entity {0}".format(node))
    
    logger.info("描述构建完成")


def prepare_amazon_hit_data(meta_file_path, tokenizer, output_dir=None, use_hard_negatives=False, 
                          test_size=0.1, val_size=0.1, random_seed=42, num_negatives=10):
    """
    准备Amazon层次编码器训练数据
    
    Args:
        meta_file_path: 元数据文件路径
        tokenizer: 用于编码文本的tokenizer
        output_dir: 输出目录，如果提供则保存处理后的数据
        use_hard_negatives: 是否使用硬负样本
        test_size: 测试集比例
        val_size: 验证集比例
        random_seed: 随机种子
        num_negatives: 每个正样本对应的负样本数量
        
    Returns:
        datasets: 数据集字典 {'train': dataset, 'eval': dataset, 'test': dataset}
    """
    logger.info("准备Amazon层次编码器训练数据...")
    
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 1. 提取层次结构
    taxonomy = extract_amazon_hierarchy(meta_file_path)
    
    # 2. 构建实体描述
    build_entity_descriptions(meta_file_path, taxonomy)
    
    # 3. 使用HierarchyTransformers的数据集构造器
    constructor = HierarchyDatasetConstructor(taxonomy)
    
    # 如果提供了输出目录，则保存数据
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        constructor.construct(output_dir, num_negative=num_negatives, eval_size=test_size+val_size)
        
        # 确定数据任务类型
        task_type = "mixed"  # "multi"用于多跳推理，"mixed"用于混合跳预测
        
        # 加载保存的数据文件
        train_file = os.path.join(output_dir, task_type, "train.jsonl")
        val_file = os.path.join(output_dir, task_type, "val.jsonl")
        test_file = os.path.join(output_dir, task_type, "test.jsonl")
        
        # 加载数据
        with open(train_file, 'r') as f:
            train_examples = [json.loads(line) for line in f]
        
        with open(val_file, 'r') as f:
            val_examples = [json.loads(line) for line in f]
            
        with open(test_file, 'r') as f:
            test_examples = [json.loads(line) for line in f]
    else:
        # 如果没有提供输出目录，则直接构建数据
        # 获取所有边
        all_edges = [(child, parent) for parent, child in taxonomy.edges]
        
        # 构建训练数据
        all_examples = []
        for child, parent in tqdm(all_edges, desc="构建样本"):
            all_examples.append(constructor.construct_example(child, parent, num_negatives))
        
        # 拆分数据
        train_val_test_split = int(len(all_examples) * (1 - test_size - val_size))
        val_test_split = int(len(all_examples) * (1 - test_size))
        
        all_examples_shuffled = all_examples.copy()
        random.shuffle(all_examples_shuffled)
        
        train_examples = all_examples_shuffled[:train_val_test_split]
        val_examples = all_examples_shuffled[train_val_test_split:val_test_split]
        test_examples = all_examples_shuffled[val_test_split:]
    
    # 将HierarchyTransformers格式转换为我们的数据集格式
    negative_type = "hard_negatives" if use_hard_negatives else "random_negatives"
    
    # 展开样本
    train_triplets = []
    for example in train_examples:
        child = taxonomy.get_node_attributes(example["child"]).get('description', example["child"])
        parent = taxonomy.get_node_attributes(example["parent"]).get('description', example["parent"])
        
        for neg in example[negative_type]:
            negative = taxonomy.get_node_attributes(neg).get('description', neg)
            train_triplets.append({"child": child, "parent": parent, "negative": negative})
    
    val_triplets = []
    for example in val_examples:
        child = taxonomy.get_node_attributes(example["child"]).get('description', example["child"])
        parent = taxonomy.get_node_attributes(example["parent"]).get('description', example["parent"])
        
        for neg in example[negative_type]:
            negative = taxonomy.get_node_attributes(neg).get('description', neg)
            val_triplets.append({"child": child, "parent": parent, "negative": negative})
    
    test_triplets = []
    for example in test_examples:
        child = taxonomy.get_node_attributes(example["child"]).get('description', example["child"])
        parent = taxonomy.get_node_attributes(example["parent"]).get('description', example["parent"])
        
        for neg in example[negative_type]:
            negative = taxonomy.get_node_attributes(neg).get('description', neg)
            test_triplets.append({"child": child, "parent": parent, "negative": negative})
    
    # 创建数据集
    train_dataset = AmazonHierarchyDataset(train_triplets, tokenizer)
    val_dataset = AmazonHierarchyDataset(val_triplets, tokenizer)
    test_dataset = AmazonHierarchyDataset(test_triplets, tokenizer)
    
    # 准备数据
    datasets = {
        'train': train_dataset,
        'eval': val_dataset,
        'test': test_dataset
    }
    
    logger.info("数据准备完成: {0}训练样本, {1}验证样本, {2}测试样本".format(
        len(train_dataset), len(val_dataset), len(test_dataset)))
    
    return datasets


# 简单测试函数
def test_amazon_data(meta_file_path, output_dir=None):
    """
    测试Amazon数据处理功能
    
    Args:
        meta_file_path: 元数据文件路径
        output_dir: 输出目录
    """
    from transformers import AutoTokenizer
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 准备数据
    datasets = prepare_amazon_hit_data(meta_file_path, tokenizer, output_dir)
    
    # 打印数据集信息
    print("\n=== 数据集信息 ===")
    print("训练集大小: {0}".format(len(datasets['train'])))
    print("验证集大小: {0}".format(len(datasets['eval'])))
    print("测试集大小: {0}".format(len(datasets['test'])))
    
    # 测试数据样本
    if len(datasets['train']) > 0:
        print("\n=== 训练样本示例 ===")
        sample = datasets['train'][0]
        print("样本键: {0}".format(list(sample.keys())))
        print("实体输入ID形状: {0}".format(sample['e_input_ids'].shape))
    
    return datasets


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试Amazon数据处理")
    parser.add_argument("--meta_file", type=str, required=True, help="元数据文件路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    
    args = parser.parse_args()
    test_amazon_data(args.meta_file, args.output_dir) 