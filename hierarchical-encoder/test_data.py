#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Amazon数据处理模块测试脚本

这个脚本测试Amazon数据处理模块的功能，包括创建模拟数据和测试数据处理。
"""

import os
import sys
import argparse
import json
import gzip
import random

# 添加源码目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 从src导入数据处理函数
try:
    from src.data import prepare_amazon_hit_data, test_amazon_data, AmazonTaxonomy
except ImportError:
    # 尝试直接导入
    sys.path.append(os.path.join(current_dir, 'src'))
    from data import prepare_amazon_hit_data, test_amazon_data, AmazonTaxonomy


def create_mock_meta_data(output_path, num_categories=50, num_products=200):
    """
    创建模拟的Amazon元数据文件
    
    Args:
        output_path: 输出文件路径
        num_categories: 类别数量
        num_products: 产品数量
    """
    # 定义类别层次结构
    main_categories = ["Electronics", "Books", "Clothing"]
    sub_categories = {
        "Electronics": ["Computers", "Audio", "Cameras", "Accessories", "TVs"],
        "Books": ["Fiction", "Non-fiction", "Comics", "Education", "Kids"],
        "Clothing": ["Men", "Women", "Kids", "Sportswear", "Accessories"]
    }
    
    detailed_categories = {
        "Computers": ["Laptops", "Desktops", "Tablets", "Monitors", "Printers"],
        "Audio": ["Headphones", "Speakers", "MP3 Players", "Microphones"],
        "Fiction": ["Fantasy", "Sci-Fi", "Mystery", "Romance", "Thriller"],
        "Non-fiction": ["History", "Science", "Biography", "Self-help", "Travel"],
        "Men": ["Shirts", "Pants", "Jackets", "Shoes", "Underwear"],
        "Women": ["Dresses", "Tops", "Skirts", "Shoes", "Accessories"]
    }
    
    # 生成产品品牌
    brands = [
        "Sony", "Samsung", "Apple", "Dell", "LG", 
        "Penguin Books", "Harper Collins", "Random House",
        "Nike", "Adidas", "Puma", "Levi's", "H&M"
    ]
    
    # 生成产品特性
    features = [
        "high quality and durable construction",
        "excellent performance for the price",
        "very popular among customers",
        "highly rated and reviewed",
        "award-winning design",
        "cutting-edge technology",
        "environmentally friendly",
        "stylish and elegant",
        "comfortable to use",
        "reliable and long-lasting"
    ]
    
    # 生成模拟数据
    meta_data = []
    
    # 为了确保所有类别都有产品，先确保每个详细类别至少有一个产品
    for main_cat in main_categories:
        for sub_cat in sub_categories[main_cat]:
            if sub_cat in detailed_categories:
                for detail_cat in detailed_categories[sub_cat]:
                    product = {
                        "asin": "P{0:05d}".format(len(meta_data)),
                        "title": "{0} {1}".format(random.choice(brands), detail_cat),
                        "brand": random.choice(brands),
                        "categories": [[main_cat, sub_cat, detail_cat]],
                        "description": random.choice(features),
                        "feature": [random.choice(features), random.choice(features)]
                    }
                    meta_data.append(product)
    
    # 生成剩余的随机产品
    remaining_products = max(0, num_products - len(meta_data))
    for i in range(remaining_products):
        main_cat = random.choice(main_categories)
        sub_cat = random.choice(sub_categories[main_cat])
        
        if sub_cat in detailed_categories:
            detail_cat = random.choice(detailed_categories[sub_cat])
            
            # 有些产品可能属于多个类别
            categories = [[main_cat, sub_cat, detail_cat]]
            if random.random() < 0.3:  # 30%的概率有第二个类别
                second_main = random.choice(main_categories)
                second_sub = random.choice(sub_categories[second_main])
                if second_sub in detailed_categories:
                    second_detail = random.choice(detailed_categories[second_sub])
                    categories.append([second_main, second_sub, second_detail])
            
            product = {
                "asin": "P{0:05d}".format(len(meta_data)),
                "title": "{0} {1}".format(random.choice(brands), detail_cat),
                "brand": random.choice(brands),
                "categories": categories,
                "description": random.choice(features),
                "feature": [random.choice(features), random.choice(features)]
            }
            meta_data.append(product)
    
    # 保存到gzip压缩的JSON文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        for product in meta_data:
            f.write(json.dumps(product) + '\n')
    
    print("创建了模拟元数据文件: {0}, 共{1}个产品".format(output_path, len(meta_data)))
    return output_path


def test_pipeline(output_dir="test_output", use_hard_negatives=False):
    """
    测试完整的数据处理流程
    
    Args:
        output_dir: 输出目录
        use_hard_negatives: 是否使用硬负样本
    """
    from transformers import AutoTokenizer
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1: 创建模拟数据
    meta_file = os.path.join(output_dir, "mock_meta.json.gz")
    create_mock_meta_data(meta_file, num_categories=20, num_products=100)
    
    # 步骤2: 加载tokenizer
    print("加载tokenizer: bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 步骤3: 准备数据
    print("准备训练数据...")
    datasets = prepare_amazon_hit_data(
        meta_file_path=meta_file,
        tokenizer=tokenizer,
        output_dir=output_dir,
        use_hard_negatives=use_hard_negatives
    )
    
    # 步骤4: 打印数据集信息
    print("\n=== 数据集信息 ===")
    print("训练集大小: {0}".format(len(datasets['train'])))
    print("验证集大小: {0}".format(len(datasets['eval'])))
    print("测试集大小: {0}".format(len(datasets['test'])))
    
    # 步骤5: 测试数据样本
    print("\n=== 训练样本示例 ===")
    if len(datasets['train']) > 0:
        sample = datasets['train'][0]
        print("样本键: {0}".format(list(sample.keys())))
        print("实体输入ID形状: {0}".format(sample['e_input_ids'].shape))
    
    print("\n测试完成!")
    return datasets


def test_amazon_taxonomy():
    """测试AmazonTaxonomy类"""
    # 创建分类法
    taxonomy = AmazonTaxonomy()
    
    # 添加节点
    taxonomy.add_node("Electronics", name="Electronics")
    taxonomy.add_node("Computers", name="Computers")
    taxonomy.add_node("Laptops", name="Laptops")
    taxonomy.add_node("Gaming_Laptops", name="Gaming Laptops")
    
    # 添加边
    taxonomy.add_edge("Electronics", "Computers")
    taxonomy.add_edge("Computers", "Laptops")
    taxonomy.add_edge("Laptops", "Gaming_Laptops")
    
    # 测试获取子节点
    children = taxonomy.get_children("Computers")
    print("Computers的子节点: {0}".format(children))
    assert "Laptops" in children
    
    # 测试获取父节点
    parents = taxonomy.get_parents("Laptops")
    print("Laptops的父节点: {0}".format(parents))
    assert "Computers" in parents
    
    # 测试获取所有祖先
    ancestors = taxonomy.get_parents("Gaming_Laptops", transitive=True)
    print("Gaming_Laptops的所有祖先: {0}".format(ancestors))
    assert "Computers" in ancestors
    assert "Electronics" in ancestors
    
    print("AmazonTaxonomy测试通过!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试Amazon数据处理")
    parser.add_argument("--output_dir", type=str, default="test_output", help="输出目录")
    parser.add_argument("--use_hard_negatives", action="store_true", help="是否使用硬负样本")
    parser.add_argument("--test_taxonomy", action="store_true", help="测试AmazonTaxonomy类")
    
    args = parser.parse_args()
    
    if args.test_taxonomy:
        test_amazon_taxonomy()
    else:
        test_pipeline(args.output_dir, args.use_hard_negatives) 