#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试数据集加载和格式
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志 - 设置为DEBUG级别以获取更多信息
logging.basicConfig(
    level=logging.DEBUG,  # 改为DEBUG级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 确保日志输出到标准输出
    ]
)
logger = logging.getLogger(__name__)

# 强制刷新日志
print("测试脚本开始执行...", flush=True)
logger.debug("日志系统初始化完成")

from src.data import load_processed_datasets

def print_full_sample(example):
    """详细打印一个完整的样本内容"""
    print("\n" + "=" * 50, flush=True)
    print("完整样本内容:", flush=True)
    print("=" * 50, flush=True)
    
    # 打印基本文本字段
    text_fields = ['child', 'parent', 'negative']
    for field in text_fields:
        if field in example:
            print(f"\n{field}:", flush=True)
            print("-" * 30, flush=True)
            print(example[field], flush=True)
            print("-" * 30, flush=True)
    
    # 打印编码后的字段
    encoded_fields = ['input_ids', 'token_type_ids', 'attention_mask']
    for field in encoded_fields:
        if field in example:
            print(f"\n{field}:", flush=True)
            print("-" * 30, flush=True)
            # 如果是列表的列表，分别打印每个部分
            if isinstance(example[field], list) and all(isinstance(item, list) for item in example[field]):
                for i, item in enumerate(example[field]):
                    print(f"序列 {i+1}: {item}", flush=True)
            else:
                print(example[field], flush=True)
            print("-" * 30, flush=True)
    
    # 打印其他字段
    other_fields = [k for k in example.keys() if k not in text_fields and k not in encoded_fields]
    for field in other_fields:
        print(f"\n{field}:", flush=True)
        print("-" * 30, flush=True)
        print(example[field], flush=True)
        print("-" * 30, flush=True)
    
    print("=" * 50, flush=True)

def test_load_dataset():
    """测试加载处理后的数据集并检查格式"""
    print("开始执行test_load_dataset函数...", flush=True)
    
    # 获取处理后的数据集目录
    processed_dir = os.path.join(project_root, "datasets", "amazon_beauty_dataset", "processed")
    print(f"处理后的数据集目录: {processed_dir}", flush=True)
    
    # 检查目录是否存在
    if not os.path.exists(processed_dir):
        print(f"错误: 处理后的数据集目录不存在: {processed_dir}", flush=True)
        logger.error(f"处理后的数据集目录不存在: {processed_dir}")
        logger.error("请先运行数据处理脚本生成处理后的数据集")
        return
    
    # 检查子目录
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(processed_dir, split)
        if os.path.exists(split_dir):
            print(f"找到{split}数据集目录: {split_dir}", flush=True)
            # 列出目录内容
            files = os.listdir(split_dir)
            print(f"{split}目录内容: {files}", flush=True)
        else:
            print(f"警告: 找不到{split}数据集目录: {split_dir}", flush=True)
    
    logger.info(f"从处理后的目录加载数据集: {processed_dir}")
    print(f"尝试加载数据集...", flush=True)
    
    # 加载处理后的数据集
    try:
        print("调用load_processed_datasets函数...", flush=True)
        datasets = load_processed_datasets(processed_dir)
        print(f"load_processed_datasets返回结果: {datasets is not None}", flush=True)
        
        if datasets is None:
            print("错误: 处理后的数据集加载失败", flush=True)
            logger.error("处理后的数据集加载失败")
            return
            
        # 显示数据集基本信息
        print("=" * 50, flush=True)
        print("数据集加载成功", flush=True)
        for split, dataset in datasets.items():
            print(f"{split}数据集: {len(dataset)} 样本", flush=True)
        print("=" * 50, flush=True)
        
        # 检查数据格式
        print("\n数据样本格式:", flush=True)
        if 'train' in datasets and len(datasets['train']) > 0:
            example = datasets['train'][0]
            print("\n示例数据字段:", flush=True)
            for key, value in example.items():
                if isinstance(value, (list, dict)):
                    print(f"- {key}: {type(value).__name__} [长度: {len(value)}]", flush=True)
                elif hasattr(value, 'shape'):
                    print(f"- {key}: {type(value).__name__} [shape: {value.shape}]", flush=True)
                else:
                    # 对于文本内容，只显示前100个字符
                    display_value = str(value)[:100] + "..." if len(str(value)) > 100 else value
                    print(f"- {key}: {display_value}", flush=True)
            
            # 打印完整样本内容
            print_full_sample(example)
            
            # 再打印几个随机样本
            import random
            if len(datasets['train']) > 10:
                random_idx = random.randint(1, min(100, len(datasets['train'])-1))
                print(f"\n随机样本 (索引 {random_idx}):", flush=True)
                print_full_sample(datasets['train'][random_idx])
    except Exception as e:
        print(f"测试过程出错: {str(e)}", flush=True)
        logger.error(f"测试过程出错: {str(e)}")
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str, flush=True)
        logger.error(traceback_str)

if __name__ == "__main__":
    print("脚本主函数开始执行...", flush=True)
    logger.info("开始测试数据集加载和格式...")
    test_load_dataset()
    logger.info("测试完成")
    print("脚本执行完毕", flush=True) 