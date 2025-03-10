#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据加载模块 - 专门用于加载已处理好的数据集
"""

import os
import sys
import logging
from datasets import load_from_disk

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def load_processed_datasets(processed_dir):
    """
    从磁盘加载已处理的Dataset对象
    
    Args:
        processed_dir: 处理后的数据集目录，包含train, val, test子目录
        
    Returns:
        包含train, val, test的数据集字典
    """
    print(f"从磁盘加载处理后的数据集: {processed_dir}", flush=True)
    logger.info(f"从磁盘加载处理后的数据集: {processed_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(processed_dir):
        print(f"错误: 处理后的数据集目录不存在: {processed_dir}", flush=True)
        logger.error(f"处理后的数据集目录不存在: {processed_dir}")
        return None
    
    datasets = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(processed_dir, split)
        print(f"检查{split}数据集目录: {split_dir}", flush=True)
        
        if os.path.exists(split_dir):
            print(f"找到{split}数据集目录，尝试加载...", flush=True)
            try:
                # 列出目录内容
                files = os.listdir(split_dir)
                print(f"{split}目录内容: {files}", flush=True)
                
                # 尝试加载数据集
                print(f"开始加载{split}数据集...", flush=True)
                datasets[split] = load_from_disk(split_dir)
                print(f"成功加载{split}数据集，包含 {len(datasets[split])} 个样本", flush=True)
                logger.info(f"成功加载{split}数据集，包含 {len(datasets[split])} 个样本")
            except Exception as e:
                print(f"加载{split}数据集时出错: {str(e)}", flush=True)
                logger.error(f"加载{split}数据集时出错: {str(e)}")
                import traceback
                traceback_str = traceback.format_exc()
                print(traceback_str, flush=True)
                logger.error(traceback_str)
        else:
            print(f"警告: 找不到{split}数据集目录: {split_dir}", flush=True)
            logger.warning(f"找不到{split}数据集目录: {split_dir}")
    
    if not datasets:
        print("错误: 没有成功加载任何数据集", flush=True)
        logger.error("没有成功加载任何数据集")
        return None
        
    print(f"所有数据集加载完成: {', '.join([f'{k}={len(v)}' for k, v in datasets.items()])}", flush=True)
    logger.info(f"所有数据集加载完成")
    return datasets 