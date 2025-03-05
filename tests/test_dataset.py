import json
import os
import sys
import logging
import random
from typing import List, Dict
from tqdm import tqdm
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.libs.tokenizer import AmazonDistilBertTokenizer
from src.data import AmazonBeautyMLMDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_mlm_samples(samples: List[str], tokenizer: AmazonDistilBertTokenizer, max_samples: int = 1000) -> Dict:
    """测试MLM样本的tokenization质量
    
    Args:
        samples: MLM样本文本列表
        tokenizer: 使用的tokenizer
        max_samples: 最大测试样本数
        
    Returns:
        Dict: 测试结果统计
    """
    # 如果样本数超过max_samples，随机选择max_samples个样本
    if len(samples) > max_samples:
        logger.info(f"随机选择 {max_samples} 个样本进行测试...")
        samples = random.sample(samples, max_samples)
    
    stats = {
        "total_samples": len(samples),
        "samples_with_user_tokens": 0,
        "samples_with_item_tokens": 0,
        "samples_with_category_tokens": 0,
        "avg_sequence_length": 0,
        "max_sequence_length": 0,
        "token_stats": {
            "user_tokens": set(),
            "item_tokens": set(),
            "category_tokens": set()
        }
    }
    
    total_length = 0
    
    for sample in tqdm(samples, desc="分析样本"):
        # 编码样本
        encoded = tokenizer.encode_plus(
            sample.strip(),
            return_tensors='pt',
            padding=False,
            truncation=True,  # 添加截断
            max_length=512    # 设置最大长度
        )
        
        # 计算序列长度
        seq_length = len(encoded['input_ids'][0])
        total_length += seq_length
        stats['max_sequence_length'] = max(stats['max_sequence_length'], seq_length)
        
        # 检查特殊token（直接从文本中提取，不需要解码）
        if '[user_' in sample:
            stats['samples_with_user_tokens'] += 1
            user_tokens = [t for t in sample.split() if t.startswith('[user_')]
            stats['token_stats']['user_tokens'].update(user_tokens)
            
        if '[item_' in sample:
            stats['samples_with_item_tokens'] += 1
            item_tokens = [t for t in sample.split() if t.startswith('[item_')]
            stats['token_stats']['item_tokens'].update(item_tokens)
            
        if '[category_' in sample:
            stats['samples_with_category_tokens'] += 1
            category_tokens = [t for t in sample.split() if t.startswith('[category_')]
            stats['token_stats']['category_tokens'].update(category_tokens)
    
    stats['avg_sequence_length'] = total_length / len(samples)
    
    # 转换set为长度
    stats['token_stats'] = {
        k: len(v) for k, v in stats['token_stats'].items()
    }
    
    return stats

def print_sample_analysis(sample: str, tokenizer: AmazonDistilBertTokenizer):
    """详细分析单个样本的tokenization结果"""
    print("\n" + "="*100)
    print("原始文本:", sample)
    
    # 编码
    encoded = tokenizer.encode_plus(
        sample.strip(),
        return_tensors='pt',
        padding=False,
        truncation=True,
        max_length=512
    )
    
    # 只对示例样本进行解码
    decoded = tokenizer.decode(encoded['input_ids'][0])
    print("\n解码后文本:", decoded)
    
    # 分析特殊token
    special_tokens = {
        'user': [t for t in sample.split() if t.startswith('[user_')],
        'item': [t for t in sample.split() if t.startswith('[item_')],
        'category': [t for t in sample.split() if t.startswith('[category_')]
    }
    
    print("\n特殊Token统计:")
    for token_type, tokens in special_tokens.items():
        if tokens:
            print(f"{token_type}_tokens ({len(tokens)}): {tokens}")
    
    print(f"\n序列长度: {len(encoded['input_ids'][0])}")
    print("="*100)

def analyze_text_lengths(samples: List[str], tokenizer: AmazonDistilBertTokenizer) -> Dict:
    """分析文本长度分布
    
    Args:
        samples: MLM样本文本列表
        tokenizer: 使用的tokenizer
        
    Returns:
        Dict: 长度分析统计
    """
    stats = {
        "total_samples": len(samples),
        "length_distribution": [],
        "truncated_samples": 0,
        "max_length": 0,
        "min_length": float('inf'),
        "avg_length": 0,
        "length_percentiles": {}
    }
    
    lengths = []
    truncated_examples = []
    
    for sample in tqdm(samples, desc="分析文本长度"):
        # 获取完整长度（不截断）
        encoded = tokenizer.encode_plus(
            sample.strip(),
            truncation=False,
            return_tensors='pt'
        )
        length = len(encoded['input_ids'][0])
        lengths.append(length)
        
        # 更新统计信息
        stats["max_length"] = max(stats["max_length"], length)
        stats["min_length"] = min(stats["min_length"], length)
        
        # 如果长度超过512（BERT的标准长度限制），记录为被截断的样本
        if length > 512:
            stats["truncated_samples"] += 1
            if len(truncated_examples) < 5:  # 保存前5个被截断的样本作为示例
                truncated_examples.append({
                    "text": sample[:100] + "...",  # 只显示前100个字符
                    "original_length": length
                })
    
    # 计算平均长度
    stats["avg_length"] = sum(lengths) / len(lengths)
    
    # 计算分位数
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        stats["length_percentiles"][f"p{p}"] = int(np.percentile(lengths, p))
    
    # 添加长度分布的直方图数据
    hist, bins = np.histogram(lengths, bins=50)
    stats["length_distribution"] = [{
        "bin_start": int(bins[i]),
        "bin_end": int(bins[i+1]),
        "count": int(hist[i])
    } for i in range(len(hist))]
    
    # 添加被截断的样本示例
    if truncated_examples:
        stats["truncated_examples"] = truncated_examples
    
    return stats

def main():
    """测试MLM数据集的主函数"""
    try:
        # 1. 加载tokenizer
        logger.info("加载tokenizer...")
        tokenizer = AmazonDistilBertTokenizer.from_pretrained('data/beauty_tokenizer')
        
        # 2. 读取MLM样本
        logger.info("读取MLM训练样本...")
        with open('data/mlm_data/train_mlm.txt', 'r', encoding='utf-8') as f:
            samples = f.readlines()[:100]  # 只分析前100个样本
        
        logger.info(f"读取了 {len(samples)} 个样本用于快速分析")
        
        # 3. 分析文本长度
        logger.info("分析文本长度分布...")
        length_stats = analyze_text_lengths(samples, tokenizer)
        
        # 4. 打印长度统计信息
        print("\n=== 文本长度统计 ===")
        print(f"样本总数: {length_stats['total_samples']:,}")
        print(f"被截断的样本数: {length_stats['truncated_samples']:,} ({length_stats['truncated_samples']/length_stats['total_samples']*100:.2f}%)")
        print(f"\n长度统计:")
        print(f"- 最短: {length_stats['min_length']} tokens")
        print(f"- 最长: {length_stats['max_length']} tokens")
        print(f"- 平均: {length_stats['avg_length']:.1f} tokens")
        
        print(f"\n分位数统计:")
        for p, value in length_stats['length_percentiles'].items():
            print(f"- {p}: {value} tokens")
            
        if length_stats.get('truncated_examples'):
            print(f"\n被截断样本示例:")
            for i, example in enumerate(length_stats['truncated_examples'], 1):
                print(f"\n{i}. 长度: {example['original_length']} tokens")
                print(f"   文本: {example['text']}")
        
        # 5. 继续其他测试...
        logger.info("分析样本统计信息...")
        stats = test_mlm_samples(samples, tokenizer, max_samples=50)
        
        # 4. 打印统计信息
        print("\n=== MLM数据集统计 ===")
        print(f"分析的样本数: {stats['total_samples']}")
        print(f"包含用户token的样本数: {stats['samples_with_user_tokens']}")
        print(f"包含物品token的样本数: {stats['samples_with_item_tokens']}")
        print(f"包含类别token的样本数: {stats['samples_with_category_tokens']}")
        print(f"\n平均序列长度: {stats['avg_sequence_length']:.2f}")
        print(f"最大序列长度: {stats['max_sequence_length']}")
        
        print("\n特殊Token统计:")
        print(f"独特用户token数量: {stats['token_stats']['user_tokens']}")
        print(f"独特物品token数量: {stats['token_stats']['item_tokens']}")
        print(f"独特类别token数量: {stats['token_stats']['category_tokens']}")
        
        # 5. 详细分析少量随机样本
        print("\n=== 随机样本详细分析 ===")
        # 只分析5个随机样本
        for i, sample in enumerate(random.sample(samples, min(5, len(samples))), 1):
            print(f"\n样本 {i}:")
            print_sample_analysis(sample, tokenizer)
            
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 