import json
import os
import sys
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.libs.tokenizer import AmazonDistilBertTokenizer
from src.data import AmazonBooksBertDataset

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def print_text_sample(text):
    print("\n=== 生成的自然语言文本 ===")
    print(text)
    print("=" * 50)

def main():
    # 1. 创建自定义tokenizer
    model_path = "huggingface/models/distilbert-base-uncased"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径 {model_path} 不存在！")
        
    tokenizer = AmazonDistilBertTokenizer(
        pretrained_model_name_or_path=model_path,
        metadata_file="example_metadata.jsonl",
        reviews_file="example_reviews.jsonl"
    )
    
    # 2. 创建数据集
    dataset = AmazonBooksBertDataset(
        tokenizer=tokenizer,
        metadata_file="example_metadata.jsonl",
        reviews_file="example_reviews.jsonl",
        max_length=256,
        mlm_probability=0.15
    )
    
    # 3. 获取一个样本并打印原始文本
    if len(dataset.reviews) == 0:
        print("错误：数据集为空！")
        print(f"metadata数量: {len(dataset.asin2meta)}")
        print(f"reviews数量: {len(dataset.reviews)}")
        return
        
    sample = dataset[0]
    
    # 先打印未mask的原始文本
    review_data = dataset.reviews[0]
    asin = review_data.get("asin") or review_data.get("parent_asin")
    if not asin or asin not in dataset.asin2meta:
        print(f"错误：找不到对应的metadata！asin={asin}")
        print(f"可用的asin: {list(dataset.asin2meta.keys())}")
        return
        
    text_str = dataset._build_natural_sentence(review_data, dataset.asin2meta[asin])
    print_text_sample(text_str)
    
    # 4. 检查特殊token的分词结果
    tokens = tokenizer.tokenize_with_special_ids(text_str)
    special_tokens = [token for token in tokens if any(x in token for x in ['user_', 'item_', 'category_'])]
    print("\n=== 特殊Token ===")
    print(special_tokens)
    
    # 5. 解码masked文本
    masked_text = tokenizer.decode(sample["input_ids"])
    original_text = tokenizer.decode(sample["labels"][sample["labels"] != -100])
    
    print("\n=== 数据样本信息 ===")
    print(f"输入序列长度: {len(sample['input_ids'])}")
    print(f"注意力掩码长度: {len(sample['attention_mask'])}")
    print(f"标签序列长度: {len(sample['labels'])}")
    
    print("\n=== Masked文本 ===")
    print(masked_text)
    
    print("\n=== 被Mask的原始标记 ===")
    print(original_text)

if __name__ == "__main__":
    main() 