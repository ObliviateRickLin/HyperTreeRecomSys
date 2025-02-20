import logging
import os
from src.libs.tokenizer import AmazonDistilBertTokenizer
from src.data import AmazonBooksBertDataset
import torch
from torch.utils.data import DataLoader
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def save_sample_to_file(sample_text: str, file_path: str):
    """保存样本文本到文件"""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(sample_text + "\n" + "="*100 + "\n")

def main():
    # 1. 初始化tokenizer
    model_path = os.path.abspath("huggingface/models/distilbert-base-uncased")
    tokenizer = AmazonDistilBertTokenizer(
        pretrained_model_name_or_path=model_path,
        num_users=100000,    # 假设10万用户
        num_items=100000,    # 10万商品
        num_categories=1000  # 1000个分类
    )
    
    # 2. 创建数据集
    dataset = AmazonBooksBertDataset(
        tokenizer=tokenizer,
        metadata_file="amazon_books_processed/books_metadata.jsonl",
        reviews_file="amazon_books_processed/books_reviews.jsonl",
        max_length=512,          # 使用更长的序列长度
        mlm_probability=0.15,
        max_samples=10000,       # 限制样本数
        min_rating=4.0,          # 只选择4星以上的评论
        min_reviews=10,          # 至少有10条评论的书
        chunk_size=1000          # 每次处理1000条数据
    )
    
    # 3. 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    # 4. 保存一些样本用于检查
    logging.info("生成样本数据...")
    sample_file = "dataset_samples.txt"
    
    # 清空样本文件
    if os.path.exists(sample_file):
        os.remove(sample_file)
    
    # 获取10个样本
    for i in range(10):
        sample = dataset[i]
        # 获取原始文本
        review_data = dataset.reviews[i]
        meta = dataset.asin2meta[review_data["asin"]]
        text = dataset._build_natural_sentence(review_data, meta)
        
        # 保存样本
        save_sample_to_file(
            f"Sample {i+1}:\n{text}\n\n"
            f"Sequence length: {len(sample['input_ids'])}\n"
            f"Masked tokens: {(sample['labels'] != -100).sum().item()}\n",
            sample_file
        )
    
    # 5. 测试DataLoader
    logging.info("测试DataLoader...")
    batch = next(iter(dataloader))
    batch_ids, batch_attention, batch_labels = batch
    
    logging.info(f"Batch shape: {batch_ids.shape}")
    logging.info(f"Number of masked tokens: {(batch_labels != -100).sum().item()}")
    
    # 6. 输出数据集统计信息
    logging.info("\n=== 数据集统计 ===")
    logging.info(f"总样本数: {len(dataset)}")
    logging.info(f"独立书籍数: {len(dataset.asin2meta)}")
    logging.info(f"平均序列长度: {sum(len(dataset[i]['input_ids']) for i in range(min(100, len(dataset)))) / min(100, len(dataset)):.2f}")

if __name__ == "__main__":
    main() 