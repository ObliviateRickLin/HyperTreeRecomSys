import os
import logging
from libs.tokenizer import AmazonDistilBertTokenizer
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_tokenizer(
    pretrained_model_name: str = "/root/huggingface/models/distilbert-base-uncased",
    data_dir: str = "amazon_books_processed",
) -> AmazonDistilBertTokenizer:
    """
    初始化AmazonDistilBertTokenizer，使用本地模型
    
    Args:
        pretrained_model_name: 预训练模型路径
        data_dir: 数据目录路径
    
    Returns:
        AmazonDistilBertTokenizer: 初始化好的tokenizer
    """
    # 构建文件路径
    metadata_file = os.path.join(data_dir, "books_metadata.jsonl")
    reviews_file = os.path.join(data_dir, "books_reviews.jsonl")
    
    # 检查文件是否存在
    for file_path in [metadata_file, reviews_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")
    
    # 检查预训练模型路径
    if not os.path.exists(pretrained_model_name):
        raise FileNotFoundError(f"找不到预训练模型: {pretrained_model_name}")
    
    logger.info("=" * 50)
    logger.info("开始初始化tokenizer...")
    logger.info(f"预训练模型路径: {pretrained_model_name}")
    logger.info(f"metadata文件: {metadata_file}")
    logger.info(f"reviews文件: {reviews_file}")
    logger.info("=" * 50)
    
    try:
        logger.info("1. 加载预训练模型...")
        tokenizer = AmazonDistilBertTokenizer(
            pretrained_model_name_or_path=pretrained_model_name,
            metadata_file=metadata_file,
            reviews_file=reviews_file,
            local_files_only=True,
            use_fast=True
        )
        logger.info("Tokenizer初始化成功！")
        logger.info("=" * 50)
        return tokenizer
        
    except Exception as e:
        logger.error(f"初始化tokenizer时出错: {str(e)}")
        raise

if __name__ == "__main__":
    # 设置torch使用CPU，避免GPU内存问题
    torch.cuda.is_available = lambda: False
    
    # 尝试初始化tokenizer
    try:
        logger.info("开始初始化过程...")
        tokenizer = init_tokenizer()
        
        # 打印一些基本信息
        logger.info("Tokenizer信息:")
        logger.info("-" * 30)
        logger.info(f"词表大小: {len(tokenizer.base_tokenizer):,}")
        logger.info(f"用户token数量: {len(tokenizer.user_tokens):,}")
        logger.info(f"物品token数量: {len(tokenizer.item_tokens):,}")
        logger.info(f"类别token数量: {len(tokenizer.category_tokens):,}")
        logger.info("-" * 30)
        
        # 测试tokenizer的基本功能
        test_text = "这是一个测试文本 user_123 item_456 category_0"
        logger.info("测试tokenizer功能:")
        logger.info(f"输入文本: {test_text}")
        encoded = tokenizer.encode_plus(test_text)
        logger.info(f"编码结果: {encoded['input_ids']}")
        logger.info("=" * 50)
        logger.info("初始化完成！")
        
    except Exception as e:
        logger.error(f"运行失败: {str(e)}")
        raise