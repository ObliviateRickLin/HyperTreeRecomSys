import os
import logging
from src.libs.tokenizer import AmazonDistilBertTokenizer
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_beauty_tokenizer(
    pretrained_model_name: str = "distilbert-base-uncased",
    data_dir: str = "data",
) -> AmazonDistilBertTokenizer:
    """
    初始化Beauty数据集的AmazonDistilBertTokenizer
    
    Args:
        pretrained_model_name: 预训练模型名称或路径
        data_dir: 数据目录路径
    
    Returns:
        AmazonDistilBertTokenizer: 初始化好的tokenizer
    """
    # 构建文件路径
    metadata_file = os.path.join(data_dir, "meta_Beauty_2014.json.gz")
    reviews_file = os.path.join(data_dir, "reviews_Beauty_5.json.gz")
    
    # 检查文件是否存在
    for file_path in [metadata_file, reviews_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")
    
    logger.info("=" * 50)
    logger.info("开始初始化Beauty数据集tokenizer...")
    logger.info(f"预训练模型: {pretrained_model_name}")
    logger.info(f"metadata文件: {metadata_file}")
    logger.info(f"reviews文件: {reviews_file}")
    logger.info("=" * 50)
    
    try:
        # 加载预训练模型的tokenizer
        tokenizer = AmazonDistilBertTokenizer(
            pretrained_model_name_or_path=pretrained_model_name,
            metadata_file=metadata_file,
            reviews_file=reviews_file
        )
        
        # 保存tokenizer
        output_dir = os.path.join(data_dir, "beauty_tokenizer")
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Tokenizer已保存到: {output_dir}")
        return tokenizer
        
    except Exception as e:
        logger.error(f"初始化tokenizer时出错: {str(e)}")
        raise

if __name__ == "__main__":
    init_beauty_tokenizer() 