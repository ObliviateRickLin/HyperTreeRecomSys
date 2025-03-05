import os
import logging
from src.libs.tokenizer import AmazonDistilBertTokenizer
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # 强制更新日志配置
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
    
    print("=" * 50)  # 使用print确保一定会输出
    print("开始初始化Beauty数据集tokenizer...")
    print(f"预训练模型: {pretrained_model_name}")
    print(f"metadata文件: {metadata_file}")
    print(f"reviews文件: {reviews_file}")
    print("=" * 50)
    
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
        
        print(f"Tokenizer已保存到: {output_dir}")
        return tokenizer
        
    except Exception as e:
        print(f"初始化tokenizer时出错: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("开始初始化过程...")
        tokenizer = init_beauty_tokenizer()
        
        # 打印token统计信息
        print("\nTokenizer信息:")
        print("-" * 30)
        print(f"词表大小: {len(tokenizer.base_tokenizer):,}")
        print(f"用户token数量: {len(tokenizer.user_tokens):,}")
        print(f"物品token数量: {len(tokenizer.item_tokens):,}")
        print(f"类别token数量: {len(tokenizer.category_tokens):,}")
        print("-" * 30)
        
        # 测试tokenizer的基本功能
        test_text = "这是一个测试文本 user_123 item_456 category_Beauty"
        print("\n测试tokenizer功能:")
        print(f"输入文本: {test_text}")
        encoded = tokenizer.encode_plus(test_text)
        print(f"编码结果: {encoded['input_ids']}")
        print("=" * 50)
        print("初始化完成！")
        
    except Exception as e:
        print(f"运行失败: {str(e)}")
        import traceback
        print(traceback.format_exc())  # 打印完整的错误堆栈
        raise 