import torch
from transformers import DistilBertModel, DistilBertTokenizer
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model():
    """测试本地模型"""
    try:
        logger.info("加载本地tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained(
            './huggingface/models/distilbert-base-uncased',
            local_files_only=True
        )
        logger.info("Tokenizer加载成功！")
        
        logger.info("加载本地模型...")
        model = DistilBertModel.from_pretrained(
            './huggingface/models/distilbert-base-uncased',
            torch_dtype=torch.float16,
            local_files_only=True
        )
        logger.info("模型加载成功！")
        
        # 测试推理
        test_texts = [
            "Hello, world!",
            "This is a test sentence.",
            "Let's see if the model works."
        ]
        
        logger.info("\n开始测试推理...")
        for text in test_texts:
            logger.info(f"\n处理文本: {text}")
            inputs = tokenizer(text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 显示输出形状
            logger.info(f"输出张量形状: {outputs.last_hidden_state.shape}")
            logger.info(f"输出示例（前5个值）: {outputs.last_hidden_state[0][0][:5]}")
        
        logger.info("\n所有测试通过！")
        return True
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        return False

def main():
    logger.info("=== DistilBert本地模型测试 ===")
    
    if test_model():
        logger.info("\n恭喜！模型可以正常使用。")
    else:
        logger.error("\n模型测试失败。")
        logger.info("""
建议：
1. 检查模型文件是否完整
2. 确保所有依赖都已正确安装
3. 检查GPU内存是否足够（如果使用GPU）
""")

if __name__ == "__main__":
    main() 