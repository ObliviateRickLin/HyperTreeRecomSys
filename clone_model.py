import os
import subprocess
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd):
    """运行shell命令并返回输出"""
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"命令失败: {e.stderr}"

def setup_directories():
    """设置必要的目录"""
    base_dir = Path('./huggingface/models')
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def clone_model():
    """克隆模型仓库"""
    model_name = "distilbert-base-uncased"
    base_dir = setup_directories()
    model_dir = base_dir / model_name
    
    if model_dir.exists():
        logger.info(f"模型目录已存在: {model_dir}")
        return True
    
    # 尝试从不同的镜像克隆
    mirrors = [
        f"https://hf-mirror.com/{model_name}",
        f"https://huggingface.co/{model_name}"
    ]
    
    for mirror in mirrors:
        logger.info(f"尝试从 {mirror} 克隆模型...")
        success, output = run_command(f"git clone {mirror} {model_dir}")
        
        if success:
            logger.info("克隆成功！")
            return True
        else:
            logger.warning(f"从 {mirror} 克隆失败: {output}")
    
    return False

def main():
    logger.info("=== 开始克隆DistilBert模型 ===")
    
    if clone_model():
        logger.info("\n模型克隆成功！")
        logger.info("现在您可以使用以下代码加载模型：")
        logger.info("""
from transformers import DistilBertModel, DistilBertTokenizer

model = DistilBertModel.from_pretrained('./huggingface/models/distilbert-base-uncased',
                                      local_files_only=True)
tokenizer = DistilBertTokenizer.from_pretrained('./huggingface/models/distilbert-base-uncased',
                                               local_files_only=True)
""")
    else:
        logger.error("\n模型克隆失败。")
        logger.info("""
建议：
1. 检查网络连接
2. 确保git已正确安装
3. 尝试手动下载模型文件
""")

if __name__ == "__main__":
    main() 