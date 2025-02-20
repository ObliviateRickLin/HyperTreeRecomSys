import os
import sys
import json
from pathlib import Path
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_hf_environment():
    """设置Hugging Face环境"""
    # 创建必要的目录
    base_dir = Path('./huggingface')
    base_dir.mkdir(exist_ok=True)
    for subdir in ['cache', 'models', 'tokens']:
        (base_dir / subdir).mkdir(exist_ok=True)
    
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HOME'] = str(base_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(base_dir / 'models')
    
    # 创建或更新配置文件
    config = {
        "default_endpoint": "https://hf-mirror.com",
        "cache_dir": str(base_dir / 'cache'),
        "model_cache_dir": str(base_dir / 'models'),
        "token_cache_dir": str(base_dir / 'tokens')
    }
    
    config_file = base_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Hugging Face 配置已设置在: {config_file}")

def test_connection(url, timeout=5):
    """测试连接"""
    try:
        response = requests.get(url, timeout=timeout)
        return True, response.status_code
    except Exception as e:
        return False, str(e)

def test_model_download():
    """测试模型下载"""
    models_to_try = [
        "bert-base-chinese",
        "clue/albert_chinese_tiny",
        "uer/chinese_roberta_L-2_H-128"
    ]
    
    for model_name in models_to_try:
        logger.info(f"\n尝试下载模型: {model_name}")
        try:
            # 只下载配置文件和词汇表
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=False,
                resume_download=True,
                use_auth_token=False
            )
            logger.info(f"成功下载 {model_name} 的tokenizer!")
            return True
        except Exception as e:
            logger.error(f"下载 {model_name} 失败: {str(e)}")
    
    return False

def main():
    logger.info("=== Hugging Face 连接测试 ===")
    
    # 设置环境
    setup_hf_environment()
    
    # 测试镜像连接
    mirrors = [
        "https://hf-mirror.com",
        "https://mirror.sjtu.edu.cn/hugging-face-models",
        "https://huggingface.tuna.tsinghua.edu.cn"
    ]
    
    logger.info("\n测试镜像站点连接：")
    working_mirrors = []
    for mirror in mirrors:
        success, result = test_connection(mirror)
        if success:
            logger.info(f"{mirror}: 连接成功 (状态码: {result})")
            working_mirrors.append(mirror)
        else:
            logger.error(f"{mirror}: 连接失败 ({result})")
    
    if not working_mirrors:
        logger.error("没有可用的镜像站点！")
        return
    
    # 使用第一个可用的镜像
    os.environ['HF_ENDPOINT'] = working_mirrors[0]
    logger.info(f"\n使用镜像: {working_mirrors[0]}")
    
    # 测试模型下载
    if test_model_download():
        logger.info("\n成功下载测试模型！")
    else:
        logger.error("\n所有模型下载尝试均失败。")
        logger.info("""
建议：
1. 检查网络连接和防火墙设置
2. 尝试使用代理服务器
3. 考虑手动下载模型文件
4. 如果问题持续，可以使用离线模式：
   - 从其他机器下载模型文件
   - 将文件放在正确的缓存目录中
   - 使用 local_files_only=True 参数
""")

if __name__ == "__main__":
    main() 