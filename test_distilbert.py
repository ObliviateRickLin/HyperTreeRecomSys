import os
import torch
from transformers import DistilBertModel, DistilBertTokenizer
import logging
from huggingface_hub import snapshot_download, HfApi
import requests
import socket

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_running_in_docker():
    """检查是否在Docker容器中运行"""
    try:
        with open('/proc/1/cgroup', 'r') as f:
            return any('docker' in line for line in f)
    except:
        return False

def setup_environment():
    """设置环境变量"""
    # 基本的HuggingFace设置
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HOME'] = './huggingface'
    os.environ['TRANSFORMERS_CACHE'] = './huggingface/models'
    os.environ['HF_CACHE_DIR'] = './huggingface/cache'
    os.environ['HF_HUB_CACHE'] = './huggingface/hub'
    os.environ['HF_ASSETS_CACHE'] = './huggingface/assets'
    
    # 检查是否在Docker中运行
    in_docker = is_running_in_docker()
    logger.info(f"运行环境: {'Docker容器内' if in_docker else '主机系统'}")
    
    if in_docker:
        # 在Docker中，尝试获取主机的代理设置
        host_ip = socket.gethostbyname(socket.gethostname())
        logger.info(f"容器IP: {host_ip}")
        
        # 设置NO_PROXY
        os.environ['NO_PROXY'] = f'localhost,127.0.0.1,{host_ip}'
        
        # 检查现有的代理设置
        current_http_proxy = os.environ.get('HTTP_PROXY', '')
        current_https_proxy = os.environ.get('HTTPS_PROXY', '')
        logger.info(f"当前HTTP代理: {current_http_proxy}")
        logger.info(f"当前HTTPS代理: {current_https_proxy}")
    
    # 确保所有缓存目录存在
    for path in ['./huggingface/models', './huggingface/cache', 
                 './huggingface/hub', './huggingface/assets']:
        os.makedirs(path, exist_ok=True)
    
    logger.info("环境变量和缓存目录已设置")
    
    # 测试网络连接
    try:
        # 测试镜像站点
        logger.info("测试镜像站点连接...")
        response = requests.get('https://hf-mirror.com', timeout=5)
        logger.info(f"镜像站点连接测试: {response.status_code}")
        
        # 测试原始站点
        logger.info("测试原始站点连接...")
        response = requests.get('https://huggingface.co', timeout=5)
        logger.info(f"原始站点连接测试: {response.status_code}")
    except Exception as e:
        logger.warning(f"连接测试失败: {str(e)}")
        logger.info("尝试不使用代理直接连接...")
        try:
            # 临时移除代理设置
            original_http_proxy = os.environ.pop('HTTP_PROXY', None)
            original_https_proxy = os.environ.pop('HTTPS_PROXY', None)
            
            response = requests.get('https://hf-mirror.com', timeout=5)
            logger.info(f"直接连接测试: {response.status_code}")
            
            # 恢复代理设置
            if original_http_proxy:
                os.environ['HTTP_PROXY'] = original_http_proxy
            if original_https_proxy:
                os.environ['HTTPS_PROXY'] = original_https_proxy
        except Exception as e2:
            logger.warning(f"直接连接也失败: {str(e2)}")

def download_model():
    """下载并加载模型"""
    try:
        logger.info("开始下载DistilBert模型...")
        model_id = "distilbert-base-uncased"
        
        # 配置下载参数
        download_kwargs = {
            'cache_dir': './huggingface/models',
            'local_files_only': False,
            'mirror': 'https://hf-mirror.com',
            'force_download': True,
            'resume_download': True,
            'trust_remote_code': True,
            'use_auth_token': False
        }
        
        # 尝试下载
        logger.info("下载tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained(
            model_id,
            **download_kwargs
        )
        logger.info("Tokenizer下载成功！")
        
        logger.info("下载模型...")
        model = DistilBertModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            **download_kwargs
        )
        logger.info("模型下载成功！")
        
        # 测试模型
        logger.info("测试模型...")
        test_input = tokenizer("Hello, world!", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**test_input)
        logger.info("模型测试成功！")
        
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    logger.info("=== DistilBert模型下载测试 ===")
    
    # 设置环境
    setup_environment()
    
    # 下载并测试模型
    success, error = download_model()
    
    if success:
        logger.info("\n恭喜！模型已成功下载并可以使用。")
        logger.info("模型文件保存在 ./huggingface/models 目录下")
    else:
        logger.error(f"\n模型下载失败: {error}")
        # 显示更详细的错误信息
        logger.info("\n当前环境变量设置：")
        for var in ['HF_ENDPOINT', 'HF_HOME', 'TRANSFORMERS_CACHE', 
                   'HF_CACHE_DIR', 'HF_HUB_CACHE', 'HF_ASSETS_CACHE',
                   'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY']:
            logger.info(f"{var}: {os.environ.get(var, '未设置')}")

if __name__ == "__main__":
    main() 