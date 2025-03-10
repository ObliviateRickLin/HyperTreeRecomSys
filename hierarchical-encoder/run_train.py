#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run training script for the hierarchical encoder model.
"""

import os
import sys
import logging
import argparse
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments and pass them to the training script."""
    parser = argparse.ArgumentParser(description="Run training for hierarchical encoder model")
    
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, default="datasets/amazon_beauty_dataset/processed", 
                        help="Directory containing the processed dataset files")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L12-v2",
                        help="Pre-trained model name or path")
    parser.add_argument("--output_dir", type=str, default="output/amazon-hierarchy",
                        help="Directory to save the model and results")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=20, 
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=128,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Loss function arguments
    parser.add_argument("--clustering_loss_weight", type=float, default=1.0,
                        help="Weight for clustering loss")
    parser.add_argument("--clustering_loss_margin", type=float, default=3.0,
                        help="Margin for clustering loss")
    parser.add_argument("--centripetal_loss_weight", type=float, default=1.0,
                        help="Weight for centripetal loss")
    parser.add_argument("--centripetal_loss_margin", type=float, default=0.5,
                        help="Margin for centripetal loss")
    
    args = parser.parse_args()
    return args

def main():
    """Main function to run training."""
    args = parse_args()
    
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建训练脚本的绝对路径
    script_path = os.path.join(current_dir, "src", "train.py")
    
    # 构建命令行参数
    cmd_args = [sys.executable, script_path]  # 使用当前Python解释器
    
    # 添加所有参数
    for arg_name, arg_value in vars(args).items():
        cmd_args.append(f"--{arg_name}={arg_value}")
    
    # 记录完整命令
    cmd_str = " ".join(cmd_args)
    logger.info(f"运行命令: {cmd_str}")
    
    # 使用subprocess运行命令（直接输出到控制台，不捕获）
    try:
        process = subprocess.run(
            cmd_args,
            check=True,
            # 不再捕获输出，让其直接显示在控制台
        )
        logger.info("训练完成")
    except subprocess.CalledProcessError as e:
        logger.error(f"训练失败，错误代码: {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main() 