"""
简化版MLM模型训练测试，使用最小化的配置进行测试。
"""

import os
import sys
import tempfile
import shutil
import torch
from tqdm import tqdm
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("导入依赖...")
try:
    from src.mlm_model import SimpleMLMDataset, set_seed
    from src.libs.tokenizer import AmazonDistilBertTokenizer
    from src.libs.model import AmazonDistilBertBaseModel, AmazonDistilBertForMLM
    from transformers import DistilBertConfig, DistilBertModel
    print("依赖导入成功")
except Exception as e:
    print(f"导入依赖失败: {e}")
    traceback.print_exc()
    sys.exit(1)

def main():
    print("\n" + "="*70)
    print("MLM训练简单测试")
    print("="*70)
    
    # 设置随机种子
    print("设置随机种子")
    try:
        set_seed(42)
    except Exception as e:
        print(f"设置随机种子失败: {e}")
        traceback.print_exc()
        return
    
    # 创建临时目录
    print("创建临时目录")
    try:
        temp_dir = tempfile.mkdtemp()
        print(f"临时目录: {temp_dir}")
    except Exception as e:
        print(f"创建临时目录失败: {e}")
        traceback.print_exc()
        return
    
    try:
        # 加载tokenizer
        print("加载tokenizer")
        tokenizer_path = "data/beauty_tokenizer"
        print(f"Tokenizer路径: {tokenizer_path}")
        if not os.path.exists(tokenizer_path):
            print(f"错误: tokenizer路径不存在: {tokenizer_path}")
            return
            
        try:
            tokenizer = AmazonDistilBertTokenizer.from_pretrained(tokenizer_path)
            print(f"Tokenizer加载成功，词表大小: {len(tokenizer.base_tokenizer)}")
        except Exception as e:
            print(f"Tokenizer加载失败: {e}")
            traceback.print_exc()
            return
        
        # 创建示例训练数据
        print("创建训练数据")
        train_data_path = os.path.join(temp_dir, "test_train.txt")
        try:
            with open(train_data_path, "w", encoding="utf-8") as f:
                # 用户token样本
                for token in tokenizer.user_tokens[:3]:
                    f.write(f"{token} has interacted with products.\n")
                # 物品token样本
                for token in tokenizer.item_tokens[:3]:
                    f.write(f"Many users have purchased {token}.\n")
                # 类别token样本
                for token in tokenizer.category_tokens[:3]:
                    f.write(f"{token} products are popular.\n")
            print(f"训练数据写入成功: {train_data_path}")
        except Exception as e:
            print(f"创建训练数据失败: {e}")
            traceback.print_exc()
            return
        
        # 创建数据集
        print("创建数据集")
        try:
            dataset = SimpleMLMDataset(
                tokenizer=tokenizer,
                data_file=train_data_path,
                max_length=64,
                mlm_probability=0.15
            )
            print(f"数据集创建成功，样本数: {len(dataset)}")
        except Exception as e:
            print(f"创建数据集失败: {e}")
            traceback.print_exc()
            return
        
        # 创建dataloader
        print("创建数据加载器")
        try:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=2,
                shuffle=True
            )
            print(f"数据加载器创建成功")
        except Exception as e:
            print(f"创建数据加载器失败: {e}")
            traceback.print_exc()
            return
        
        # 初始化模型
        print("初始化模型")
        try:
            config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
            base_model = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
            print(f"基础模型加载成功")
            
            base_model.resize_token_embeddings(len(tokenizer.base_tokenizer))
            print(f"词表调整成功，新大小: {base_model.embeddings.word_embeddings.num_embeddings}")
            
            amazon_base_model = AmazonDistilBertBaseModel(base_model)
            mlm_model = AmazonDistilBertForMLM(amazon_base_model)
            print(f"MLM模型创建成功")
        except Exception as e:
            print(f"初始化模型失败: {e}")
            traceback.print_exc()
            return
        
        # 冻结预训练参数
        print("冻结预训练参数")
        try:
            for param in mlm_model.parameters():
                param.requires_grad = False
            
            # 解冻embedding
            embedding_layer = mlm_model.distilbert.embeddings.word_embeddings
            embedding_layer.weight.requires_grad = True
            
            # 计算可训练参数数量
            trainable_params = sum(p.numel() for p in mlm_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in mlm_model.parameters())
            print(f"可训练参数: {trainable_params:,d} / {total_params:,d} ({trainable_params/total_params:.2%})")
        except Exception as e:
            print(f"冻结参数失败: {e}")
            traceback.print_exc()
            return
        
        # 训练
        print("准备训练")
        try:
            optimizer = torch.optim.Adam([p for p in mlm_model.parameters() if p.requires_grad], lr=1e-4)
            print(f"优化器创建成功")
            
            print("开始训练")
            mlm_model.train()
            
            for epoch in range(2):
                print(f"开始第 {epoch+1} 轮训练")
                epoch_loss = 0
                for i, batch in enumerate(dataloader):
                    print(f"处理批次 {i+1}/{len(dataloader)}")
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]
                    
                    # 前向传播
                    print("执行前向传播")
                    outputs = mlm_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    print(f"损失: {loss.item():.4f}")
                    
                    # 反向传播
                    print("执行反向传播")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                print(f"Epoch {epoch+1} 损失: {epoch_loss/len(dataloader):.4f}")
            
            print("训练完成")
        except Exception as e:
            print(f"训练过程失败: {e}")
            traceback.print_exc()
            return
        
        # 清理临时目录
        print(f"清理临时目录: {temp_dir}")
        shutil.rmtree(temp_dir)
    
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        traceback.print_exc()
        # 确保清理临时目录
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    print("="*70)
    print("测试完成")
    print("="*70)

if __name__ == "__main__":
    print("脚本开始执行...")
    try:
        main()
    except Exception as e:
        print(f"主函数执行失败: {e}")
        traceback.print_exc() 