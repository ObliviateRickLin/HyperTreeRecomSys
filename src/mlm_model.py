import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import random
import numpy as np
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertConfig
import json

# 尝试导入wandb，如果不可用则提供一个虚拟实现
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    # 提供一个虚拟的wandb实现
    class WandbMock:
        def __init__(self):
            self.run = type('obj', (object,), {'name': 'mock-run'})
            
        def init(self, **kwargs):
            print(f"[MOCK] wandb.init called with: {kwargs}")
            return self.run
            
        def log(self, data):
            print(f"[MOCK] wandb.log: {data}")
            
        def finish(self):
            print("[MOCK] wandb.finish called")
            
    wandb = WandbMock()
    print("警告: 未安装wandb，将使用模拟实现（不会实际记录训练数据）")

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.libs.tokenizer import AmazonDistilBertTokenizer
from src.libs.model import AmazonDistilBertBaseModel, AmazonDistilBertForMLM

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

#################################################
# 数据集类
#################################################

class SimpleMLMDataset(Dataset):
    """简单的MLM数据集，从文本文件中加载样本"""
    
    def __init__(self, tokenizer, data_file, max_length=128, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.examples = []
        
        # 从文件加载文本
        logger.info(f"从 {data_file} 加载数据")
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text:
                    self.examples.append(text)
        
        logger.info(f"加载了 {len(self.examples)} 个样本")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # 编码文本
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        
        # 随机mask
        input_ids, labels = self._random_mask(input_ids, labels, attention_mask)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _random_mask(self, input_ids, labels, attention_mask):
        """随机mask输入tokens的一部分"""
        # 创建概率矩阵
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        
        # 不对特殊token进行mask
        special_tokens_mask = [
            self.tokenizer.base_tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
            for val in [input_ids]
        ]
        # 确保维度匹配 - 修复shape不匹配问题
        special_tokens_mask_tensor = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        # 确保special_tokens_mask_tensor的维度与probability_matrix匹配
        if special_tokens_mask_tensor.shape != probability_matrix.shape:
            if len(special_tokens_mask_tensor.shape) == 2 and len(probability_matrix.shape) == 1:
                # 如果special_tokens_mask是2D但probability_matrix是1D
                special_tokens_mask_tensor = special_tokens_mask_tensor.squeeze(0)
            elif len(special_tokens_mask_tensor.shape) == 1 and len(probability_matrix.shape) == 2:
                # 如果special_tokens_mask是1D但probability_matrix是2D
                special_tokens_mask_tensor = special_tokens_mask_tensor.unsqueeze(0)
                
        probability_matrix.masked_fill_(special_tokens_mask_tensor, value=0.0)
        
        # 不对padding token进行mask
        if hasattr(self.tokenizer.base_tokenizer, "pad_token_id"):
            padding_mask = input_ids.eq(self.tokenizer.base_tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        # 生成mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 只对被选择的索引位置修改labels，其他位置设为-100
        labels[~masked_indices] = -100
        
        # 80%的概率用[MASK]替换
        # 10%的概率用随机token替换
        # 10%的概率保持不变
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.base_tokenizer.convert_tokens_to_ids(self.tokenizer.base_tokenizer.mask_token)
        
        # 随机替换10%
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.base_tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        return input_ids, labels

#################################################
# 模型训练相关函数
#################################################

def set_seed(seed):
    """设置随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(tokenizer_path, model_path=None, device="cpu"):
    """加载或初始化模型"""
    logger.info(f"从 {tokenizer_path} 加载tokenizer")
    tokenizer = AmazonDistilBertTokenizer.from_pretrained(tokenizer_path)
    
    # 初始化DistilBERT模型
    config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
    base_bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
    
    # 调整模型的嵌入层大小以适应扩展的tokenizer
    base_bert_model.resize_token_embeddings(len(tokenizer.base_tokenizer))
    
    # 创建模型
    base_model = AmazonDistilBertBaseModel(base_bert_model)
    mlm_model = AmazonDistilBertForMLM(base_model)
    
    # 如果提供了模型路径，加载训练好的权重
    if model_path and os.path.exists(model_path):
        logger.info(f"从 {model_path} 加载模型权重")
        
        # 检查是否是checkpoint文件
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                mlm_model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"从checkpoint加载了模型状态")
            else:
                mlm_model.load_state_dict(torch.load(model_path, map_location=device))
        except:
            mlm_model.load_state_dict(torch.load(model_path, map_location=device))
    
    mlm_model.to(device)
    
    return tokenizer, mlm_model

def train_mlm_model(args):
    """训练MLM模型的主函数"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 初始化wandb（如果启用且可用）
    if args.use_wandb:
        if WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"mlm-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=vars(args),
                tags=["mlm", "training"]
            )
            logger.info(f"Wandb初始化完成: {wandb.run.name}")
        else:
            logger.warning("未安装wandb，无法记录训练数据。可使用 'pip install wandb' 安装")
    
    # 加载tokenizer和初始化模型
    tokenizer, mlm_model = load_model(args.tokenizer_path, args.resume_from, device=device)
    
    # 冻结预训练参数，只训练新增的token embedding
    if args.freeze_pretrained:
        logger.info("冻结预训练模型参数，只训练新增token的embedding")
        # 获取embedding层
        embedding_layer = mlm_model.base_model.distilbert.embeddings.word_embeddings
        
        # 获取预训练的token数量
        pretrained_token_count = len(tokenizer.base_tokenizer) - len(tokenizer.user_tokens) - len(tokenizer.item_tokens) - len(tokenizer.category_tokens)
        logger.info(f"预训练token数量: {pretrained_token_count}")
        logger.info(f"总token数量: {len(tokenizer.base_tokenizer)}")
        
        # 冻结所有参数
        for param in mlm_model.parameters():
            param.requires_grad = False
        
        # 解冻embedding中新增token的参数
        embedding_layer.weight.requires_grad = True
        # 创建一个mask，只有新增的token部分为True
        weight_mask = torch.zeros_like(embedding_layer.weight, dtype=torch.bool)
        weight_mask[pretrained_token_count:] = True
        # 设置梯度mask
        embedding_layer.weight.register_hook(lambda grad: grad * weight_mask)
        
        # 打印可训练参数数量
        trainable_params = sum(p.numel() for p in mlm_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in mlm_model.parameters())
        logger.info(f"可训练参数: {trainable_params:,d} / {total_params:,d} ({trainable_params/total_params:.2%})")
        
        if args.use_wandb:
            wandb.log({
                "trainable_params": trainable_params,
                "total_params": total_params,
                "trainable_ratio": trainable_params/total_params
            })
    
    # 加载训练集和验证集
    train_dataset = SimpleMLMDataset(
        tokenizer=tokenizer,
        data_file=args.train_file,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability
    )
    
    val_dataset = SimpleMLMDataset(
        tokenizer=tokenizer,
        data_file=args.val_file,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 创建优化器和学习率调度器
    optimizer = optim.AdamW(
        mlm_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 加载checkpoint（如果存在）
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        logger.info(f"从checkpoint加载训练状态: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        else:
            best_val_loss = float('inf')
        logger.info(f"从第 {start_epoch} 轮继续训练")
    else:
        best_val_loss = float('inf')
    
    # 训练模型
    logger.info("开始训练")
    
    for epoch in range(start_epoch, args.num_epochs):
        # 训练
        mlm_model.train()
        train_loss = 0
        train_steps = 0
        
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs} [Train]")
        for batch in epoch_iterator:
            # 将数据移动到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播和计算损失
            outputs = mlm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新统计
            train_loss += loss.item()
            train_steps += 1
            
            # 更新进度条
            epoch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 记录到wandb
            if args.use_wandb and train_steps % args.wandb_log_interval == 0:
                wandb.log({"train_loss": loss.item(), 
                          "train_step": epoch * len(train_loader) + train_steps})
        
        # 计算平均训练损失
        avg_train_loss = train_loss / train_steps
        logger.info(f"Epoch {epoch + 1} 训练损失: {avg_train_loss:.4f}")
        
        # 验证
        mlm_model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            epoch_iterator = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs} [Validation]")
            for batch in epoch_iterator:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = mlm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                val_loss += loss.item()
                val_steps += 1
                
                epoch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 计算平均验证损失
        avg_val_loss = val_loss / val_steps
        logger.info(f"Epoch {epoch + 1} 验证损失: {avg_val_loss:.4f}")
        
        # 记录到wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss
            })
        
        # 保存模型
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # 保存最佳模型和当前模型
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            logger.info(f"发现新的最佳模型，验证损失: {best_val_loss:.4f}")
            
            # 保存最佳模型
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(mlm_model.state_dict(), best_model_path)
            logger.info(f"最佳模型已保存到: {best_model_path}")
        
        # 总是保存最近的checkpoint，便于恢复训练
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': mlm_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'args': vars(args)
        }, checkpoint_path)
        logger.info(f"Checkpoint已保存到: {checkpoint_path}")
        
        # 保存最新的模型
        latest_model_path = os.path.join(args.output_dir, "latest_model.pth")
        torch.save(mlm_model.state_dict(), latest_model_path)
    
    # 训练结束
    logger.info("训练完成!")
    
    # 关闭wandb
    if args.use_wandb:
        wandb.finish()
    
    return best_val_loss

#################################################
# 模型测试相关函数
#################################################

def mask_tokens(tokenizer, input_ids, mask_token_index=None):
    """对输入的token序列进行掩码，用于MLM测试"""
    # 如果指定了mask_token_index，只掩码该位置；否则随机选择一个非特殊token位置掩码
    if mask_token_index is None:
        # 找出非特殊token的位置
        special_tokens_mask = tokenizer.base_tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        # 找出padding token的位置
        padding_mask = input_ids.eq(tokenizer.base_tokenizer.pad_token_id)
        
        # 可掩码的位置：非特殊token且非padding
        maskable_indices = ~(special_tokens_mask | padding_mask)
        
        # 随机选择一个位置
        valid_indices = torch.where(maskable_indices)[0]
        if len(valid_indices) == 0:
            return input_ids, None
        mask_token_index = valid_indices[random.randint(0, len(valid_indices) - 1)].item()
    
    # 获取原始token，用于后续比较
    original_token = input_ids[mask_token_index].item()
    
    # 创建labels，只对mask位置进行预测
    labels = torch.full_like(input_ids, -100)
    labels[mask_token_index] = original_token
    
    # 掩码输入
    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_token_index] = tokenizer.base_tokenizer.convert_tokens_to_ids(tokenizer.base_tokenizer.mask_token)
    
    return masked_input_ids, labels, mask_token_index, original_token

def predict_masked_token(model, tokenizer, input_ids, attention_mask, labels, mask_index, top_k=5):
    """预测被掩码的token"""
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, labels)
        loss, logits = outputs
    
    # 获取被掩码位置的logits
    mask_logits = logits[0, mask_index, :]
    
    # 获取top-k预测
    topk_values, topk_indices = torch.topk(mask_logits, top_k)
    topk_tokens = [tokenizer.decode([idx.item()]) for idx in topk_indices]
    
    # 获取真实token
    true_token = tokenizer.decode([labels[mask_index].item()])
    
    # 计算预测准确率
    correct = topk_indices[0].item() == labels[mask_index].item()
    
    return {
        "loss": loss.item(),
        "top_k_tokens": list(zip(topk_tokens, topk_values.tolist())),
        "true_token": true_token,
        "correct": correct
    }

def test_specific_token_types(tokenizer, model, device="cpu"):
    """测试模型对特定类型token的掩码预测能力"""
    logger.info("测试模型对特定类型token的预测能力")
    
    results = {
        "user_tokens": [],
        "item_tokens": [],
        "category_tokens": [],
        "regular_tokens": []
    }
    
    # 1. 测试用户token
    if tokenizer.user_tokens:
        for _ in range(5):
            user_token = random.choice(tokenizer.user_tokens)
            text = f"{user_token} has interacted with several beauty products."
            
            # 编码
            encoded = tokenizer.encode_plus(
                text,
                return_tensors="pt",
                max_length=20,
                padding="max_length",
                truncation=True
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # 掩码用户token（通常是位置1，位置0是[CLS]）
            masked_ids, labels, mask_index, original_token = mask_tokens(tokenizer, input_ids[0], mask_token_index=1)
            masked_ids = masked_ids.unsqueeze(0)
            if labels is not None:
                labels = labels.unsqueeze(0)
                
                # 预测
                result = predict_masked_token(model, tokenizer, masked_ids, attention_mask, labels, mask_index)
                
                # 记录结果
                results["user_tokens"].append({
                    "text": text,
                    "masked_text": tokenizer.decode(masked_ids[0]),
                    "prediction": result
                })
    
    # 2. 测试物品token
    if tokenizer.item_tokens:
        for _ in range(5):
            item_token = random.choice(tokenizer.item_tokens)
            text = f"Many users have purchased {item_token} which is a popular product."
            
            # 编码
            encoded = tokenizer.encode_plus(
                text,
                return_tensors="pt",
                max_length=20,
                padding="max_length",
                truncation=True
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # 找到物品token的位置
            item_token_id = tokenizer.base_tokenizer.convert_tokens_to_ids(item_token)
            if item_token_id in input_ids[0]:
                mask_index = (input_ids[0] == item_token_id).nonzero(as_tuple=True)[0][0].item()
                
                # 掩码物品token
                masked_ids, labels, mask_index, original_token = mask_tokens(tokenizer, input_ids[0], mask_token_index=mask_index)
                masked_ids = masked_ids.unsqueeze(0)
                if labels is not None:
                    labels = labels.unsqueeze(0)
                    
                    # 预测
                    result = predict_masked_token(model, tokenizer, masked_ids, attention_mask, labels, mask_index)
                    
                    # 记录结果
                    results["item_tokens"].append({
                        "text": text,
                        "masked_text": tokenizer.decode(masked_ids[0]),
                        "prediction": result
                    })
    
    # 3. 测试类别token
    if tokenizer.category_tokens:
        for _ in range(5):
            category_token = random.choice(tokenizer.category_tokens)
            text = f"This is a {category_token} product with high ratings."
            
            # 编码
            encoded = tokenizer.encode_plus(
                text,
                return_tensors="pt",
                max_length=20,
                padding="max_length",
                truncation=True
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # 找到类别token的位置
            category_token_id = tokenizer.base_tokenizer.convert_tokens_to_ids(category_token)
            if category_token_id in input_ids[0]:
                mask_index = (input_ids[0] == category_token_id).nonzero(as_tuple=True)[0][0].item()
                
                # 掩码类别token
                masked_ids, labels, mask_index, original_token = mask_tokens(tokenizer, input_ids[0], mask_token_index=mask_index)
                masked_ids = masked_ids.unsqueeze(0)
                if labels is not None:
                    labels = labels.unsqueeze(0)
                    
                    # 预测
                    result = predict_masked_token(model, tokenizer, masked_ids, attention_mask, labels, mask_index)
                    
                    # 记录结果
                    results["category_tokens"].append({
                        "text": text,
                        "masked_text": tokenizer.decode(masked_ids[0]),
                        "prediction": result
                    })
    
    # 4. 测试普通token
    common_words = ["beauty", "skin", "hair", "care", "product", "review", "rating", "good", "effective", "popular"]
    for word in random.sample(common_words, 5):
        text = f"This {word} is highly recommended by many users."
        
        # 编码
        encoded = tokenizer.encode_plus(
            text,
            return_tensors="pt",
            max_length=20,
            padding="max_length",
            truncation=True
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # 找到普通词的位置
        word_ids = tokenizer.base_tokenizer.encode(word, add_special_tokens=False)
        if word_ids[0] in input_ids[0]:
            mask_index = (input_ids[0] == word_ids[0]).nonzero(as_tuple=True)[0][0].item()
            
            # 掩码普通词
            masked_ids, labels, mask_index, original_token = mask_tokens(tokenizer, input_ids[0], mask_token_index=mask_index)
            masked_ids = masked_ids.unsqueeze(0)
            if labels is not None:
                labels = labels.unsqueeze(0)
                
                # 预测
                result = predict_masked_token(model, tokenizer, masked_ids, attention_mask, labels, mask_index)
                
                # 记录结果
                results["regular_tokens"].append({
                    "text": text,
                    "masked_text": tokenizer.decode(masked_ids[0]),
                    "prediction": result
                })
    
    return results

def test_mixed_token_scenario(tokenizer, model, device="cpu"):
    """测试包含多种特殊token的复杂场景"""
    logger.info("测试包含多种特殊token的复杂场景")
    
    results = []
    
    # 生成包含用户、物品和类别token的样本
    if tokenizer.user_tokens and tokenizer.item_tokens and tokenizer.category_tokens:
        for _ in range(5):
            user_token = random.choice(tokenizer.user_tokens)
            item_token = random.choice(tokenizer.item_tokens)
            category_token = random.choice(tokenizer.category_tokens)
            
            text = f"{user_token} purchased {item_token} which is a {category_token} product. It has good quality."
            
            # 编码
            encoded = tokenizer.encode_plus(
                text,
                return_tensors="pt",
                max_length=30,
                padding="max_length",
                truncation=True
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # 随机掩码一个token
            masked_ids, labels, mask_index, original_token = mask_tokens(tokenizer, input_ids[0])
            masked_ids = masked_ids.unsqueeze(0)
            if labels is not None and mask_index is not None:
                labels = labels.unsqueeze(0)
                
                # 预测
                result = predict_masked_token(model, tokenizer, masked_ids, attention_mask, labels, mask_index)
                
                # 记录结果
                token_type = "unknown"
                original_token_str = tokenizer.decode([original_token])
                if original_token_str.startswith("[user_"):
                    token_type = "user"
                elif original_token_str.startswith("[item_"):
                    token_type = "item"
                elif original_token_str.startswith("[category_"):
                    token_type = "category"
                else:
                    token_type = "regular"
                
                results.append({
                    "text": text,
                    "masked_text": tokenizer.decode(masked_ids[0]),
                    "masked_token_type": token_type,
                    "prediction": result
                })
    
    return results

def calculate_accuracy(results):
    """计算各种token类型的预测准确率"""
    accuracy_stats = {}
    
    # 计算特定类型token的准确率
    for token_type, entries in results.items():
        if entries:
            correct = sum(1 for entry in entries if entry["prediction"]["correct"])
            accuracy = correct / len(entries) if entries else 0
            accuracy_stats[token_type] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(entries)
            }
    
    return accuracy_stats

def test_trained_model(args):
    """测试训练好的MLM模型"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    tokenizer, model = load_model(args.tokenizer_path, args.model_path, device)
    
    # 打印模型信息
    logger.info(f"模型词表大小: {model.extended_vocab_size}")
    
    # 测试特定类型token
    token_type_results = test_specific_token_types(tokenizer, model, device)
    
    # 测试混合token场景
    mixed_results = test_mixed_token_scenario(tokenizer, model, device)
    
    # 计算准确率
    accuracy_stats = calculate_accuracy(token_type_results)
    
    # 打印结果
    logger.info("\n===== 特定类型Token预测结果 =====")
    for token_type, stats in accuracy_stats.items():
        logger.info(f"{token_type}: 准确率 {stats['accuracy']:.2f} ({stats['correct']}/{stats['total']})")
        
        # 打印一些示例
        if token_type_results[token_type]:
            for i, result in enumerate(token_type_results[token_type][:2]):  # 只显示前两个示例
                logger.info(f"  示例 {i+1}:")
                logger.info(f"    原文: {result['text']}")
                logger.info(f"    掩码后: {result['masked_text']}")
                logger.info(f"    真实token: {result['prediction']['true_token']}")
                logger.info(f"    预测结果 (Top-5): {result['prediction']['top_k_tokens']}")
                logger.info(f"    是否正确: {result['prediction']['correct']}")
    
    # 打印混合token场景结果
    logger.info("\n===== 混合Token场景预测结果 =====")
    if mixed_results:
        mixed_correct = sum(1 for result in mixed_results if result["prediction"]["correct"])
        mixed_accuracy = mixed_correct / len(mixed_results) if mixed_results else 0
        logger.info(f"混合场景准确率: {mixed_accuracy:.2f} ({mixed_correct}/{len(mixed_results)})")
        
        # 按token类型分组
        token_type_counts = {}
        token_type_correct = {}
        for result in mixed_results:
            token_type = result["masked_token_type"]
            token_type_counts[token_type] = token_type_counts.get(token_type, 0) + 1
            if result["prediction"]["correct"]:
                token_type_correct[token_type] = token_type_correct.get(token_type, 0) + 1
        
        # 打印各类型准确率
        for token_type, count in token_type_counts.items():
            correct = token_type_correct.get(token_type, 0)
            type_accuracy = correct / count if count > 0 else 0
            logger.info(f"  {token_type} token: 准确率 {type_accuracy:.2f} ({correct}/{count})")
        
        # 打印一些示例
        for i, result in enumerate(mixed_results[:3]):  # 只显示前三个示例
            logger.info(f"  示例 {i+1} ({result['masked_token_type']} token):")
            logger.info(f"    原文: {result['text']}")
            logger.info(f"    掩码后: {result['masked_text']}")
            logger.info(f"    真实token: {result['prediction']['true_token']}")
            logger.info(f"    预测结果 (Top-5): {result['prediction']['top_k_tokens']}")
            logger.info(f"    是否正确: {result['prediction']['correct']}")
    
    logger.info("\n===== 总结 =====")
    # 计算总体准确率
    all_correct = 0
    all_total = 0
    for stats in accuracy_stats.values():
        all_correct += stats["correct"]
        all_total += stats["total"]
    
    if mixed_results:
        all_correct += mixed_correct
        all_total += len(mixed_results)
    
    overall_accuracy = all_correct / all_total if all_total > 0 else 0
    logger.info(f"总体准确率: {overall_accuracy:.4f} ({all_correct}/{all_total})")
    
    return overall_accuracy

#################################################
# 命令行入口函数
#################################################

def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(
        description="训练和测试MLM模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 通用参数
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="运行模式：train（训练模型）或test（测试模型）"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default="data/beauty_tokenizer",
        help="已初始化的tokenizer路径"
    )
    parser.add_argument(
        "--no_cuda", 
        action="store_true",
        help="不使用CUDA（即使可用）"
    )
    
    # 训练参数
    train_group = parser.add_argument_group("训练参数")
    train_group.add_argument(
        "--train_file", 
        type=str, 
        default="data/mlm_data/train_mlm.txt",
        help="训练数据文件路径"
    )
    train_group.add_argument(
        "--val_file", 
        type=str, 
        default="data/mlm_data/val_mlm.txt",
        help="验证数据文件路径"
    )
    train_group.add_argument(
        "--output_dir", 
        type=str, 
        default="models/mlm",
        help="输出目录，用于保存训练的模型"
    )
    train_group.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="训练批次大小"
    )
    train_group.add_argument(
        "--max_length", 
        type=int, 
        default=128,
        help="最大序列长度"
    )
    train_group.add_argument(
        "--mlm_probability", 
        type=float, 
        default=0.15,
        help="MLM任务中mask的概率"
    )
    train_group.add_argument(
        "--num_epochs", 
        type=int, 
        default=5,
        help="训练轮数"
    )
    train_group.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5,
        help="学习率"
    )
    train_group.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01,
        help="权重衰减"
    )
    train_group.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="数据加载的工作线程数"
    )
    train_group.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="随机种子"
    )
    
    # Checkpoint 参数
    train_group.add_argument(
        "--resume_from", 
        type=str, 
        default="",
        help="从指定的checkpoint恢复训练"
    )
    
    # 参数冻结选项
    train_group.add_argument(
        "--freeze_pretrained", 
        action="store_true",
        help="冻结预训练的参数，只训练新增token的embedding"
    )
    
    # Wandb参数
    train_group.add_argument(
        "--use_wandb", 
        action="store_true",
        help="使用Weights & Biases记录训练过程 (需要安装wandb)"
    )
    train_group.add_argument(
        "--wandb_project", 
        type=str, 
        default="amazon-beauty-mlm",
        help="Wandb项目名称"
    )
    train_group.add_argument(
        "--wandb_run_name", 
        type=str, 
        default="",
        help="Wandb运行名称，留空则自动生成"
    )
    train_group.add_argument(
        "--wandb_log_interval", 
        type=int, 
        default=10,
        help="Wandb记录间隔（步数）"
    )
    
    # 测试参数
    test_group = parser.add_argument_group("测试参数")
    test_group.add_argument(
        "--model_path", 
        type=str, 
        default="models/mlm/best_mlm_model.pt",
        help="训练好的模型路径"
    )
    
    args = parser.parse_args()
    
    # 记录开始时间
    start_time = datetime.now()
    
    if args.mode == "train":
        logger.info(f"开始MLM模型训练: {start_time}")
        model_path = train_mlm_model(args)
        logger.info(f"MLM模型训练完成！模型保存在: {model_path}")
    
    elif args.mode == "test":
        logger.info(f"开始MLM模型测试: {start_time}")
        accuracy = test_trained_model(args)
        logger.info(f"MLM模型测试完成！总体准确率: {accuracy:.4f}")
    
    # 记录结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"总耗时: {duration}")

if __name__ == "__main__":
    main() 