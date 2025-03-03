import logging
import os
import sys
import torch
import random
from transformers import DistilBertModel, DistilBertConfig

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def test_model_with_extended_tokenizer():
    """测试使用扩展词表的模型功能"""
    
    print("\n" + "="*70)
    print("Testing Model with Extended Tokenizer")
    print("="*70)
    
    try:
        # 导入依赖
        import sys
        import os
        import torch
        from transformers import DistilBertConfig, DistilBertModel
        
        # 添加项目根目录到Python路径
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
        from src.libs.tokenizer import AmazonDistilBertTokenizer
        from src.libs.model import AmazonDistilBertBaseModel, AmazonDistilBertForMLM
        
        # 加载tokenizer
        tokenizer_path = "data/beauty_tokenizer"
        print(f"Loading tokenizer from: {tokenizer_path}")
        
        tokenizer = AmazonDistilBertTokenizer.from_pretrained(tokenizer_path)
        print(f"Base tokenizer vocabulary size: {len(tokenizer.base_tokenizer)}")
        
        # 1. 初始化DistilBERT
        print("\n1. Initializing DistilBERT model")
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        base_bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
        
        # 2. 调整embedding大小以适应扩展的tokenizer
        print("\n2. Resizing token embeddings")
        original_size = base_bert_model.embeddings.word_embeddings.num_embeddings
        print(f"Original embedding size: {original_size}")
        
        base_bert_model.resize_token_embeddings(len(tokenizer.base_tokenizer))
        new_size = base_bert_model.embeddings.word_embeddings.num_embeddings
        print(f"New embedding size: {new_size}")
        
        # 3. 创建AmazonDistilBertBaseModel
        print("\n3. Creating AmazonDistilBertBaseModel")
        base_model = AmazonDistilBertBaseModel(base_bert_model)
        
        # 4. 创建带MLM头的模型
        print("\n4. Creating AmazonDistilBertForMLM")
        mlm_model = AmazonDistilBertForMLM(base_model)
        print(f"MLM model vocabulary size: {mlm_model.extended_vocab_size}")
        
        # 5. 测试模型前向传播
        print("\n5. Testing model forward pass")
        # 构造一个测试句子，包含特殊token
        test_text = f"{tokenizer.user_tokens[0]} purchased {tokenizer.item_tokens[0]} which is a {tokenizer.category_tokens[0]} category product"
        print(f"Test text: '{test_text}'")
        
        # 编码文本
        encoding = tokenizer.encode_plus(
            test_text,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        print(f"Input shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        
        # 前向传播 - 基础模型
        with torch.no_grad():
            hidden_states = base_model(input_ids, attention_mask)
            print(f"Base model output shape: {hidden_states.shape}")
        
        # 前向传播 - MLM模型
        with torch.no_grad():
            outputs = mlm_model(input_ids, attention_mask)
            print(f"MLM model output shape: {outputs.logits.shape}")
            assert outputs.logits.shape[-1] == mlm_model.extended_vocab_size, "MLM output dimension mismatch"
            print("✓ MLM output dimension matches vocabulary size")
        
        # 6. 测试MLM任务
        print("\n6. Testing MLM task")
        # 创建MLM标签 - 随机mask一些token
        labels = input_ids.clone()
        # 创建概率矩阵，用于随机mask
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = tokenizer.base_tokenizer.get_special_tokens_mask(
            labels.squeeze().tolist(), already_has_special_tokens=True
        )
        probability_matrix[0, torch.tensor(special_tokens_mask, dtype=torch.bool)] = 0.0
        
        # 为简化测试，只mask一个token
        masked_indices = torch.zeros_like(labels, dtype=torch.bool)
        # 找到一个非特殊token进行mask
        for i in range(1, labels.size(1) - 1):
            if special_tokens_mask[i] == 0:  # 不是特殊token
                masked_indices[0, i] = True
                break
                
        # 设置不被mask的token的label为-100
        labels_clone = labels.clone()
        labels_clone[~masked_indices] = -100
        
        # 记录要mask的token数量
        num_masked = masked_indices.sum().item()
        print(f"Number of masked tokens: {num_masked}")
        
        # 在input_ids中应用mask
        input_ids_masked = input_ids.clone()
        input_ids_masked[masked_indices] = tokenizer.base_tokenizer.convert_tokens_to_ids(tokenizer.base_tokenizer.mask_token)
        
        # 前向传播 - MLM任务
        with torch.no_grad():
            outputs = mlm_model(
                input_ids=input_ids_masked, 
                attention_mask=attention_mask, 
                labels=labels_clone
            )
            
            mlm_loss = outputs.loss
            mlm_logits = outputs.logits
            
            print(f"MLM loss: {mlm_loss.item()}")
            
            # 获取预测结果
            predictions = torch.argmax(mlm_logits, dim=-1)
            
            # 计算准确率
            correct = (predictions == labels) & (labels_clone != -100)
            if (labels_clone != -100).sum() > 0:
                accuracy = correct.sum().float() / (labels_clone != -100).sum().float()
                print(f"MLM accuracy: {accuracy.item():.4f}")
            else:
                print("MLM accuracy: N/A (no masked tokens)")
                
        print("\nModel test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

def test_mlm_train_flow():
    """测试MLM模型训练流程"""
    
    print("\n" + "="*70)
    print("测试MLM模型训练流程")
    print("="*70)
    
    try:
        # 导入依赖
        import sys
        import os
        import shutil
        import tempfile
        import torch
        from torch.utils.data import Dataset, DataLoader
        
        # 添加项目根目录到Python路径
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
        from src.mlm_model import SimpleMLMDataset, set_seed
        from src.libs.model import AmazonDistilBertBaseModel, AmazonDistilBertForMLM
        from src.libs.tokenizer import AmazonDistilBertTokenizer
        from transformers import DistilBertConfig, DistilBertModel
        
        # 设置随机种子
        set_seed(42)
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        try:
            # 创建微型数据集
            tokenizer_path = "data/beauty_tokenizer"
            
            if not os.path.exists(tokenizer_path):
                print(f"错误: 找不到tokenizer路径: {tokenizer_path}")
                return
                
            # 加载tokenizer
            print(f"加载tokenizer: {tokenizer_path}")
            tokenizer = AmazonDistilBertTokenizer.from_pretrained(tokenizer_path)
            
            # 创建临时训练和验证数据
            train_data_path = os.path.join(temp_dir, "mini_train.txt")
            val_data_path = os.path.join(temp_dir, "mini_val.txt")
            
            # 创建小型训练数据
            print("创建微型训练数据...")
            with open(train_data_path, "w", encoding="utf-8") as f:
                # 添加一些包含用户token的样本
                for i, user_token in enumerate(tokenizer.user_tokens[:5]):
                    f.write(f"{user_token} has interacted with several products.\n")
                
                # 添加一些包含物品token的样本
                for i, item_token in enumerate(tokenizer.item_tokens[:5]):
                    f.write(f"Many users have purchased {item_token} which is a popular product.\n")
                
                # 添加一些包含类别token的样本
                for i, category_token in enumerate(tokenizer.category_tokens[:5]):
                    f.write(f"Products in {category_token} are popular in the market.\n")
            
            # 创建小型验证数据
            print("创建微型验证数据...")
            with open(val_data_path, "w", encoding="utf-8") as f:
                # 添加一些混合样本
                f.write(f"{tokenizer.user_tokens[10]} has purchased {tokenizer.item_tokens[10]} which is a {tokenizer.category_tokens[10]} product.\n")
                f.write(f"{tokenizer.user_tokens[11]} has purchased {tokenizer.item_tokens[11]} which is a {tokenizer.category_tokens[11]} product.\n")
            
            # 设置训练参数
            output_dir = os.path.join(temp_dir, "models")
            os.makedirs(output_dir, exist_ok=True)
            batch_size = 4
            max_length = 64
            num_epochs = 1  # 减少训练轮数，加快测试
            learning_rate = 1e-4
            mlm_probability = 0.15
            
            print(f"微型训练设置: 批次大小={batch_size}, 最大长度={max_length}, 轮数={num_epochs}")
            
            # 加载数据集
            print("加载数据集...")
            train_dataset = SimpleMLMDataset(
                tokenizer=tokenizer,
                data_file=train_data_path,
                max_length=max_length,
                mlm_probability=mlm_probability
            )
            
            val_dataset = SimpleMLMDataset(
                tokenizer=tokenizer,
                data_file=val_data_path,
                max_length=max_length,
                mlm_probability=mlm_probability
            )
            
            print(f"训练样本数: {len(train_dataset)}")
            print(f"验证样本数: {len(val_dataset)}")
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            # 加载模型
            print("初始化模型...")
            device = torch.device("cpu")  # 使用CPU进行测试
            
            # 初始化DistilBERT模型
            config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
            base_bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
            
            # 调整embedding大小
            base_bert_model.resize_token_embeddings(len(tokenizer.base_tokenizer))
            
            # 创建模型
            base_model = AmazonDistilBertBaseModel(base_bert_model)
            mlm_model = AmazonDistilBertForMLM(base_model)
            mlm_model.to(device)
            
            # 冻结预训练参数，只训练新增的token embedding
            print("冻结预训练参数...")
            embedding_layer = mlm_model.distilbert.embeddings.word_embeddings
            
            # 获取预训练的token数量
            pretrained_token_count = 30522  # 原始DistilBERT词表大小
            print(f"预训练token数量: {pretrained_token_count}")
            print(f"总token数量: {len(tokenizer.base_tokenizer)}")
            
            # 冻结所有参数
            for param in mlm_model.parameters():
                param.requires_grad = False
            
            # 解冻embedding中新增token的参数
            embedding_layer.weight.requires_grad = True
            
            # 创建一个mask，只有新增的token部分为True
            weight_mask = torch.zeros_like(embedding_layer.weight, dtype=torch.bool)
            weight_mask[pretrained_token_count:] = True
            
            # 设置梯度mask的钩子函数
            def grad_hook(grad):
                return grad * weight_mask
                
            embedding_layer.weight.register_hook(grad_hook)
            
            # 打印可训练参数数量
            trainable_params = sum(p.numel() for p in mlm_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in mlm_model.parameters())
            print(f"可训练参数: {trainable_params:,d} / {total_params:,d} ({trainable_params/total_params:.2%})")
            
            # 创建优化器
            optimizer = torch.optim.AdamW(
                [p for p in mlm_model.parameters() if p.requires_grad],
                lr=learning_rate,
                weight_decay=0.01
            )
            
            # 简单训练循环
            print("开始训练...")
            for epoch in range(num_epochs):
                # 训练
                mlm_model.train()
                train_loss = 0
                train_steps = 0
                
                for batch in train_loader:
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
                    
                    # 仅更新新token的embedding
                    optimizer.step()
                    
                    # 更新统计
                    train_loss += loss.item()
                    train_steps += 1
                
                # 计算平均训练损失
                avg_train_loss = train_loss / max(train_steps, 1)
                print(f"Epoch {epoch + 1} 训练损失: {avg_train_loss:.4f}")
                
                # 验证
                mlm_model.eval()
                val_loss = 0
                val_steps = 0
                
                with torch.no_grad():
                    for batch in val_loader:
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
                
                # 计算平均验证损失
                avg_val_loss = val_loss / max(val_steps, 1)
                print(f"Epoch {epoch + 1} 验证损失: {avg_val_loss:.4f}")
                
                # 保存checkpoint
                checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': mlm_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }, checkpoint_path)
                print(f"保存Checkpoint: {checkpoint_path}")
            
            # 测试从checkpoint恢复
            print("测试从checkpoint恢复...")
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{num_epochs}.pth")
            
            # 初始化新模型
            new_base_model = AmazonDistilBertBaseModel(base_bert_model)
            new_model = AmazonDistilBertForMLM(new_base_model)
            new_model.to(device)
            
            # 从checkpoint加载
            checkpoint = torch.load(checkpoint_path, map_location=device)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            
            # 验证模型加载成功
            new_model.eval()
            with torch.no_grad():
                batch = next(iter(val_loader))
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = new_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
            print(f"从checkpoint加载后的验证损失: {loss.item():.4f}")
            print("微型训练测试成功完成!")
            
        finally:
            # 清理临时目录
            print(f"清理临时目录: {temp_dir}")
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("="*70)
    print("测试MLM模型训练流程结束")
    print("="*70)

if __name__ == "__main__":
    # 运行所有测试
    test_model_with_extended_tokenizer()
    test_mlm_train_flow() 