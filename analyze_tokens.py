import re
import os
import sys
from collections import Counter

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.libs.tokenizer import AmazonDistilBertTokenizer

def analyze_tokens():
    """分析训练数据中的token与tokenizer中的token的差异"""
    print("加载tokenizer...")
    tokenizer = AmazonDistilBertTokenizer.from_pretrained('data/beauty_tokenizer')
    
    # 使用集合确保唯一性
    unique_user_tokens = set(tokenizer.user_tokens)
    unique_item_tokens = set(tokenizer.item_tokens)
    unique_category_tokens = set(tokenizer.category_tokens)
    
    print(f"Tokenizer中的用户token数量: {len(unique_user_tokens)}")
    print(f"Tokenizer中的物品token数量: {len(unique_item_tokens)}")
    print(f"Tokenizer中的类别token数量: {len(unique_category_tokens)}")
    
    # 读取训练数据
    print("读取训练数据...")
    with open('data/mlm_data/train_mlm.txt', 'r', encoding='utf-8') as f:
        train_content = f.read()
    
    # 读取验证数据
    print("读取验证数据...")
    with open('data/mlm_data/val_mlm.txt', 'r', encoding='utf-8') as f:
        val_content = f.read()
    
    # 提取token
    train_user_tokens = set(re.findall(r'\[user_[^\]]+\]', train_content))
    train_item_tokens = set(re.findall(r'\[item_[^\]]+\]', train_content))
    train_category_tokens = set(re.findall(r'\[category_[^\]]+\]', train_content))
    
    val_user_tokens = set(re.findall(r'\[user_[^\]]+\]', val_content))
    val_item_tokens = set(re.findall(r'\[item_[^\]]+\]', val_content))
    val_category_tokens = set(re.findall(r'\[category_[^\]]+\]', val_content))
    
    # 合并训练集和验证集的token
    all_user_tokens = train_user_tokens | val_user_tokens
    all_item_tokens = train_item_tokens | val_item_tokens
    all_category_tokens = train_category_tokens | val_category_tokens
    
    # 统计token数量
    print(f"训练集中的用户token数量: {len(train_user_tokens)}")
    print(f"训练集中的物品token数量: {len(train_item_tokens)}")
    print(f"训练集中的类别token数量: {len(train_category_tokens)}")
    
    print(f"验证集中的用户token数量: {len(val_user_tokens)}")
    print(f"验证集中的物品token数量: {len(val_item_tokens)}")
    print(f"验证集中的类别token数量: {len(val_category_tokens)}")
    
    print(f"所有数据中的用户token数量: {len(all_user_tokens)}")
    print(f"所有数据中的物品token数量: {len(all_item_tokens)}")
    print(f"所有数据中的类别token数量: {len(all_category_tokens)}")
    
    # 检查token覆盖率
    user_coverage = len(train_user_tokens) / len(unique_user_tokens) * 100
    item_coverage = len(train_item_tokens) / len(unique_item_tokens) * 100
    category_coverage = len(train_category_tokens) / len(unique_category_tokens) * 100
    
    print(f"训练集中用户token覆盖率: {user_coverage:.2f}%")
    print(f"训练集中物品token覆盖率: {item_coverage:.2f}%")
    print(f"训练集中类别token覆盖率: {category_coverage:.2f}%")
    
    # 检查只在验证集中出现的token
    only_in_val_user = val_user_tokens - train_user_tokens
    only_in_val_item = val_item_tokens - train_item_tokens
    only_in_val_category = val_category_tokens - train_category_tokens
    
    print(f"只在验证集中出现的用户token数量: {len(only_in_val_user)}")
    print(f"只在验证集中出现的物品token数量: {len(only_in_val_item)}")
    print(f"只在验证集中出现的类别token数量: {len(only_in_val_category)}")
    
    # 检查tokenizer中没有的token
    extra_user_tokens = all_user_tokens - unique_user_tokens
    extra_item_tokens = all_item_tokens - unique_item_tokens
    extra_category_tokens = all_category_tokens - unique_category_tokens
    
    print(f"数据中存在但tokenizer中没有的用户token数量: {len(extra_user_tokens)}")
    print(f"数据中存在但tokenizer中没有的物品token数量: {len(extra_item_tokens)}")
    print(f"数据中存在但tokenizer中没有的类别token数量: {len(extra_category_tokens)}")
    
    # 检查tokenizer中有但数据中没有的token
    missing_user_tokens = unique_user_tokens - train_user_tokens
    missing_item_tokens = unique_item_tokens - train_item_tokens
    missing_category_tokens = unique_category_tokens - train_category_tokens
    
    print(f"tokenizer中有但训练集中没有的用户token数量: {len(missing_user_tokens)}")
    print(f"tokenizer中有但训练集中没有的物品token数量: {len(missing_item_tokens)}")
    print(f"tokenizer中有但训练集中没有的类别token数量: {len(missing_category_tokens)}")
    
    # 输出缺失的类别token
    if missing_category_tokens:
        print("\n缺失的类别token:")
        for token in missing_category_tokens:
            print(f"  {token}")
    
    # 输出详细的类别token信息
    print("\n详细的类别token信息:")
    print(f"Tokenizer中的类别token数量(原始): {len(tokenizer.category_tokens)}")
    print(f"Tokenizer中的类别token数量(唯一): {len(unique_category_tokens)}")
    print(f"训练集中的类别token数量: {len(train_category_tokens)}")
    
    # 检查是否有格式问题
    print("\n类别token格式检查:")
    tokenizer_category_tokens_list = list(unique_category_tokens)
    train_category_tokens_list = list(train_category_tokens)
    
    print("Tokenizer中的类别token示例:")
    for token in tokenizer_category_tokens_list[:5]:
        print(f"  {token}")
    
    print("训练集中的类别token示例:")
    for token in train_category_tokens_list[:5]:
        print(f"  {token}")
    
    # 检查正则表达式是否正确
    print("\n正则表达式检查:")
    sample_text = "This is a [category_Beauty] product."
    found_tokens = re.findall(r'\[category_[^\]]+\]', sample_text)
    print(f"样本文本: {sample_text}")
    print(f"找到的token: {found_tokens}")
    
    # 检查tokenizer中的类别token是否符合正则表达式
    matched_tokens = 0
    for token in tokenizer.category_tokens:
        if re.match(r'\[category_[^\]]+\]', token):
            matched_tokens += 1
    
    print(f"Tokenizer中符合正则表达式的类别token数量: {matched_tokens}/{len(tokenizer.category_tokens)}")
    
    # 如果有不符合正则表达式的token，输出它们
    if matched_tokens < len(tokenizer.category_tokens):
        print("不符合正则表达式的类别token:")
        for token in tokenizer.category_tokens:
            if not re.match(r'\[category_[^\]]+\]', token):
                print(f"  {token}")

if __name__ == "__main__":
    analyze_tokens() 