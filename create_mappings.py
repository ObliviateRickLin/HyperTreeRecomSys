import json
from collections import defaultdict
from tqdm import tqdm
import os

def count_unique_users(reviews_file):
    """统计唯一用户数量"""
    unique_users = set()
    with open(reviews_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="统计用户数量"):
            try:
                review = json.loads(line)
                if 'user_id' in review:
                    unique_users.add(review['user_id'])
            except:
                continue
    return unique_users

def create_mappings(reviews_file, metadata_file, output_dir="mappings", resume=True):
    """创建用户和物品的token映射，支持断点续传"""
    print("开始创建映射...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否存在已保存的映射文件
    user_mapping_file = os.path.join(output_dir, "user_mapping.json")
    item_mapping_file = os.path.join(output_dir, "item_mapping.json")
    
    if resume and os.path.exists(user_mapping_file) and os.path.exists(item_mapping_file):
        print("发现已存在的映射文件，从第三阶段继续...")
        with open(user_mapping_file, 'r', encoding='utf-8') as f:
            user_mapping = json.load(f)
        with open(item_mapping_file, 'r', encoding='utf-8') as f:
            item_mapping = json.load(f)
        
        unique_users = set(user_mapping.keys())
        unique_items = set(item_mapping.keys())
        print(f"已加载 {len(unique_users):,} 个用户和 {len(unique_items):,} 个物品的映射")
    else:
    # 1. 收集所有unique用户
    print("\n收集用户ID...")
    unique_users = count_unique_users(reviews_file)
    print(f"找到 {len(unique_users):,} 个唯一用户")
    
    # 2. 收集所有unique物品
    print("\n收集物品ID...")
    unique_items = set()
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="统计物品"):
            try:
                item = json.loads(line)
                if 'asin' in item:
                    unique_items.add(item['asin'])
                if 'parent_asin' in item:
                    unique_items.add(item['parent_asin'])
            except:
                continue
    print(f"找到 {len(unique_items):,} 个唯一物品")
    
    # 3. 创建映射
    print("\n创建映射...")
        user_mapping = {}
        print("创建用户映射...")
        for i, user_id in enumerate(tqdm(sorted(unique_users), desc="用户映射")):
            user_mapping[user_id] = f"user_{i}"
            # 每10000个用户保存一次，防止内存不足
            if (i + 1) % 10000 == 0:
                with open(user_mapping_file, 'w', encoding='utf-8') as f:
                    json.dump(user_mapping, f)
        
        item_mapping = {}
        print("创建物品映射...")
        for i, item_id in enumerate(tqdm(sorted(unique_items), desc="物品映射")):
            item_mapping[item_id] = f"item_{i}"
            # 每10000个物品保存一次
            if (i + 1) % 10000 == 0:
                with open(item_mapping_file, 'w', encoding='utf-8') as f:
                    json.dump(item_mapping, f)
    
    # 4. 保存最终映射
    print("\n保存映射文件...")
    with open(user_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(user_mapping, f, indent=2)
    
    with open(item_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(item_mapping, f, indent=2)
    
    # 5. 分批创建反向映射
    print("\n创建反向映射...")
    reverse_user_mapping = {}
    reverse_item_mapping = {}
    
    batch_size = 100000
    print("创建用户反向映射...")
    items = list(user_mapping.items())
    for i in tqdm(range(0, len(items), batch_size), desc="用户反向映射"):
        batch = items[i:i+batch_size]
        for k, v in batch:
            reverse_user_mapping[v] = k
    
    print("创建物品反向映射...")
    items = list(item_mapping.items())
    for i in tqdm(range(0, len(items), batch_size), desc="物品反向映射"):
        batch = items[i:i+batch_size]
        for k, v in batch:
            reverse_item_mapping[v] = k
    
    with open(os.path.join(output_dir, "reverse_user_mapping.json"), 'w', encoding='utf-8') as f:
        json.dump(reverse_user_mapping, f, indent=2)
    
    with open(os.path.join(output_dir, "reverse_item_mapping.json"), 'w', encoding='utf-8') as f:
        json.dump(reverse_item_mapping, f, indent=2)
    
    print("\n映射文件已保存到", output_dir)
    return len(unique_users), len(unique_items)

if __name__ == "__main__":
    reviews_file = "amazon_books_processed/books_reviews.jsonl"
    metadata_file = "amazon_books_processed/books_metadata.jsonl"
    
    n_users, n_items = create_mappings(reviews_file, metadata_file)
    print(f"\n总结：")
    print(f"- 用户数量: {n_users:,}")
    print(f"- 物品数量: {n_items:,}") 