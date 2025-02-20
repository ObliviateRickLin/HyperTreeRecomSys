import json
from collections import defaultdict
from tqdm import tqdm

def count_categories(data, prefix=''):
    """递归统计类别数量"""
    total = 0
    for category, info in data.items():
        total += 1
        if info.get('subcategories'):
            total += count_categories(info['subcategories'], prefix + '  ')
    return total

# 读取分析文件
print("正在读取 category_analysis.json...")
with open('category_analysis.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 基本统计
print(f"\n=== 类别统计 ===")
print(f"总商品数: {data['total_items']:,}")
print(f"有类别信息的商品数: {data['items_with_categories']:,}")
print(f"类别覆盖率: {data['items_with_categories']/data['total_items']*100:.2f}%")

# 统计类别数量
category_count = count_categories(data['category_counts'])
print(f"\n总类别数: {category_count:,}")

# 打印顶级类别及其商品数
print("\n顶级类别统计:")
for category, info in sorted(data['category_counts'].items(), 
                           key=lambda x: x[1]['total_items'], 
                           reverse=True):
    print(f"{category}: {info['direct_items']:,}/{info['total_items']:,} (直接/总计)")

# 分析原始数据中的unique items
print("\n=== 商品统计 ===")
print("正在分析商品元数据...")

unique_items = set()
unique_authors = set()
stores = defaultdict(int)

with open('amazon_books_processed/books_metadata.jsonl', 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="分析商品"):
        try:
            item = json.loads(line)
            # 收集unique items (使用asin或parent_asin)
            if 'parent_asin' in item:
                unique_items.add(item['parent_asin'])
            if 'asin' in item:
                unique_items.add(item['asin'])
            
            # 收集作者信息
            if 'author' in item and isinstance(item['author'], dict):
                author_name = item['author'].get('name')
                if author_name:
                    unique_authors.add(author_name)
            
            # 收集store信息
            if 'store' in item:
                stores[item['store']] += 1
                
        except json.JSONDecodeError:
            continue
        except Exception as e:
            continue

print(f"\nUnique Items数量: {len(unique_items):,}")
print(f"Unique Authors数量: {len(unique_authors):,}")
print(f"Unique Stores数量: {len(stores):,}")

# 打印top stores
print("\nTop 10 Stores:")
for store, count in sorted(stores.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{store}: {count:,} 商品")

# 分析评论数据中的unique users
print("\n=== 用户统计 ===")
print("正在分析评论数据...")

unique_users = set()
user_reviews = defaultdict(int)
total_reviews = 0

with open('amazon_books_processed/books_reviews.jsonl', 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="分析评论"):
        try:
            review = json.loads(line)
            if 'user_id' in review:
                unique_users.add(review['user_id'])
                user_reviews[review['user_id']] += 1
            total_reviews += 1
        except json.JSONDecodeError:
            continue
        except Exception as e:
            continue

print(f"\nUnique Users数量: {len(unique_users):,}")
print(f"总评论数: {total_reviews:,}")
print(f"平均每用户评论数: {total_reviews/len(unique_users):.2f}")

# 计算用户评论分布
review_counts = defaultdict(int)
for count in user_reviews.values():
    review_counts[count] += 1

print("\n用户评论数分布:")
print("评论数\t用户数\t占比")
for count in sorted(review_counts.keys())[:10]:
    users = review_counts[count]
    percentage = users / len(unique_users) * 100
    print(f"{count}\t{users:,}\t{percentage:.2f}%") 