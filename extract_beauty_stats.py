import json
import gzip
from collections import defaultdict
from tqdm import tqdm
import os
import ast

# 数据文件路径
METADATA_FILE = "data/meta_Beauty_2014.json.gz"
REVIEWS_FILE = "data/reviews_Beauty_5.json.gz"
OUTPUT_FILE = "data/beauty_stats_result.json"

def extract_stats():
    """提取用户数量和类别信息"""
    print("开始提取统计信息...")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 初始化统计数据
    stats = {
        "users": set(),
        "products_with_reviews": set(),
        "categories": defaultdict(int),
        "total_reviews": 0,
        "total_products": 0,
        "total_products_with_reviews": 0,
        "products_with_metadata": 0
    }
    
    # 第一步：处理评论文件，提取用户信息和有评论的产品ID
    print(f"处理评论文件: {REVIEWS_FILE}")
    with gzip.open(REVIEWS_FILE, 'rt', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                data = json.loads(line.strip())
                stats["total_reviews"] += 1
                
                # 提取用户ID
                if "reviewerID" in data:
                    stats["users"].add(data["reviewerID"])
                
                # 提取产品ID
                if "asin" in data:
                    stats["products_with_reviews"].add(data["asin"])
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"处理评论时出错: {str(e)}")
    
    stats["total_products_with_reviews"] = len(stats["products_with_reviews"])
    print(f"找到 {stats['total_products_with_reviews']} 个有评论的产品")
    
    # 打印前5个有评论的产品ID用于调试
    print("前5个有评论的产品ID:")
    for asin in list(stats["products_with_reviews"])[:5]:
        print(asin)
    
    # 第二步：处理元数据文件，只统计有评论的产品
    print(f"\n处理元数据文件: {METADATA_FILE}")
    products_found = 0
    line_count = 0
    
    with gzip.open(METADATA_FILE, 'rt', encoding='utf-8') as f:
        for line in tqdm(f):
            line_count += 1
            try:
                # 使用ast.literal_eval处理单引号的字典
                data = ast.literal_eval(line.strip())
                stats["total_products"] += 1
                
                # 调试前几行
                if line_count <= 5:
                    print(f"元数据样例 {line_count}: {data.get('asin', 'No ASIN')}")
                
                # 只处理有评论的产品
                if "asin" in data and data["asin"] in stats["products_with_reviews"]:
                    products_found += 1
                    if products_found <= 5:
                        print(f"找到匹配产品: {data['asin']}")
                    
                    # 提取类别信息
                    if "categories" in data and data["categories"]:
                        for cat_list in data["categories"]:
                            if cat_list:  # 确保类别列表不为空
                                for i in range(len(cat_list)):
                                    # 记录每个层级的类别
                                    category_path = " > ".join(cat_list[:i+1])
                                    stats["categories"][category_path] += 1
            except SyntaxError as e:
                print(f"语法错误 (行 {line_count}): {str(e)}")
                print(f"问题数据: {line[:100]}...")
                if line_count <= 10:
                    continue
                else:
                    break
            except Exception as e:
                print(f"处理元数据时出错 (行 {line_count}): {str(e)}")
                print(f"问题数据: {line[:100]}...")
                if line_count <= 10:
                    continue
                else:
                    break
    
    stats["products_with_metadata"] = products_found
    print(f"在元数据中找到 {products_found} 个有评论的产品")
    
    # 转换结果为可序列化格式
    result = {
        "total_users": len(stats["users"]),
        "total_reviews": stats["total_reviews"],
        "total_products": stats["total_products"],
        "total_products_with_reviews": stats["total_products_with_reviews"],
        "products_with_metadata": stats["products_with_metadata"],
        "total_categories": len(stats["categories"]),
        "top_categories": sorted(stats["categories"].items(), key=lambda x: x[1], reverse=True)[:50]
    }
    
    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n统计结果已保存到: {OUTPUT_FILE}")
    print(f"总用户数: {result['total_users']}")
    print(f"总评论数: {result['total_reviews']}")
    print(f"元数据中的总产品数: {result['total_products']}")
    print(f"有评论的产品数: {result['total_products_with_reviews']}")
    print(f"有评论且有元数据的产品数: {result['products_with_metadata']}")
    print(f"总类别数: {result['total_categories']}")
    print("\n前10个最常见类别:")
    for i, (category, count) in enumerate(result["top_categories"][:10], 1):
        print(f"{i}. {category}: {count}个产品")

if __name__ == "__main__":
    extract_stats() 