"""
Amazon Beauty数据集处理工具

这个脚本提供了处理Amazon Beauty数据集的功能，包括：
1. 下载元数据
2. 提取产品信息
3. 创建产品映射
4. 分析数据集

使用方法:
    py beauty_data_processor.py [command]

命令:
    download    - 下载元数据
    extract     - 提取产品信息
    categories  - 提取类别信息
    map         - 创建产品映射
    analyze     - 分析数据集
    all         - 执行所有操作
"""

import os
import sys
import gzip
import json
import time
import requests
from tqdm import tqdm
from collections import Counter

# 常量定义
METADATA_URLS = [
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz",
    "https://jmcauley.ucsd.edu/data/amazon/meta_Beauty.json.gz"
]
REVIEWS_FILE = "data/reviews_Beauty_5.json.gz"
DATA_DIR = "data"
METADATA_FILE = f"{DATA_DIR}/meta_Beauty.json.gz"
STATS_FILE = f"{DATA_DIR}/beauty_stats.json"
MAPPING_FILE = f"{DATA_DIR}/product_mapping.json"
CATEGORIES_FILE = f"{DATA_DIR}/beauty_categories.json"

def ensure_dir():
    """确保数据目录存在"""
    os.makedirs(DATA_DIR, exist_ok=True)

def download_metadata():
    """下载元数据文件"""
    ensure_dir()
    
    if os.path.exists(METADATA_FILE):
        print(f"元数据文件 {METADATA_FILE} 已存在，跳过下载")
        return True
    
    print("尝试下载美妆元数据...")
    
    for url in METADATA_URLS:
        print(f"尝试从 {url} 下载...")
        if download_file(url, METADATA_FILE):
            print(f"成功从 {url} 下载元数据到 {METADATA_FILE}")
            return True
    
    print("所有URL下载尝试均失败")
    return False

def download_file(url, filename, max_retries=3):
    """下载文件并显示进度条，支持重试"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 对错误状态码抛出异常
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            
            print(f"\n下载 {filename}...")
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            
            with open(filename, 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
                    
            progress_bar.close()
            
            # 验证文件大小
            if total_size != 0 and progress_bar.n != total_size:
                print("下载的文件大小与预期不符，可能下载不完整")
                if attempt < max_retries - 1:
                    print(f"尝试重新下载 (尝试 {attempt+2}/{max_retries})...")
                    time.sleep(2)  # 等待2秒后重试
                    continue
                return False
            
            return True
        except Exception as e:
            print(f"下载 {filename} 时出错: {str(e)}")
            if attempt < max_retries - 1:
                print(f"尝试重新下载 (尝试 {attempt+2}/{max_retries})...")
                time.sleep(2)  # 等待2秒后重试
            else:
                return False
    
    return False

def extract_metadata_info():
    """从元数据文件中提取所有可用的信息"""
    ensure_dir()
    
    if not os.path.exists(METADATA_FILE):
        print(f"元数据文件 {METADATA_FILE} 不存在，请先下载")
        return False
    
    try:
        print(f"\n从 {METADATA_FILE} 提取产品信息...")
        
        # 统计变量
        total_products = 0
        fields_count = Counter()
        brands = Counter()
        price_ranges = {
            "0-10": 0,
            "10-20": 0,
            "20-50": 0,
            "50-100": 0,
            "100+": 0
        }
        
        # 收集样本
        samples = []
        
        with gzip.open(METADATA_FILE, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="处理元数据"):
                total_products += 1
                try:
                    # 使用eval解析，因为之前的测试显示这种方法最有效
                    data = eval(line.strip())
                    
                    # 统计每个字段的出现次数
                    for field in data:
                        fields_count[field] += 1
                    
                    # 统计品牌
                    if 'brand' in data and data['brand']:
                        brands[data['brand']] += 1
                    
                    # 统计价格范围
                    if 'price' in data and isinstance(data['price'], (int, float)):
                        price = float(data['price'])
                        if price <= 10:
                            price_ranges["0-10"] += 1
                        elif price <= 20:
                            price_ranges["10-20"] += 1
                        elif price <= 50:
                            price_ranges["20-50"] += 1
                        elif price <= 100:
                            price_ranges["50-100"] += 1
                        else:
                            price_ranges["100+"] += 1
                    
                    # 收集样本
                    if len(samples) < 10 and 'brand' in data and 'price' in data:
                        samples.append(data)
                        
                except Exception:
                    continue
        
        # 准备统计数据
        stats = {
            "total_products": total_products,
            "fields_stats": {field: count for field, count in fields_count.most_common()},
            "top_brands": {brand: count for brand, count in brands.most_common(50)},
            "price_ranges": price_ranges,
            "samples": samples[:10]
        }
        
        # 保存统计数据
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 显示统计摘要
        print(f"\n成功处理 {total_products} 个产品")
        print(f"统计信息已保存到 {STATS_FILE}")
        
        print("\n字段统计:")
        for field, count in fields_count.most_common(10):
            print(f"  {field}: {count} ({count/total_products*100:.1f}%)")
        
        print("\n前10个最常见的品牌:")
        for brand, count in brands.most_common(10):
            print(f"  {brand}: {count} 个产品")
        
        print("\n价格范围分布:")
        for price_range, count in price_ranges.items():
            print(f"  ${price_range}: {count} 个产品 ({count/total_products*100:.1f}%)")
        
        return True
    except Exception as e:
        print(f"提取元数据信息时出错: {str(e)}")
        return False

def extract_categories():
    """从元数据文件中提取类别信息"""
    ensure_dir()
    
    if not os.path.exists(METADATA_FILE):
        print(f"元数据文件 {METADATA_FILE} 不存在，请先下载")
        return False
    
    try:
        print(f"\n从 {METADATA_FILE} 提取类别信息...")
        
        all_categories = set()
        category_hierarchy = {}
        product_count_by_category = {}
        
        with gzip.open(METADATA_FILE, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="处理元数据"):
                try:
                    data = eval(line.strip())
                    if 'categories' in data and data['categories']:
                        for category_path in data['categories']:
                            # 添加完整路径
                            if isinstance(category_path, list):
                                category_str = " > ".join(category_path)
                                all_categories.add(category_str)
                                
                                # 统计每个类别的产品数量
                                if category_str not in product_count_by_category:
                                    product_count_by_category[category_str] = 0
                                product_count_by_category[category_str] += 1
                                
                                # 构建层次结构
                                current_level = category_hierarchy
                                for category in category_path:
                                    if category not in current_level:
                                        current_level[category] = {}
                                    current_level = current_level[category]
                except Exception:
                    continue
        
        # 按产品数量排序类别
        sorted_categories = sorted(
            product_count_by_category.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 保存类别信息
        categories_data = {
            "total_categories": len(all_categories),
            "categories_list": sorted(list(all_categories)),
            "category_hierarchy": category_hierarchy,
            "top_categories": sorted_categories[:50],  # 前50个最常见的类别
            "product_count_by_category": product_count_by_category
        }
        
        with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(categories_data, f, indent=2, ensure_ascii=False)
            
        print(f"\n成功提取了 {len(all_categories)} 个类别路径")
        print(f"类别信息已保存到 {CATEGORIES_FILE}")
        
        # 显示一些统计信息
        print("\n前10个最常见的类别:")
        for i, (category, count) in enumerate(sorted_categories[:10], 1):
            print(f"{i}. {category} ({count}个产品)")
            
        return True
    except Exception as e:
        print(f"提取类别信息时出错: {str(e)}")
        return False

def create_product_mapping():
    """创建产品ID到元数据的映射，并与评论数据关联"""
    ensure_dir()
    
    if not os.path.exists(METADATA_FILE):
        print(f"元数据文件 {METADATA_FILE} 不存在，请先下载")
        return False
    
    if not os.path.exists(REVIEWS_FILE):
        print(f"评论文件 {REVIEWS_FILE} 不存在")
        return False
    
    try:
        print(f"\n创建产品ID映射...")
        
        # 从元数据中提取产品信息
        product_info = {}
        with gzip.open(METADATA_FILE, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取元数据"):
                try:
                    data = eval(line.strip())
                    if 'asin' in data:
                        asin = data['asin']
                        product_info[asin] = {
                            'title': data.get('title', ''),
                            'brand': data.get('brand', ''),
                            'price': data.get('price', 0),
                            'description': data.get('description', '')[:200] if data.get('description') else '',
                            'categories': data.get('categories', [])
                        }
                except Exception:
                    continue
        
        print(f"从元数据中提取了 {len(product_info)} 个产品信息")
        
        # 统计评论中的产品
        review_products = Counter()
        review_count = 0
        
        with gzip.open(REVIEWS_FILE, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取评论"):
                review_count += 1
                try:
                    data = json.loads(line.strip())
                    if 'asin' in data:
                        review_products[data['asin']] += 1
                except Exception:
                    continue
        
        print(f"评论数据中有 {len(review_products)} 个不同的产品，共 {review_count} 条评论")
        
        # 创建映射
        mapping = {
            'total_products_in_metadata': len(product_info),
            'total_products_in_reviews': len(review_products),
            'products_in_both': sum(1 for asin in review_products if asin in product_info),
            'top_reviewed_products': [
                {
                    'asin': asin,
                    'review_count': count,
                    'info': product_info.get(asin, {'title': 'Unknown', 'brand': 'Unknown'})
                }
                for asin, count in review_products.most_common(100)
                if asin in product_info
            ]
        }
        
        # 保存映射
        with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        print(f"\n映射信息已保存到 {MAPPING_FILE}")
        print(f"元数据和评论数据中共有 {mapping['products_in_both']} 个产品重叠")
        
        return True
    except Exception as e:
        print(f"创建产品映射时出错: {str(e)}")
        return False

def analyze_data():
    """分析数据集并生成报告"""
    ensure_dir()
    
    if not os.path.exists(STATS_FILE):
        print(f"统计文件 {STATS_FILE} 不存在，请先运行提取功能")
        return False
    
    if not os.path.exists(MAPPING_FILE):
        print(f"映射文件 {MAPPING_FILE} 不存在，请先运行映射功能")
        return False
    
    try:
        # 读取统计数据
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        # 读取映射数据
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        # 读取类别数据（如果存在）
        categories_data = None
        if os.path.exists(CATEGORIES_FILE):
            with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
                categories_data = json.load(f)
        
        # 生成报告
        report = f"""# Amazon Beauty 数据集分析报告

## 数据集概述

本报告分析了Amazon Beauty（美妆）产品数据集，包括评论数据和元数据。

### 评论数据集 (`{REVIEWS_FILE}`)
- **总评论数**: {mapping['total_products_in_reviews']} 条
- **不同产品数**: {len(mapping['top_reviewed_products'])} 个
- **格式**: JSON行格式，每行一条评论记录

### 元数据集 (`{METADATA_FILE}`)
- **总产品数**: {stats['total_products']} 个
- **格式**: JSON行格式，每行一个产品记录

## 数据字段分析

### 元数据字段覆盖率
"""
        
        # 添加字段统计
        for field, count in stats['fields_stats'].items():
            percentage = count / stats['total_products'] * 100
            report += f"- {field}: {percentage:.1f}%\n"
        
        report += """
### 评论数据字段
- reviewerID: 评论者ID
- asin: 产品ID
- reviewerName: 评论者名称
- helpful: 有用性评分 [有用票数, 总票数]
- reviewText: 评论文本
- overall: 总体评分 (1-5星)
- summary: 评论摘要
- unixReviewTime: Unix时间戳格式的评论时间
- reviewTime: 人类可读格式的评论时间

## 品牌分析

### 前10大品牌 (按产品数量)
"""
        
        # 添加品牌统计
        for i, (brand, count) in enumerate(list(stats['top_brands'].items())[:10], 1):
            report += f"{i}. {brand}: {count} 个产品\n"
        
        report += """
## 评论分析

### 最多评论的产品 (前5名)
"""
        
        # 添加评论最多的产品
        for i, product in enumerate(mapping['top_reviewed_products'][:5], 1):
            report += f"{i}. {product['asin']} ({product['info']['title']}): {product['review_count']} 条评论\n"
        
        report += f"""
## 数据集交叉分析

- 评论数据集中的所有{mapping['products_in_both']}个产品都在元数据集中有对应记录
- 元数据集中有{stats['total_products']}个产品，但只有{mapping['products_in_both']}个产品(约{mapping['products_in_both']/stats['total_products']*100:.1f}%)在评论数据集中出现
"""
        
        # 添加类别信息（如果存在）
        if categories_data:
            report += f"""
## 类别分析

- 总类别路径数: {categories_data['total_categories']}

### 前10个最常见的类别
"""
            for i, (category, count) in enumerate(categories_data['top_categories'][:10], 1):
                report += f"{i}. {category} ({count}个产品)\n"
        
        report += """
## 结论

Amazon Beauty数据集提供了丰富的美妆产品信息和用户评论数据，可用于多种推荐系统和数据分析任务。元数据包含了产品的详细信息，而评论数据则提供了用户对产品的评价和反馈。

这些数据可以用于：
1. 构建基于内容的推荐系统
2. 协同过滤推荐系统
3. 情感分析
4. 产品趋势分析
5. 品牌影响力研究

使用HyperTreeRecomSys项目的方法，可以将这些数据转换为适合深度学习模型的格式，构建更高效的推荐系统。
"""
        
        # 保存报告
        report_file = f"{DATA_DIR}/beauty_data_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n分析报告已生成并保存到 {report_file}")
        return True
    except Exception as e:
        print(f"生成分析报告时出错: {str(e)}")
        return False

def print_help():
    """打印帮助信息"""
    print("""
Amazon Beauty数据集处理工具

这个脚本提供了处理Amazon Beauty数据集的功能，包括：
1. 下载元数据
2. 提取产品信息
3. 创建产品映射
4. 分析数据集

使用方法:
    py beauty_data_processor.py [command]

命令:
    download    - 下载元数据
    extract     - 提取产品信息
    categories  - 提取类别信息
    map         - 创建产品映射
    analyze     - 分析数据集
    all         - 执行所有操作
""")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "download":
        download_metadata()
    elif command == "extract":
        extract_metadata_info()
    elif command == "categories":
        extract_categories()
    elif command == "map":
        create_product_mapping()
    elif command == "analyze":
        analyze_data()
    elif command == "all":
        if download_metadata():
            extract_metadata_info()
            extract_categories()
            create_product_mapping()
            analyze_data()
    else:
        print(f"未知命令: {command}")
        print_help()

if __name__ == "__main__":
    main() 