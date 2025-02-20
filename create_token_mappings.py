import json
import logging
from collections import defaultdict
from typing import Dict, Set, List, Tuple
from tqdm import tqdm
import os
import gc

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_intermediate_result(data: Dict, filename: str):
    """保存中间结果"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    logger.info(f"已保存中间结果到 {filename}")

def load_intermediate_result(filename: str) -> Dict:
    """加载中间结果"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"已加载中间结果 {filename}")
    return data

def process_metadata(metadata_file: str, chunk_size: int = 500) -> Tuple[Dict, Dict]:
    """第一阶段：处理metadata文件，获取item和category信息"""
    logger.info("开始处理metadata文件...")
    
    # 使用简化的数据结构
    item_ratings = {}  # asin -> rating
    category_counts = defaultdict(int)  # category -> count
    
    total_processed = 0
    with open(metadata_file, 'r', encoding='utf-8') as f:
        while True:
            chunk_data = []
            # 读取一个chunk的数据
            for _ in range(chunk_size):
                line = f.readline()
                if not line:
                    break
                chunk_data.append(line)
            
            if not chunk_data:
                break
                
            # 处理这个chunk的数据
            for line in chunk_data:
                try:
                    data = json.loads(line.strip())
                    asin = data.get('asin') or data.get('parent_asin')
                    if not asin:
                        continue
                    
                    # 只保存必要的信息
                    rating = float(data.get("average_rating", 0))
                    item_ratings[asin] = rating
                    
                    # 统计类别
                    if "categories" in data:
                        for cat in data["categories"]:
                            category_counts[cat] += 1
                            
                except Exception as e:
                    logger.warning(f"处理metadata记录时出错: {str(e)[:100]}")
                    continue
            
            total_processed += len(chunk_data)
            if total_processed % 10000 == 0:
                logger.info(f"已处理 {total_processed:,} 条metadata记录")
            
            # 清理内存
            chunk_data.clear()
            gc.collect()
    
    logger.info(f"metadata处理完成，共处理 {total_processed:,} 条记录")
    return item_ratings, category_counts

def process_reviews(reviews_file: str, valid_items: Set[str], chunk_size: int = 500) -> Dict[str, int]:
    """第二阶段：处理reviews文件，获取user和item的评论数"""
    logger.info("开始处理reviews文件...")
    
    user_review_counts = defaultdict(int)
    item_review_counts = defaultdict(int)
    
    total_processed = 0
    with open(reviews_file, 'r', encoding='utf-8') as f:
        while True:
            chunk_data = []
            # 读取一个chunk的数据
            for _ in range(chunk_size):
                line = f.readline()
                if not line:
                    break
                chunk_data.append(line)
            
            if not chunk_data:
                break
                
            # 处理这个chunk的数据
            for line in chunk_data:
                try:
                    review = json.loads(line.strip())
                    
                    # 统计用户评论数
                    if 'user_id' in review:
                        user_review_counts[review['user_id']] += 1
                    
                    # 统计商品评论数
                    asin = review.get('asin') or review.get('parent_asin')
                    if asin and asin in valid_items:
                        item_review_counts[asin] += 1
                        
                except Exception as e:
                    logger.warning(f"处理review记录时出错: {str(e)[:100]}")
                    continue
            
            total_processed += len(chunk_data)
            if total_processed % 10000 == 0:
                logger.info(f"已处理 {total_processed:,} 条review记录")
            
            # 清理内存
            chunk_data.clear()
            gc.collect()
    
    logger.info(f"reviews处理完成，共处理 {total_processed:,} 条记录")
    return user_review_counts, item_review_counts

def create_filtered_mappings(
    user_reviews: Dict[str, int],
    item_reviews: Dict[str, int],
    item_ratings: Dict[str, float],
    category_counts: Dict[str, int],
    min_user_reviews: int = 5,
    min_item_reviews: int = 10,
    min_item_rating: float = 3.5,
    min_category_items: int = 100
) -> Dict:
    """第三阶段：根据统计信息创建映射"""
    logger.info("开始创建映射...")
    
    # 1. 筛选用户
    logger.info("筛选用户...")
    valid_users = {
        user_id: f"user_{i}"
        for i, (user_id, review_count) in enumerate(
            sorted(
                user_reviews.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )
        if review_count >= min_user_reviews
    }
    gc.collect()
    
    # 2. 筛选商品
    logger.info("筛选商品...")
    valid_items = {
        item_id: f"item_{i}"
        for i, (item_id, review_count) in enumerate(
            sorted(
                item_reviews.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )
        if (review_count >= min_item_reviews and 
            item_ratings.get(item_id, 0) >= min_item_rating)
    }
    gc.collect()
    
    # 3. 筛选类别
    logger.info("筛选类别...")
    valid_categories = {
        category: f"category_{i}"
        for i, (category, count) in enumerate(
            sorted(
                category_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )
        if count >= min_category_items
    }
    gc.collect()
    
    # 4. 创建反向映射
    logger.info("创建反向映射...")
    reverse_user_mapping = {v: k for k, v in valid_users.items()}
    reverse_item_mapping = {v: k for k, v in valid_items.items()}
    reverse_category_mapping = {v: k for k, v in valid_categories.items()}
    
    # 5. 输出统计信息
    logger.info("\n=== 映射统计 ===")
    logger.info(f"用户数: {len(valid_users):,}/{len(user_reviews):,} "
               f"(筛选条件: >= {min_user_reviews}条评论)")
    logger.info(f"商品数: {len(valid_items):,}/{len(item_reviews):,} "
               f"(筛选条件: >= {min_item_reviews}条评论, >= {min_item_rating}平均评分)")
    logger.info(f"类别数: {len(valid_categories):,}/{len(category_counts):,} "
               f"(筛选条件: >= {min_category_items}个商品)")
    
    return {
        "user": {
            "id_to_token": valid_users,
            "token_to_id": reverse_user_mapping,
            "stats": {
                "total": len(user_reviews),
                "filtered": len(valid_users),
                "min_reviews": min_user_reviews
            }
        },
        "item": {
            "id_to_token": valid_items,
            "token_to_id": reverse_item_mapping,
            "stats": {
                "total": len(item_reviews),
                "filtered": len(valid_items),
                "min_reviews": min_item_reviews,
                "min_rating": min_item_rating
            }
        },
        "category": {
            "id_to_token": valid_categories,
            "token_to_id": reverse_category_mapping,
            "stats": {
                "total": len(category_counts),
                "filtered": len(valid_categories),
                "min_items": min_category_items
            }
        }
    }

def save_mappings(mappings: Dict, output_dir: str = "token_mappings"):
    """保存最终的映射文件"""
    logger.info(f"保存映射文件到 {output_dir} 目录...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 分别保存每个部分
    for key in ["user", "item", "category"]:
        logger.info(f"保存 {key} 相关的映射...")
        
        # 保存ID到token的映射
        with open(os.path.join(output_dir, f"{key}_id_to_token.json"), 'w', encoding='utf-8') as f:
            json.dump(mappings[key]["id_to_token"], f, indent=2)
        
        # 保存token到ID的映射
        with open(os.path.join(output_dir, f"{key}_token_to_id.json"), 'w', encoding='utf-8') as f:
            json.dump(mappings[key]["token_to_id"], f, indent=2)
        
        # 保存统计信息
        with open(os.path.join(output_dir, f"{key}_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(mappings[key]["stats"], f, indent=2)
        
        gc.collect()
    
    logger.info("所有映射文件保存完成")

def main():
    # 数据文件路径
    metadata_file = "amazon_books_processed/books_metadata.jsonl"
    reviews_file = "amazon_books_processed/books_reviews.jsonl"
    
    try:
        # 1. 处理metadata
        logger.info("=== 第一阶段：处理metadata ===")
        item_ratings, category_counts = process_metadata(metadata_file, chunk_size=500)
        
        # 保存中间结果
        save_intermediate_result(item_ratings, "item_ratings.json")
        save_intermediate_result(category_counts, "category_counts.json")
        
        # 清理内存
        valid_items = set(item_ratings.keys())
        del item_ratings
        gc.collect()
        
        # 2. 处理reviews
        logger.info("\n=== 第二阶段：处理reviews ===")
        user_reviews, item_reviews = process_reviews(reviews_file, valid_items, chunk_size=500)
        
        # 保存中间结果
        save_intermediate_result(user_reviews, "user_reviews.json")
        save_intermediate_result(item_reviews, "item_reviews.json")
        
        # 3. 重新加载所有数据并创建映射
        logger.info("\n=== 第三阶段：创建映射 ===")
        item_ratings = load_intermediate_result("item_ratings.json")
        category_counts = load_intermediate_result("category_counts.json")
        
        # 创建最终映射
        mappings = create_filtered_mappings(
            user_reviews, item_reviews, item_ratings, category_counts,
            min_user_reviews=5,
            min_item_reviews=10,
            min_item_rating=3.5,
            min_category_items=100
        )
        
        # 4. 保存最终结果
        save_mappings(mappings)
        
        # 5. 清理中间文件
        logger.info("\n清理中间文件...")
        for f in ["item_ratings.json", "category_counts.json", 
                 "user_reviews.json", "item_reviews.json"]:
            if os.path.exists(f):
                os.remove(f)
        
    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}")
        raise
    finally:
        # 清理内存
        gc.collect()

if __name__ == "__main__":
    main() 