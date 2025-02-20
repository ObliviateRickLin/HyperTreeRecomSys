import json
from collections import defaultdict
from pathlib import Path
import logging
from tqdm import tqdm
import traceback
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_category_dict():
    """创建一个新的类别字典，确保包含_items列表和subcategories字典"""
    return {
        '_items': [],
        'subcategories': {}
    }

def insert_into_tree(tree, category_path, item_id):
    """将商品插入到类别树中"""
    logger.debug(f"插入商品 {item_id} 到路径: {' > '.join(category_path)}")
    current = tree
    for level in category_path:
        if not level or not level.strip():  # 跳过空类别
            logger.debug(f"跳过空类别")
            continue
            
        if level not in current['subcategories']:
            current['subcategories'][level] = create_category_dict()
        current = current['subcategories'][level]
    
    if item_id not in current['_items']:
        current['_items'].append(item_id)
        logger.debug(f"商品 {item_id} 已添加到类别 {' > '.join(category_path)}")

def count_items_in_category(tree):
    """计算每个类别下的商品数量"""
    try:
        counts = {}
        for category, data in tree['subcategories'].items():
            # 计算直接商品数
            direct_items = len(data['_items'])
            # 递归计算子类别
            sub_counts = count_items_in_category(data)
            # 计算总商品数（直接商品 + 子类别商品）
            total_items = direct_items
            for sub_data in sub_counts.values():
                total_items += sub_data['total_items']
            
            counts[category] = {
                'direct_items': direct_items,
                'total_items': total_items,
                'subcategories': sub_counts
            }
            logger.debug(f"类别 {category}: 直接商品 {direct_items}, 总商品 {total_items}")
        return counts
    except Exception as e:
        logger.error(f"计算类别数量时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

def print_tree(tree, indent=0, min_items=0):
    """打印类别树，只显示包含超过min_items个商品的类别"""
    try:
        sorted_categories = sorted(
            tree.items(),
            key=lambda x: x[1]['total_items'],
            reverse=True
        )
        for category, data in sorted_categories:
            total = data['total_items']
            direct = data['direct_items']
            if total >= min_items:
                print(f"{'  ' * indent}{category} ({direct}/{total})")
                if 'subcategories' in data:
                    print_tree(data['subcategories'], indent + 1, min_items)
    except Exception as e:
        logger.error(f"打印类别树时出错: {str(e)}")
        logger.error(traceback.format_exc())

def analyze_metadata(file_path):
    """分析元数据文件中的类别信息"""
    category_tree = create_category_dict()
    
    logger.info(f"开始分析文件: {file_path}")
    total_items = 0
    items_with_categories = 0
    last_report_time = time.time()
    start_time = time.time()
    
    try:
        # 直接处理数据，使用tqdm但不指定total
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="处理商品", unit="条"), 1):
                total_items += 1
                
                # 每100000行报告一次进度
                if total_items % 100000 == 0:
                    current_time = time.time()
                    elapsed = current_time - last_report_time
                    total_elapsed = current_time - start_time
                    speed = 100000 / elapsed
                    logger.info(f"已处理 {total_items:,} 条记录, "
                              f"处理速度: {speed:.0f} 条/秒, "
                              f"总用时: {total_elapsed:.0f} 秒")
                    last_report_time = current_time
                
                try:
                    item = json.loads(line.strip())
                    categories = item.get('categories', [])
                    
                    if categories:
                        items_with_categories += 1
                        item_id = item.get('parent_asin', '') or item.get('asin', '')
                        
                        # 处理类别路径
                        # 检查是否是二维数组
                        if categories and isinstance(categories[0], list):
                            # 如果是二维数组，处理每个类别路径
                            for category_path in categories:
                                insert_into_tree(category_tree, category_path, item_id)
                        else:
                            # 如果是一维数组，直接处理
                            insert_into_tree(category_tree, categories, item_id)
                
                except json.JSONDecodeError:
                    logger.warning(f"第 {line_num} 行: JSON解析错误")
                except Exception as e:
                    logger.warning(f"第 {line_num} 行处理出错: {str(e)}")
                    logger.warning(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    
    total_elapsed = time.time() - start_time
    logger.info(f"\n处理完成:")
    logger.info(f"总商品数: {total_items:,}")
    logger.info(f"有类别信息的商品数: {items_with_categories:,}")
    logger.info(f"总用时: {total_elapsed:.0f} 秒")
    logger.info(f"平均处理速度: {total_items/total_elapsed:.0f} 条/秒")
    
    # 计算统计信息
    try:
        logger.info("开始计算类别统计信息...")
        start_time = time.time()
        category_counts = count_items_in_category(category_tree)
        stats = {
            'total_items': total_items,
            'items_with_categories': items_with_categories,
            'category_counts': category_counts
        }
        elapsed = time.time() - start_time
        logger.info(f"类别统计信息计算完成，用时 {elapsed:.1f} 秒")
        return category_tree, stats
    except Exception as e:
        logger.error(f"计算统计信息时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def save_category_examples(tree, output_file, num_examples=5):
    """保存每个类别的示例商品"""
    try:
        examples = {}
        
        def collect_examples(node, path=[]):
            if not path:  # 跳过根节点
                for category, data in node['subcategories'].items():
                    collect_examples(data, [category])
                return
                
            # 保存当前类别的示例
            category_path = ' > '.join(path)
            if node['_items']:
                examples[category_path] = {
                    'items': node['_items'][:num_examples],
                    'total_items': len(node['_items'])
                }
            
            # 递归处理子类别
            for category, data in node['subcategories'].items():
                collect_examples(data, path + [category])
        
        collect_examples(tree)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        logger.info(f"类别示例已保存到: {output_file}")
        return True
    except Exception as e:
        logger.error(f"保存类别示例时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    # 使用正式数据文件
    metadata_file = "amazon_books_processed/books_metadata.jsonl"
    
    # 分析元数据
    logger.info("开始分析元数据...")
    result = analyze_metadata(metadata_file)
    
    if not result:
        logger.error("元数据分析失败")
        return
    
    tree, stats = result
    
    logger.info("\n=== 分析结果 ===")
    logger.info(f"总商品数: {stats['total_items']}")
    logger.info(f"有类别信息的商品数: {stats['items_with_categories']}")
    logger.info(f"类别覆盖率: {stats['items_with_categories']/stats['total_items']*100:.2f}%")
    
    logger.info("\n=== 类别层级 (类别名 (直接商品数/总商品数)) ===")
    print_tree(stats['category_counts'], min_items=1000)
    
    # 保存分析结果
    try:
        output_file = "category_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"\n详细分析结果已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存分析结果时出错: {str(e)}")
        logger.error(traceback.format_exc())
    
    # 保存类别示例
    examples_file = "category_examples.json"
    if save_category_examples(tree, examples_file):
        logger.info(f"类别示例已保存到: {examples_file}")
    else:
        logger.error("保存类别示例失败")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        logger.error(traceback.format_exc()) 