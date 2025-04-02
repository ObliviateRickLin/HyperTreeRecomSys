#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Amazon产品层次结构分类法模块
"""

import os
import sys
import json
import gzip
import ast
import logging
from collections import defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# 导入Taxonomy
from deeponto.onto import Taxonomy

# 定义固定路径
BASE_DIR = "/u/home/r/ricklin/HyperTreeRecomSys"
META_FILE = os.path.join(BASE_DIR, "data", "meta_Beauty_2014.json.gz")
REVIEWS_FILE = os.path.join(BASE_DIR, "data", "reviews_Beauty_5.json.gz")
OUTPUT_DIR = os.path.join(BASE_DIR, "hierarchical-encoder", "datasets", "amazon_beauty_dataset")

class AmazonTaxonomy(Taxonomy):
    """
    Amazon产品层次结构分类法
    
    简化版本：不区分节点类型，只保留必要的描述属性。
    """
    
    def __init__(self, meta_file_path=META_FILE, reviews_file_path=REVIEWS_FILE):
        """
        从Amazon元数据和评论数据初始化分类法
        
        Args:
            meta_file_path: Amazon元数据文件路径，默认使用预设路径
            reviews_file_path: Amazon评论数据文件路径，默认使用预设路径
        """
        # 记录数据源
        self.meta_file_path = meta_file_path
        self.reviews_file_path = reviews_file_path
        
        # 加载数据并构建边
        logger.info(f"从Amazon数据构建分类法...")
        categories, products, category_to_products = self._extract_hierarchy()
        edges = self._build_edges(categories, products, category_to_products)
        
        # 调用父类初始化
        super().__init__(edges=edges)
        
        # 设置节点描述属性
        self._set_node_descriptions(categories, products)
        
        logger.info(f"Amazon分类法构建完成：{len(self.nodes)}个节点，{len(self.edges)}条边")
    
    def _extract_hierarchy(self):
        """提取Amazon层次结构数据"""
        logger.info("提取Amazon层次结构数据...")
        categories = set()
        products = set()
        category_to_products = defaultdict(set)
        products_with_reviews = set()
        
        # 存储产品和类别的元数据信息
        self.product_metadata = {}
        
        # 1. 首先获取有评论的产品ID
        try:
            with gzip.open(self.reviews_file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        review = json.loads(line.strip())
                        if 'asin' in review:
                            products_with_reviews.add(review['asin'])
                    except:
                        continue
    
            logger.info(f"找到{len(products_with_reviews)}个有评论的产品")
        except Exception as e:
            logger.error(f"处理评论文件时出错: {str(e)}")
            raise
        
        # 2. 从元数据中提取产品和类别关系
        matched_products = 0
        products_with_categories = 0
        products_without_categories = 0
        
        try:
            with gzip.open(self.meta_file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        # 解析JSON或字典字符串
                        try:
                            item = ast.literal_eval(line.strip())
                        except:
                            item = json.loads(line.strip())
                        
                        # 处理匹配的产品
                        if 'asin' in item and item['asin'] in products_with_reviews:
                            product_id = item['asin']
                            matched_products += 1
                    
                            # 存储产品元数据
                            self.product_metadata[product_id] = {
                                'title': item.get('title', f"Product {product_id}"),
                                'description': item.get('description', ''),
                                'brand': item.get('brand', '')
                            }
                            
                            # 处理类别信息
                            if 'categories' in item and item['categories']:
                                products.add(product_id)
                                products_with_categories += 1
                        
                                # 处理每个类别路径
                                for path_list in item['categories']:
                                    if not path_list:
                                        continue
                            
                                    # 构建类别层次结构
                                    for i in range(len(path_list)):
                                        sub_path = " > ".join(path_list[:i+1])
                                        categories.add(sub_path)
                                        category_to_products[sub_path].add(product_id)
                    except Exception as e:
                        # 跳过处理有问题的行
                        continue
                        
                # 输出统计信息
                logger.info(f"元数据产品匹配: 匹配了{matched_products}个有评论的产品")
                logger.info(f"类别信息: {products_with_categories}个产品有类别信息, {products_without_categories}个产品无类别信息")
                logger.info(f"从元数据中找到{len(categories)}个类别路径，{len(products)}个产品")
        except Exception as e:
            logger.error(f"处理元数据文件时出错: {str(e)}")
            raise
        
        return categories, products, category_to_products
    
    def _build_edges(self, categories, products, category_to_products):
        """构建层次结构的边"""
        # 为所有ID添加统一前缀
        self.node_map = {}
        
        # 创建节点ID映射
        for product in products:
            self.node_map[product] = f"node_{product}"
        for category in categories:
            self.node_map[category] = f"node_{category}"
        
        edges = []
        
        # 添加类别之间的层次关系
        for category in categories:
            parts = category.split(" > ")
            if len(parts) > 1:
                for i in range(len(parts) - 1):
                    parent = " > ".join(parts[:i+1])
                    child = " > ".join(parts[:i+2])
                    parent_id = self.node_map[parent]
                    child_id = self.node_map[child]
                    edges.append((parent_id, child_id))
        
        # 添加产品与类别之间的关系
        for category, product_set in category_to_products.items():
            category_id = self.node_map[category]
            for product in product_set:
                product_id = self.node_map[product]
                edges.append((category_id, product_id))
                
        return edges
    
    def _set_node_descriptions(self, categories, products):
        """设置节点描述属性 - 使用完整的描述文本，不截断"""
        # 为所有节点设置描述
        for node_id in self.graph.nodes:
            original_id = None
            
            # 查找原始ID
            for orig, mapped in self.node_map.items():
                if mapped == node_id:
                    original_id = orig
                    break
            
            if original_id is None:
                continue
    
            # 判断是产品还是类别
            if original_id in products:
                # 产品节点 - 使用产品完整信息
                metadata = self.product_metadata.get(original_id, {})
                title = metadata.get('title', f"Product {original_id}")
                brand = metadata.get('brand', '')
                product_desc = metadata.get('description', '')
                
                # 构建产品描述，保留所有信息
                parts = []
                if brand:
                    parts.append(brand)
                parts.append(title)
                
                # 添加完整的产品描述，不截断
                if product_desc:
                    parts.append(product_desc)
                
                # 组合所有部分
                description = " - ".join(parts)
                    
                self.graph.nodes[node_id]['description'] = description
            else:
                # 类别节点 - 使用完整的路径并添加"商品种类"提示语
                self.graph.nodes[node_id]['description'] = f"Product Category: {original_id}"
    
    def get_node_id(self, original_id):
        """获取转换后的节点ID"""
        return self.node_map.get(original_id)
    
    def export_entity_lexicon(self, output_dir):
        """
        导出实体词典到entity_lexicon.json
        
        Args:
            output_dir: 输出目录路径
        """
        entity_lexicon = {}
        for node_id in self.graph.nodes:
            if 'description' in self.graph.nodes[node_id]:
                entity_lexicon[node_id] = self.graph.nodes[node_id]['description']
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'entity_lexicon.json')
        
        # 保存词典
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entity_lexicon, f, ensure_ascii=False, indent=2)
        
        logger.info(f"实体词典已保存到: {output_path}")
        logger.info(f"词典包含 {len(entity_lexicon)} 个实体")
        
        return output_path

if __name__ == "__main__":
    # 设置日志级别为INFO
    logging.getLogger(__name__).setLevel(logging.INFO)
    
    logger.info("=" * 60)
    logger.info("开始生成实体词典")
    logger.info("=" * 60)
    
    try:
        # 构建分类法
        taxonomy = AmazonTaxonomy()
        # 导出词典
        taxonomy.export_entity_lexicon(OUTPUT_DIR)
        
        logger.info("=" * 60)
        logger.info("实体词典生成完成")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"生成实体词典时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1) 