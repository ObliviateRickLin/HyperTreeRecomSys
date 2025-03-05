import json
import gzip
import random
import os
import sys
import ast  # 用于处理Python字典字符串
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Tuple, Set
import logging
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse
from datetime import datetime
import re

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.libs.tokenizer import AmazonDistilBertTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AmazonBeautyMLMDataset(Dataset):
    """
    为Beauty数据集创建MLM训练数据。
    
    混合生成5种模板的数据:
    1. 历史交互记录: [user_X] has interacted with [item_Y1] [item_Y2]...
    2. 物品特征描述: The content of [item_Y] is: This is a [category_Z1] > [category_Z2]... product.
    3. 用户评论: [user_X] writes the review for [item_Y]: [review_text]
    4. 类别层次结构: [category_Z1] includes subcategories such as [category_Z1_Z2_1]...
    5. 多级类别产品关系: Products in [category_Z1] > [category_Z2]... include [item_Y1], [item_Y2]...
    
    每条样本通过以下流程处理:
    1. 随机选择一个模板类型
    2. 生成模板文本
    3. 编码并随机mask => (input_ids, labels)
    """

    def __init__(
        self,
        tokenizer,
        metadata_file,
        reviews_file,
        max_length=128,
        mlm_probability=0.15,
        max_samples=None,         # 最大样本数，None表示不限制
        min_rating=None,          # 最小评分过滤
        min_reviews_per_item=5,   # 每个物品的最小评论数
        min_items_per_user=5,     # 每个用户的最小物品数
        template_weights=None,    # 各模板权重，默认均匀分布
        seed=42                   # 随机种子
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.max_samples = max_samples
        self.min_rating = min_rating
        self.min_reviews_per_item = min_reviews_per_item
        self.min_items_per_user = min_items_per_user
        self.template_weights = template_weights or [0.2, 0.2, 0.2, 0.2, 0.2]
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 初始化数据存储
        self.item2meta = {}                # 物品ID到元数据的映射
        self.user2items = defaultdict(set) # 用户购买的物品集合
        self.item2users = defaultdict(set) # 物品被哪些用户购买
        self.item2reviews = defaultdict(list) # 物品的评论列表
        self.user2reviews = defaultdict(list) # 用户的评论列表
        self.category_hierarchy = defaultdict(set) # 类别层次结构
        self.category2items = defaultdict(set) # 类别包含的物品集合
        
        # 用于生成的过滤后实体集合
        self.valid_users = set()
        self.valid_items = set()
        self.valid_categories = set()
        
        # 生成的文本样本
        self.text_samples = []
        
        logger.info("开始加载和处理数据...")
        self._load_data(metadata_file, reviews_file)
        logger.info("生成训练样本...")
        self._generate_text_samples()
        
    def _load_data(self, metadata_file: str, reviews_file: str):
        """加载并处理metadata和reviews数据"""
        # 1. 处理评论数据，建立用户-物品交互关系
        logger.info("处理评论数据...")
        user_item_count = Counter()  # 统计每个用户的物品数量
        item_review_count = Counter() # 统计每个物品的评论数量
        
        # 检查文件是否为gzip格式
        reviews_open_func = gzip.open if reviews_file.endswith('.gz') else open
        review_mode = 'rt' if reviews_file.endswith('.gz') else 'r'
        
        with reviews_open_func(reviews_file, review_mode, encoding='utf-8') as f:
            for line in tqdm(f, desc="处理评论数据"):
                try:
                    review = json.loads(line.strip())
                    user_id = review.get('reviewerID')
                    item_id = review.get('asin')
                    rating = review.get('overall', 0)
                    
                    # 应用过滤条件
                    if self.min_rating and rating < self.min_rating:
                        continue
                    
                    # 记录用户-物品交互
                    if user_id and item_id:
                        self.user2items[user_id].add(item_id)
                        self.item2users[item_id].add(user_id)
                        user_item_count[user_id] += 1
                        item_review_count[item_id] += 1
                        
                        # 保存评论数据
                        self.item2reviews[item_id].append(review)
                        self.user2reviews[user_id].append(review)
                except Exception as e:
                    logger.warning(f"处理评论记录时出错: {str(e)}")
        
        # 2. 过滤掉交互次数过少的用户和物品
        logger.info("过滤用户和物品...")
        for user_id, count in user_item_count.items():
            if count >= self.min_items_per_user:
                self.valid_users.add(user_id)
        
        for item_id, count in item_review_count.items():
            if count >= self.min_reviews_per_item:
                self.valid_items.add(item_id)
        
        logger.info(f"过滤后有效用户数: {len(self.valid_users)}")
        logger.info(f"过滤后有效物品数: {len(self.valid_items)}")
        
        # 3. 处理元数据
        logger.info("处理元数据...")
        metadata_open_func = gzip.open if metadata_file.endswith('.gz') else open
        meta_mode = 'rt' if metadata_file.endswith('.gz') else 'r'
        
        valid_meta_count = 0
        error_count = 0
        with metadata_open_func(metadata_file, meta_mode, encoding='utf-8') as f:
            for line in tqdm(f, desc="处理元数据"):
                try:
                    # 确保行不为空
                    line = line.strip()
                    if not line:
                        continue
                        
                    # 尝试使用ast.literal_eval处理单引号格式的字典
                    try:
                        meta = ast.literal_eval(line)
                    except:
                        # 如果ast.literal_eval失败，尝试使用json.loads
                        meta = json.loads(line)
                    
                    item_id = meta.get('asin')
                    
                    # 只处理有效物品
                    if item_id in self.valid_items:
                        self.item2meta[item_id] = meta
                        valid_meta_count += 1
                        
                        # 处理类别信息
                        categories = meta.get('categories', [])
                        if categories:
                            # 处理多层级类别
                            for category_path in categories:
                                if isinstance(category_path, list):
                                    # 记录类别层次结构
                                    for i in range(len(category_path) - 1):
                                        parent = category_path[i]
                                        child = category_path[i + 1]
                                        self.category_hierarchy[parent].add(child)
                                    
                                    # 记录每个层级类别包含的物品
                                    for i, category in enumerate(category_path):
                                        self.category2items[category].add(item_id)
                                        self.valid_categories.add(category)
                                        
                                        # 构建完整路径作为多级类别
                                        if i > 0:
                                            full_path = " > ".join(category_path[:i+1])
                                            self.category2items[full_path].add(item_id)
                                            self.valid_categories.add(full_path)
                except json.JSONDecodeError:
                    # 忽略无效的JSON行
                    error_count += 1
                    if error_count <= 10:  # 只显示前10个错误
                        logger.warning(f"跳过无效的JSON行: {line[:50]}...")
                    elif error_count == 11:
                        logger.warning("后续JSON解析错误将不再显示...")
                except Exception as e:
                    error_count += 1
                    if error_count <= 10:
                        logger.warning(f"处理元数据记录时出错: {str(e)}")
                    elif error_count == 11:
                        logger.warning("后续处理错误将不再显示...")
        
        logger.info(f"加载了 {valid_meta_count} 个物品元数据，跳过了 {error_count} 个无效记录")
        logger.info(f"提取了 {len(self.valid_categories)} 个有效类别")
        
        # 4. 进一步过滤用户和物品，只保留有完整信息的
        logger.info("最终过滤...")
        self.valid_items = {item_id for item_id in self.valid_items if item_id in self.item2meta}
        self.valid_users = {user_id for user_id in self.valid_users 
                          if any(item_id in self.valid_items for item_id in self.user2items[user_id])}
        
        logger.info(f"最终有效用户数: {len(self.valid_users)}")
        logger.info(f"最终有效物品数: {len(self.valid_items)}")
        logger.info(f"最终有效类别数: {len(self.valid_categories)}")

    def _generate_text_samples(self):
        """生成混合模板的文本样本"""
        # 限制样本数量
        max_samples = self.max_samples or (len(self.valid_users) + len(self.valid_items) + len(self.valid_categories))
        
        # 计算每种模板的样本数
        template_counts = [int(max_samples * weight) for weight in self.template_weights]
        # 确保总和等于max_samples
        template_counts[-1] += max_samples - sum(template_counts)
        
        logger.info(f"生成 {max_samples} 个文本样本，模板分布: {template_counts}")
        
        # 1. 生成历史交互记录样本
        self._generate_interaction_history_samples(template_counts[0])
        
        # 2. 生成物品特征描述样本
        self._generate_item_description_samples(template_counts[1])
        
        # 3. 生成用户评论样本
        self._generate_review_samples(template_counts[2])
        
        # 4. 生成类别层次结构样本
        self._generate_category_hierarchy_samples(template_counts[3])
        
        # 5. 生成多级类别产品关系样本
        self._generate_category_item_samples(template_counts[4])
        
        # 打乱样本顺序
        random.shuffle(self.text_samples)
        logger.info(f"生成了 {len(self.text_samples)} 个混合文本样本")

    def _generate_interaction_history_samples(self, count):
        """生成用户交互历史样本"""
        logger.info(f"生成 {count} 个用户交互历史样本...")
        users = list(self.valid_users)
        
        for _ in tqdm(range(min(count, len(users))), desc="生成交互历史"):
            user_id = random.choice(users)
            items = list(self.user2items[user_id] & self.valid_items)
            
            if len(items) >= 3:  # 确保有足够的交互物品
                # 随机选择3-8个物品
                selected_items = random.sample(items, min(random.randint(3, 8), len(items)))
                
                # 构建文本
                items_str = " ".join([f"[item_{item_id}]" for item_id in selected_items])
                text = f"[user_{user_id}] has interacted with {items_str}."
                
                # 可选：添加购买时间或购买顺序信息
                if random.random() < 0.3:
                    text += f" These products were purchased within the last few months."
                
                self.text_samples.append(text)

    def _generate_item_description_samples(self, count):
        """生成物品特征描述样本"""
        logger.info(f"生成 {count} 个物品特征描述样本...")
        items = list(self.valid_items)
        
        for _ in tqdm(range(min(count, len(items))), desc="生成物品描述"):
            item_id = random.choice(items)
            meta = self.item2meta.get(item_id, {})
            
            # 提取物品信息
            title = meta.get('title', '')
            if title:
                # 清理标题文本
                title = ' '.join(title.strip().split('\n')).strip()
            
            description = meta.get('description', '')
            if isinstance(description, list):
                description = ' '.join(description)
            # 清理描述文本
            if description:
                description = ' '.join(description.strip().split('\n')).strip()
                # 限制描述长度
                description = description[:150]
            
            # 获取物品类别
            categories = []
            for cat_list in meta.get('categories', []):
                if isinstance(cat_list, list) and len(cat_list) > 0:
                    categories.append(cat_list)
            
            if categories:
                # 随机选择一个类别路径
                category_path = random.choice(categories)
                category_str = " > ".join(category_path)
                
                # 构建物品特征
                features = []
                if 'feature' in meta:
                    features = meta['feature']
                elif 'features' in meta:
                    features = meta['features']
                
                if isinstance(features, str):
                    features = [features]
                
                # 清理特性文本
                if features:
                    cleaned_features = []
                    for feature in features:
                        # 确保特性文本是字符串并清理
                        if isinstance(feature, str):
                            cleaned_feature = ' '.join(feature.strip().split('\n')).strip()
                            cleaned_features.append(cleaned_feature)
                    features = cleaned_features
                
                # 提取评分信息
                avg_rating = 0
                num_reviews = 0
                reviews = self.item2reviews.get(item_id, [])
                if reviews:
                    ratings = [r.get('overall', 0) for r in reviews]
                    avg_rating = sum(ratings) / len(ratings)
                    num_reviews = len(reviews)
                
                # 构建描述文本
                text = f"The content of [item_{item_id}] is: This is a [category_{self._safe_category_name(category_str)}] product. "
                
                # 添加标题
                if title:
                    text += f"The product name is '{title}'. "
                
                # 添加详细描述
                if description:
                    text += f"{description} "
                
                # 添加特性
                if features:
                    # 选择最多3个特性
                    selected_features = random.sample(features, min(3, len(features)))
                    # 确保每个特性不超过50个字符
                    selected_features = [f[:50] for f in selected_features]
                    features_str = ", ".join(selected_features)
                    text += f"It has features like {features_str}. "
                
                # 添加评分信息
                if num_reviews > 0:
                    text += f"User ratings average {avg_rating:.1f} stars based on {num_reviews} reviews."
                
                self.text_samples.append(text)

    def _generate_review_samples(self, count):
        """生成用户评论样本"""
        logger.info(f"生成 {count} 个用户评论样本...")
        # 收集所有有效的评论
        valid_reviews = []
        
        for user_id in self.valid_users:
            for review in self.user2reviews[user_id]:
                item_id = review.get('asin')
                if item_id in self.valid_items:
                    valid_reviews.append((user_id, item_id, review))
        
        # 随机选择评论
        selected_reviews = random.sample(valid_reviews, min(count, len(valid_reviews)))
        
        for user_id, item_id, review in tqdm(selected_reviews, desc="生成评论样本"):
            # 提取评论内容
            review_text = review.get('reviewText', '')
            rating = review.get('overall', 0)
            
            if review_text:
                # 清理评论文本，确保没有换行
                review_text = ' '.join(review_text.strip().split('\n'))
                # 限制评论长度，避免过长
                review_text = review_text[:200]
                
                # 获取物品类别
                meta = self.item2meta.get(item_id, {})
                category = ""
                for cat_list in meta.get('categories', []):
                    if isinstance(cat_list, list) and len(cat_list) > 0:
                        category = cat_list[-1]  # 使用最具体的类别
                    break
                
                # 构建评论文本
                text = f"[user_{user_id}] writes the review for [item_{item_id}]: {review_text}"
                
                # 添加类别信息
                if category:
                    text = f"[user_{user_id}] writes the review for [item_{item_id}], a [category_{self._safe_category_name(category)}] product: {review_text}"
                
                # 可选：添加评分信息
                if random.random() < 0.5:
                    text += f" The user rated this product {rating} out of 5 stars."
                
                self.text_samples.append(text)

    def _generate_category_hierarchy_samples(self, count):
        """生成类别层次结构样本"""
        logger.info(f"生成 {count} 个类别层次结构样本...")
        # 找出有子类别的父类别
        parent_categories = [cat for cat, children in self.category_hierarchy.items() if children]
        
        if not parent_categories:
            logger.warning("没有找到类别层次结构，跳过此类样本生成")
            return
        
        for _ in tqdm(range(min(count, len(parent_categories))), desc="生成类别层次"):
            parent = random.choice(parent_categories)
            children = list(self.category_hierarchy[parent])
            
            # 最多选择5个子类别
            selected_children = random.sample(children, min(5, len(children)))
            
            # 确保所有类别名称是单行的
            parent = ' '.join(parent.strip().split('\n')).strip()
            selected_children = [' '.join(child.strip().split('\n')).strip() for child in selected_children]
            
            # 构建子类别列表
            child_tokens = []
            for child in selected_children:
                # 构建完整路径
                full_path = f"{parent} > {child}"
                # 获取类别token
                child_token = f"[category_{self._safe_category_name(full_path)}]"
                child_tokens.append(child_token)
            
            # 构建层次结构文本
            parent_token = f"[category_{self._safe_category_name(parent)}]"
            
            # 限制子类别数量，确保文本不会过长
            if len(child_tokens) > 3:
                child_tokens = child_tokens[:3]
            
            children_str = ", ".join(child_tokens)
            text = f"{parent_token} includes subcategories such as {children_str}."
            
            self.text_samples.append(text)

    def _clean_text(self, text):
        """清理文本，确保没有换行符并限制长度"""
        if not text:
            return ""
        # 移除换行，合并多个空格
        text = ' '.join(text.strip().split('\n'))
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _generate_category_item_samples(self, count):
        """生成多级类别产品关系样本"""
        logger.info(f"生成 {count} 个多级类别产品关系样本...")
        # 找出有物品的类别
        categories_with_items = [(cat, items) for cat, items in self.category2items.items() 
                               if items and cat in self.valid_categories and " > " in cat]
        
        if not categories_with_items:
            logger.warning("没有找到多级类别物品关系，跳过此类样本生成")
            return
        
        # 筛选出物品数量足够的类别
        categories_with_items = [(cat, items) for cat, items in categories_with_items 
                               if len(items & self.valid_items) >= 3]
        
        if not categories_with_items:
            logger.warning("没有找到包含足够物品的多级类别，跳过此类样本生成")
            return
        
        for _ in tqdm(range(min(count, len(categories_with_items))), desc="生成类别物品关系"):
            category, all_items = random.choice(categories_with_items)
            
            # 清理类别名称
            category = self._clean_text(category)
            
            # 确保只使用有效物品
            items = list(all_items & self.valid_items)
            
            if len(items) >= 3:
                # 随机选择3-5个物品
                selected_items = random.sample(items, min(random.randint(3, 5), len(items)))
                
                # 获取类别token
                category_token = f"[category_{self._safe_category_name(category)}]"
                
                # 构建物品token列表
                item_tokens = [f"[item_{item_id}]" for item_id in selected_items]
                
                # 构建文本
                items_str = ", ".join(item_tokens)
                text = f"Products in {category_token} category include {items_str}."
                
                # 可选：添加额外信息
                if random.random() < 0.3:
                    text += f" These products are popular in the beauty market."
                
                self.text_samples.append(text)

    def _safe_category_name(self, category: str) -> str:
        """从类别名构建安全的token名
        
        Args:
            category: 类别名
            
        Returns:
            str: 符合tokenizer格式的类别名
        """
        # 清理输入文本
        category = self._clean_text(category)
        
        # 首先尝试直接从tokenizer获取
        if hasattr(self, 'tokenizer'):
            safe_name = self.tokenizer.get_category_token(category)
            if safe_name:
                # 去掉前缀和后缀
                safe_name = safe_name.replace("[category_", "").replace("]", "")
                return safe_name
            
            # 如果未找到完全匹配，尝试寻找类似的类别
            for cat in self.tokenizer.category_to_idx.keys():
                if category in cat or cat in category:
                    token = self.tokenizer.get_category_token(cat)
                    if token:
                        simplified_name = token.replace("[category_", "").replace("]", "")
                        return simplified_name
            
            # 如果还找不到，返回"Beauty"作为默认类别
            if "[category_Beauty]" in self.tokenizer.category_tokens:
                return "Beauty"
        
        # 备选方案：自己处理类别名
        safe_name = category.replace(" > ", "_").replace(" ", "_")
        safe_name = re.sub(r'[^\w]', '_', safe_name)
        safe_name = re.sub(r'_+', '_', safe_name)  # 替换连续的下划线
        
        # 如果类别名太长，进行截断
        if len(safe_name) > 32:
            parts = safe_name.split("_")
            if len(parts) > 3:
                # 保留第一部分和最后两部分
                safe_name = f"{parts[0]}_.._{parts[-2]}_{parts[-1]}"
        
        return safe_name

    def __len__(self):
        return len(self.text_samples)

    def __getitem__(self, idx):
        text = self.text_samples[idx]

        # encode => input_ids, attention_mask
        encoded = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # 随机mask一部分token
        input_ids, labels = self._random_mask(input_ids, labels, attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def _random_mask(self, input_ids, labels, attention_mask):
        """随机mask输入tokens的一部分"""
        # 创建概率矩阵进行随机mask
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        
        # 不对特殊token进行mask
        special_tokens_mask = [
            self.tokenizer.base_tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
            for val in [input_ids]
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        # 不对padding token进行mask
        if hasattr(self.tokenizer.base_tokenizer, "pad_token_id"):
            padding_mask = input_ids.eq(self.tokenizer.base_tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        # 生成mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 只对被选择的索引位置修改labels，其他位置设为-100，在损失计算时会被忽略
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

    @staticmethod
    def collate_fn(batch):
        """合并batch数据"""
        input_ids = torch.stack([example["input_ids"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        labels = torch.stack([example["labels"] for example in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def create_mlm_training_data(
    tokenizer,
    metadata_file,
    reviews_file,
    output_dir,
    max_samples=100000,
    max_length=128,
    train_ratio=0.9,
    seed=42,
    ensure_all_tokens=True  # 新增参数，确保所有token都被覆盖
):
    """
    创建MLM训练数据并保存为文件
    
    Args:
        tokenizer: AmazonDistilBertTokenizer实例
        metadata_file: 元数据文件路径
        reviews_file: 评论数据文件路径
        output_dir: 输出目录
        max_samples: 最大样本数
        max_length: 最大序列长度
        train_ratio: 训练集比例
        seed: 随机种子
        ensure_all_tokens: 是否确保所有token都被覆盖
    """
    logger.info("创建MLM训练数据...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化所有样本列表
    all_samples = []
    token_coverage_samples = []  # 专门用于确保token覆盖的样本
    
    # 创建数据集实例
    logger.info("初始化数据集...")
    dataset = AmazonBeautyMLMDataset(
        tokenizer=tokenizer,
        metadata_file=metadata_file,
        reviews_file=reviews_file,
        max_length=max_length,
        max_samples=max_samples,
        seed=seed
    )
    
    # 如果需要确保所有token都被覆盖
    if ensure_all_tokens:
        logger.info("确保所有token都被覆盖...")
        
        # 1. 用户token覆盖
        logger.info(f"生成用户token样本 ({len(tokenizer.user_tokens)}个)...")
        for user_token in tqdm(tokenizer.user_tokens, desc="生成用户token样本"):
            # 提取用户ID
            user_id = user_token.replace("[user_", "").replace("]", "")
            # 获取用户的所有交互历史
            user_items = list(dataset.user2items.get(user_id, set()) & dataset.valid_items)
            if user_items:
                # 使用所有交互物品
                items_str = " ".join([f"[item_{item}]" for item in user_items])
                # 生成包含完整交互历史的样本
                sample = f"{user_token} has interacted with {items_str}."
                token_coverage_samples.append(sample)
        
        # 2. 物品token覆盖
        logger.info(f"生成物品token样本 ({len(tokenizer.item_tokens)}个)...")
        for item_token in tqdm(tokenizer.item_tokens, desc="生成物品token样本"):
            # 提取物品ID
            item_id = item_token.replace("[item_", "").replace("]", "")
            # 获取物品元数据和所有购买用户
            meta = dataset.item2meta.get(item_id, {})
            item_users = list(dataset.item2users.get(item_id, set()) & dataset.valid_users)
            
            if meta and item_users:
                # 获取物品类别
                categories = []
                for cat_list in meta.get('categories', []):
                    if isinstance(cat_list, list) and cat_list:
                        # 获取完整的类别层次结构
                        for i in range(len(cat_list)):
                            # 构建从根到当前层级的完整路径
                            current_path = cat_list[:i+1]
                            full_path = " > ".join(current_path)
                            category_token = f"[category_{dataset._safe_category_name(full_path)}]"
                            categories.append(category_token)
                
                # 获取物品评分信息
                reviews = dataset.item2reviews.get(item_id, [])
                if reviews:
                    ratings = [r.get('overall', 0) for r in reviews]
                    avg_rating = sum(ratings) / len(ratings)
                    num_reviews = len(reviews)
                    
                    # 生成包含所有用户信息的样本
                    users_str = " ".join([f"[user_{user}]" for user in item_users])
                    
                    # 生成包含完整类别层次和用户列表的样本
                    if categories:
                        # 使用完整的类别层次结构
                        category_hierarchy = " > ".join(categories)
                        sample = f"{item_token} belongs to category hierarchy: {category_hierarchy}, with {avg_rating:.1f} average rating based on {num_reviews} reviews. It has been purchased by the following users: {users_str}."
                    else:
                        sample = f"{item_token} has received {avg_rating:.1f} average rating from {num_reviews} users. It has been purchased by: {users_str}."
                    token_coverage_samples.append(sample)
    
    # 生成自然分布样本
    logger.info("生成自然分布样本...")
    natural_samples_count = max(0, max_samples - len(token_coverage_samples)) if max_samples else 10000
    
    # 添加自然分布样本，并确保格式化一致
    for sample in dataset.text_samples:
        # 清理和规范化样本文本
        cleaned_sample = _clean_and_normalize_text(sample)
        all_samples.append(cleaned_sample)
    
    # 打乱自然分布样本
    random.seed(seed)
    random.shuffle(all_samples)
    
    # 计算训练集和验证集大小
    dataset_size = len(all_samples)
    train_size = int(dataset_size * train_ratio)
    
    # 分割自然分布样本为训练集和验证集
    train_texts = all_samples[:train_size]
    val_texts = all_samples[train_size:]
    
    # 清理并规范化token覆盖样本
    token_coverage_samples = [_clean_and_normalize_text(sample) for sample in token_coverage_samples]
    
    # 将token覆盖样本添加到训练集
    train_texts = token_coverage_samples + train_texts
    
    logger.info(f"总样本数: {len(train_texts) + len(val_texts)}")
    logger.info(f"训练集样本数: {len(train_texts)} (包含 {len(token_coverage_samples)} 个token覆盖样本)")
    logger.info(f"验证集样本数: {len(val_texts)}")
    
    # 保存为文本文件
    train_path = os.path.join(output_dir, "train_mlm.txt")
    val_path = os.path.join(output_dir, "val_mlm.txt")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for text in train_texts:
            f.write(text + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for text in val_texts:
            f.write(text + "\n")
    
    logger.info(f"保存了 {len(train_texts)} 个训练样本到 {train_path}")
    logger.info(f"保存了 {len(val_texts)} 个验证样本到 {val_path}")
    
    return train_path, val_path

def _clean_and_normalize_text(text):
    """
    清理和规范化文本，确保没有多行问题，但保留所有内容
    
    Args:
        text: 输入文本
        
    Returns:
        str: 清理后的单行文本
    """
    if not text:
        return ""
    
    # 1. 将多行文本合并为单行
    text = ' '.join(text.strip().split('\n'))
    
    # 2. 替换连续的空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 3. 确保特殊token（如[user_x]）前后有空格
    text = re.sub(r'(\S)(\[(?:user|item|category)_)', r'\1 \2', text)
    text = re.sub(r'(\])(\S)', r'\1 \2', text)
    
    return text.strip()

def main():
    """
    命令行入口点，用于生成Amazon Beauty数据集的MLM训练数据
    """
    parser = argparse.ArgumentParser(
        description="生成Amazon Beauty数据集的MLM训练数据"
    )
    
    # 数据参数
    parser.add_argument(
        "--metadata_file", 
        type=str, 
        default="data/meta_Beauty_2014.json.gz",
        help="Amazon Beauty元数据文件路径"
    )
    parser.add_argument(
        "--reviews_file", 
        type=str, 
        default="data/reviews_Beauty_5.json.gz",
        help="Amazon Beauty评论文件路径"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default="data/beauty_tokenizer",
        help="已初始化的tokenizer路径"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/mlm_data",
        help="输出目录，用于保存生成的训练数据"
    )
    
    # 生成参数
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=100000,
        help="生成的最大样本数"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=128,
        help="每个样本的最大token长度"
    )
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=0.9,
        help="训练集比例，剩余为验证集"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--ensure_all_tokens", 
        action="store_true",
        help="确保所有token都被覆盖（会生成大量样本）"
    )
    
    args = parser.parse_args()
    
    # 记录开始时间
    start_time = datetime.now()
    logger.info(f"开始生成MLM训练数据: {start_time}")
    
    # 加载tokenizer
    logger.info(f"从 {args.tokenizer_path} 加载tokenizer")
    tokenizer = AmazonDistilBertTokenizer.from_pretrained(args.tokenizer_path)
    
    # 创建数据目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成训练数据
    logger.info("开始生成MLM训练数据")
    train_path, val_path = create_mlm_training_data(
        tokenizer=tokenizer,
        metadata_file=args.metadata_file,
        reviews_file=args.reviews_file,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        seed=args.seed,
        ensure_all_tokens=args.ensure_all_tokens
    )
    
    # 记录结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"MLM训练数据生成完成！耗时: {duration}")
    logger.info(f"训练数据保存在: {train_path}")
    logger.info(f"验证数据保存在: {val_path}")

if __name__ == "__main__":
    main()
