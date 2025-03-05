import re
import torch
import logging
import json
import ast
from typing import List, Tuple, Set
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import Counter
import gzip
from tokenizers import decoders

logger = logging.getLogger(__name__)

class AmazonDistilBertTokenizer:
    """
    在DistilBERT的Tokenizer上扩展 [user_x], [item_y], [category_z] 三种特殊token。
    基于实际数据中的ID构建特殊token。
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        metadata_file: str = None,
        reviews_file: str = None,
        max_users: int = None,
        max_items: int = None,
        max_categories: int = None,
        **kwargs
    ):
        self.base_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # 只有当提供了数据文件时才进行ID收集
        if metadata_file and reviews_file:
            # 从实际数据中收集ID
            user_ids, item_ids, categories = self._collect_ids(
                metadata_file, 
                reviews_file,
                max_users=max_users,
                max_items=max_items,
                max_categories=max_categories
            )
            
            # 构造特殊token - 使用方括号格式，与BERT保持一致
            self.user_tokens = [f"[user_{uid}]" for uid in user_ids]
            self.item_tokens = [f"[item_{iid}]" for iid in item_ids]
            
            # 为类别创建有意义的token名
            self.categories = categories  # 保存原始类别列表
            self.category_tokens = []
            self.category_to_idx = {}
            self.token_to_category = {}  # 添加token到类别的映射，用于检测重复
            
            # 首先检查是否有重复的原始类别名
            category_counter = Counter(categories)
            if any(count > 1 for count in category_counter.values()):
                duplicates = [cat for cat, count in category_counter.items() if count > 1]
                logger.warning(f"检测到 {len(duplicates)} 个重复的原始类别名: {duplicates[:5]}...")
            
            # 记录已使用的token，用于检测碰撞
            used_tokens = set()
            
            for i, category in enumerate(categories):
                # 使用'|'作为层级分隔符，这更适合类别层级结构且不易与其他符号混淆
                safe_category = category.replace(" > ", "|").replace(" ", "_")
                
                # 替换可能导致分词问题的特殊字符
                # 保留字母、数字、下划线和竖线，将其他字符替换为下划线
                safe_category = re.sub(r'[^\w\|]', '_', safe_category)
                
                # 避免连续的下划线，将多个下划线替换为一个
                safe_category = re.sub(r'_+', '_', safe_category)
                
                # 如果类别名太长，使用截断版本
                if len(safe_category) > 32:
                    parts = safe_category.split("|")
                    if len(parts) > 2:
                        # 保留第一部分和最后一部分
                        safe_category = f"{parts[0]}|{parts[-1]}"
                    # 如果还是太长，则进一步截断
                    if len(safe_category) > 32:
                        safe_category = safe_category[:29] + "..."
                
                # 使用方括号格式
                token = f"[category_{safe_category}]"
                
                # 检查是否存在重复token
                if token in used_tokens:
                    logger.warning(f"检测到重复的类别token: '{token}' 对应于类别 '{category}'")
                    # 在token中添加索引以避免重复
                    token = f"[category_{safe_category}_{i}]"
                
                self.category_tokens.append(token)
                self.category_to_idx[category] = i
                self.token_to_category[token] = category
                used_tokens.add(token)
            
            # 检查生成的tokens，确保没有重复
            if len(set(self.category_tokens)) != len(self.category_tokens):
                duplicates = [token for token, count in Counter(self.category_tokens).items() if count > 1]
                logger.error(f"生成了重复的类别token: {duplicates[:5]}...")
                raise ValueError("类别token生成出错：存在重复token")
                
            # 同时保存反向映射用于解码
            self.idx_to_category = {i: cat for cat, i in self.category_to_idx.items()}
            
            # 将这些token作为特殊token添加
            all_special_tokens = self.user_tokens + self.item_tokens + self.category_tokens
            logger.info(f"Adding {len(all_special_tokens)} special tokens...")
            logger.info(f"- User tokens: {len(self.user_tokens)}")
            logger.info(f"- Item tokens: {len(self.item_tokens)}")
            logger.info(f"- Category tokens: {len(self.category_tokens)}")
            
            # 创建特殊token到ID的映射 - 使用special_tokens确保它们不会被分词
            special_tokens_dict = {
                "additional_special_tokens": all_special_tokens
            }
            num_added = self.base_tokenizer.add_special_tokens(special_tokens_dict)
            logger.info(f"Added {num_added} special tokens to the vocabulary")
        else:
            # 初始化为空，等待from_pretrained加载
            self.user_tokens = []
            self.item_tokens = []
            self.category_tokens = []
            self.category_to_idx = {}
            self.idx_to_category = {}
            self.categories = []
            
        # 创建特殊token的正则表达式模式 - 更新为方括号格式
        self.special_token_pattern = '|'.join([
            r'\[user_\w+\]',        # 匹配 [user_后跟任意字母数字]
            r'\[item_\w+\]',        # 匹配 [item_后跟任意字母数字]
            r'\[category_[a-zA-Z0-9_\.\-]+\]' # 匹配 [category_后跟字母数字下划线和点]
        ])
        self.special_token_regex = re.compile(f'({self.special_token_pattern})')

    def _collect_ids(
        self, 
        metadata_file: str, 
        reviews_file: str, 
        min_interactions: int = 5,
        max_users: int = None,
        max_items: int = None,
        max_categories: int = None
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """从实际数据中收集所有unique的user_id、item_id和categories
        
        Args:
            metadata_file: metadata文件路径
            reviews_file: reviews文件路径
            min_interactions: 最小交互次数阈值，默认为5
            max_users: 最大用户数量限制，默认为None（不限制）
            max_items: 最大物品数量限制，默认为None（不限制）
            max_categories: 最大类别数量限制，默认为None（不限制）
        """
        from collections import Counter
        
        # 使用Counter来选择最频繁的用户和物品
        user_counter = Counter()
        item_counter = Counter()
        categories = set()
        
        # 1. 从reviews中收集user_id和item_id
        logger.info("从reviews中收集用户和物品ID频次...")
        
        # 使用批处理方式处理大文件，减少内存占用
        batch_size = 10000  # 每次处理10000行
        
        with gzip.open(reviews_file, 'rt', encoding='utf-8') as f:
            batch = []
            for line in tqdm(f, desc="统计交互数据"):
                batch.append(line)
                
                if len(batch) >= batch_size:
                    self._process_reviews_batch(batch, user_counter, item_counter)
                    batch = []  # 清空批次
                    
            # 处理最后一个不完整的批次
            if batch:
                self._process_reviews_batch(batch, user_counter, item_counter)
        
        # 根据频次筛选用户和物品
        sorted_users = user_counter.most_common()
        sorted_items = item_counter.most_common()
        
        logger.info(f"在评论数据中找到 {len(sorted_users)} 个不同用户和 {len(sorted_items)} 个不同物品")
        
        # 应用最大数量限制，保留最常见的
        if max_users is not None and max_users > 0:
            sorted_users = sorted_users[:max_users]
        if max_items is not None and max_items > 0:
            sorted_items = sorted_items[:max_items]
            
        # 转换为ID集合
        user_ids = {uid for uid, _ in sorted_users}
        item_ids = {iid for iid, _ in sorted_items}
        
        logger.info(f"保留 {len(user_ids)} 个用户和 {len(item_ids)} 个物品")
        
        # 2. 从metadata中收集类别信息，只处理在评论中出现的物品
        logger.info("从metadata中收集类别信息...")
        
        category_counter = Counter()  # 用于统计类别频次
        
        # 使用批处理方式处理元数据
        with gzip.open(metadata_file, 'rt', encoding='utf-8') as f:
            batch = []
            for line in tqdm(f, desc="处理metadata"):
                batch.append(line)
                
                if len(batch) >= batch_size:
                    self._process_metadata_batch(batch, item_ids, category_counter)
                    batch = []
                    
            # 处理最后一个批次
            if batch:
                self._process_metadata_batch(batch, item_ids, category_counter)
                
        # 根据频次选择类别，按照出现频率排序
        sorted_categories = category_counter.most_common()
        logger.info(f"共收集到 {len(sorted_categories)} 个不同类别")
        
        # 应用最大类别数限制
        if max_categories is not None and max_categories > 0:
            sorted_categories = sorted_categories[:max_categories]
            
        # 转换为类别集合
        categories = [cat for cat, _ in sorted_categories]
        
        logger.info("收集到的统计:")
        logger.info(f"- 用户数: {len(user_ids):,}")
        logger.info(f"- 物品数: {len(item_ids):,}")
        logger.info(f"- 类别数: {len(categories):,}")
        
        return user_ids, item_ids, categories
        
    def _process_reviews_batch(self, batch, user_counter, item_counter):
        """处理一批评论数据"""
        for line in batch:
            try:
                review = json.loads(line.strip())
                if 'reviewerID' in review:
                    user_counter[review['reviewerID']] += 1
                if 'asin' in review:
                    item_counter[review['asin']] += 1
            except Exception as e:
                logger.warning(f"处理review记录时出错: {e}")
                
    def _process_metadata_batch(self, batch, item_ids, category_counter):
        """处理一批元数据"""
        for line in batch:
            try:
                # 使用ast.literal_eval而不是json.loads来处理单引号格式的字典
                data = ast.literal_eval(line.strip())
                
                # 只处理在评论中出现的物品
                if 'asin' in data and data['asin'] in item_ids:
                    # 收集类别信息
                    if 'categories' in data and data['categories']:
                        # 扁平化多级类别列表
                        for cat_list in data['categories']:
                            if cat_list:  # 确保类别列表不为空
                                for i in range(len(cat_list)):
                                    # 添加每个层级的类别路径
                                    category_path = " > ".join(cat_list[:i+1])
                                    category_counter[category_path] += 1
            except Exception as e:
                logger.warning(f"处理metadata记录时出错: {e}")

    def get_category_token(self, category: str) -> str:
        """根据类别名获取对应的token
        
        Args:
            category: 原始类别名
            
        Returns:
            str: 对应的token，如果找不到则返回None
        """
        if category in self.category_to_idx:
            idx = self.category_to_idx[category]
            return self.category_tokens[idx] if idx < len(self.category_tokens) else None
        return None
        
    def get_category_from_token(self, token: str) -> str:
        """从token反向获取原始类别名
        
        Args:
            token: 类别token
            
        Returns:
            str: 原始类别名，如果找不到则返回None
        """
        # 首先使用token_to_category直接映射查找
        if hasattr(self, 'token_to_category') and token in self.token_to_category:
            return self.token_to_category[token]
        
        # 如果没有直接映射，使用索引方法
        if token in self.category_tokens:
            idx = self.category_tokens.index(token)
            return self.idx_to_category.get(idx)
            
        return None

    def tokenize_with_special_ids(self, text: str) -> List[str]:
        """
        对包含特殊token的文本进行分词。
        确保特殊token（user_x, item_y, category_z）作为整体被处理。
        """
        # 使用正则表达式分割文本，保留特殊token
        pieces = self.special_token_regex.split(text)
        tokens = []
        
        for piece in pieces:
            if self.special_token_regex.match(piece):
                # 如果是特殊token，直接添加
                tokens.append(piece)
            else:
                # 如果是普通文本，使用基础tokenizer处理
                if piece.strip():
                    tokens.extend(self.base_tokenizer.tokenize(piece.strip()))
        
        return tokens

    def encode_plus(self, text: str, **kwargs):
        """
        确保encode时也能正确处理特殊token
        """
        # 先进行特殊token的分词
        tokens = self.tokenize_with_special_ids(text)
        
        # 检查是否有未知的特殊token
        unknown_tokens = []
        for token in tokens:
            # 更新检查条件，匹配方括号格式
            if any(x in token for x in ['[user_', '[item_', '[category_']):
                if token not in self.base_tokenizer.get_added_vocab():
                    unknown_tokens.append(token)
        
        if unknown_tokens:
            logger.warning(f"发现未知的特殊token: {unknown_tokens}")
            
        # 将tokens转换为ids
        return self.base_tokenizer.encode_plus(
            tokens,
            is_split_into_words=True,
            **kwargs
        )

    def encode_batch(
        self,
        texts: List[str],
        max_length: int = None,
        padding: bool = True,
        truncation: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量编码，确保特殊token的处理一致性
        """
        encoded = self.base_tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def decode(self, *args, **kwargs):
        """使用WordPiece decoder进行解码"""
        return self.base_tokenizer.decode(*args, **kwargs)

    def convert_tokens_to_ids(self, tokens):
        """将token转换为id"""
        return self.base_tokenizer.convert_tokens_to_ids(tokens)

    @property
    def mask_token_id(self):
        return self.base_tokenizer.mask_token_id

    @property
    def pad_token_id(self):
        return self.base_tokenizer.pad_token_id
        
    def save_pretrained(self, save_directory: str):
        """保存tokenizer到指定目录
        
        Args:
            save_directory: 保存目录
        """
        import os
        import pickle
        
        # 确保目录存在
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存基础tokenizer
        self.base_tokenizer.save_pretrained(save_directory)
        
        # 去重类别token - 保持原始索引映射
        unique_category_tokens = []
        seen_tokens = set()
        
        for token in self.category_tokens:
            if token not in seen_tokens:
                unique_category_tokens.append(token)
                seen_tokens.add(token)
        
        # 保存去重后的token列表
        if len(unique_category_tokens) != len(self.category_tokens):
            logger.warning(f"去重前类别token数量: {len(self.category_tokens)}, 去重后: {len(unique_category_tokens)}")
            self.category_tokens = unique_category_tokens
        
        # 保存额外的特殊token信息
        special_tokens_info = {
            'user_tokens': self.user_tokens,
            'item_tokens': self.item_tokens,
            'category_tokens': self.category_tokens,
            'category_to_idx': self.category_to_idx,
            'idx_to_category': self.idx_to_category,
            'categories': self.categories,
            'special_token_pattern': self.special_token_pattern,
            'token_to_category': self.token_to_category  # 添加token到类别的映射
        }
        
        # 保存到文件
        with open(os.path.join(save_directory, 'amazon_special_tokens.pkl'), 'wb') as f:
            pickle.dump(special_tokens_info, f)
            
        # 创建类别映射的可读版本
        if self.categories:
            with open(os.path.join(save_directory, 'category_mapping.txt'), 'w', encoding='utf-8') as f:
                f.write("Token\tOriginal Category\n")
                for i, category in enumerate(self.categories):
                    token = self.category_tokens[i]
                    f.write(f"{token}\t{category}\n")
            
        logger.info(f"Tokenizer已保存到: {save_directory}")
        logger.info(f"- 保存了 {len(self.user_tokens)} 个用户token")
        logger.info(f"- 保存了 {len(self.item_tokens)} 个物品token")
        logger.info(f"- 保存了 {len(self.category_tokens)} 个类别token")
        
    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs):
        """从保存的目录加载tokenizer
        
        Args:
            pretrained_path: 预训练tokenizer目录
            
        Returns:
            AmazonDistilBertTokenizer: 加载的tokenizer
        """
        import os
        import pickle
        
        # 加载基础tokenizer
        base_tokenizer = AutoTokenizer.from_pretrained(pretrained_path, **kwargs)
        
        # 加载特殊token信息
        special_tokens_path = os.path.join(pretrained_path, 'amazon_special_tokens.pkl')
        if not os.path.exists(special_tokens_path):
            raise ValueError(f"找不到特殊token文件: {special_tokens_path}")
            
        with open(special_tokens_path, 'rb') as f:
            special_tokens_info = pickle.load(f)
            
        # 创建tokenizer实例
        tokenizer = cls.__new__(cls)
        tokenizer.base_tokenizer = base_tokenizer
        tokenizer.user_tokens = special_tokens_info['user_tokens']
        tokenizer.item_tokens = special_tokens_info['item_tokens']
        tokenizer.category_tokens = special_tokens_info['category_tokens']
        tokenizer.category_to_idx = special_tokens_info['category_to_idx']
        tokenizer.idx_to_category = special_tokens_info.get('idx_to_category', {})
        tokenizer.categories = special_tokens_info.get('categories', [])
        tokenizer.special_token_pattern = special_tokens_info.get('special_token_pattern')
        tokenizer.token_to_category = special_tokens_info.get('token_to_category', {})
        
        # 兼容旧版本，如果没有token_to_category，则构建它
        if not tokenizer.token_to_category and tokenizer.category_tokens:
            tokenizer.token_to_category = {}
            for i, token in enumerate(tokenizer.category_tokens):
                if i in tokenizer.idx_to_category:
                    tokenizer.token_to_category[token] = tokenizer.idx_to_category[i]
        
        # 重新构建category_to_idx以确保正确的双向映射
        if tokenizer.token_to_category and tokenizer.category_tokens:
            # 使用token_to_category作为权威来源
            tokenizer.category_to_idx = {category: tokenizer.category_tokens.index(token) 
                                        for token, category in tokenizer.token_to_category.items() 
                                        if token in tokenizer.category_tokens}
            
            # 更新idx_to_category
            tokenizer.idx_to_category = {idx: category for category, idx in tokenizer.category_to_idx.items()}
        
        # 做一个去重操作
        if len(set(tokenizer.category_tokens)) != len(tokenizer.category_tokens):
            unique_tokens = []
            seen = set()
            token_to_category = {}
            category_to_idx = {}
            idx_to_category = {}
            
            for i, token in enumerate(tokenizer.category_tokens):
                if token not in seen:
                    unique_tokens.append(token)
                    seen.add(token)
                    
                    # 更新映射
                    if token in tokenizer.token_to_category:
                        category = tokenizer.token_to_category[token]
                        new_idx = len(unique_tokens) - 1
                        token_to_category[token] = category
                        category_to_idx[category] = new_idx
                        idx_to_category[new_idx] = category
            
            logger.warning(f"从加载的tokenizer中去除了 {len(tokenizer.category_tokens) - len(unique_tokens)} 个重复的类别token")
            tokenizer.category_tokens = unique_tokens
            
            # 更新映射
            if token_to_category:
                tokenizer.token_to_category = token_to_category
                tokenizer.category_to_idx = category_to_idx
                tokenizer.idx_to_category = idx_to_category
        
        if not tokenizer.special_token_pattern:
            # 默认模式
            tokenizer.special_token_pattern = '|'.join([
                r'\[user_\w+\]',        # 匹配 [user_后跟任意字母数字]
                r'\[item_\w+\]',        # 匹配 [item_后跟任意字母数字]
                r'\[category_[\w\._]+]' # 匹配 [category_后跟字母数字下划线和点]
            ])
            
        tokenizer.special_token_regex = re.compile(f'({tokenizer.special_token_pattern})')
        
        logger.info(f"从 {pretrained_path} 加载tokenizer")
        logger.info(f"- 加载了 {len(tokenizer.user_tokens)} 个用户token")
        logger.info(f"- 加载了 {len(tokenizer.item_tokens)} 个物品token")
        logger.info(f"- 加载了 {len(tokenizer.category_tokens)} 个类别token")
        
        return tokenizer
