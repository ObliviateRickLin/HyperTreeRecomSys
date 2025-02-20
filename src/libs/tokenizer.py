import re
import torch
import logging
import json
from typing import List, Tuple, Set
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import Counter

logger = logging.getLogger(__name__)

class AmazonDistilBertTokenizer:
    """
    在DistilBERT的Tokenizer上扩展 user_x, item_y, category_z 三种特殊token。
    基于实际数据中的ID构建特殊token。
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        metadata_file: str,
        reviews_file: str,
        **kwargs
    ):
        self.base_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # 从实际数据中收集ID
        user_ids, item_ids, categories = self._collect_ids(metadata_file, reviews_file)
        
        # 构造特殊token
        self.user_tokens = [f"user_{uid}" for uid in user_ids]
        self.item_tokens = [f"item_{iid}" for iid in item_ids]
        self.category_tokens = [f"category_{i}" for i, _ in enumerate(categories)]
        
        # 保存category到index的映射，用于后续转换
        self.category_to_idx = {cat: i for i, cat in enumerate(categories)}
        
        # 将这些token作为特殊token添加
        all_special_tokens = self.user_tokens + self.item_tokens + self.category_tokens
        logger.info(f"Adding {len(all_special_tokens)} special tokens...")
        logger.info(f"- User tokens: {len(self.user_tokens)}")
        logger.info(f"- Item tokens: {len(self.item_tokens)}")
        logger.info(f"- Category tokens: {len(self.category_tokens)}")
        
        # 创建特殊token到ID的映射
        special_tokens_dict = {
            "additional_special_tokens": all_special_tokens
        }
        num_added = self.base_tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added} special tokens to the vocabulary")
        
        # 创建特殊token的正则表达式模式
        self.special_token_pattern = '|'.join([
            r'user_\w+',     # 匹配 user_后跟任意字母数字
            r'item_\w+',     # 匹配 item_后跟任意字母数字
            r'category_\d+'  # 匹配 category_后跟数字
        ])
        self.special_token_regex = re.compile(f'({self.special_token_pattern})')

    def _collect_ids(self, metadata_file: str, reviews_file: str, min_interactions: int = 5) -> Tuple[Set[str], Set[str], Set[str]]:
        """从实际数据中收集所有unique的user_id、item_id和categories，并过滤低频项
        
        Args:
            metadata_file: metadata文件路径
            reviews_file: reviews文件路径
            min_interactions: 最小交互次数阈值，默认为5
        """
        # 用Counter来统计频次
        user_counter = Counter()
        item_counter = Counter()
        categories = set()
        
        # 1. 从reviews中收集user_id和item_id的频次
        logger.info("从reviews中收集交互统计...")
        total_lines = sum(1 for _ in open(reviews_file, 'r', encoding='utf-8'))
        with open(reviews_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="统计交互次数"):
                try:
                    review = json.loads(line.strip())
                    if 'user_id' in review:
                        user_counter[review['user_id']] += 1
                    if 'asin' in review:
                        item_counter[review['asin']] += 1
                except Exception as e:
                    logger.warning(f"处理review记录时出错: {e}")
                    continue
        
        # 2. 从metadata中补充item信息和收集categories
        logger.info("从metadata中收集补充信息...")
        total_lines = sum(1 for _ in open(metadata_file, 'r', encoding='utf-8'))
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="处理metadata"):
                try:
                    data = json.loads(line.strip())
                    # 收集categories
                    if 'categories' in data:
                        categories.update(data['categories'])
                except Exception as e:
                    logger.warning(f"处理metadata记录时出错: {e}")
                    continue
        
        # 3. 根据阈值过滤
        logger.info(f"根据交互阈值({min_interactions})过滤...")
        
        # 过滤前的统计
        logger.info("过滤前统计:")
        logger.info(f"- 总用户数: {len(user_counter):,}")
        logger.info(f"- 总物品数: {len(item_counter):,}")
        logger.info(f"- 总类别数: {len(categories):,}")
        
        # 过滤低频用户和物品
        filtered_users = {user for user, count in user_counter.items() if count >= min_interactions}
        filtered_items = {item for item, count in item_counter.items() if count >= min_interactions}
        
        # 过滤后的统计
        logger.info("过滤后统计:")
        logger.info(f"- 保留用户数: {len(filtered_users):,}")
        logger.info(f"- 保留物品数: {len(filtered_items):,}")
        logger.info(f"- 类别数: {len(categories):,}")
        
        # 计算过滤比例
        user_filter_ratio = len(filtered_users) / len(user_counter) * 100
        item_filter_ratio = len(filtered_items) / len(item_counter) * 100
        logger.info(f"过滤比例:")
        logger.info(f"- 用户保留比例: {user_filter_ratio:.2f}%")
        logger.info(f"- 物品保留比例: {item_filter_ratio:.2f}%")
        
        return filtered_users, filtered_items, sorted(categories)  # 对categories排序以保持一致性

    def get_category_token(self, category: str) -> str:
        """将原始category转换为对应的token"""
        if category in self.category_to_idx:
            return f"category_{self.category_to_idx[category]}"
        logger.warning(f"未知的category: {category}")
        return category

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
            if any(x in token for x in ['user_', 'item_', 'category_']):
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
        """代理到base_tokenizer的decode方法"""
        return self.base_tokenizer.decode(*args, **kwargs)

    def convert_tokens_to_ids(self, tokens):
        """确保特殊token能正确转换为id"""
        if isinstance(tokens, str):
            tokens = [tokens]
        return self.base_tokenizer.convert_tokens_to_ids(tokens)

    @property
    def mask_token_id(self):
        return self.base_tokenizer.mask_token_id

    @property
    def pad_token_id(self):
        return self.base_tokenizer.pad_token_id
