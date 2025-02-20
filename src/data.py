import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class AmazonBooksBertDataset(Dataset):
    """
    将 `books_metadata.jsonl` 和 `books_reviews.jsonl` 合并后,
    用于 BERT/DistilBert 做 MLM (Masked Language Modeling)。

    每条样本：
      - 取自一条 review (包含 user_id, asin, rating, title, text等),
      - 通过 asin 找到相应图书元数据 (title, author, desc, categories, price等),
      - 拼成一段**自然语言**文本串: 不再是 "key:value" 列表,
        而是连贯的句子/段落,
      - 然后对这段文本做随机mask => (input_ids, labels).

    训练时:
      - __getitem__() 里将review+metadata合并成一句/几句自然语言
      - encode => 进行mask => 返回 batch
      - 在collate_fn里做padding
    """

    def __init__(
        self,
        tokenizer,
        metadata_file,
        reviews_file,
        max_length=256,
        mlm_probability=0.15,
        chunk_size=10000,          # 每次加载的数据块大小
        max_samples=None,          # 最大样本数，None表示不限制
        min_rating=None,           # 最小评分过滤
        min_reviews=None,          # 最小评论数过滤
        seed=42                    # 随机种子
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.chunk_size = chunk_size
        self.max_samples = max_samples
        self.min_rating = min_rating
        self.min_reviews = min_reviews
        
        random.seed(seed)
        
        # 初始化数据存储
        self.asin2meta = {}
        self.reviews = []
        
        logger.info("开始加载数据...")
        self._load_data(metadata_file, reviews_file)
        
    def _load_data(self, metadata_file: str, reviews_file: str):
        """分块加载数据，并进行必要的过滤"""
        # 1. 首先扫描metadata获取合格的asin
        logger.info("扫描metadata文件...")
        valid_asins = set()
        meta_count = 0
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                meta_count += 1
                if meta_count % 100000 == 0:
                    logger.info(f"已处理 {meta_count} 条metadata记录")
                
                try:
                    data = json.loads(line.strip())
                    # 应用过滤条件
                    if self.min_rating and data.get('average_rating', 0) < self.min_rating:
                        continue
                    if self.min_reviews and data.get('rating_number', 0) < self.min_reviews:
                        continue
                    
                    # 获取asin - 注意：在示例数据中，asin在details中
                    asin = data.get('asin') or data.get('parent_asin') or data.get('details', {}).get('ISBN-10')
                    if asin:
                        valid_asins.add(asin)
                        self.asin2meta[asin] = data
                        logger.debug(f"添加了asin: {asin}")
                except Exception as e:
                    logger.warning(f"处理metadata记录时出错: {e}")
                    continue
                
                # 如果达到最大样本数，提前停止
                if self.max_samples and len(valid_asins) >= self.max_samples:
                    break
        
        logger.info(f"找到 {len(valid_asins)} 个有效的图书记录")
        logger.debug(f"有效的asins: {valid_asins}")
        
        # 2. 然后加载对应的reviews
        logger.info("加载reviews文件...")
        review_count = 0
        
        with open(reviews_file, 'r', encoding='utf-8') as f:
            for line in f:
                review_count += 1
                if review_count % 100000 == 0:
                    logger.info(f"已处理 {review_count} 条review记录")
                
                try:
                    review = json.loads(line.strip())
                    asin = review.get('asin') or review.get('parent_asin')
                    logger.debug(f"处理review的asin: {asin}")
                    
                    # 只保留有效asin的评论
                    if asin in valid_asins:
                        self.reviews.append(review)
                        logger.debug(f"添加了review，当前reviews数量: {len(self.reviews)}")
                except Exception as e:
                    logger.warning(f"处理review记录时出错: {e}")
                    continue
                
                # 如果达到最大样本数，提前停止
                if self.max_samples and len(self.reviews) >= self.max_samples:
                    break
        
        # 3. 随机采样（如果需要）
        if self.max_samples and len(self.reviews) > self.max_samples:
            self.reviews = random.sample(self.reviews, self.max_samples)
        
        logger.info(f"最终加载了 {len(self.reviews)} 条评论")

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review_data = self.reviews[idx]
        # 取 asin
        asin = review_data.get("asin") or review_data.get("parent_asin")
        if not asin:
            return self._empty_instance()

        meta = self.asin2meta.get(asin, None)
        if meta is None:
            return self._empty_instance()

        text_str = self._build_natural_sentence(review_data, meta)

        # encode => input_ids, attention_mask
        encoded = self.tokenizer.encode_plus(
            text_str,
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            padding=False  # 在collate_fn再统一pad
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # 随机mask
        input_ids, labels = self._random_mask(input_ids, labels, attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def _empty_instance(self):
        """
        如果遇到找不到asin或meta的情况, 
        返回一个最短的空样本
        """
        return {
            "input_ids": torch.zeros((1,), dtype=torch.long),
            "attention_mask": torch.zeros((1,), dtype=torch.long),
            "labels": torch.tensor([-100], dtype=torch.long),
        }

    def _build_natural_sentence(self, rv, meta):
        """
        将书本元数据 + 用户评论 => 一段(或多段)通顺的自然语言。
        使用特殊标记 user_x, item_y, category_z
        """
        # 从meta取字段
        title = meta.get("title", "")
        subtitle = meta.get("subtitle", "")
        author_name = meta.get("author", {}).get("name", "")
        average_rating = meta.get("average_rating", "")
        rating_number = meta.get("rating_number", "")
        desc_list = meta.get("description", [])
        description_str = " ".join(desc_list)
        features = meta.get("features", [])
        price = meta.get("price", "")
        categories = meta.get("categories", [])

        # 从review取字段
        user_id = rv.get("user_id", "")
        rv_rating = rv.get("rating", "")
        rv_title = rv.get("title", "")
        rv_text = rv.get("text", "")
        verified = rv.get("verified_purchase", False)
        item_id = rv.get("asin") or rv.get("parent_asin", "")

        # 处理特殊标记
        user_token = f"user_{user_id}"  # 如 user_USER123
        item_token = f"item_{item_id}"  # 如 item_1234567890

        sents = []

        # 1. 介绍用户对该书的评论（使用特殊标记）
        sents.append(
            f"{user_token} posted a review about {item_token} ('{title}'"
            + (f" {subtitle}" if subtitle else "")
            + f") by {author_name}."
        )

        # 2. 补充一些书信息
        sents.append(
            f"This book is described as: {description_str} "
            f"and is priced around ${price}. "
            f"It has an average rating of {average_rating} based on {rating_number} reviews."
        )

        # 3. 如果有features/categories
        if features:
            feats_str = ", ".join(features)
            sents.append(
                f"Some notable features are: {feats_str}."
            )
        if categories:
            # 处理分类信息，从大类到小类
            category_tokens = []
            for i, cat in enumerate(categories):
                category_tokens.append(f"category_{i}")
            
            # 构建层次化的分类描述
            if len(categories) > 1:
                main_cat = categories[0]  # 主分类（如Books）
                
                if len(categories) >= 3:
                    # 完整的层次结构：主分类 -> 领域分类 -> 具体主题
                    domain_cat = categories[1]    # 领域分类（如Computer Science）
                    topic_cats = categories[2:]   # 具体主题（如Programming, Algorithms等）
                    
                    cat_desc = (
                        f"This book belongs to the main category {category_tokens[0]} ({main_cat}), "
                        f"in the domain of {category_tokens[1]} ({domain_cat})"
                    )
                    
                    if topic_cats:
                        topic_desc = []
                        for i, (cat, token) in enumerate(zip(topic_cats, category_tokens[2:])):
                            topic_desc.append(f"{token} ({cat})")
                        cat_desc += f", specifically focusing on: {', '.join(topic_desc)}"
                    
                elif len(categories) == 2:
                    # 只有两级分类：主分类 -> 领域分类
                    domain_cat = categories[1]
                    cat_desc = (
                        f"This book belongs to the main category {category_tokens[0]} ({main_cat}), "
                        f"specifically in the field of {category_tokens[1]} ({domain_cat})"
                    )
                
                sents.append(cat_desc + ".")
            else:
                # 如果只有一个分类
                sents.append(
                    f"This book belongs to the general category {category_tokens[0]} ({categories[0]})."
                )

        # 4. 用户的评论信息（使用特殊标记）
        sents.append(
            f"{user_token} gave a rating of {rv_rating} stars to {item_token}, titled '{rv_title}'. "
            f"Review text: {rv_text}"
        )

        # 5. 是否verified
        if verified:
            sents.append(f"The purchase of {item_token} by {user_token} was verified.")
        else:
            sents.append(f"The purchase of {item_token} by {user_token} may not be verified.")

        # 把所有句子合并成1~2段
        full_text = " ".join(sents)
        return full_text

    def _random_mask(self, input_ids, labels, attention_mask):
        """
        简易随机mask, 直接演示用. 
        你也可使用官方 DataCollatorForLanguageModeling.
        """
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix = probability_matrix * attention_mask

        # 形成mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # 其余地方 label=-100
        labels[~masked_indices] = -100

        # Mask token
        mask_token_id = self.tokenizer.mask_token_id
        input_ids[masked_indices] = mask_token_id

        return input_ids, labels

    @staticmethod
    def collate_fn(batch):
        # batch是list, 内含N个dict => pad
        input_ids_list = [x["input_ids"] for x in batch]
        attn_list = [x["attention_mask"] for x in batch]
        labels_list = [x["labels"] for x in batch]

        max_len = max(seq.size(0) for seq in input_ids_list)
        padded_ids = []
        padded_attn = []
        padded_lbl = []

        for i in range(len(batch)):
            seq_len = input_ids_list[i].size(0)
            diff = max_len - seq_len
            pad_ids = torch.zeros(diff, dtype=torch.long)
            pad_attn = torch.zeros(diff, dtype=torch.long)
            pad_lbls = torch.full((diff,), -100, dtype=torch.long)

            new_ids = torch.cat([input_ids_list[i], pad_ids], dim=0)
            new_attn = torch.cat([attn_list[i], pad_attn], dim=0)
            new_lbls = torch.cat([labels_list[i], pad_lbls], dim=0)

            padded_ids.append(new_ids.unsqueeze(0))
            padded_attn.append(new_attn.unsqueeze(0))
            padded_lbl.append(new_lbls.unsqueeze(0))

        batch_ids = torch.cat(padded_ids, dim=0)     # [B, max_len]
        batch_attn = torch.cat(padded_attn, dim=0)   # [B, max_len]
        batch_lbls = torch.cat(padded_lbl, dim=0)    # [B, max_len]

        return batch_ids, batch_attn, batch_lbls
