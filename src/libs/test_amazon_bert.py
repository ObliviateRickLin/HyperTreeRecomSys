import logging
import torch
import random

from transformers import DistilBertConfig, DistilBertModel
from model import AmazonDistilBertBaseModel, AmazonDistilBertForMLM
from tokenizer import AmazonDistilBertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tokenizer():
    """测试tokenizer的功能"""
    logger.info("测试tokenizer...")

    # DistilBert 原始 vocab_size=30522，一般 hidden_dim=768
    config = DistilBertConfig(
        vocab_size=30522,
        dim=768,
        num_users=1000,
        num_items=5000,
        num_categories=100
    )

    # 初始化 tokenizer
    tokenizer = AmazonDistilBertTokenizer(
        vocab_file='./huggingface/models/distilbert-base-uncased/vocab.txt',
        num_users=config.num_users,
        num_items=config.num_items,
        num_categories=config.num_categories
    )

    # 测试文本
    texts = [
        "user_1 reviewed item_2 this is a great book",
        "user_3 bought item_4 category_1 excellent read",
        "user_5 commented on item_6 not recommended"
    ]
    try:
        # 批量编码
        input_ids, attention_mask = tokenizer.encode_batch(texts)
        logger.info(f"输入ID形状: {input_ids.shape}")
        logger.info(f"注意力掩码形状: {attention_mask.shape}")

        # 解码测试
        decoded_texts = []
        for ids in input_ids:
            tokens = tokenizer.convert_ids_to_tokens(ids)
            decoded_texts.append(" ".join(tokens))

        logger.info("\n解码结果示例:")
        for orig, decoded in zip(texts, decoded_texts):
            logger.info(f"原始文本: {orig}")
            logger.info(f"解码文本: {decoded}\n")

        logger.info("Tokenizer测试通过！")
        return True
    except Exception as e:
        logger.error(f"Tokenizer测试失败: {str(e)}")
        return False

def test_mlm_model():
    """测试MLM模型的功能"""
    logger.info("测试MLM模型...")

    # 创建配置
    config = DistilBertConfig(
        vocab_size=30522,
        dim=768,
        num_users=1000,
        num_items=5000,
        num_categories=100,
        initializer_range=0.02
    )

    # 1) 初始化 tokenizer，并增加 user/item/category token
    tokenizer = AmazonDistilBertTokenizer(
        vocab_file='./huggingface/models/distilbert-base-uncased/vocab.txt',
        num_users=config.num_users,
        num_items=config.num_items,
        num_categories=config.num_categories
    )

    # 2) 创建 distilbert_model
    distilbert_model = DistilBertModel.from_pretrained(
        './huggingface/models/distilbert-base-uncased',
        config=config,
        local_files_only=True
    )

    # 3) resize embedding
    new_size = len(tokenizer)  # tokenizer 新的词表大小(含 user/item/category)
    logger.info(f"Resizing DistilBERT embeddings to new size={new_size}")
    distilbert_model.resize_token_embeddings(new_size)

    # 4) 构造base_model & MLM模型
    base_model = AmazonDistilBertBaseModel(distilbert_model)
    model = AmazonDistilBertForMLM(base_model)

    # 创建测试输入
    batch_size = 2
    seq_length = 10

    # 这里故意制造一些超出原始vocab的index
    # config.vocab_size=30522 => if you add 1000 + 5000 + 100=6100 => 36622
    # 这里 index 30522 + 10, 代表 user_10
    # 这里 index 30522 + config.num_users + 20 => item_20
    # ...
    input_ids = torch.tensor([
        [101, 2003, 2023, 102, config.vocab_size, config.vocab_size + 10,
         config.vocab_size + config.num_users + 20,
         config.vocab_size + config.num_users + config.num_items + 5, 102, 0],
        [101, 2023, 2003, 102, config.vocab_size + 1,
         config.vocab_size + config.num_users + 30,
         config.vocab_size + config.num_users + config.num_items + 2, 2009, 102, 0]
    ])

    attention_mask = (input_ids != 0).long()

    # 创建MLM标签，-100 表示忽略计算损失的位置
    labels = input_ids.clone()
    labels[0, 2] = -100  # mask 掉位置 2
    labels[1, 6] = -100  # mask 掉位置 6

    try:
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss, logits = outputs
        logger.info(f"损失值: {loss.item()}")
        logger.info(f"输出logits形状: {logits.shape}")

        # 验证输出形状： (batch_size, seq_length, extended_vocab_size)
        # extended_vocab_size = 30522 + 1000 + 5000 + 100 = 36,622(或更多)
        assert logits.shape[0] == batch_size
        assert logits.shape[1] == seq_length
        assert logits.shape[2] == distilbert_model.embeddings.word_embeddings.num_embeddings

        assert not torch.isnan(loss).any()

        # 测试预测结果
        predictions = torch.argmax(logits, dim=-1)
        logger.info(f"预测结果形状: {predictions.shape}")

        logger.info("MLM模型测试通过！")
        return True
    except Exception as e:
        logger.error(f"MLM模型测试失败: {str(e)}")
        return False

def main():
    logger.info("=== 开始Amazon BERT测试 ===")

    tests = [
        ("Tokenizer测试", test_tokenizer),
        ("MLM模型测试", test_mlm_model)
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n执行{test_name}...")
        success = test_func()
        results.append((test_name, success))

    logger.info("\n=== 测试结果汇总 ===")
    all_passed = True
    for test_name, success in results:
        status = "通过" if success else "失败"
        logger.info(f"{test_name}: {status}")
        if not success:
            all_passed = False

    if all_passed:
        logger.info("\n所有测试通过！")
    else:
        logger.error("\n存在失败的测试，请检查日志获取详细信息。")

if __name__ == "__main__":
    main()
