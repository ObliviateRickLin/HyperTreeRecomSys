from src.libs.tokenizer import AmazonDistilBertTokenizer
import torch
import os

def test_special_tokens():
    # 1. 初始化我们的自定义tokenizer
    model_path = os.path.abspath("huggingface/models/distilbert-base-uncased")
    custom_tokenizer = AmazonDistilBertTokenizer(
        pretrained_model_name_or_path=model_path,
        num_users=1000,    # 假设我们有1000个用户
        num_items=1000,    # 1000个商品
        num_categories=50  # 50个分类
    )
    
    # 2. 构造一个测试文本（使用我们的示例数据格式）
    test_text = "user_123 posted a review about item_456. The book belongs to category_7 and category_12. user_123 gave item_456 a rating of 5 stars."
    
    # 3. 使用自定义tokenizer进行分词
    tokens = custom_tokenizer.tokenize_with_special_ids(test_text)
    print("\n=== 分词结果 ===")
    print(tokens)
    
    # 4. 检查特殊token是否被正确保留
    special_tokens = [token for token in tokens if any(x in token for x in ['user_', 'item_', 'category_'])]
    print("\n=== 特殊Token ===")
    print(special_tokens)
    
    # 5. 转换为ids并解码回文本
    input_ids = custom_tokenizer.convert_tokens_to_ids(tokens)
    decoded_text = custom_tokenizer.decode(input_ids)
    print("\n=== 解码后的文本 ===")
    print(decoded_text)
    
    # 6. 测试批量编码功能
    test_texts = [
        "user_123 reviewed item_456",
        "item_789 belongs to category_5",
        "user_999 liked item_111 in category_45"
    ]
    
    input_ids, attention_mask = custom_tokenizer.encode_batch(
        test_texts,
        max_length=32,
        padding=True,
        truncation=True
    )
    
    print("\n=== 批量编码结果 ===")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # 解码每个序列
    print("\n=== 批量解码结果 ===")
    for i, ids in enumerate(input_ids):
        text = custom_tokenizer.decode(ids[attention_mask[i] == 1])
        print(f"Text {i+1}: {text}")

if __name__ == "__main__":
    test_special_tokens() 