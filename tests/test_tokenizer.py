import logging
import re
import os
import sys
import random
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.libs.tokenizer import AmazonDistilBertTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 确保日志输出到标准输出
    ]
)
logger = logging.getLogger(__name__)

def test_tokenizer_special_tokens():
    """测试tokenizer的特殊token功能"""
    
    print("\n" + "="*70)
    print("Testing Tokenizer Special Token Functionality")
    print("="*70)
    
    # 加载保存的tokenizer
    tokenizer_path = "data/beauty_tokenizer"
    
    if not os.path.exists(tokenizer_path):
        print(f"Error: tokenizer path does not exist: {tokenizer_path}")
        return
    
    try:
        print(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = AmazonDistilBertTokenizer.from_pretrained(tokenizer_path)
        
        # 1. 基础统计信息
        print(f"\n1. Basic Statistics")
        print(f"Number of user tokens: {len(tokenizer.user_tokens)}")
        print(f"Number of item tokens: {len(tokenizer.item_tokens)}")
        print(f"Number of category tokens: {len(tokenizer.category_tokens)}")
        
        # 2. 测试用户token
        print(f"\n2. User Token Tests")
        # 随机选择5个用户token
        if tokenizer.user_tokens:
            user_samples = random.sample(tokenizer.user_tokens, min(5, len(tokenizer.user_tokens)))
            print(f"Random user token samples: {user_samples}")
            
            # 测试编码和解码
            for user_token in user_samples:
                # 测试简单句子中的用户token
                test_text = f"{user_token} purchased a product"
                encoded = tokenizer.encode_plus(test_text, return_tensors="pt")
                decoded = tokenizer.decode(encoded['input_ids'][0])
                
                print(f"\nOriginal text: '{test_text}'")
                print(f"Encoded IDs: {encoded['input_ids'][0][:10].tolist()}...")
                print(f"Decoded result: '{decoded}'")
                
                # 验证用户token是否被保留（没有被分词）
                assert user_token in decoded, f"User token {user_token} was lost in decoding"
                print(f"✓ User token preservation test passed")
        else:
            print("No user tokens available")
        
        # 3. 测试物品token
        print(f"\n3. Item Token Tests")
        # 随机选择5个物品token
        if tokenizer.item_tokens:
            item_samples = random.sample(tokenizer.item_tokens, min(5, len(tokenizer.item_tokens)))
            print(f"Random item token samples: {item_samples}")
            
            # 测试编码和解码
            for item_token in item_samples:
                # 测试简单句子中的物品token
                test_text = f"User reviewed {item_token}"
                encoded = tokenizer.encode_plus(test_text, return_tensors="pt")
                decoded = tokenizer.decode(encoded['input_ids'][0])
                
                print(f"\nOriginal text: '{test_text}'")
                print(f"Encoded IDs: {encoded['input_ids'][0][:10].tolist()}...")
                print(f"Decoded result: '{decoded}'")
                
                # 验证物品token是否被保留
                assert item_token in decoded, f"Item token {item_token} was lost in decoding"
                print(f"✓ Item token preservation test passed")
        else:
            print("No item tokens available")
        
        # 4. 测试类别token
        print(f"\n4. Category Token Tests")
        # 随机选择5个类别token
        if tokenizer.category_tokens:
            cat_samples = random.sample(tokenizer.category_tokens, min(5, len(tokenizer.category_tokens)))
            print(f"Random category token samples: {cat_samples}")
            
            # 测试每个类别token
            for cat_token in cat_samples:
                # 测试简单句子中的类别token
                test_text = f"This product belongs to the {cat_token} category"
                encoded = tokenizer.encode_plus(test_text, return_tensors="pt")
                decoded = tokenizer.decode(encoded['input_ids'][0])
                
                print(f"\nOriginal text: '{test_text}'")
                print(f"Encoded IDs: {encoded['input_ids'][0][:10].tolist()}...")
                print(f"Decoded result: '{decoded}'")
                
                # 验证类别token是否被保留
                assert cat_token in decoded, f"Category token {cat_token} was lost in decoding"
                print(f"✓ Category token preservation test passed")
                
                # 尝试获取原始类别名
                original_category = tokenizer.get_category_from_token(cat_token)
                if original_category:
                    print(f"Category token mapping: {cat_token} -> {original_category}")
                    
                    # 反向测试：从原始类别获取token
                    reverse_token = tokenizer.get_category_token(original_category)
                    assert reverse_token == cat_token, f"Category reverse mapping failed: {original_category} -> {reverse_token} != {cat_token}"
                    print(f"✓ Category reverse mapping test passed")
        else:
            print("No category tokens available")
        
        # 5. 测试多种token混合
        print(f"\n5. Mixed Token Tests")
        if tokenizer.user_tokens and tokenizer.item_tokens and tokenizer.category_tokens:
            user_token = random.choice(tokenizer.user_tokens)
            item_token = random.choice(tokenizer.item_tokens)
            cat_token = random.choice(tokenizer.category_tokens)
            
            test_text = f"{user_token} purchased {item_token} which is a {cat_token} category product"
            print(f"Mixed token test text: '{test_text}'")
            
            encoded = tokenizer.encode_plus(test_text, return_tensors="pt")
            decoded = tokenizer.decode(encoded['input_ids'][0])
            
            print(f"Decoded result: '{decoded}'")
            
            # 验证所有特殊token是否被保留
            assert user_token in decoded, f"User token {user_token} was lost in decoding"
            assert item_token in decoded, f"Item token {item_token} was lost in decoding"
            assert cat_token in decoded, f"Category token {cat_token} was lost in decoding"
            print(f"✓ Mixed token preservation test passed")
        
        # 6. 测试不存在的token
        print(f"\n6. Non-existent Token Tests")
        
        # 测试不存在的用户
        fake_user = "[user_THIS_USER_DOES_NOT_EXIST]"
        test_text = f"{fake_user} purchased a product"
        encoded = tokenizer.encode_plus(test_text, return_tensors="pt")
        decoded = tokenizer.decode(encoded['input_ids'][0])
        print(f"Non-existent user test: '{test_text}'")
        print(f"Decoded result: '{decoded}'")
        print(f"Fictional user handling: {fake_user in decoded}")
        
        # 测试不存在的类别
        fake_category = "Beauty > Fake > Category"
        cat_token = tokenizer.get_category_token(fake_category)
        print(f"Non-existent category test: '{fake_category}'")
        print(f"Retrieved token: {cat_token}")
        
        # 7. 测试特殊字符
        print(f"\n7. Special Characters Tests")
        # 测试包含特殊字符的类别名
        special_chars_category = "Beauty & Health > Personal Care > Special_Characters!@#$%^&*()"
        safe_category = re.sub(r'[^\w]', '_', special_chars_category.replace(" > ", "_").replace(" ", "_"))
        special_token = f"[category_{safe_category}]"
        test_text = f"This product belongs to {special_token}"
        
        print(f"Special character category: '{special_chars_category}'")
        print(f"Converted token: '{special_token}'")
        
        encoded = tokenizer.encode_plus(test_text, return_tensors="pt")
        decoded = tokenizer.decode(encoded['input_ids'][0])
        print(f"Encoded result: {encoded['input_ids'][0][:10].tolist()}...")
        print(f"Decoded result: '{decoded}'")
        
        # 8. 测试批量编码和长文本
        print(f"\n8. Batch Encoding and Long Text Tests")
        
        # 准备一个长文本，包含特殊token
        if tokenizer.user_tokens and tokenizer.item_tokens:
            user_token = random.choice(tokenizer.user_tokens)
            item_tokens = random.sample(tokenizer.item_tokens, min(10, len(tokenizer.item_tokens)))
            
            long_text = f"{user_token} has the following purchase history: " + ", ".join(item_tokens)
            print(f"Long text example ({len(long_text.split())} words): '{long_text[:100]}...'")
            
            # 测试不同的最大长度设置
            for max_len in [10, 20, 50]:
                encoded = tokenizer.encode_plus(
                    long_text, 
                    max_length=max_len,
                    truncation=True,
                    return_tensors="pt"
                )
                print(f"Max length {max_len}, encoded shape: {encoded['input_ids'].shape}")
            
            # 批量编码测试
            texts = [
                f"{user_token} reviewed {item_tokens[0]}",
                f"{user_token} browsed {item_tokens[1]}",
                f"{item_tokens[2]} belongs to a category"
            ]
            batch_encoded = tokenizer.encode_batch(
                texts, 
                max_length=20,
                padding=True,
                truncation=True
            )
            
            print(f"Batch encoding test, input {len(texts)} texts:")
            print(f"Encoded shape: {batch_encoded[0].shape}")
            print(f"Attention mask shape: {batch_encoded[1].shape}")
            
        # 9. 测试文本保真度和特殊字符处理
        print(f"\n9. Text Fidelity and Special Character Tests")
        
        # 测试包含逗号、括号和特殊格式的文本
        test_cases = [
            # 测试逗号后的空格保留
            "This is a test, with commas, and spaces",
            
            # 测试产品评论格式
            "[user_TEST] writes the review for [item_TEST], a [category_TEST] product: Look at the ingredients: Water, Aloe Vera, Glycerin",
            
            # 测试括号内的文本
            "Contains (Certified Organic) ingredients and [special] tokens",
            
            # 测试数字和标点
            "Rated 4.5 out of 5 stars, with 1,234 reviews!",
            
            # 测试长单词和专业术语
            "Contains glycerin, propylparaben, and methylchloroisothiazolinone"
        ]
        
        print("\nTesting text preservation in tokenization:")
        for i, test_text in enumerate(test_cases, 1):
            print(f"\nTest case {i}:")
            print(f"Original: {test_text}")
            
            # 编码然后解码
            encoded = tokenizer.encode_plus(test_text, return_tensors="pt")
            decoded = tokenizer.decode(encoded['input_ids'][0])
            
            print(f"Decoded : {decoded}")
            
            # 检查关键特征是否保留
            # 1. 逗号后的空格
            if "," in test_text:
                comma_spaces_original = len([x for x in test_text.split(",") if x.startswith(" ")])
                comma_spaces_decoded = len([x for x in decoded.split(",") if x.startswith(" ")])
                print(f"✓ Comma-space preservation: {comma_spaces_original == comma_spaces_decoded}")
            
            # 2. 特殊token完整性
            if "[" in test_text and "]" in test_text:
                tokens_original = [t for t in test_text.split() if t.startswith("[") and t.endswith("]")]
                tokens_decoded = [t for t in decoded.split() if t.startswith("[") and t.endswith("]")]
                print(f"✓ Special token preservation: {set(tokens_original) == set(tokens_decoded)}")
            
            # 3. 数字和小数点完整性
            numbers_original = re.findall(r'\d+\.?\d*', test_text)
            numbers_decoded = re.findall(r'\d+\.?\d*', decoded)
            if numbers_original:
                print(f"✓ Number preservation: {numbers_original == numbers_decoded}")
            
            # 4. 检查是否有不当的词分割（用#号分割）
            if "#" in decoded:
                print("⚠ Warning: Detected word splitting with # symbols")
                split_words = [w for w in decoded.split() if "#" in w]
                print(f"   Split words: {split_words}")
        
        print("\nTests completed!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    test_tokenizer_special_tokens() 