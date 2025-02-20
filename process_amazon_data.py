import gzip
import json
from tqdm import tqdm
import os

def count_lines(filename):
    """计算文件的行数"""
    count = 0
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count

def process_and_show_sample(gz_file, output_file, num_sample=5):
    """处理压缩文件并显示样本数据"""
    print(f"\n处理文件: {gz_file}")
    
    # 首先计算总行数用于进度条
    print("计算文件大小...")
    total_lines = count_lines(gz_file)
    print(f"总计 {total_lines} 条记录")
    
    # 读取并解压数据
    print(f"开始解压到 {output_file}")
    samples = []
    with gzip.open(gz_file, 'rt', encoding='utf-8') as gz, \
         open(output_file, 'w', encoding='utf-8') as out:
        for line in tqdm(gz, total=total_lines, desc="解压进度"):
            data = json.loads(line.strip())
            out.write(line)
            if len(samples) < num_sample:
                samples.append(data)
    
    # 显示样本数据
    print(f"\n显示前 {num_sample} 条记录示例:")
    for i, sample in enumerate(samples, 1):
        print(f"\n样本 #{i}:")
        print(json.dumps(sample, indent=2, ensure_ascii=False))

def main():
    # 创建输出目录
    output_dir = "amazon_books_processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理评论数据
    reviews_gz = "amazon_books_data/books_reviews.jsonl.gz"
    reviews_output = f"{output_dir}/books_reviews.jsonl"
    
    # 处理元数据
    metadata_gz = "amazon_books_data/books_metadata.jsonl.gz"
    metadata_output = f"{output_dir}/books_metadata.jsonl"
    
    # 处理两个文件
    process_and_show_sample(reviews_gz, reviews_output)
    process_and_show_sample(metadata_gz, metadata_output)
    
    print(f"\n处理完成！解压后的文件保存在 {output_dir} 目录下")
    print(f"- 评论数据: {reviews_output}")
    print(f"- 元数据: {metadata_output}")

if __name__ == "__main__":
    main() 