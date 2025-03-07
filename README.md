# Beauty数据集Tokenizer

这个项目包含用于处理Amazon Beauty数据集的tokenizer和相关工具。

## 目录结构

- `src/libs/` - 核心库文件
  - `tokenizer.py` - 自定义tokenizer实现
  - `model.py` - 模型定义
- `data/` - 数据目录
  - `meta_Beauty_2014.json.gz` - 元数据文件
  - `reviews_Beauty_5.json.gz` - 评论数据文件
  - `beauty_stats_result.json` - 统计结果
- `beauty_data_processor.py` - 数据处理脚本
- `extract_beauty_stats.py` - 数据统计脚本
- `beauty_tokenizer_init.py` - tokenizer初始化脚本
- `test_tokenizer.py` - tokenizer测试脚本
- `test_dataset.py` - 数据集测试脚本

## 使用方法

1. 提取统计信息:
   ```
   py extract_beauty_stats.py
   ```

2. 初始化tokenizer:
   ```
   py beauty_tokenizer_init.py
   ```
