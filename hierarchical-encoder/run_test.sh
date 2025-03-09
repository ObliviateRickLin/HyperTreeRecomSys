#!/bin/bash
#$ -cwd
#$ -j y
#$ -o test_output.$JOB_ID
#$ -l h_rt=1:00:00

# 选择是否需要GPU，取消注释其中一行
# 使用GPU
##$ -l gpu,cuda=1,h_data=8G
# 不使用GPU，仅CPU
#$ -l h_data=8G

# 使用系统Python，不加载模块
# module load python/3.9.6

# 设置环境变量
export PATH=/usr/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# 输出Python和环境信息
echo "=== 系统信息 ==="
which python
python --version
echo "环境路径: $PATH"
echo "库路径: $LD_LIBRARY_PATH"

# 设置环境
source ../venv/bin/activate

# 安装依赖包
echo -e "\n=== 安装依赖包 ==="
pip install tqdm transformers geoopt torch==1.10.0

# 创建输出目录
mkdir -p test_output

# 运行简化版测试，只测试AmazonTaxonomy类
echo -e "\n=== 测试AmazonTaxonomy类 ==="
python -c "
import sys
from src.data import AmazonTaxonomy

# 创建分类法
taxonomy = AmazonTaxonomy()

# 添加节点
taxonomy.add_node('Electronics', name='Electronics')
taxonomy.add_node('Computers', name='Computers')
taxonomy.add_node('Laptops', name='Laptops')

# 添加边
taxonomy.add_edge('Electronics', 'Computers')
taxonomy.add_edge('Computers', 'Laptops')

# 测试获取子节点
children = taxonomy.get_children('Computers')
print('Computers的子节点:', children)
assert 'Laptops' in children

# 测试获取父节点
parents = taxonomy.get_parents('Laptops')
print('Laptops的父节点:', parents)
assert 'Computers' in parents

print('AmazonTaxonomy测试通过!')
"

echo -e "\n测试完成！" 