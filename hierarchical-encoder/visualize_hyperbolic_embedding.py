#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双曲空间层次结构可视化：直接使用模型学到的双曲距离作为深度指标
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
from typing import Dict, List, Tuple

# 导入项目模块
from src.taxonomy import AmazonTaxonomy
from hierarchy_transformers.models import HierarchyTransformer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# 定义路径
BASE_DIR = "/u/home/r/ricklin/HyperTreeRecomSys"
DATA_DIR = os.path.join(BASE_DIR, "hierarchical-encoder", "datasets", "amazon_beauty_dataset")
MODEL_DIR = os.path.join(BASE_DIR, "hierarchical-encoder", "output", "amazon-hierarchy", "final")
OUTPUT_DIR = os.path.join(BASE_DIR, "hierarchical-encoder", "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_entity_lexicon() -> Dict[str, str]:
    """加载实体词典"""
    lexicon_path = os.path.join(DATA_DIR, 'entity_lexicon.json')
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        entity_lexicon = json.load(f)
    logger.info(f"加载了 {len(entity_lexicon)} 个实体")
    return entity_lexicon

def load_taxonomy() -> AmazonTaxonomy:
    """加载Amazon产品分类法"""
    logger.info("加载Amazon产品分类法...")
    taxonomy = AmazonTaxonomy()
    logger.info(f"分类法加载完成，包含 {len(taxonomy.nodes)} 个节点和 {len(taxonomy.edges)} 条边")
    return taxonomy

def load_model() -> HierarchyTransformer:
    """加载训练好的模型"""
    logger.info(f"从 {MODEL_DIR} 加载模型")
    model = HierarchyTransformer.from_pretrained(MODEL_DIR)
    logger.info(f"模型加载成功: {model.__class__.__name__}")
    return model

def get_embeddings(
    model: HierarchyTransformer, 
    entity_lexicon: Dict[str, str],
    nodes: List[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """获取节点的双曲嵌入和双曲距离"""
    logger.info(f"计算 {len(nodes)} 个节点的嵌入...")
    
    # 获取节点描述文本
    node_texts = []
    for node_id in nodes:
        if node_id in entity_lexicon:
            node_texts.append(entity_lexicon[node_id])
        else:
            node_texts.append(f"Unknown {node_id}")
    
    # 使用模型获取嵌入
    with torch.no_grad():
        embeddings = model.encode(
            node_texts, 
            batch_size=32, 
            show_progress_bar=True,
            convert_to_tensor=True
        )
    
    # 获取模型使用的双曲流形
    manifold = model.manifold  # 这是PoincareBall实例
    
    # 计算双曲距离和嵌入
    hyperbolic_norms = {}
    embedding_dict = {}
    
    for i, node_id in enumerate(nodes):
        # 计算到原点的双曲距离
        # 使用geoopt.manifolds.PoincareBall的dist方法
        # 原点表示为torch.zeros_like(embeddings[i])
        origin = torch.zeros_like(embeddings[i])
        norm = manifold.dist(embeddings[i], origin).item()
        hyperbolic_norms[node_id] = norm
        
        # 获取欧氏空间中的坐标用于可视化
        # 直接取前两维即可，因为在PoincareBall中已经是双曲空间的表示
        embedding_2d = embeddings[i][:2].cpu().numpy()
        embedding_dict[node_id] = embedding_2d
    
    logger.info(f"嵌入和双曲距离计算完成")
    return embedding_dict, hyperbolic_norms

def visualize_hierarchy(
    taxonomy: AmazonTaxonomy,
    embedding_dict: Dict[str, np.ndarray],
    hyperbolic_norms: Dict[str, float],
    entity_lexicon: Dict[str, str],
    output_file: str,
    max_nodes: int = 500,
    max_labels: int = 100
):
    """使用双曲嵌入可视化层次结构"""
    logger.info(f"创建可视化图...")
    
    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # 绘制Poincaré圆盘边界
    circle = Circle((0, 0), 1, fill=False, color='black')
    ax.add_patch(circle)
    
    # 设置坐标轴
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    # 获取所有可视化的节点
    graph = taxonomy.graph
    all_nodes = list(graph.nodes())
    
    # 如果节点太多，按双曲范数选择
    if len(all_nodes) > max_nodes:
        nodes_by_norm = sorted(
            [(node, hyperbolic_norms.get(node, float('inf'))) for node in all_nodes],
            key=lambda x: x[1]
        )
        selected_nodes = [node for node, _ in nodes_by_norm[:max_nodes]]
    else:
        selected_nodes = all_nodes
    
    # 获取范数范围用于颜色映射
    norm_values = [v for k, v in hyperbolic_norms.items() if k in selected_nodes]
    min_norm = min(norm_values) if norm_values else 0
    max_norm = max(norm_values) if norm_values else 1
    norm_range = max_norm - min_norm
    
    # 创建颜色映射规范化器
    norm = plt.Normalize(vmin=min_norm, vmax=max_norm)
    
    # 绘制节点
    scatter_points = []
    scatter_colors = []
    for node in selected_nodes:
        if node in embedding_dict:
            x, y = embedding_dict[node]
            scatter_points.append([x, y])
            
            if node in hyperbolic_norms:
                norm_val = hyperbolic_norms[node]
                normalized_norm = (norm_val - min_norm) / norm_range if norm_range > 0 else 0.5
            else:
                normalized_norm = 0.5
            scatter_colors.append(normalized_norm)
    
    if scatter_points:
        scatter_points = np.array(scatter_points)
        scatter = ax.scatter(
            scatter_points[:, 0], 
            scatter_points[:, 1], 
            c=scatter_colors,
            cmap='viridis',
            norm=norm,
            s=30, 
            alpha=0.8
        )
    
    # 绘制边
    for parent, child in taxonomy.edges:
        if (parent in embedding_dict and child in embedding_dict and 
            parent in selected_nodes and child in selected_nodes):
            parent_pos = embedding_dict[parent]
            child_pos = embedding_dict[child]
            
            parent_norm = hyperbolic_norms.get(parent, min_norm)
            child_norm = hyperbolic_norms.get(child, min_norm)
            avg_norm = (parent_norm + child_norm) / 2
            normalized_avg = (avg_norm - min_norm) / norm_range if norm_range > 0 else 0.5
            
            ax.plot(
                [parent_pos[0], child_pos[0]], 
                [parent_pos[1], child_pos[1]], 
                color=plt.cm.viridis(normalized_avg), 
                alpha=0.5, 
                linewidth=0.8
            )
    
    # 添加标签
    if len(selected_nodes) > max_labels:
        nodes_to_label = sorted(
            [(node, hyperbolic_norms.get(node, float('inf'))) for node in selected_nodes],
            key=lambda x: x[1]
        )[:max_labels]
        nodes_to_label = [node for node, _ in nodes_to_label]
    else:
        nodes_to_label = selected_nodes
    
    for node in nodes_to_label:
        if node in embedding_dict:
            x, y = embedding_dict[node]
            
            if node in entity_lexicon:
                description = entity_lexicon[node]
                if description.startswith("Product Category:"):
                    label = description.replace("Product Category: ", "").split(" > ")[-1]
                else:
                    label = description.split(" - ")[0]
            else:
                label = node.split('_')[-1]
            
            if len(label) > 25:
                label = label[:22] + '...'
            
            if node in hyperbolic_norms:
                norm_val = hyperbolic_norms[node]
                normalized_norm = (norm_val - min_norm) / norm_range if norm_range > 0 else 0.5
                text_color = 'white' if normalized_norm > 0.5 else 'black'
            else:
                text_color = 'black'
                normalized_norm = 0.5
            
            ax.annotate(
                label, 
                (x, y), 
                fontsize=6, 
                color=text_color,
                ha='center', 
                va='center', 
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc=plt.cm.viridis(normalized_norm),
                    ec="none", 
                    alpha=0.7
                )
            )
    
    # 添加颜色条
    if scatter_points is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('双曲范数（层次深度）')
    
    ax.set_title("Amazon产品层次结构在双曲空间中的嵌入", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"可视化图已保存至: {output_file}")
    plt.close()

def main():
    """主函数"""
    try:
        # 1. 加载实体词典
        entity_lexicon = load_entity_lexicon()
        
        # 2. 加载分类法
        taxonomy = load_taxonomy()
        
        # 3. 加载训练好的模型
        model = load_model()
        
        # 4. 获取节点嵌入和双曲范数（真实层次深度）
        embedding_dict, hyperbolic_norms = get_embeddings(
            model, 
            entity_lexicon, 
            list(taxonomy.nodes)
        )
        
        # 5. 可视化层次结构
        output_file = os.path.join(OUTPUT_DIR, "amazon_hierarchy_hyperbolic.png")
        visualize_hierarchy(
            taxonomy, 
            embedding_dict, 
            hyperbolic_norms, 
            entity_lexicon, 
            output_file
        )
        
        # 6. 分析双曲范数分布
        norm_values = list(hyperbolic_norms.values())
        plt.figure(figsize=(10, 6))
        plt.hist(norm_values, bins=30, alpha=0.7, color='blue')
        plt.xlabel('双曲范数（到原点的距离）')
        plt.ylabel('节点数量')
        plt.title('节点双曲范数分布 - 体现层次深度')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "hyperbolic_norm_distribution.png"), dpi=300)
        
        logger.info("可视化完成!")
        
    except Exception as e:
        logger.error(f"可视化过程出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 