#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交互式双曲空间层次结构可视化：允许点击节点查看详细信息
"""

import os
import json
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import webbrowser

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
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, str]]:
    """
    获取节点的双曲嵌入和双曲距离
    
    返回:
        embedding_dict: 节点ID到嵌入的映射
        hyperbolic_norms: 节点ID到双曲范数的映射
        node_descriptions: 节点ID到原始描述文本的映射
    """
    logger.info(f"计算 {len(nodes)} 个节点的嵌入...")
    
    # 获取节点描述文本
    node_texts = []
    node_descriptions = {}
    
    for node_id in nodes:
        if node_id in entity_lexicon:
            text = entity_lexicon[node_id]
            node_texts.append(text)
            node_descriptions[node_id] = text
        else:
            node_texts.append(f"Unknown {node_id}")
            node_descriptions[node_id] = f"Unknown {node_id}"
    
    # 使用模型获取嵌入
    with torch.no_grad():
        embeddings = model.encode(
            node_texts, 
            batch_size=32, 
            show_progress_bar=True,
            convert_to_tensor=True
        )
    
    # 获取模型使用的双曲流形
    manifold = model.manifold
    
    # 计算双曲距离和嵌入
    hyperbolic_norms = {}
    embedding_dict = {}
    
    for i, node_id in enumerate(nodes):
        # 计算到原点的双曲距离
        origin = torch.zeros_like(embeddings[i])
        norm = manifold.dist(embeddings[i], origin).item()
        hyperbolic_norms[node_id] = norm
        
        # 获取欧氏空间中的坐标用于可视化
        embedding_2d = embeddings[i][:2].cpu().numpy()
        embedding_dict[node_id] = embedding_2d
    
    logger.info(f"嵌入和双曲距离计算完成")
    return embedding_dict, hyperbolic_norms, node_descriptions

def visualize_interactive(
    taxonomy: AmazonTaxonomy,
    embedding_dict: Dict[str, np.ndarray],
    hyperbolic_norms: Dict[str, float],
    node_descriptions: Dict[str, str],
    output_file: str,
    max_nodes: int = 1000,
    max_edges: int = 3000
):
    """
    创建交互式双曲空间可视化
    
    参数:
        taxonomy: 分类法对象
        embedding_dict: 节点ID到嵌入的映射
        hyperbolic_norms: 节点ID到双曲范数的映射
        node_descriptions: 节点ID到原始描述文本的映射
        output_file: 输出HTML文件路径
        max_nodes: 最大显示节点数
        max_edges: 最大显示边数
    """
    logger.info(f"创建交互式可视化...")
    
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
    
    # 为有效节点创建数据框
    node_data = []
    for node in selected_nodes:
        if node in embedding_dict:
            x, y = embedding_dict[node]
            norm = hyperbolic_norms.get(node, 0)
            desc = node_descriptions.get(node, "Unknown")
            
            # 提取简短标签用于显示
            if desc.startswith("Product Category:"):
                short_label = desc.replace("Product Category: ", "").split(" > ")[-1]
                node_type = "category"
                full_category = desc.replace("Product Category: ", "")
                level = len(full_category.split(" > "))
            else:
                parts = desc.split(" - ")
                short_label = parts[0]
                node_type = "product"
                level = 0
            
            node_data.append({
                'node_id': node,
                'x': x,
                'y': y,
                'norm': norm,
                'description': desc,
                'short_label': short_label,
                'node_type': node_type,
                'level': level
            })
    
    # 创建节点数据框
    nodes_df = pd.DataFrame(node_data)
    
    # 获取范数范围用于颜色映射
    min_norm = nodes_df['norm'].min()
    max_norm = nodes_df['norm'].max()
    
    # 创建边数据
    edge_x = []
    edge_y = []
    edge_data = []
    
    # 按照双曲范数的和排序边，优先显示连接高层节点的边
    edges_with_score = []
    
    for parent, child in taxonomy.edges:
        if (parent in embedding_dict and child in embedding_dict and
            parent in selected_nodes and child in selected_nodes):
            parent_pos = embedding_dict[parent]
            child_pos = embedding_dict[child]
            
            parent_norm = hyperbolic_norms.get(parent, 0)
            child_norm = hyperbolic_norms.get(child, 0)
            # 优先选择双曲范数小的边（层次结构中更高级的关系）
            score = parent_norm + child_norm
            
            edges_with_score.append((parent, child, score, parent_pos, child_pos))
    
    # 按分数排序并取前max_edges个边
    sorted_edges = sorted(edges_with_score, key=lambda x: x[2])[:max_edges]
    
    # 添加边
    for parent, child, score, parent_pos, child_pos in sorted_edges:
        edge_x.extend([parent_pos[0], child_pos[0], None])
        edge_y.extend([parent_pos[1], child_pos[1], None])
        
        parent_desc = node_descriptions.get(parent, "Unknown")
        child_desc = node_descriptions.get(child, "Unknown")
        
        # 每条边添加一次，用于悬停信息
        norm_value = (hyperbolic_norms.get(parent, 0) + hyperbolic_norms.get(child, 0)) / 2
        normalized_norm = (norm_value - min_norm) / (max_norm - min_norm) if max_norm > min_norm else 0.5
        
        edge_data.append({
            'parent': parent,
            'child': child,
            'parent_desc': parent_desc,
            'child_desc': child_desc,
            'norm': norm_value,
            'normalized_norm': normalized_norm
        })
    
    # 创建Plotly图形
    fig = make_subplots(specs=[[{"type": "scatter"}]])
    
    # 添加边线
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='rgba(0,150,100,0.2)'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    fig.add_trace(edge_trace)
    
    # 为不同类型的节点创建散点图
    # 1. 类别节点
    category_nodes = nodes_df[nodes_df['node_type'] == 'category']
    
    if not category_nodes.empty:
        # 颜色映射到层次深度
        category_nodes['color'] = category_nodes['norm'].apply(
            lambda x: (x - min_norm) / (max_norm - min_norm) if max_norm > min_norm else 0.5
        )
        
        categories_trace = go.Scatter(
            x=category_nodes['x'], 
            y=category_nodes['y'],
            mode='markers+text',
            marker=dict(
                size=10,
                color=category_nodes['norm'],
                colorscale='Viridis',
                colorbar=dict(title='双曲范数(深度)', x=1.02),
                cmin=min_norm,
                cmax=max_norm,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=category_nodes['short_label'],
            textposition="top center",
            textfont=dict(size=8, color='gray'),
            hovertemplate='<b>%{customdata[0]}</b><br>类型: 类别<br>完整路径: %{customdata[1]}<br>双曲范数: %{marker.color:.3f}<extra></extra>',
            customdata=category_nodes[['short_label', 'description']].values,
            name='类别',
            showlegend=True
        )
        
        fig.add_trace(categories_trace)
    
    # 2. 产品节点 (如果需要)
    product_nodes = nodes_df[nodes_df['node_type'] == 'product']
    
    if not product_nodes.empty:
        product_nodes['color'] = product_nodes['norm'].apply(
            lambda x: (x - min_norm) / (max_norm - min_norm) if max_norm > min_norm else 0.5
        )
        
        products_trace = go.Scatter(
            x=product_nodes['x'],
            y=product_nodes['y'],
            mode='markers',
            marker=dict(
                size=6,
                color=product_nodes['norm'],
                colorscale='Viridis',
                cmin=min_norm,
                cmax=max_norm,
                opacity=0.7,
                symbol='circle',
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            hovertemplate='<b>产品</b><br>%{customdata[0]}<extra></extra>',
            customdata=product_nodes[['description']].values,
            name='产品',
            showlegend=True
        )
        
        fig.add_trace(products_trace)
    
    # 绘制Poincaré圆盘边界
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    
    circle_trace = go.Scatter(
        x=x_circle,
        y=y_circle,
        mode='lines',
        line=dict(color='black', width=1),
        hoverinfo='none',
        showlegend=False
    )
    
    fig.add_trace(circle_trace)
    
    # 配置布局
    fig.update_layout(
        title='Amazon产品层次结构在双曲空间中的交互式可视化',
        width=1000,
        height=1000,
        xaxis=dict(
            range=[-1.1, 1.1],
            zeroline=False,
            showgrid=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[-1.1, 1.1],
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1
        ),
        showlegend=True,
        legend=dict(x=1, y=0.5),
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        hovermode='closest',
        annotations=[
            dict(
                text="点击节点可查看详细信息<br>使用工具栏可缩放、平移和下载",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.01, y=0.99,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=4
            )
        ]
    )
    
    # 添加交互功能提示
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": [True, True, True, True]}],
                        label="显示所有",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True, True, False, True]}],
                        label="仅显示类别",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True, False, True, True]}],
                        label="仅显示产品",
                        method="update"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.02,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ]
    )
    
    # 保存为HTML文件
    fig.write_html(
        output_file,
        full_html=True,
        include_plotlyjs='cdn',
        include_mathjax='cdn'
    )
    
    logger.info(f"交互式可视化已保存到: {output_file}")
    
    # 尝试自动打开浏览器
    try:
        webbrowser.open('file://' + os.path.abspath(output_file), new=2)
    except Exception as e:
        logger.warning(f"无法自动打开浏览器: {str(e)}")
        logger.info(f"请手动打开文件: {os.path.abspath(output_file)}")

def main():
    """主函数"""
    try:
        # 1. 加载实体词典
        entity_lexicon = load_entity_lexicon()
        
        # 2. 加载分类法
        taxonomy = load_taxonomy()
        
        # 3. 加载训练好的模型
        model = load_model()
        
        # 4. 获取节点嵌入、双曲范数和描述
        embedding_dict, hyperbolic_norms, node_descriptions = get_embeddings(
            model, 
            entity_lexicon, 
            list(taxonomy.nodes)
        )
        
        # 5. 创建交互式可视化
        output_file = os.path.join(OUTPUT_DIR, "amazon_hierarchy_interactive.html")
        visualize_interactive(
            taxonomy, 
            embedding_dict, 
            hyperbolic_norms, 
            node_descriptions,
            output_file
        )
        
        logger.info("交互式可视化完成!")
        
    except Exception as e:
        logger.error(f"可视化过程出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()