"""
层次编码器损失函数

定义用于层次编码器的各种损失函数。
基于HiT论文中的超球面聚类损失和超球面向心损失。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperbolicClusteringLoss(nn.Module):
    """
    超球面聚类损失
    
    将相关实体在超球面空间中聚集在一起，将无关实体分开。
    基于HiT论文中的超球面聚类损失。
    """
    
    def __init__(self, manifold, margin=5.0):
        """
        初始化
        
        Args:
            manifold: 超球面流形
            margin: 边界
        """
        super().__init__()
        self.manifold = manifold
        self.margin = margin
    
    def forward(self, e, e_pos, e_neg):
        """
        前向传播
        
        Args:
            e: 实体嵌入 [batch_size, embed_dim]
            e_pos: 正样本嵌入 [batch_size, embed_dim]
            e_neg: 负样本嵌入 [batch_size, embed_dim]
            
        Returns:
            loss: 损失值
        """
        # 计算超球面距离
        d_pos = self.manifold.dist(e, e_pos)
        d_neg = self.manifold.dist(e, e_neg)
        
        # 计算三元组损失
        loss = F.relu(d_pos - d_neg + self.margin)
        
        return loss.mean()


class HyperbolicCentripetalLoss(nn.Module):
    """
    超球面向心损失
    
    确保父实体位于超球面空间中靠近原点的位置。
    基于HiT论文中的超球面向心损失。
    """
    
    def __init__(self, manifold, margin=0.1):
        """
        初始化
        
        Args:
            manifold: 超球面流形
            margin: 边界
        """
        super().__init__()
        self.manifold = manifold
        self.margin = margin
    
    def forward(self, e, e_pos):
        """
        前向传播
        
        Args:
            e: 实体嵌入 [batch_size, embed_dim]
            e_pos: 正样本嵌入 [batch_size, embed_dim]
            
        Returns:
            loss: 损失值
        """
        # 计算超球面范数（到原点的距离）
        norm_e = self.manifold.dist0(e)
        norm_pos = self.manifold.dist0(e_pos)
        
        # 计算向心损失
        loss = F.relu(norm_pos - norm_e + self.margin)
        
        return loss.mean()


class HierarchyEncoderLoss(nn.Module):
    """
    层次编码器损失
    
    结合超球面聚类损失和超球面向心损失。
    """
    
    def __init__(self, manifold, clustering_margin=5.0, centripetal_margin=0.1):
        """
        初始化
        
        Args:
            manifold: 超球面流形
            clustering_margin: 聚类损失边界
            centripetal_margin: 向心损失边界
        """
        super().__init__()
        self.clustering_loss = HyperbolicClusteringLoss(manifold, clustering_margin)
        self.centripetal_loss = HyperbolicCentripetalLoss(manifold, centripetal_margin)
    
    def forward(self, e, e_pos, e_neg):
        """
        前向传播
        
        Args:
            e: 实体嵌入 [batch_size, embed_dim]
            e_pos: 正样本嵌入 [batch_size, embed_dim]
            e_neg: 负样本嵌入 [batch_size, embed_dim]
            
        Returns:
            loss: 总损失值
            losses: 各损失组件
        """
        clustering_loss = self.clustering_loss(e, e_pos, e_neg)
        centripetal_loss = self.centripetal_loss(e, e_pos)
        
        total_loss = clustering_loss + centripetal_loss
        
        losses = {
            'total': total_loss.item(),
            'clustering': clustering_loss.item(),
            'centripetal': centripetal_loss.item()
        }
        
        return total_loss, losses 