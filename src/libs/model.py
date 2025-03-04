import torch
import torch.nn as nn
from transformers import DistilBertModel

class AmazonDistilBertBaseModel(nn.Module):
    """
    可选的基类，记录扩表后的DistilBertModel。
    注意：如果只想直接用DistilBertModel，也可以不写此类。
    """
    def __init__(self, distilbert_model: DistilBertModel):
        super().__init__()
        # distilbert_model 应该是已做 resize_token_embeddings(...) 的
        self.distilbert = distilbert_model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state

class AmazonDistilBertForMLM(nn.Module):
    """
    MLM 模型：DistilBert + Linear Head
    输出维度 = distilbert_model.embeddings.word_embeddings.num_embeddings
    """
    def __init__(self, base_model: AmazonDistilBertBaseModel):
        super().__init__()
        self.base_model = base_model
        self.distilbert = base_model.distilbert
        
        # 获取原始和扩展后的词表大小
        self.original_vocab_size = 30522  # DistilBERT原始词表大小
        self.extended_vocab_size = self.distilbert.embeddings.word_embeddings.num_embeddings
        
        hidden_size = self.distilbert.config.dim
        
        # 创建新的预测头
        self.mlm_head = nn.Linear(hidden_size, self.extended_vocab_size, bias=False)
        
        # 如果可能，加载原始预测头的权重
        if hasattr(base_model.distilbert, 'mlm_head'):
            with torch.no_grad():
                self.mlm_head.weight[:self.original_vocab_size].copy_(
                    base_model.distilbert.mlm_head.weight[:self.original_vocab_size]
                )

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        labels: 与 input_ids 同形状，对被mask位置是原token id，其他位置=-100
        
        Returns:
            与HuggingFace Transformers兼容的输出格式：
            - 包含loss和logits属性的对象(MaskedLMOutput)
        """
        hidden_states = self.base_model(input_ids, attention_mask=attention_mask)
        logits = self.mlm_head(hidden_states)  # (batch_size, seq_len, extended_vocab_size)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.extended_vocab_size),
                labels.view(-1)
            )
        
        # 返回一个与HuggingFace模型兼容的输出格式
        # 使用带命名参数的类似命名元组的对象
        return type('MaskedLMOutput', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        })
