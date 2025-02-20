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
        # 词表大小（含扩展后的 user/item/category）
        self.extended_vocab_size = self.distilbert.embeddings.word_embeddings.num_embeddings

        hidden_size = self.distilbert.config.dim
        # 线性映射到 extended_vocab_size
        self.mlm_head = nn.Linear(hidden_size, self.extended_vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        labels: 与 input_ids 同形状，对被mask位置是原token id，其他位置=-100
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
            return (loss, logits)
        return (logits,)
