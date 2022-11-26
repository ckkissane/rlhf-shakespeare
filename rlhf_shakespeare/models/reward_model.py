import torch
from torch import nn
import copy


class RewardModel(nn.Module):
    def __init__(self, pretrained_model: nn.Module, num_classes: int):
        super().__init__()
        self.token_embedding = copy.deepcopy(pretrained_model.token_embedding)
        self.pos_embedding = copy.deepcopy(pretrained_model.pos_embedding)
        self.dropout = copy.deepcopy(pretrained_model.dropout)
        self.blocks = copy.deepcopy(pretrained_model.blocks)
        self.ln = copy.deepcopy(pretrained_model.ln)

        self.sentiment_head = nn.Linear(pretrained_model.hidden_size, num_classes)

    def forward(self, input_ids):
        """
        Args:
            input_ids : torch.tensor of shape (batch_size, seq_len)
        Returns:
            logits: torch.tensor of shape (batch_size, seq_len, 2)
        """
        batch_size, seq_len = input_ids.shape
        pos = torch.arange(seq_len).to(input_ids.device)
        x = self.dropout(self.token_embedding(input_ids) + self.pos_embedding(pos))
        x = self.blocks(x)
        x = self.ln(x)
        out = self.sentiment_head(x)
        return out
