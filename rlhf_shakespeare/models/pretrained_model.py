import torch
from torch import nn
import torch.nn.functional as F
import math


def mask_scores(attn_scores):
    """
    Args:
        attn_scores: torch.tensor of shape (batch_size, num_heads, seq_len, seq_len)
    Returns:
        out: torch.tensor of shape (batch_size, num_heads, seq_len, seq_len)
    """
    seq_len = attn_scores.shape[-2]
    neg_inf = torch.tensor(-1e9).to(attn_scores.device)
    q_ind = torch.arange(seq_len).unsqueeze(1)
    k_ind = torch.arange(seq_len).unsqueeze(0)
    mask = (q_ind < k_ind).to(attn_scores.device)
    attn_scores = torch.where(mask, neg_inf, attn_scores)
    return attn_scores


def masked_attn(q, k, v):
    """
    Args:
        q: torch.tensor of shape (batch_size, num_heads, seq_len, headsize)
        k: torch.tensor of shape (batch_size, num_heads, seq_len, headsize)
        v: torch.tensor of shape (batch_size, num_heads, seq_len, headsize)
    Returns:
        out: torch.tensor of shape (batch_size, num_heads, seq_len, headsize)
    """
    headsize = q.shape[-1]
    attn_scores = q.matmul(k.transpose(-1, -2)) / math.sqrt(headsize)
    attn_scores = mask_scores(attn_scores)
    attn_scores = attn_scores.softmax(dim=-1)
    out = attn_scores.matmul(v)
    return out


class MaskedMultiHeadedAttn(nn.Module):
    def __init__(self, num_heads, hidden_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.headsize = hidden_size // self.num_heads

    def forward(self, x):
        """
        Args:
            x: torch.tensor of shape (batch_size, seq_len, hidden_size)
        Returns:
            out: torch.tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.headsize)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.headsize)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.headsize)
            .transpose(1, 2)
        )
        out = masked_attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, num_heads, hidden_size, dropout):
        super().__init__()
        self.attn = MaskedMultiHeadedAttn(num_heads, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.lin1 = nn.Linear(hidden_size, hidden_size * 4)
        self.lin2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x : torch.tensor of shape (batch_size, seq_len, emb_dim)
        Returns:
            out: torch.tensor of shape (batch_size, seq_len, emb_dim)
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.dropout(self.lin2(F.gelu(self.lin1(self.ln2(x)))))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_pos_embeddings,
        num_heads,
        hidden_size,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_pos_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[DecoderBlock(num_heads, hidden_size, dropout) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        """
        Args:
            input_ids : torch.tensor of shape (batch_size, seq_len)
        Returns:
            logits: torch.tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        pos = torch.arange(seq_len).to(input_ids.device)
        x = self.dropout(self.token_embedding(input_ids) + self.pos_embedding(pos))
        x = self.blocks(x)
        x = self.ln(x)
        out = self.lm_head(x)
        return out
