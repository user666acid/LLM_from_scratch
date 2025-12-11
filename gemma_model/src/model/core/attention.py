import torch
import torch.nn as nn

from typing import Tuple, Optional

from .positional_encoding import RoPE

class MultiQueryAttention(nn.Module):
    def __init__(self,
                 num_q_heads:int,
                 emb_size: int,
                 head_size: int,
                 max_seq_len: int,
                 rope: RoPE,
                 dropout: float=0.1):
        """Слой для MultiQueryAttention.
        Вариация MHA, призванная повысить эффективность (в частности, уменьшить размер KV-кэша).
        Матрицы запросов уникальны для каждой головы внимания.
        Матрицы ключей и значений являются общими для все голов.

        Args:
            num_q_heads: Количество голов внимания для запросов.
            emb_size: Размерность внутреннего представления.
            head_size: Размерность головы внимания.
            max_seq_len: Максимальная длина последовательности.
            rope: Объект для слоя позиционного кодирования RoPE.
            dropout: Доля зануляемых элементов.
        """
        super().__init__()
        
        self.num_q_heads = num_q_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len

        self.rope = rope
        
        self.W_q = nn.Linear(emb_size, num_q_heads * head_size)
        self.W_k = nn.Linear(emb_size, head_size)
        self.W_v = nn.Linear(emb_size, head_size)
        
        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        
        self.W_o = nn.Linear(num_q_heads * head_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor,
                                     cache: Optional[Tuple[torch.Tensor, torch.Tensor]]
                                     ) -> torch.Tensor:
        """Вычисление матрицы внимания.

        Args:
            Q: Матрица запросов.
            K: Матрица ключей.
            V: Матрица значений.
            cache: Содержит предпосчитанные матрицы ключей и значений.

        Returns:
            Взвешанные по вниманию значения.
        """
        _, _, seq_len, _ = Q.shape
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_scores = attn_scores / self.head_size ** 0.5
        if not cache:
            mask = (self.mask[:seq_len, :seq_len] == 0)
            attn_scores = attn_scores.masked_fill(mask, float('-Inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_values = torch.matmul(attn_weights, V)

        return weighted_values

    def forward(self,
                x: torch.Tensor,
                use_cache: bool=True,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]]=None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Определяет логику вычислений в слое.

        Args:
            x: Исходное представление последовательности.
            use_cache: Флаг, контролирующий использование KV-кэша.
            cache: Содержит предпосчитанные матрицы ключей и значений.

        Returns:
            Преобразованное представление, KV-кэш.
        """
        batch_size, seq_len, _ = x.size()
        
        Q = self.W_q(x).reshape(batch_size, seq_len, self.num_q_heads, self.head_size).transpose(1, 2)
        K = self.W_k(x).reshape(batch_size, seq_len, 1, self.head_size).transpose(1, 2)
        V = self.W_v(x).reshape(batch_size, seq_len, 1, self.head_size).transpose(1, 2)
        
        if cache:
            start_pos = cache[0].size(2)
            Q_rotated = self.rope(Q, start_pos)
            K_rotated = self.rope(K, start_pos)
        else:
            Q_rotated = self.rope(Q)
            K_rotated = self.rope(K)
        
        if use_cache:
            if cache:
                K_rotated = torch.cat([cache[0], K_rotated], dim=2)
                V = torch.cat([cache[1], V], dim=2)
                         
            attn = self.scaled_dot_product_attention(Q_rotated, K_rotated, V, cache)
            cache = (K_rotated, V)    
        else:
            attn = self.scaled_dot_product_attention(Q_rotated, K_rotated, V, cache)
        
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, self.num_q_heads * self.head_size)
        attn = self.W_o(attn)
        attn = self.dropout(attn)
        
        return (attn, cache)