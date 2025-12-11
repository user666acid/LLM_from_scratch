import torch
import torch.nn as nn

import numpy as np

class GELU(nn.Module):
    def __init__(self):
        """Слой для функции активации Gaussian Error Linear Units.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Определяет логику вычислений в слое.

        Args:
            x: Исходное представление последовательности.

        Returns:
            Преобразованное представление.
        """
        tanh_value = torch.tanh(torch.sqrt(torch.as_tensor(2 / np.pi)) * (x + 0.044715 * x ** 3))
        activation = 0.5 * x * (1 + tanh_value)

        return activation

class GeGLU(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float=0.1):
        """Слой для функции активации GeLU Gated Linear Unit.

        Args:
            emb_size: Размерность внутреннего представления.
            dropout: Доля зануляемых элементов.
        """
        super().__init__()

        self.gate = nn.Linear(emb_size, 4 * emb_size)
        self.up = nn.Linear(emb_size, 4 * emb_size)
        self.down = nn.Linear(4 * emb_size, emb_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Определяет логику вычислений в слое.

        Args:
            x: Исходное представление последовательности.

        Returns:
            Преобразованное представление.
        """
        activation = self.down(self.up(x) * self.activation(self.gate(x)))
        activation = self.dropout(activation)

        return activation