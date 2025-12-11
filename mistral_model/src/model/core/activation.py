import torch
import torch.nn as nn

class SiLU(nn.Module):
    def __init__(self):
        """Слой для функции активации Sigmoid Linear Unit.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Определяет логику вычислений в слое.

        Args:
            x: Исходное представление последовательности.

        Returns:
            Активированное представление.
        """
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float=0.1):
        """Слой для функции активации Swish Gated Linear Unit.

        Args:
            emb_size: Размерность внутреннего представления.
            dropout: Доля зануляемых элементов.
        """
        super().__init__()

        self.gate = nn.Linear(emb_size, 4 * emb_size)
        self.up = nn.Linear(emb_size, 4 * emb_size)
        self.down = nn.Linear(4 * emb_size, emb_size)

        self.activation = SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Определяет логику вычислений в слое.

        Args:
            x: Исходное представление последовательности.

        Returns:
            Активированное представление.
        """
        activation = self.down(self.up(x) * self.activation(self.gate(x)))
        activation = self.dropout(activation)

        return activation