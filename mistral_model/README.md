### Структура проекта

```
src
└── model
    ├── core
    │   ├── activation.py
    │   └── attention.py
    │   └── normalization.py
    │   └── positional_encoding.py
    ├── layers
    │   ├── decoder.py
    │   └── embeddings.py
    └── mistral.py
```

### Функционал

*activation.py*
- Функция активации SiLU
- Слой активации SwiGLU

*attention.py*
- Grouped Query архитектура внимания
- Возможность ограничения длины контекста (sliding window attention)
- Возможность использования KV-кэширования

*normalization.py*
- Root Mean Square нормализация 

*positional_encoding.py*
- Роторное позиционное кодирование (RoPE)

*decoder.py*
- Блок декодера
- pre-layer нормализация
- Остаточная связь (residual connection)
- SwiGLU вместо простой полносвязной сети (FFN)

*embeddings.py*
- Отображение из пространства словаря модели

*mistral.py*
- Логика обработки исходной последовательности моделью
- Генерация токенов: возможность применения top-k и top-p стратегий, контроля температуры
- Базовые процедуры обучения, сохранения и загрузки модели

![mistral.png](mistral.png)