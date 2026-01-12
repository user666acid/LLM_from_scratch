### Структура проекта

```
src
└── model
    ├── core
    │   ├── activation.py
    │   └── attention.py
    │   └── feedforward.py
    │   └── positional_encoding.py
    ├── layers
    │   ├── decoder.py
    │   └── embeddings.py
    └── gpt2.py
```

### Функционал

*activation.py*
- Функция активации GELU

*attention.py*
- Multi Head архитектура внимания
- Возможность использования KV-кэширования

*feedforward.py*
- Полносвязная сеть (FFN) с активацией GELU

*positional_encoding.py*
- Обучаемое позиционное кодирование

*decoder.py*
- Блок декодера
- pre-layer нормализация
- Остаточная связь (residual connection)
- FFN

*embeddings.py*
- Отображение из пространства словаря модели

*gpt2.py*
- Логика обработки исходной последовательности моделью
- Генерация токенов: возможность применения top-k и top-p стратегий, контроля температуры
- Базовые процедуры обучения, сохранения и загрузки модели

![gpt2.png](gpt2.png)