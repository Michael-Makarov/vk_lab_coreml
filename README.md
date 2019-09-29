# VK lab coreml task

## Требования

```
pytorch >=1.2
implicit >=0.4.0
gensim >=3.8.0
```

## Запуск

Чтобы обучить модель, запустите

```
    python3 prepare_model.py --model-path=<where to save model> --data-path=<path to dataset>
```

Датасет должен иметь вид csv-таблицы с 4 столбцами: `userId`,`movieId`,`rating`,`timestamp`.

Отчёт в файле `report.ipynb`.
