# Credit Default

Предсказание дефолта действующего клиента кредитной карты в следующем месяце.

Данные: [Default of Credit Card Clients (UCI)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
Цель: `default.payment.next.month`. 1 означает дефолт, 0 означает его отсутствие.

## Запуск

Python 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,eda,experiments]'
GX_ANALYTICS_ENABLED=false python -m credit_default.data
GX_ANALYTICS_ENABLED=false python -m credit_default.split
python -m credit_default.hypotheses
python -m credit_default.hypotheses --model boosting
OMP_NUM_THREADS=2 python -m credit_default.experiments
OMP_NUM_THREADS=2 python -m credit_default.experiments --seed 17 \
    --variants hgb_base cat_base cat_monthly xgb_base
OMP_NUM_THREADS=2 python -m credit_default.calibration --variant cat_base
jupyter lab notebooks/
```

## Анализ

| Ноутбук | Содержание |
|---|---|
| [01_eda](notebooks/01_eda.ipynb) | Качество данных, разделение, анализ train и проверка гипотез |
| [02_experiments](notebooks/02_experiments.ipynb) | Сравнение моделей, калибровка и ошибки |

Train/test: около 80/20, seed=42. Одинаковые исходные профили остаются
в одной выборке и одном блоке CV. Test не используется для выбора модели.
Временная проверка невозможна: данные представлены одним общим срезом.

Кандидат: CatBoost. Средний ROC-AUC 0,791 на пяти блоках train.
При другом разбиении получаем 0,790. Это результаты CV, не итоговая оценка на test.
Параметры в [baseline.json](reports/baseline.json).

## Файлы

| Папка | Содержание |
|---|---|
| `notebooks/` | Два последовательных этапа: анализ данных и выбор модели |
| `src/credit_default/` | Загрузка и проверки данных, разделение, признаки, расчёты экспериментов |
| `tests/` | Проверки данных, разделения, признаков и расчёта метрик |
| `reports/` | Таблицы анализа и параметры выбранной модели в `baseline.json` |
| `reports/figures/` | Графики анализа и экспериментов |
| `reports/experiments/` | Метрики и настройки моделей по seed |
| `data/` | Исходные и подготовленные данные, вне Git |
| `.github/workflows/` | Автоматические проверки |

Результаты экспериментов в `reports/experiments/`. Данные, списки ID
и подробные отчёты создаются командами запуска. В Git сохранены основные метрики,
два графика и параметры кандидата. Новые ячейки EDA сохранены без выводов.

## Проверки

```bash
black --check src tests
flake8 src tests
pytest -q
```
