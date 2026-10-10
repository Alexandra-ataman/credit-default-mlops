# Credit Default

Предсказание дефолта клиента кредитной карты.

## Данные

[Default of Credit Card Clients (UCI)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
Цель — `default.payment.next.month`: 1 - дефолт, 0 - его отсутствие.

## Установка и проверки

Требуется Python 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
black --check src tests
flake8 src tests
```

## Структура

```text
src/credit_default/  код проекта
tests/               тесты
notebooks/           EDA
data/                данные
models/              модели
reports/             результаты
.github/workflows/   CI
```

## EDA

```bash
python -m pip install -e '.[eda]'
GX_ANALYTICS_ENABLED=false python -m credit_default.data
jupyter lab notebooks/01_eda.ipynb
```

Графики сохраняются в `reports/figures/`.
