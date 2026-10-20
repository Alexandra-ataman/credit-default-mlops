# Credit Default

Прогноз дефолта клиента кредитной карты в следующем месяце.
Данные: [Default of Credit Card Clients (UCI)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
Цель `default.payment.next.month`: 1 означает дефолт, 0 означает его отсутствие.

## Установка и обучение

Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,training,api]'
GX_ANALYTICS_ENABLED=false python -m credit_default.data
GX_ANALYTICS_ENABLED=false OMP_NUM_THREADS=2 dvc repro train
```

DVC запускает подготовку и обучение, хранит версии данных и моделей.
Кеш локальный. Для общего хранилища нужно настроить свой DVC remote.

Train/test: около 80/20, seed=42. Одинаковые профили остаются вместе.
Временной test недоступен: данные содержат один временной срез.
RandomizedSearchCV перебирает шесть вариантов CatBoost на пяти блоках train.
Признаки входят в сохранённый Pipeline. Test оцениваем после выбора параметров.

| CV ROC-AUC | Test ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| 0,7914 | 0,7784 | 0,6756 | 0,3722 | 0,4800 |

Порог 0,5 фиксирован; стоимость ошибок неизвестна.
Результаты в `reports/training/`, модель в `models/model.joblib`.

## Базовая модель

```bash
OMP_NUM_THREADS=2 dvc repro logreg
```

LogisticRegression использует Imputer, StandardScaler и OneHotEncoder.
Они обучаются внутри Pipeline на тех же пяти блоках train.
Подбираем C из трёх значений в `params.json`, затем оцениваем test.
Параметры, метрики, ROC и модель записываются в MLflow (`logreg-baseline`).
Отчёты в `reports/logreg/`, модель в `models/logreg.joblib`.
CV ROC-AUC: 0,7779; test ROC-AUC: 0,7651. У CatBoost CV выше (0,7914),
поэтому API использует CatBoost.

## Расширенный подбор

```bash
OMP_NUM_THREADS=2 dvc repro tune
```

Сравниваем 15 случайных сочетаний и текущий контроль на пяти блоках train.
Шесть параметров и их диапазоны заданы в `params.json`.
Код общий с обучением: `train.py --tune`. Test не читается.
Результаты в `reports/tuning/`, кандидат в `models/candidate.joblib`.

CV ROC-AUC: 0,791576 против 0,791415 у контроля. Выигрыш на двух блоках из пяти.
Прибавка мала, поэтому сохраняем исходную модель. Test выше относится к ней.

## MLflow

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

[MLflow](http://127.0.0.1:5000): эксперимент `credit-default` хранит параметры,
CV-метрики каждого варианта, итоговые test-метрики, ROC и модель.
Сравнивать варианты нужно по `cv_auc`.

## API

После обучения:

```bash
uvicorn credit_default.api:app --host 127.0.0.1 --port 8000
```

Или в Docker: `./scripts/run_api.sh`.

[Документация API](http://127.0.0.1:8000/docs).
`POST /predict` принимает `{"clients": [...]}`: от 1 до 1000 записей,
каждая содержит 23 исходных признака без ID и цели.
Ответ: `class` и `probability` для каждого клиента.
`GET /health` возвращает состояние и хеш модели.

## Мониторинг

При запущенном API:

```bash
python -m credit_default.monitor --rows 1000
```

Часть test отправляем на API и считаем PSI вероятностей относительно train.
Фон привязан к версии модели. Результат в `reports/monitoring.json`.
Это имитация потока из того же среза; фон рассчитан на обучающем train.
PSI показывает сдвиг распределения, но без новых ответов не измеряет качество.

## Анализ

| Ноутбук | Содержание |
|---|---|
| [01_eda](notebooks/01_eda.ipynb) | Качество данных, анализ train и гипотезы |
| [02_experiments](notebooks/02_experiments.ipynb) | Сравнение моделей и выбор признаков |

Для пересчёта исследования:

```bash
python -m pip install -e '.[eda,experiments]'
python -m credit_default.hypotheses
python -m credit_default.hypotheses --model boosting
OMP_NUM_THREADS=2 python -m credit_default.experiments
OMP_NUM_THREADS=2 python -m credit_default.experiments --seed 17 \
    --variants hgb_base cat_base cat_monthly xgb_base
OMP_NUM_THREADS=2 python -m credit_default.calibration --variant cat_base
jupyter lab notebooks/
```

## Структура и проверки

`src/credit_default/`: данные, признаки, обучение, API и мониторинг.
`tests/`: тесты. `notebooks/`: анализ. `reports/`: результаты.
`data/` и `models/`: данные и модели под управлением DVC.
Данные, модели и журнал MLflow не хранятся в Git.

```bash
black --check src tests
flake8 src tests
pytest -q
```

GitHub Actions запускает эти проверки и валидацию Great Expectations.
