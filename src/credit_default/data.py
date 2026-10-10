import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
TARGET = "default.payment.next.month"
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
# PAY = статусы погашения, BILL_AMT = счета, PAY_AMT = суммы платежей.
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]
COLUMNS = ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
COLUMNS += PAY + BILLS + PAYMENTS + [TARGET]


def download(destination: Path) -> None:
    # Уже скачанный файл используем повторно.
    if destination.exists():
        return
    with urlopen(URL, timeout=60) as response:
        archive = response.read()
    with ZipFile(io.BytesIO(archive)) as zipped:
        content = zipped.read("default of credit card clients.xls")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def read_source(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, header=1).rename(
        columns={"default payment next month": TARGET}
    )


def summarize(frame: pd.DataFrame) -> dict:
    # Краткая сводка: размер, пропуски, повторы и частоты категорий.
    return {
        "rows": len(frame),
        "features": len(frame.columns) - 2,  # Не считаем ID и таргет.
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_ids": int(frame["ID"].duplicated().sum()),
        "duplicate_profiles": int(frame.drop(columns="ID").duplicated().sum()),
        "default_rate": float(frame[TARGET].mean()),
        "target_counts": frame[TARGET].value_counts().sort_index().to_dict(),
        "categories": {
            col: frame[col].value_counts().sort_index().to_dict()
            for col in ["SEX", "EDUCATION", "MARRIAGE"] + PAY
        },
    }


def main() -> None:
    from credit_default.validation import validate

    parser = argparse.ArgumentParser()
    # Пути можно переопределить аргументами командной строки
    parser.add_argument("--raw", type=Path, default=Path("data/raw/clients.xls"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/clients.csv")
    )
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    download(args.raw)
    frame = read_source(args.raw)
    args.reports.mkdir(parents=True, exist_ok=True)
    # Сохраняем результат проверки, даже если данные её не прошли
    result = validate(frame)
    (args.reports / "validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    if not result["success"]:
        raise ValueError(f"Data validation failed: {result['failures']}")
    # CSV создаём только после успешной проверки
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    summary = summarize(frame)
    # Хеш позволяет сравнить, один и тот же ли исходный файл использовался
    summary["source_sha256"] = hashlib.sha256(args.raw.read_bytes()).hexdigest()
    (args.reports / "data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    # Отдельная таблица диапазонов, средних и квартилей
    frame.describe().T.to_csv(args.reports / "numeric_summary.csv")
    print(f"Saved {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
