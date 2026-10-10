import argparse
import json
from pathlib import Path

import great_expectations as gx
import pandas as pd
from great_expectations import expectations as expect

from credit_default.data import COLUMNS, PAY, PAYMENTS, TARGET


def build_suite() -> gx.ExpectationSuite:
    # Собираем правила: нужные колонки, хотя бы одна строка, без пропусков.
    suite = gx.ExpectationSuite(name="credit_clients")
    suite.add_expectation(
        expect.ExpectTableColumnsToMatchOrderedList(column_list=COLUMNS)
    )
    suite.add_expectation(expect.ExpectTableRowCountToBeBetween(min_value=1))
    for col in COLUMNS:
        suite.add_expectation(expect.ExpectColumnValuesToNotBeNull(column=col))
        # В исходном датасете все значения целочисленные, включая суммы
        suite.add_expectation(
            expect.ExpectColumnValuesToBeInTypeList(
                column=col, type_list=["int64", "int32"]
            )
        )
    suite.add_expectation(expect.ExpectColumnValuesToBeUnique(column="ID"))
    # Неописанные коды EDUCATION=0/5/6 и MARRIAGE=0 тоже есть в файле.
    categories = {
        "SEX": [1, 2],
        "EDUCATION": list(range(7)),
        "MARRIAGE": list(range(4)),
        TARGET: [0, 1],
    }
    for col in PAY:
        # −2 и 0 есть в файле; задержки до 9 описаны в UCI.
        categories[col] = list(range(-2, 10))
    for col, values in categories.items():
        suite.add_expectation(
            expect.ExpectColumnValuesToBeInSet(column=col, value_set=values)
        )
    # Возраст 18–100 - личное правило. None означает отсутствие верхней границы.
    bounds = {"ID": (1, None), "AGE": (18, 100), "LIMIT_BAL": (1, None)}
    for col in PAYMENTS:
        bounds[col] = (0, None)
    # BILL_AMT может быть отрицательным, PAY_AMT нет.
    for col, (low, high) in bounds.items():
        suite.add_expectation(
            expect.ExpectColumnValuesToBeBetween(
                column=col, min_value=low, max_value=high
            )
        )
    return suite


def validate(frame: pd.DataFrame) -> dict:
    # Сначала проверяем структуру, чтобы GX не обращался к отсутствующим колонкам.
    if list(frame.columns) != COLUMNS:
        return {"success": False, "failures": ["Unexpected columns or column order"]}
    # Контекст GX живёт в памяти
    context = gx.get_context(mode="ephemeral")
    context.enable_analytics(False)
    source = context.data_sources.add_pandas(name="clients")
    asset = source.add_dataframe_asset(name="incoming")
    # Проверяем весь DataFrame одним набором правил
    batch = asset.add_batch_definition_whole_dataframe("all_rows")
    suite = context.suites.add(build_suite())
    definition = gx.ValidationDefinition(name="incoming", data=batch, suite=suite)
    definition = context.validation_definitions.add(definition)
    result = definition.run(batch_parameters={"dataframe": frame})
    # В отчёт включаем только нарушенные правила и названия колонок.
    failures = []
    for item in result.results:
        if not item.success:
            failures.append(
                {
                    "expectation": item.expectation_config.type,
                    "column": item.expectation_config.kwargs.get("column"),
                }
            )
    return {"success": bool(result.success), "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    result = validate(pd.read_csv(args.path))
    print(json.dumps(result, indent=2))
    if not result["success"]:
        # Ненулевой код завершения останавливает шаг CI.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
