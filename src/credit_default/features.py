import numpy as np
import pandas as pd

from credit_default.data import BILLS, PAY, PAYMENTS


AGE_LABELS = ["До 30", "30–39", "40–49", "50–59", "60+"]
NEW_NUMERIC = [
    "delay_months",
    "severe_delay_months",
    "max_delay",
    "previous_delay_months",
    "bill_to_limit",
    "mean_bill_to_limit",
    "bill_change_to_limit",
    "payment_sum",
    "zero_payment_months",
    "payment_change_to_limit",
    "payment_to_previous_bill",
]


def add_features(df):
    result = df.copy()
    # Только данные одной строки: средние по выборке и таргет не используем.
    result["AGE_BAND"] = pd.cut(
        df["AGE"], bins=[0, 29, 39, 49, 59, np.inf], labels=AGE_LABELS
    )
    result["delay_months"] = df[PAY].gt(0).sum(axis=1)
    result["severe_delay_months"] = df[PAY].ge(2).sum(axis=1)
    result["max_delay"] = df[PAY].clip(lower=0).max(axis=1)
    result["previous_delay_months"] = df[PAY[1:]].gt(0).sum(axis=1)
    result["bill_to_limit"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
    result["mean_bill_to_limit"] = df[BILLS].mean(axis=1) / df["LIMIT_BAL"]
    # BILL_AMT1 — сентябрь, BILL_AMT6 — апрель.
    result["bill_change_to_limit"] = (df["BILL_AMT1"] - df["BILL_AMT6"]) / df[
        "LIMIT_BAL"
    ]
    result["payment_sum"] = df[PAYMENTS].sum(axis=1)
    result["zero_payment_months"] = df[PAYMENTS].eq(0).sum(axis=1)
    recent = df[PAYMENTS[:3]].mean(axis=1)
    older = df[PAYMENTS[3:]].mean(axis=1)
    result["payment_change_to_limit"] = (recent - older) / df["LIMIT_BAL"]
    # Условное соотношение сумм, не доля погашения обязательного платежа.
    previous_bill = df["BILL_AMT2"].where(df["BILL_AMT2"] > 0)
    result["payment_to_previous_bill"] = df["PAY_AMT1"] / previous_bill
    return result


SEQUENCE_NUMERIC = [
    "longest_delay_run",
    "current_delay_run",
    "months_since_delay",
    "delay_count_change",
    "known_status_change",
]
MONTHLY_NUMERIC = (
    [f"bill_to_limit_{i}" for i in range(2, 7)]
    + [f"payment_to_limit_{i}" for i in range(1, 7)]
    + ["bill_std_to_limit", "payment_std_to_limit"]
)


def add_history_features(df):
    result = add_features(df)
    # PAY идёт от сентября к апрелю. Серия — только соседние положительные коды.
    delayed = df[PAY].gt(0).to_numpy()
    current = np.zeros(len(df), dtype=int)
    longest = np.zeros(len(df), dtype=int)
    running = np.zeros(len(df), dtype=int)
    for i in range(len(PAY)):
        running = np.where(delayed[:, i], running + 1, 0)
        longest = np.maximum(longest, running)
        current += delayed[:, : i + 1].all(axis=1)
    result["longest_delay_run"] = longest
    result["current_delay_run"] = current
    # 6 означает: положительного кода задержки нет во всём окне.
    result["months_since_delay"] = np.where(
        delayed.any(axis=1), delayed.argmax(axis=1), 6
    )
    result["delay_count_change"] = df[PAY[:3]].gt(0).sum(axis=1) - df[PAY[3:]].gt(
        0
    ).sum(axis=1)
    # Для изменения тяжести используем только описанные коды: -1 и >0.
    known = df[PAY].where(df[PAY].eq(-1) | df[PAY].gt(0)).clip(lower=0)
    result["known_status_change"] = known[PAY[0]] - known[PAY[1]]
    for i in range(2, 7):
        result[f"bill_to_limit_{i}"] = df[f"BILL_AMT{i}"] / df["LIMIT_BAL"]
    for i in range(1, 7):
        result[f"payment_to_limit_{i}"] = df[f"PAY_AMT{i}"] / df["LIMIT_BAL"]
    result["bill_std_to_limit"] = df[BILLS].std(axis=1) / df["LIMIT_BAL"]
    result["payment_std_to_limit"] = df[PAYMENTS].std(axis=1) / df["LIMIT_BAL"]
    return result
