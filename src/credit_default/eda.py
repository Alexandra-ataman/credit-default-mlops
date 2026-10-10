"""Разведочный анализ данных.

На основе EDAProcessor (NeKonnnn):
https://github.com/NeKonnnn/Exploratory_Data_Analysis/blob/main/eda_script.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


TITLES = {
    "SEX": "Пол",
    "EDUCATION": "Образование",
    "MARRIAGE": "Семейное положение",
    "PAY_0": "Статус погашения в сентябре 2005 года",
}
LABELS = {
    "SEX": {1: "Мужской", 2: "Женский"},
    "EDUCATION": {
        0: "Код 0: не описан",
        1: "Последипломное образование",
        2: "Университет",
        3: "Старшая школа",
        4: "Другое",
        5: "Код 5: не описан",
        6: "Код 6: не описан",
    },
    "MARRIAGE": {0: "Код 0: не описан", 1: "В браке", 2: "Не в браке", 3: "Другое"},
    "PAY_0": {
        -2: "Код −2: не описан",
        -1: "Платёж вовремя",
        0: "Код 0: не описан",
        **{i: f"Задержка: {i} мес." for i in range(1, 9)},
        9: "Задержка: 9 мес. и более",
    },
}


class EDAProcessor:
    def __init__(self, df, output_dir="reports/figures"):
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_eda_summary(self):
        summary = pd.DataFrame(
            {
                "Тип": self.df.dtypes.astype(str),
                "Пропуски": self.df.isna().sum(),
                "Пропуски, %": (self.df.isna().mean() * 100).round(2),
                "Уникальных": self.df.nunique(),
            }
        )
        columns = self.df.columns
        duplicate_columns = []
        for i, left in enumerate(columns):
            for right in columns[i + 1 :]:
                if self.df[left].equals(self.df[right]):
                    duplicate_columns.append((left, right))
        return {
            "summary": summary,
            "duplicate_count": int(self.df.duplicated().sum()),
            "duplicates": self.df[self.df.duplicated()],
            "duplicate_columns": duplicate_columns,
        }

    def _save(self, ax, name):
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{name}.png", dpi=120)
        plt.show()
        plt.close(fig)

    def plot_target_distribution(self, target_column):
        counts = self.df[target_column].value_counts().sort_index()
        summary = pd.DataFrame(
            {"Количество": counts, "Доля, %": counts / len(self.df) * 100}
        )
        _, ax = plt.subplots(figsize=(5, 3))
        counts.rename(index={0: "Без дефолта", 1: "Дефолт"}).plot.bar(
            ax=ax, rot=0, title="Дефолт в следующем месяце"
        )
        ax.set_xlabel("")
        for bar, count in zip(ax.patches, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                count,
                f"{count:,} ({count / len(self.df):.1%})",
                ha="center",
                va="bottom",
            )
        ax.set_ylim(0, counts.max() * 1.15)
        ax.set_ylabel("Количество клиентов")
        self._save(ax, "target_distribution")
        return summary.round(2)

    def plot_categorical_distributions(self, categorical_columns):
        tables = []
        for col in categorical_columns:
            counts = self.df[col].value_counts(dropna=False).sort_index()
            table = counts.rename("Количество").to_frame()
            table["Доля, %"] = counts / len(self.df) * 100
            table.index = [
                LABELS.get(col, {}).get(value, str(value)) for value in table.index
            ]
            table.insert(0, "Признак", TITLES.get(col, col))
            tables.append(table)
            _, ax = plt.subplots(figsize=(9, max(3, len(counts) * 0.45)))
            table["Доля, %"].plot.barh(ax=ax, title=TITLES.get(col, col))
            ax.invert_yaxis()
            ax.set_xlabel("Доля клиентов, %")
            ax.set_ylabel("")
            ax.set_xlim(0, table["Доля, %"].max() * 1.25)
            for bar, count, share in zip(ax.patches, counts, table["Доля, %"]):
                ax.text(
                    share,
                    bar.get_y() + bar.get_height() / 2,
                    f" {count:,} ({share:.2f}%)",
                    va="center",
                    fontsize=9,
                )
            self._save(ax, f"categories_{col}")
        return pd.concat(tables).round(2)

    def plot_categorical_vs_target(self, target_column, categorical_columns):
        tables = []
        for col in categorical_columns:
            table = self.df.groupby(col, dropna=False)[target_column].agg(
                ["size", "mean"]
            )
            table.columns = ["Количество", "Доля дефолта"]
            table.index = [
                LABELS.get(col, {}).get(value, str(value)) for value in table.index
            ]
            table.insert(0, "Признак", TITLES.get(col, col))
            tables.append(table)
            _, ax = plt.subplots(figsize=(6, 3))
            table["Доля дефолта"].plot.bar(ax=ax, rot=30, title=TITLES.get(col, col))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlabel("")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Доля дефолта")
            self._save(ax, f"default_by_{col}")
        return pd.concat(tables)

    def find_rare_categories(self, categorical_columns, threshold=0.05):
        rare = {}
        for col in categorical_columns:
            shares = self.df[col].value_counts(normalize=True)
            rare[TITLES.get(col, col)] = [
                LABELS.get(col, {}).get(value, str(value))
                for value in shares[shares < threshold].index
            ]
        return rare
