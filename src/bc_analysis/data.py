"""数据加载与清洗。

默认使用 scikit-learn 自带的 Wisconsin Diagnostic Breast Cancer (WDBC) 数据集，
无需联网即可复现；也支持从同结构的 CSV 文件读取。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

TARGET = "diagnosis"
POSITIVE_LABEL = "malignant"
NEGATIVE_LABEL = "benign"


def _snake(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def load_raw(csv_path: str | Path | None = None) -> pd.DataFrame:
    """读取原始数据。

    csv_path 为空时使用 scikit-learn 内置数据集；否则读取 CSV，
    CSV 需包含 30 个特征列和一个 `diagnosis` 列（值为 M/B 或 malignant/benign）。
    """
    if csv_path is None:
        bunch = load_breast_cancer(as_frame=True)
        df = bunch.frame.copy()
        # sklearn 中 target: 0 = malignant, 1 = benign
        df[TARGET] = df.pop("target").map({0: POSITIVE_LABEL, 1: NEGATIVE_LABEL})
        return df
    return pd.read_csv(csv_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名、标准化标签、去除重复与缺失、剔除无关列。"""
    out = df.copy()
    out.columns = [_snake(c) for c in out.columns]
    # 常见公开版本（如 Kaggle）带有 id 和空的 Unnamed 列
    drop = [c for c in out.columns if c == "id" or c.startswith("unnamed")]
    out = out.drop(columns=drop)

    if TARGET not in out.columns:
        raise ValueError(f"数据中缺少标签列 '{TARGET}'")

    out[TARGET] = (
        out[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"m": POSITIVE_LABEL, "b": NEGATIVE_LABEL})
    )
    unknown = set(out[TARGET].unique()) - {POSITIVE_LABEL, NEGATIVE_LABEL}
    if unknown:
        raise ValueError(f"未知的诊断标签: {sorted(unknown)}")

    out = out.drop_duplicates().dropna().reset_index(drop=True)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """返回特征矩阵与二值标签（1 = 恶性）。"""
    X = df[feature_columns(df)]
    y = (df[TARGET] == POSITIVE_LABEL).astype(int).rename("is_malignant")
    return X, y


def load_dataset(csv_path: str | Path | None = None) -> pd.DataFrame:
    return clean(load_raw(csv_path))
