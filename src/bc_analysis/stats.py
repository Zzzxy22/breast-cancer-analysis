"""描述统计与假设检验。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import NEGATIVE_LABEL, POSITIVE_LABEL, TARGET, feature_columns


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET].value_counts()
    return pd.DataFrame(
        {"count": counts, "share": (counts / counts.sum()).round(4)}
    ).rename_axis(TARGET)


def summary_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """每个特征在良性 / 恶性两组中的均值与标准差。"""
    grouped = df.groupby(TARGET)[feature_columns(df)].agg(["mean", "std"])
    return grouped.T.unstack(level=1)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """合并标准差下的 Cohen's d（a 相对 b）。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def compare_groups(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """对每个特征做 Mann-Whitney U 检验（不假设正态），并给出 Cohen's d。

    p 值使用 Benjamini-Hochberg 方法校正多重比较。
    """
    mal = df[df[TARGET] == POSITIVE_LABEL]
    ben = df[df[TARGET] == NEGATIVE_LABEL]
    rows = []
    for col in feature_columns(df):
        _, p = stats.mannwhitneyu(mal[col], ben[col], alternative="two-sided")
        rows.append(
            {
                "feature": col,
                "mean_malignant": mal[col].mean(),
                "mean_benign": ben[col].mean(),
                "cohens_d": cohens_d(mal[col], ben[col]),
                "p_value": p,
            }
        )
    res = pd.DataFrame(rows)
    res["p_adj"] = stats.false_discovery_control(res["p_value"], method="bh")
    res["significant"] = res["p_adj"] < alpha
    res["abs_d"] = res["cohens_d"].abs()
    return res.sort_values("abs_d", ascending=False).drop(columns="abs_d").reset_index(drop=True)


def correlated_pairs(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """列出相关系数绝对值超过阈值的特征对（多重共线性提示）。"""
    corr = df[feature_columns(df)].corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("r").reset_index()
    pairs.columns = ["feature_a", "feature_b", "r"]
    pairs = pairs[pairs["r"].abs() >= threshold]
    return pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)
