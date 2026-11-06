"""把分析结果写成 Markdown 报告。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .data import NEGATIVE_LABEL, POSITIVE_LABEL, TARGET
from .model import ModelResults


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """不依赖 tabulate 的简易 Markdown 表格。"""
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_report(
    df: pd.DataFrame,
    balance: pd.DataFrame,
    tests: pd.DataFrame,
    pairs: pd.DataFrame,
    results: ModelResults,
    figures: dict[str, Path],
    out_path: Path,
) -> Path:
    out_dir = out_path.parent
    rel = {k: v.relative_to(out_dir).as_posix() for k, v in figures.items()}

    n = len(df)
    n_mal = int(balance.loc[POSITIVE_LABEL, "count"])
    n_ben = int(balance.loc[NEGATIVE_LABEL, "count"])
    n_sig = int(tests["significant"].sum())
    top = tests.head(5)
    weak = tests.tail(3)
    tm = results.test_metrics
    tn, fp, fn, tp = results.confusion.ravel()

    top_tbl = top.assign(
        feature=top["feature"].str.replace("_", " "),
        p_adj=top["p_adj"].map(lambda p: f"{p:.1e}"),
    )[["feature", "mean_malignant", "mean_benign", "cohens_d", "p_adj"]]
    top_tbl.columns = ["特征", "恶性均值", "良性均值", "Cohen's d", "校正后 p 值"]

    cv = results.cv_scores
    cv_tbl = pd.DataFrame(
        {
            "模型": cv["model"],
            "ROC AUC": [f"{m:.3f} ± {s:.3f}" for m, s in zip(cv["roc_auc_mean"], cv["roc_auc_std"])],
            "准确率": [f"{m:.3f} ± {s:.3f}" for m, s in zip(cv["accuracy_mean"], cv["accuracy_std"])],
            "召回率": [f"{m:.3f} ± {s:.3f}" for m, s in zip(cv["recall_mean"], cv["recall_std"])],
            "F1": [f"{m:.3f} ± {s:.3f}" for m, s in zip(cv["f1_mean"], cv["f1_std"])],
        }
    )

    coefs = results.coefficients
    pos = coefs.sort_values(ascending=False).head(3)
    neg = coefs.sort_values().head(2)

    pairs_txt = ""
    if len(pairs):
        examples = "、".join(
            f"`{a}` 与 `{b}`（r = {r:.2f}）" for a, b, r in pairs.head(3).itertuples(index=False)
        )
        pairs_txt = (
            f"共有 **{len(pairs)}** 对特征的相关系数绝对值 ≥ 0.9，例如 {examples}。"
            "半径、周长和面积本质上描述的是同一件事（细胞核大小），"
            "因此建模时使用了带正则化的模型，解读单个系数时也需要谨慎。"
        )

    text = f"""# 乳腺肿瘤诊断数据分析报告

> 自动生成于 {date.today().isoformat()} · 数据：Wisconsin Diagnostic Breast Cancer (WDBC)

## 1. 核心结论

- 数据共 **{n}** 个样本、{df.shape[1] - 1} 个特征，其中恶性 **{n_mal}** 例（{_pct(n_mal / n)}），良性 **{n_ben}** 例（{_pct(n_ben / n)}），类别轻度不平衡。
- 30 个特征中有 **{n_sig}** 个在两组之间存在显著差异（Mann-Whitney U 检验，Benjamini-Hochberg 校正，α = 0.05）。
- 区分能力最强的是描述**细胞核大小与轮廓凹陷程度**的特征：`{top.iloc[0]['feature']}`（d = {top.iloc[0]['cohens_d']:.2f}）、`{top.iloc[1]['feature']}`（d = {top.iloc[1]['cohens_d']:.2f}）、`{top.iloc[2]['feature']}`（d = {top.iloc[2]['cohens_d']:.2f}）。
- 交叉验证表现最好的模型是 **{results.best_name}**。在留出的 {tm['n_test']} 个测试样本上：ROC AUC **{tm['roc_auc']:.3f}**，准确率 **{_pct(tm['accuracy'])}**，恶性召回率 **{_pct(tm['recall'])}**（漏诊 {fn} 例，误报 {fp} 例）。

## 2. 数据概况与清洗

- 来源：scikit-learn 内置的 WDBC 数据集（UCI 机器学习库），每个样本由乳腺肿块细针穿刺的数字化图像计算得到。
- 每类细胞核测量（半径、纹理、周长、面积、光滑度、紧密度、凹度、凹点、对称性、分形维数）各有三种统计量：`mean`（均值）、`error`（标准误）、`worst`（最差三个值的均值）。
- 清洗步骤：列名统一为 snake_case；诊断标签统一为 `malignant` / `benign`；去除重复行与缺失值（本数据集均为 0）。

## 3. 哪些特征能区分恶性与良性？

对每个特征比较恶性组和良性组。由于多数特征呈右偏分布，使用非参数的 Mann-Whitney U 检验，并用 Cohen's d 衡量差异大小（|d| ≥ 0.8 通常视为大效应）。

![效应量]({rel['effect_sizes']})

效应量最大的 5 个特征：

{md_table(top_tbl)}

差异最小的特征是 {"、".join(f"`{f}`" for f in weak["feature"])}，它们几乎不携带诊断信息。

![分布对比]({rel['distributions']})

恶性样本的细胞核明显更大、轮廓凹陷更多，两组分布在这些特征上几乎可以被一条阈值分开。

## 4. 特征之间的相关性

![相关性热力图]({rel['correlation']})

{pairs_txt}

## 5. 预测模型

把数据按 8 : 2 分层划分为训练集和测试集，在训练集上用 5 折分层交叉验证比较三种模型（均值 ± 标准差）：

{md_table(cv_tbl)}

![ROC 曲线]({rel['roc']})

最佳模型 **{results.best_name}** 在测试集上的混淆矩阵：

![混淆矩阵]({rel['confusion']})

| 指标 | 数值 |
| --- | --- |
| ROC AUC | {tm['roc_auc']:.3f} |
| 准确率 | {_pct(tm['accuracy'])} |
| 召回率（恶性被识别出的比例） | {_pct(tm['recall'])} |
| 精确率（判为恶性中真正恶性的比例） | {_pct(tm['precision'])} |
| F1 | {tm['f1']:.3f} |

在医疗筛查场景中，**漏诊（假阴性）的代价远高于误报**，因此召回率比准确率更值得关注；实际应用中可以通过降低判定阈值来进一步提高召回率。

## 6. 模型在看什么？

逻辑回归的标准化系数表示：特征每增加一个标准差，恶性的对数几率变化多少。

![特征系数]({rel['coefficients']})

推动预测偏向恶性的主要特征是 {"、".join(f"`{f}`" for f in pos.index)}；{"、".join(f"`{f}`" for f in neg.index)} 的系数为负。由于特征间存在强共线性，单个系数的正负不宜孤立解读，应结合第 3 节的单变量分析一起看。

## 7. 局限性

- 样本量较小（{n} 例）且来自单一机构，模型的泛化能力需要在外部数据上验证。
- 本项目用于学习数据分析方法，**不构成任何医疗建议**。
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
