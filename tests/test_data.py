import pandas as pd
import pytest

from bc_analysis.data import TARGET, clean, feature_columns, load_dataset, split_xy


def test_load_dataset_shape_and_labels():
    df = load_dataset()
    assert df.shape == (569, 31)
    assert set(df[TARGET]) == {"malignant", "benign"}
    assert (df[TARGET] == "malignant").sum() == 212
    assert all(" " not in c for c in df.columns)


def test_clean_handles_kaggle_style_csv():
    raw = pd.DataFrame(
        {
            "id": [1, 2, 3, 3],
            "diagnosis": ["M", "B", " b ", " b "],
            "radius_mean": [17.9, 12.0, 13.1, 13.1],
            "Unnamed: 32": [None, None, None, None],
        }
    )
    out = clean(raw)
    assert list(out.columns) == ["diagnosis", "radius_mean"]
    assert list(out[TARGET]) == ["malignant", "benign", "benign"]
    assert len(out) == 3  # 去掉 id 后第 3、4 行重复


def test_clean_rejects_unknown_labels():
    with pytest.raises(ValueError):
        clean(pd.DataFrame({"diagnosis": ["X"], "f": [1.0]}))


def test_split_xy_positive_class_is_malignant():
    df = load_dataset()
    X, y = split_xy(df)
    assert list(X.columns) == feature_columns(df)
    assert y.sum() == 212
