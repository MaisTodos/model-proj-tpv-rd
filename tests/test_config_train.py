"""
split_data: separa X/y conforme project.yml. O caminho de erro (target ausente)
nunca tinha teste - é exatamente o tipo de "processo crítico não validado".
"""
import pandas as pd
import pytest

from src.config.train import split_data

DF = pd.DataFrame({
    "feature_1": [1.0, 2.0],
    "feature_2": [3.0, 4.0],
    "target_column": [10, 20],
})


def test_split_data_raises_when_target_column_missing():
    config = {"target": "coluna_que_nao_existe", "features": []}
    with pytest.raises(ValueError, match="coluna_que_nao_existe"):
        split_data(DF, config)


def test_split_data_with_empty_features_drops_only_target():
    X, y = split_data(DF, {"target": "target_column", "features": []})
    assert list(X.columns) == ["feature_1", "feature_2"]
    assert y.tolist() == [10, 20]


def test_split_data_with_explicit_features_selects_only_those():
    X, y = split_data(DF, {"target": "target_column", "features": ["feature_1"]})
    assert list(X.columns) == ["feature_1"]
    assert y.tolist() == [10, 20]
