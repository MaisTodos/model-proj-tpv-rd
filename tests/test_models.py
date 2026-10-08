"""
TemplatePreprocessor/TemplateModelTrainer: o exemplo de referência do template.
Cobre justamente os casos que já quebraram o pipeline de inferência nesta sessão
(coluna extra tipo target aprendida no fit, categoria nova na inferência) para
não regredir silenciosamente.
"""
import pandas as pd
import pytest

from src.models.model import TemplateModelTrainer, TemplatePreprocessor

TRAIN_DF = pd.DataFrame({
    "feature_1": [1.0, 2.0, 3.0, 4.0],
    "feature_2": [4.0, 3.0, 2.0, 1.0],
    "feature_3": ["a", "b", "a", "b"],
    "target_column": [10, 20, 15, 25],
})


def test_fit_transform_one_hot_encodes_categorical_feature():
    preprocessor = TemplatePreprocessor()
    out = preprocessor.fit_transform(TRAIN_DF)
    assert "feature_3_a" in out.columns
    assert "feature_3_b" in out.columns
    assert "feature_3" not in out.columns


def test_fit_transform_preserves_columns_outside_feature_set():
    """A coluna target (presente só em treino) precisa sobreviver ao fit_transform
    para o split_data conseguir separar X/y depois."""
    preprocessor = TemplatePreprocessor()
    out = preprocessor.fit_transform(TRAIN_DF)
    assert "target_column" in out.columns
    assert out["target_column"].tolist() == TRAIN_DF["target_column"].tolist()


def test_transform_after_fit_does_not_require_target_column():
    """Regressão do bug real: o ColumnTransformer não pode exigir a coluna target
    na inferência (ela nunca existe lá)."""
    preprocessor = TemplatePreprocessor()
    preprocessor.fit_transform(TRAIN_DF)

    infer_df = pd.DataFrame({
        "feature_1": [5.0],
        "feature_2": [6.0],
        "feature_3": ["a"],
    })
    out = preprocessor.transform(infer_df)
    assert "target_column" not in out.columns
    assert len(out) == 1


def test_transform_handles_unseen_categorical_value_without_raising():
    preprocessor = TemplatePreprocessor()
    preprocessor.fit_transform(TRAIN_DF)

    infer_df = pd.DataFrame({
        "feature_1": [5.0],
        "feature_2": [6.0],
        "feature_3": ["categoria_nunca_vista"],
    })
    out = preprocessor.transform(infer_df)  # não deve lançar (handle_unknown="ignore")
    assert out["feature_3_a"].iloc[0] == 0
    assert out["feature_3_b"].iloc[0] == 0


def test_transform_without_fit_raises_not_fitted_error():
    """Garante que usar um preprocessor não ajustado falha alto - é exatamente o
    bug que quebrava a inferência antes de o preprocessor passar a ser carregado
    do MLflow em vez de reinstanciado do zero (ver src/config/predict.py)."""
    from sklearn.exceptions import NotFittedError

    preprocessor = TemplatePreprocessor()
    with pytest.raises(NotFittedError):
        preprocessor.transform(TRAIN_DF)


def test_model_trainer_get_params_reports_hyperparameters():
    trainer = TemplateModelTrainer()
    params = trainer.get_params()
    assert params["model_type"] == "RandomForestRegressor"
    assert params["n_estimators"] == 100


def test_model_trainer_train_then_predict_roundtrip():
    preprocessor = TemplatePreprocessor()
    processed = preprocessor.fit_transform(TRAIN_DF)
    X = processed.drop(columns=["target_column"])
    y = processed["target_column"]

    trainer = TemplateModelTrainer()
    fitted_model = trainer.train(X, y)
    predictions = trainer.predict(fitted_model, X)

    assert len(predictions) == len(TRAIN_DF)
