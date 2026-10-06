"""
Garante que os contratos do ADR-002 realmente travam: uma classe de Cientista de
Dados que esqueça de implementar um método abstrato deve falhar na instanciação
(erro claro e imediato), não silenciosamente mais adiante com um AttributeError
em algum ponto profundo do pipeline.
"""
import pytest

from src.core.base import BaseModel, BasePreprocessor


def test_base_preprocessor_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BasePreprocessor()


def test_base_model_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseModel()


def test_preprocessor_missing_transform_fails_at_instantiation():
    class Incomplete(BasePreprocessor):
        def fit_transform(self, df):
            return df
        # transform() não implementado

    with pytest.raises(TypeError):
        Incomplete()


def test_model_missing_predict_fails_at_instantiation():
    class Incomplete(BaseModel):
        def get_params(self):
            return {}

        def train(self, X_train, y_train):
            return None
        # predict() não implementado

    with pytest.raises(TypeError):
        Incomplete()


def test_fully_implemented_preprocessor_instantiates():
    class Complete(BasePreprocessor):
        def fit_transform(self, df):
            return df

        def transform(self, df):
            return df

    Complete()  # não deve lançar


def test_fully_implemented_model_instantiates():
    class Complete(BaseModel):
        def get_params(self):
            return {}

        def train(self, X_train, y_train):
            return None

        def predict(self, model, X):
            return None

    Complete()  # não deve lançar
