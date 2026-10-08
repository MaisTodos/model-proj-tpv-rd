"""
Contratos abstratos para os Cientistas de Dados (ADR-002).
Garante padronização e permite execução batch isolada da infraestrutura.
"""
import abc
from typing import Any, Dict
import pandas as pd

class BasePreprocessor(abc.ABC):
    """
    Contrato para processamento de dados e feature engineering.
    O Cientista de Dados implementa a lógica específica (one-hot encoding, scaling, etc.).
    """
    
    @abc.abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformações na base de treino e guarda estados."""
        pass

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformações na base de inferência (batch) usando estados do fit."""
        pass


class BaseModel(abc.ABC):
    """
    Contrato para algoritmos de Machine Learning.
    Força a padronização na devolução de hiperparâmetros e execução.
    """

    @abc.abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Retorna hiperparâmetros (ex: test_size, random_state, max_depth).
        Usado automaticamente pelo Core para o MLflow.
        """
        pass

    @abc.abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Executa o treinamento e retorna o objeto do modelo treinado.
        """
        pass

    @abc.abstractmethod
    def predict(self, model: Any, X: pd.DataFrame) -> Any:
        """
        Executa a inferência batch sobre os dados de entrada.
        """
        pass