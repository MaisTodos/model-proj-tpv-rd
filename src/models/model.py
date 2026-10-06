"""
Utilizar este arquivo para modificar e implementar as regras de negócio,
transformações e algoritmos de Machine Learning.
"""
import pandas as pd
from typing import Any, Dict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Importação dos contratos de MLOps
from src.core.base import BasePreprocessor, BaseModel


class TemplatePreprocessor(BasePreprocessor):
    """
    Define a engenharia de features e o tratamento de variáveis.
    """
    def __init__(self):
        # Definição das colunas que sofrerão transformação
        self.numeric_features = ["feature_1", "feature_2"]
        self.categorical_features = ["feature_3"]
        self.feature_columns = self.numeric_features + self.categorical_features

        # Pipeline de transformações do Scikit-Learn. Opera só sobre feature_columns:
        # colunas extras (ex: a coluna target, presente só em treino) nunca entram aqui,
        # para que o schema ajustado no fit seja o mesmo em treino e inferência.
        self.transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_features)
            ],
            remainder="drop"
        )

    def _to_output_df(self, transformed_array, df: pd.DataFrame) -> pd.DataFrame:
        """Reconstrói o DataFrame de saída, reanexando colunas fora de feature_columns."""
        feature_names = self.numeric_features + \
                        list(self.transformer.named_transformers_["cat"].get_feature_names_out())
        result = pd.DataFrame(transformed_array, columns=feature_names, index=df.index)
        extra_cols = [c for c in df.columns if c not in self.feature_columns]
        return pd.concat([result, df[extra_cols]], axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executado pela pipeline de treino.
        Limpa os dados, ajusta os transformadores (fit) e aplica a transformação.
        """
        df_clean = df.fillna(0)
        transformed_array = self.transformer.fit_transform(df_clean[self.feature_columns])
        return self._to_output_df(transformed_array, df_clean)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executado pela pipeline de inferência batch.
        Utiliza o estado previamente ajustado (fit) para transformar novos dados.
        """
        df_clean = df.fillna(0)
        transformed_array = self.transformer.transform(df_clean[self.feature_columns])
        return self._to_output_df(transformed_array, df_clean)


class TemplateModelTrainer(BaseModel):
    """
    Define o algoritmo, hiperparâmetros e a lógica de inferência.
    """
    def __init__(self):
        # Hiperparâmetros
        self.n_estimators = 100
        self.max_depth = 5
        self.random_state = 42
        
        # Instância do algoritmo
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def get_params(self) -> Dict[str, Any]:
        """
        A infraestrutura (TrackerClient) lê este método para registrar no MLflow.
        """
        return {
            "model_type": "RandomForestRegressor",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state
        }

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        A pipeline injeta o X_train e y_train já separados.
        """
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, model: Any, X: pd.DataFrame) -> Any:
        """
        A pipeline passa o modelo carregado do MLflow e os dados processados para inferência.
        """
        return model.predict(X)