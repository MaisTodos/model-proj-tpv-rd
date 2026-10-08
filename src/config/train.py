"""
Orquestrador da Pipeline Batch: Integração de Feature Engineering, Treino e MLflow.
"""
from typing import Tuple
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from src.core.base import BaseModel, BasePreprocessor
from src.core.tracker import TrackerClient
from src.core.telemetria import get_logger
from src.config.feature import instantiate_class, load_config, run_feature_engineering


logger = get_logger("mlops.training")

def split_data(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa features e target conforme definido no project.yml."""
    target_col = config["target"]
    features_cols = config["features"]
    
    if target_col not in df.columns:
        raise ValueError(f"Coluna target '{target_col}' não encontrada no DataFrame.")
        
    X = df[features_cols] if features_cols and features_cols[0] != "" else df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def run_training(df: pd.DataFrame, preprocessor: BasePreprocessor, config: dict) -> None:
    """
    Executa o treinamento do modelo encapsulado pelo Tracker do MLflow.
    """
    logger.info("Iniciando pipeline de treinamento")
    
    # 1. Preparação dos Dados
    X_train, y_train = split_data(df, config)
    
    # 2. Instanciação do Modelo Analítico
    module_name = config["model_entrypoint"]["module"]
    model_class_name = config["model_entrypoint"]["model_class"]
    model_instance: BaseModel = instantiate_class(module_name, model_class_name)
    
    # 3. Configuração do Tracking (ADR-003 e ADR-004)
    experiment_name = config["tracking"]["experiment_name"]
    tracker = TrackerClient(experiment_name=experiment_name)
    
    with tracker.start_run(run_name=f"train_{config['version']}") as run:
        try:
            # Log de hiperparâmetros
            params = model_instance.get_params()
            tracker.log_params(params)
            
            # Execução do Treino (Lógica do Cientista)
            logger.info("Treinando modelo", model_class=model_class_name)
            trained_model = model_instance.train(X_train, y_train)
            
            # Validação e Métricas Mínimas de Plataforma (Exemplo: RMSE em treino)
            predictions = model_instance.predict(trained_model, X_train)
            rmse = root_mean_squared_error(y_train, predictions)
            tracker.log_metric("train_rmse", rmse)
            
            # Registro de Artefatos: modelo e preprocessor ajustado (necessário para a
            # inferência poder reconstruir o estado do `fit_transform` — ver predict.py)
            tracker.log_model(model=trained_model, artifact_path="model")
            tracker.log_model(model=preprocessor, artifact_path="preprocessor")
            tracker.register_model(run_id=run.info.run_id, model_name=config["name"])
            
            logger.info("Treinamento finalizado com sucesso", run_id=run.info.run_id, rmse=rmse)
            
        except Exception as e:
            logger.error("Falha na etapa de treinamento", error=str(e))
            raise

def main():
    """Entrypoint do Job SageMaker."""
    config = load_config()
    
    # Etapa 1: Extração e Feature Engineering
    processed_data, preprocessor = run_feature_engineering(config, is_training=True)

    # Etapa 2: Treinamento e Registro
    run_training(processed_data, preprocessor, config)

if __name__ == "__main__":
    main()