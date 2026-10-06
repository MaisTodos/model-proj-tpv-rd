"""
Orquestrador da Pipeline Batch: Extração e Feature Engineering.
"""
import yaml
import importlib
from pathlib import Path
from typing import Any, Tuple
import pandas as pd

from src.aws.athena_client import AthenaClient
from src.core.base import BasePreprocessor
from src.core.telemetria import get_logger

logger = get_logger("mlops.pipeline")

def load_config(config_path: str = "project.yml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def read_sql_file(filepath: str) -> str:
    """Lê a query do arquivo .sql bruto."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo SQL não encontrado: {filepath}")
    return path.read_text()

def instantiate_class(module_name: str, class_name: str) -> Any:
    """Instancia dinamicamente a classe analítica definida no project.yml."""
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()

def run_feature_engineering(config: dict, is_training: bool = True) -> Tuple[pd.DataFrame, BasePreprocessor]:
    """
    Executa a extração via Athena e aplica o contrato do BasePreprocessor.
    Retorna o DataFrame processado e a instância do preprocessor (ajustada, se `is_training`),
    para que o estado do `fit` possa ser persistido pelo chamador (ver `train.run_training`).
    """
    logger.info("Iniciando etapa de Extração e Feature Engineering", is_training=is_training)

    # 1. Extração de Dados (Athena)
    sql_path = config["data"]["query_file"]
    query = read_sql_file(sql_path)

    athena = AthenaClient()
    raw_df = athena.execute_query(query)

    # 2. Instanciação do Pré-processador do Cientista de Dados
    module_name = config["model_entrypoint"]["module"]
    prep_class_name = config["model_entrypoint"]["preprocessor_class"]

    preprocessor: BasePreprocessor = instantiate_class(module_name, prep_class_name)

    # 3. Execução do Contrato (Feature Engineering)
    logger.info("Aplicando transformações", preprocessor_class=prep_class_name)
    try:
        if is_training:
            processed_df = preprocessor.fit_transform(raw_df)
        else:
            processed_df = preprocessor.transform(raw_df)

        logger.info("Feature Engineering concluído", columns=len(processed_df.columns))
        return processed_df, preprocessor

    except Exception as e:
        logger.error("Falha no pré-processamento", error=str(e))
        raise

def main():
    """Entrypoint do Job de Feature Engineering no SageMaker (--step feature_engineering)."""
    config = load_config()
    run_feature_engineering(config, is_training=True)

if __name__ == "__main__":
    main()