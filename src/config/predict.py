"""
Orquestrador da Pipeline de Inferência Batch.
Carrega o modelo do registry, executa predições e salva no bucket CDT (ADR-005).
"""
import os
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime, timezone

from src.config.feature import load_config, read_sql_file, instantiate_class
from src.aws.athena_client import AthenaClient
from src.aws.s3_client import S3Client
from src.core.telemetria import get_logger
from src.core.tracker import DEFAULT_TRACKING_URI

logger = get_logger("mlops.predict")

def run_batch_predict(config: dict) -> None:
    """Executa o ciclo completo de inferência em lote."""
    logger.info("Iniciando pipeline de inferência batch")

    # 1. Extração de Dados (Athena)
    sql_path = config["data"].get("inference_query_file", "src/sql/extract_inference.sql")
    query = read_sql_file(sql_path)

    athena = AthenaClient()
    raw_df = athena.execute_query(query)

    if raw_df.empty:
        logger.warning("Nenhum dado retornado para inferência. Encerrando job.")
        return

    # 2. Resolução da versão do modelo no Registry (ADR-004)
    model_name = config["name"]
    # Pode ser injetado via variável de ambiente no CI/CD. Assumimos "Production" por padrão.
    stage = os.getenv("MODEL_STAGE", "Production")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    versions = MlflowClient().get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise ValueError(f"Nenhuma versão do modelo '{model_name}' no estágio '{stage}'.")
    run_id = versions[0].run_id

    # 3. Pré-processamento com o estado ajustado no treino (Contrato BasePreprocessor)
    # O preprocessor é carregado do mesmo run do modelo em produção — garante que o
    # estado de `fit_transform` (ex: médias do StandardScaler) seja o usado em treino.
    logger.info("Carregando preprocessor ajustado do tracking", run_id=run_id)
    preprocessor = mlflow.sklearn.load_model(f"runs:/{run_id}/preprocessor")

    logger.info("Aplicando transformações de inferência")
    processed_df = preprocessor.transform(raw_df)

    # Separação de features ignorando o target (que não existe na inferência)
    X_infer = processed_df[config["features"]] if config["features"] and config["features"][0] != "" else processed_df

    # 4. Carregamento do Modelo do MLflow Registry
    model_uri = f"models:/{model_name}/{stage}"
    logger.info("Carregando modelo do Registry", model_uri=model_uri)
    trained_model = mlflow.sklearn.load_model(model_uri)

    # 5. Inferência (Contrato BaseModel)
    module_name = config["model_entrypoint"]["module"]
    model_class_name = config["model_entrypoint"]["model_class"]
    model_instance = instantiate_class(module_name, model_class_name)

    logger.info("Executando predições batch")
    predictions = model_instance.predict(trained_model, X_infer)
    
    # 6. Formatação do Output (Shadow Logging - ADR-005)
    # Anexamos as predições aos dados brutos para viabilizar queries de Data Drift no Athena
    output_df = raw_df.copy()
    output_df["prediction"] = predictions
    output_df["inference_timestamp"] = datetime.now(timezone.utc).isoformat()
    output_df["model_version"] = stage

    # 7. Upload para o S3 (Bucket CDT)
    s3 = S3Client()
    partition_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    s3_key = f"inferences/{model_name}/dt={partition_date}/output.parquet"
    
    s3_uri = s3.upload_predictions(output_df, s3_key)
    logger.info("Inferência batch concluída com sucesso", output_uri=s3_uri, rows=len(output_df))

def main():
    """Entrypoint do Job de Inferência no SageMaker."""
    config = load_config()
    try:
        run_batch_predict(config)
    except Exception as e:
        logger.error("Falha no job de inferência", error=str(e))
        raise

if __name__ == "__main__":
    main()