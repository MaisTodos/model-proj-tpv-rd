"""
Teste recursivo de integração: roda o ciclo completo do pipeline
(feature engineering -> treino -> registro no MLflow -> promoção -> inferência batch)
exatamente como `src/main.py --step ...` executaria, com Athena mockado e MLflow/S3
reais (sqlite local + moto), para pegar gaps de integração entre as etapas.
"""
import io
from unittest import mock

import boto3
import mlflow
import pandas as pd
import pytest
from moto import mock_aws
from mlflow.tracking import MlflowClient

from src.config import feature, train, predict

TRAIN_DF = pd.DataFrame({
    "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "feature_2": [3.0, 4.0, 1.0, 2.0, 5.0, 6.0],
    "feature_3": ["a", "b", "a", "b", "a", "b"],
    "target_column": [10, 20, 15, 25, 12, 22],
})
INFER_DF = pd.DataFrame({
    "feature_1": [1.5, 2.5],
    "feature_2": [3.5, 4.5],
    "feature_3": ["a", "b"],
})


@pytest.fixture
def config():
    return feature.load_config()


@mock_aws
def test_full_cycle_feature_train_predict(config):
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-mlflow-artifacts")
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-cdt-bucket")

    # --- Etapa 1 e 2: feature engineering + treino (--step training) ---
    with mock.patch.object(feature.AthenaClient, "execute_query", return_value=TRAIN_DF):
        processed_df, preprocessor = feature.run_feature_engineering(config, is_training=True)
        assert "target_column" in processed_df.columns
        assert len(processed_df) == len(TRAIN_DF)

        train.run_training(processed_df, preprocessor, config)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(config["tracking"]["experiment_name"])
    assert experiment is not None, "Experimento não foi registrado no MLflow"

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1, "Run de treino não foi registrada"
    run = runs[0]

    # Hiperparâmetros e métrica de plataforma (ADR-003)
    assert run.data.params.get("model_type") == "RandomForestRegressor"
    assert "train_rmse" in run.data.metrics

    # Artefatos: modelo e preprocessor ajustado devem ser carregáveis do mesmo run
    # (é exatamente o que predict.py faz - ver src/config/predict.py)
    loaded_model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
    assert hasattr(loaded_model, "predict")
    loaded_preprocessor = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/preprocessor")
    assert hasattr(loaded_preprocessor, "transform")

    # Model Registry (ADR-004)
    versions = client.search_model_versions(f"name='{config['name']}'")
    assert len(versions) == 1, "Modelo não foi registrado no Registry"

    # --- Promoção manual a Production (normalmente feita via aprovação/CI) ---
    client.transition_model_version_stage(name=config["name"], version=versions[0].version, stage="Production")

    # --- Etapa 3: inferência batch (--step inference) ---
    with mock.patch.object(predict.AthenaClient, "execute_query", return_value=INFER_DF):
        predict.run_batch_predict(config)

    s3 = boto3.client("s3", region_name="us-east-1")
    objects = s3.list_objects_v2(Bucket="test-cdt-bucket", Prefix="source=parquet/database=test_ml_models/")
    assert "Contents" in objects, "Inferência não gravou output no bucket de predições"

    output_key = objects["Contents"][0]["Key"]
    body = s3.get_object(Bucket="test-cdt-bucket", Key=output_key)["Body"].read()
    output_df = pd.read_parquet(io.BytesIO(body))

    assert len(output_df) == len(INFER_DF)
    assert "prediction" in output_df.columns
    assert "target_column" not in output_df.columns  # não existe nos dados de inferência
    assert (output_df["model_version"] == "Production").all()


@mock_aws
def test_feature_engineering_step_runs_standalone(config):
    """--step feature_engineering (src/config/feature.py:main) não deve quebrar por falta de main()."""
    with mock.patch.object(feature.AthenaClient, "execute_query", return_value=TRAIN_DF):
        feature.main()
