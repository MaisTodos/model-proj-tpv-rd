"""
run_batch_predict: os dois caminhos críticos que nunca tinham teste -
(1) Athena sem dados não pode silenciosamente tentar seguir o resto do pipeline,
(2) nenhuma versão do modelo no estágio configurado tem que falhar alto, nunca
rodar predict() com um modelo errado ou None.
"""
from unittest import mock

import pandas as pd
import pytest

from src.config import feature, predict

EMPTY_DF = pd.DataFrame()
INFER_DF = pd.DataFrame({"feature_1": [1.0], "feature_2": [2.0], "feature_3": ["a"]})


@pytest.fixture
def config():
    return feature.load_config()


def test_empty_athena_result_stops_before_touching_mlflow_or_s3(config):
    with mock.patch.object(predict.AthenaClient, "execute_query", return_value=EMPTY_DF):
        with mock.patch.object(predict, "MlflowClient") as mlflow_client_cls, \
             mock.patch.object(predict, "S3Client") as s3_client_cls:
            predict.run_batch_predict(config)

    mlflow_client_cls.assert_not_called()
    s3_client_cls.assert_not_called()


def test_no_registered_model_version_raises_before_predicting(config):
    with mock.patch.object(predict.AthenaClient, "execute_query", return_value=INFER_DF):
        with mock.patch.object(predict, "MlflowClient") as mlflow_client_cls:
            mlflow_client_cls.return_value.get_latest_versions.return_value = []

            with pytest.raises(ValueError, match=config["name"]):
                predict.run_batch_predict(config)


def test_main_logs_and_reraises_on_failure(config):
    """src/config/predict.py:main não pode engolir a exceção - precisa subir para
    o src/main.py poder terminar o processo com código de erro."""
    with mock.patch.object(predict, "load_config", return_value=config):
        with mock.patch.object(predict, "run_batch_predict", side_effect=RuntimeError("falha")):
            with pytest.raises(RuntimeError):
                predict.main()
