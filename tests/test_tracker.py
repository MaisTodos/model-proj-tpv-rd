"""
TrackerClient: valida a resolução de variáveis de ambiente (ADR-003/004) e que o
experimento não é recriado a cada run - e que a lista de tipos confiáveis do skops
realmente inclui o necessário para o log_model não falhar silenciosamente depois.
"""
from unittest import mock

from src.core.tracker import DEFAULT_TRACKING_URI, TrackerClient


class DummyModel:
    pass


def test_tracking_uri_defaults_when_not_set(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)
    client = TrackerClient(experiment_name="exp_default_uri")
    assert client.tracking_uri == DEFAULT_TRACKING_URI


def test_artifact_uri_prefers_s3_uri_over_legacy_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlruns.db")
    monkeypatch.setenv("S3_URI", "s3://real-bucket/")
    monkeypatch.setenv("MLFLOW_ARTIFACT_URI", "s3://legacy-bucket/")
    client = TrackerClient(experiment_name="exp_artifact_precedence")
    assert client.artifact_uri == "s3://real-bucket/"


def test_artifact_uri_falls_back_to_legacy_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlruns.db")
    monkeypatch.delenv("S3_URI", raising=False)
    monkeypatch.setenv("MLFLOW_ARTIFACT_URI", "s3://legacy-bucket/")
    client = TrackerClient(experiment_name="exp_artifact_fallback")
    assert client.artifact_uri == "s3://legacy-bucket/"


def test_experiment_is_created_only_once(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlruns.db")
    monkeypatch.setenv("S3_URI", f"file://{tmp_path}/artifacts")

    TrackerClient(experiment_name="exp_reuse")
    with mock.patch("mlflow.create_experiment") as create_experiment:
        TrackerClient(experiment_name="exp_reuse")
        create_experiment.assert_not_called()


def test_log_model_trusts_tree_type_and_own_class(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlruns.db")
    monkeypatch.setenv("S3_URI", f"file://{tmp_path}/artifacts")
    client = TrackerClient(experiment_name="exp_log_model")

    with client.start_run():
        with mock.patch("mlflow.sklearn.log_model") as log_model:
            client.log_model(DummyModel(), artifact_path="preprocessor", trusted_types=["extra.Type"])

    trusted = log_model.call_args.kwargs["skops_trusted_types"]
    assert "sklearn.tree._tree.Tree" in trusted
    assert any("DummyModel" in t for t in trusted)
    assert "extra.Type" in trusted
