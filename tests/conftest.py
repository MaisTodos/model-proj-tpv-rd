import pytest

# Variáveis de ambiente seguras para toda a suíte, exceto testes marcados @pytest.mark.live
# (que precisam das credenciais reais do .env - ver tests/integration/test_live_connections.py).
_SAFE_ENV = {
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_SECURITY_TOKEN": "testing",
    "AWS_SESSION_TOKEN": "testing",
    "AWS_DEFAULT_REGION": "us-east-1",
    "DATALAKE_ROLE_ARN": "arn:aws:iam::123456789012:role/FakeRole",
    "S3_STAGING_DIR": "s3://test-staging/",
    "S3_PREDICT_URI": "s3://test-cdt-bucket/source=parquet/database=test_ml_models",
    "S3_CDT_BUCKET": "test-cdt-bucket",
}


@pytest.fixture(autouse=True)
def safe_env(request, monkeypatch, tmp_path):
    """Isola toda chamada a AWS/MLflow em testes locais (moto + MLflow sqlite em tmp_path)."""
    if request.node.get_closest_marker("live"):
        yield
        return

    for key, value in _SAFE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("S3_URI", "s3://test-mlflow-artifacts/")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlruns.db")
    monkeypatch.delenv("MLFLOW_ARTIFACT_URI", raising=False)
    yield
