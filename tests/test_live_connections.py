"""
Testes de conexão REAIS contra AWS/MLflow, usando as credenciais do .env.

Não rodam por padrão (`pytest` normal pula tudo aqui). Para rodar:

    RUN_LIVE_TESTS=1 pytest tests/integration/test_live_connections.py -m live -v

Cada teste pula com uma mensagem clara se a variável de ambiente necessária não
estiver configurada - isso em si documenta gaps de configuração do .env.
"""
import os
import uuid

import boto3
import pytest
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()  # carrega .env mesmo que nenhum módulo de src/ tenha sido importado ainda

pytestmark = pytest.mark.live

RUN_LIVE = os.getenv("RUN_LIVE_TESTS") == "1"
skip_unless_live = pytest.mark.skipif(not RUN_LIVE, reason="defina RUN_LIVE_TESTS=1 para rodar testes de conexão reais")


def _require_env(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        pytest.skip(f"variáveis ausentes no .env: {', '.join(missing)}")


@skip_unless_live
def test_aws_credentials_are_valid():
    """Confirma que AWS_ACCESS_KEY_ID/SECRET do .env autenticam (sts:GetCallerIdentity)."""
    _require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    identity = boto3.client("sts", region_name=os.getenv("AWS_REGION", "us-east-1")).get_caller_identity()
    assert identity["Account"]


@skip_unless_live
def test_athena_role_assumable_and_query_runs():
    """Valida o fluxo real do AthenaClient: STS AssumeRole + SELECT 1 via PyAthena."""
    _require_env("DATALAKE_ROLE_ARN", "S3_STAGING_DIR")
    from src.aws.athena_client import AthenaClient

    try:
        df = AthenaClient().execute_query("SELECT 1 AS connectivity_check")
    except ClientError as e:
        pytest.fail(
            f"Falha ao assumir DATALAKE_ROLE_ARN ou consultar o Athena: {e}. "
            "Gap de permissão IAM, não de código - ver Claude.md."
        )
    assert df["connectivity_check"].iloc[0] == 1


@skip_unless_live
def test_s3_predict_bucket_is_reachable():
    """Confirma que o bucket de S3_PREDICT_URI (saída da inferência - ADR-005) é acessível."""
    _require_env("S3_PREDICT_URI")
    bucket = os.environ["S3_PREDICT_URI"].removeprefix("s3://").split("/")[0]
    try:
        boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1")).head_bucket(Bucket=bucket)
    except ClientError as e:
        pytest.fail(f"Bucket de predições '{bucket}' inacessível com as credenciais do .env: {e}")


@skip_unless_live
def test_mlflow_tracking_server_is_reachable():
    """Conectividade de leitura com o tracking server (ARN do SageMaker, via plugin sagemaker-mlflow)."""
    _require_env("MLFLOW_TRACKING_URI")
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    try:
        MlflowClient().search_experiments(max_results=1)
    except Exception as e:
        pytest.fail(
            f"Não foi possível listar experimentos no MLFLOW_TRACKING_URI configurado: {e}. "
            "Se for 403/AccessDenied, é gap de permissão IAM no tracking server, não de código."
        )


@skip_unless_live
def test_mlflow_experiment_registration_roundtrip():
    """
    Valida o ADR-003 fim a fim: cria/reusa um experimento dedicado de smoke test,
    registra uma run com parâmetro e métrica reais, e lê de volta do tracking server -
    confirma que "o registro dos experimentos" realmente acontece, não só a conectividade.

    Efeito colateral real (intencional): cria uma run pequena e descartável no
    tracking server configurado, no experimento `_connectivity_check`.
    """
    _require_env("MLFLOW_TRACKING_URI")
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment_name = "ml_template_model_connectivity_check"

    try:
        if mlflow.get_experiment_by_name(experiment_name) is None:
            artifact_uri = os.getenv("S3_URI") or None
            mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
        mlflow.set_experiment(experiment_name)

        marker = uuid.uuid4().hex
        with mlflow.start_run(run_name="connectivity-check") as run:
            mlflow.log_param("connectivity_marker", marker)
            mlflow.log_metric("connectivity_ok", 1.0)
            run_id = run.info.run_id
    except Exception as e:
        pytest.fail(
            f"Falha ao registrar experimento/run no MLflow: {e}. "
            "Se for 403/AccessDenied, é gap de permissão IAM, não de código."
        )

    # Lê de volta do servidor (não do cache local do client) para confirmar o registro.
    persisted = MlflowClient().get_run(run_id)
    assert persisted.data.params.get("connectivity_marker") == marker
    assert persisted.data.metrics.get("connectivity_ok") == 1.0
