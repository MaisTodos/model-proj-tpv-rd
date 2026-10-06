import boto3
import pandas as pd
import pytest
from moto import mock_aws
from src.aws.s3_client import S3Client

PREDICTIONS = pd.DataFrame({"prediction": [1, 2, 3], "feature": ["A", "B", "C"]})


@mock_aws
def test_upload_predictions_uses_s3_predict_uri_prefix():
    """S3_PREDICT_URI (bucket + prefixo Hive do Glue/Athena) deve compor a key final."""
    s3_boto = boto3.client("s3", region_name="us-east-1")
    s3_boto.create_bucket(Bucket="test-cdt-bucket")

    client = S3Client()
    assert client.cdt_bucket == "test-cdt-bucket"
    assert client.cdt_prefix == "source=parquet/database=test_ml_models"

    s3_uri = client.upload_predictions(PREDICTIONS, "dt=2026-10-02/output.parquet")

    expected_key = "source=parquet/database=test_ml_models/dt=2026-10-02/output.parquet"
    assert s3_uri == f"s3://test-cdt-bucket/{expected_key}"

    response = s3_boto.list_objects_v2(Bucket="test-cdt-bucket", Prefix="source=parquet/")
    assert response["Contents"][0]["Key"] == expected_key


@mock_aws
def test_upload_predictions_falls_back_to_bare_bucket_name(monkeypatch):
    """Sem S3_PREDICT_URI, cai para S3_CDT_BUCKET (só o bucket, sem prefixo)."""
    monkeypatch.delenv("S3_PREDICT_URI", raising=False)
    monkeypatch.setenv("S3_CDT_BUCKET", "legacy-cdt-bucket")

    s3_boto = boto3.client("s3", region_name="us-east-1")
    s3_boto.create_bucket(Bucket="legacy-cdt-bucket")

    client = S3Client()
    assert client.cdt_bucket == "legacy-cdt-bucket"
    assert client.cdt_prefix == ""

    s3_uri = client.upload_predictions(PREDICTIONS, "output.parquet")
    assert s3_uri == "s3://legacy-cdt-bucket/output.parquet"
