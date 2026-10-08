"""
Cliente S3 para persistir as predições da inferência batch (ADR-005).
O bucket de artefatos do MLflow (S3_URI) é resolvido pelo próprio MLflow -
ver TrackerClient - este cliente não lida com ele.
"""
import os
import io
import boto3
import pandas as pd

from src.core.telemetria import get_logger

logger = get_logger("mlops.io.s3")

class S3Client:
    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client("s3", region_name=self.region)

        # S3_PREDICT_URI (.env) é a URI completa (bucket + prefixo no padrão de partição
        # Hive do Glue/Athena, ex: s3://bucket/source=parquet/database=...). S3_CDT_BUCKET
        # fica como alias secundário (só o nome do bucket, sem prefixo).
        predict_uri = os.getenv("S3_PREDICT_URI") or f"s3://{os.getenv('S3_CDT_BUCKET', 'default-cdt-inference')}"
        self.cdt_bucket, _, prefix = predict_uri.removeprefix("s3://").partition("/")
        self.cdt_prefix = prefix.rstrip("/")

    def _build_key(self, s3_key: str) -> str:
        return f"{self.cdt_prefix}/{s3_key}" if self.cdt_prefix else s3_key

    def upload_predictions(self, df: pd.DataFrame, s3_key: str) -> str:
        """
        Salva o resultado da inferência batch no bucket CDT em formato Parquet.
        Utiliza buffer em memória (BytesIO) para evitar I/O em disco no SageMaker.
        """
        key = self._build_key(s3_key)
        logger.info("Iniciando upload de inferência batch", bucket=self.cdt_bucket, key=key, rows=len(df))

        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)

        self.client.put_object(
            Bucket=self.cdt_bucket,
            Key=key,
            Body=buffer.getvalue()
        )

        s3_uri = f"s3://{self.cdt_bucket}/{key}"
        logger.info("Upload batch concluído", s3_uri=s3_uri)
        return s3_uri