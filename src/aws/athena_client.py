"""
Cliente Athena para extração de dados via AWS STS.
Isola a lógica de conexão e parse do source.yml.
"""
import os
import boto3
import yaml
import pandas as pd
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor

# Assume a estrutura definida anteriormente, com os logs no Datadog (ADR-001)
from src.core.telemetria import get_logger

logger = get_logger("mlops.io.athena")

class AthenaClient:
    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.role_arn = os.getenv("DATALAKE_ROLE_ARN")
        self.staging_dir = os.getenv("S3_STAGING_DIR")
        self.database = os.getenv("ATHENA_DATABASE", "todos_data_lake")
        
        if not all([self.role_arn, self.staging_dir]):
            raise ValueError("Variáveis DATALAKE_ROLE_ARN e S3_STAGING_DIR são obrigatórias.")

    def _get_cursor(self) -> PandasCursor:
        """Assume role via STS e retorna um cursor otimizado para Pandas."""
        sts = boto3.client("sts", region_name=self.region)
        creds = sts.assume_role(
            RoleArn=self.role_arn,
            RoleSessionName="sagemaker-athena-session"
        )["Credentials"]

        # Utiliza o PandasCursor do pyathena para conversão direta e eficiente
        return connect(
            s3_staging_dir=self.staging_dir,
            schema_name=self.database,
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            cursor_class=PandasCursor
        ).cursor()

    def execute_query(self, query: str) -> pd.DataFrame:
        """Executa a query e devolve um DataFrame pronto para o BasePreprocessor."""
        logger.info("Executando query no Athena", database=self.database)
        cursor = self._get_cursor()
        try:
            df = cursor.execute(query).as_pandas()
            logger.info("Query executada com sucesso", rows=len(df))
            return df
        finally:
            cursor.close()

    def get_query_from_yaml(self, yaml_path: str, table_name: str) -> str:
        """Lê o models/source.yml e constrói a query de extração."""
        with open(yaml_path, "r") as f:
            manifest = yaml.safe_load(f)
            
        for source in manifest.get("sources", []):
            if source["name"] == self.database:
                for table in source.get("tables", []):
                    if table["name"] == table_name:
                        schema = source.get("schema", self.database)
                        logger.info("Mapeamento YAML resolvido", schema=schema, table=table_name)
                        return f"SELECT * FROM {schema}.{table_name}"
                        
        raise ValueError(f"Tabela '{table_name}' não mapeada no arquivo {yaml_path}")