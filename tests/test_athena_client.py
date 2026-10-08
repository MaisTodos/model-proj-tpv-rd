"""
AthenaClient: valida que configuração incompleta falha alto e cedo (não na hora
de rodar a query), e que o cursor é sempre fechado - mesmo quando a query falha.
"""
from unittest import mock

import pandas as pd
import pytest

from src.aws.athena_client import AthenaClient


def test_missing_required_env_vars_raises_valueerror(monkeypatch):
    monkeypatch.delenv("DATALAKE_ROLE_ARN", raising=False)
    monkeypatch.delenv("S3_STAGING_DIR", raising=False)

    with pytest.raises(ValueError, match="DATALAKE_ROLE_ARN e S3_STAGING_DIR"):
        AthenaClient()


def test_missing_only_staging_dir_raises_valueerror(monkeypatch):
    monkeypatch.setenv("DATALAKE_ROLE_ARN", "arn:aws:iam::123456789012:role/FakeRole")
    monkeypatch.delenv("S3_STAGING_DIR", raising=False)

    with pytest.raises(ValueError):
        AthenaClient()


def test_execute_query_closes_cursor_on_success():
    client = AthenaClient()
    fake_cursor = mock.Mock()
    fake_cursor.execute.return_value.as_pandas.return_value = pd.DataFrame({"a": [1]})

    with mock.patch.object(AthenaClient, "_get_cursor", return_value=fake_cursor):
        df = client.execute_query("SELECT 1")

    assert len(df) == 1
    fake_cursor.close.assert_called_once()


def test_execute_query_closes_cursor_even_when_query_fails():
    """Vazamento de cursor/conexão é um dos jeitos mais comuns de falha silenciosa
    em produção - garante que o `finally` realmente protege isso."""
    client = AthenaClient()
    fake_cursor = mock.Mock()
    fake_cursor.execute.side_effect = RuntimeError("query malformada")

    with mock.patch.object(AthenaClient, "_get_cursor", return_value=fake_cursor):
        with pytest.raises(RuntimeError):
            client.execute_query("SELECT * FROM tabela_que_nao_existe")

    fake_cursor.close.assert_called_once()


def test_get_query_from_yaml_resolves_known_table(tmp_path):
    manifest = tmp_path / "source.yml"
    manifest.write_text(
        "sources:\n"
        "  - name: todos_data_lake\n"
        "    schema: todos_data_lake\n"
        "    tables:\n"
        "      - name: clientes\n"
    )
    client = AthenaClient()

    query = client.get_query_from_yaml(str(manifest), "clientes")

    assert query == "SELECT * FROM todos_data_lake.clientes"


def test_get_query_from_yaml_raises_for_unmapped_table(tmp_path):
    manifest = tmp_path / "source.yml"
    manifest.write_text(
        "sources:\n"
        "  - name: todos_data_lake\n"
        "    tables:\n"
        "      - name: clientes\n"
    )
    client = AthenaClient()

    with pytest.raises(ValueError, match="nao_existe"):
        client.get_query_from_yaml(str(manifest), "tabela_nao_existe")
