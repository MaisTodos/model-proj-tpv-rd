"""
Funções auxiliares de src/config/feature.py: devem propagar erro claro em vez de
mascarar arquivo ausente ou módulo/classe inexistente.
"""
import pytest

from src.config.feature import instantiate_class, load_config, read_sql_file


def test_read_sql_file_raises_for_missing_file():
    with pytest.raises(FileNotFoundError, match="nao_existe.sql"):
        read_sql_file("src/sql/nao_existe.sql")


def test_read_sql_file_returns_content(tmp_path):
    sql_file = tmp_path / "query.sql"
    sql_file.write_text("SELECT 1")
    assert read_sql_file(str(sql_file)) == "SELECT 1"


def test_load_config_reads_yaml(tmp_path):
    config_file = tmp_path / "project.yml"
    config_file.write_text("name: teste\nversion: '1.0.0'\n")
    config = load_config(str(config_file))
    assert config == {"name": "teste", "version": "1.0.0"}


def test_load_config_missing_file_propagates():
    with pytest.raises(FileNotFoundError):
        load_config("arquivo/que/nao/existe.yml")


def test_instantiate_class_unknown_module_propagates():
    with pytest.raises(ModuleNotFoundError):
        instantiate_class("src.models.modulo_que_nao_existe", "Qualquer")


def test_instantiate_class_unknown_class_propagates():
    with pytest.raises(AttributeError):
        instantiate_class("src.models.model", "ClasseQueNaoExiste")


def test_instantiate_class_happy_path():
    instance = instantiate_class("src.models.model", "TemplateModelTrainer")
    assert instance.get_params()["model_type"] == "RandomForestRegressor"
