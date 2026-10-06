# Comandos de Execução

Referência rápida de todos os comandos para rodar o projeto. Assume execução a partir da raiz
do repositório, com `venv` ativo (`source venv/bin/activate`).

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline (CLI local)

```bash
python src/main.py --step feature_engineering
python src/main.py --step training
python src/main.py --step inference
```

## Docker

```bash
docker build -f docker/Dockerfile -t ml-template-model .

docker run --rm --env-file .env ml-template-model --step feature_engineering
docker run --rm --env-file .env ml-template-model --step training
docker run --rm --env-file .env ml-template-model --step inference
```

## Testes

```bash
# Suíte padrão: unitários + integração mockada (sem rede real, ~7s)
pytest

# Com saída detalhada
pytest -v

# Só o teste recursivo de integração (feature -> treino -> registro MLflow -> inferência)
pytest tests/integration/test_pipeline_integration.py -v

# Testes de conexão REAL contra AWS/MLflow do .env (opt-in, nunca roda por padrão)
RUN_LIVE_TESTS=1 pytest tests/integration/test_live_connections.py -m live -v
```

## Documentação (MkDocs)

```bash
mkdocs serve   
mkdocs build
```
