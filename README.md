# ml-template-model

Template para desenvolvimento de modelos de Machine Learning da MaisTodos. Padroniza a
infraestrutura de MLOps (observabilidade, I/O com AWS, tracking e registry de experimentos) para
que o Cientista de Dados foque apenas na regra de negócio do modelo.

As decisões que sustentam esta estrutura estão documentadas em [docs/adr](docs/adr).

## Estrutura

```
src/
  core/      # Contratos (BasePreprocessor, BaseModel), logging estruturado
  aws/       # Clientes de I/O (Athena, S3)
  config/    # Orquestradores de cada etapa (feature engineering, treino, inferência)
  models/    # Código do Cientista de Dados: implementação dos contratos de src/core
  sql/       # Queries de extração usadas pelas etapas
  main.py    # Entrypoint CLI (--step)
project.yml  # Configuração do projeto: target, features, fontes de dados, mapeamento de pipelines
```

## Quickstart

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure `project.yml`: nome do projeto, `target`, `features`, arquivos de query em
   `data.query_file` / `data.inference_query_file`, e o módulo analítico em `model_entrypoint`.
3. Implemente a lógica do modelo em `src/models/model.py`, estendendo `BasePreprocessor` e
   `BaseModel` (`src/core/base.py`) — ver ADR-002.
4. Escreva as queries de extração em `src/sql/extract_features.sql` (treino) e
   `src/sql/extract_inference.sql` (inferência).
5. Execute uma etapa do pipeline:
   ```bash
   python src/main.py --step feature_engineering
   python src/main.py --step training
   python src/main.py --step inference
   ```

## Variáveis de ambiente

Lidas automaticamente de um `.env` na raiz do projeto, se existir (não versionado).

| Variável | Uso |
|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Credenciais base (usadas para assumir `DATALAKE_ROLE_ARN`) |
| `AWS_REGION` | Região AWS para clientes boto3/Athena |
| `DATALAKE_ROLE_ARN` | Role assumida via STS para consultas no Athena |
| `S3_STAGING_DIR` | Diretório de staging do Athena (obrigatório para `AthenaClient`) |
| `ATHENA_DATABASE` | Database padrão no Glue Catalog |
| `S3_URI` | Bucket de artefatos do MLflow (ADR-004) — aceita `MLFLOW_ARTIFACT_URI` como alias |
| `S3_PREDICT_URI` | URI (bucket + prefixo) de saída da inferência batch (shadow logging — ADR-005) — aceita `S3_CDT_BUCKET` (só o bucket) como alias |
| `MLFLOW_TRACKING_URI` | URI do servidor de tracking do MLflow. Aceita ARN de SageMaker MLflow Tracking Server (requer `sagemaker-mlflow`, já no `requirements.txt`). Sem valor, cai para `sqlite:///mlruns.db` (dev local) |
| `MODEL_STAGE` | Estágio do Model Registry a carregar na inferência (padrão: `Production`) |

## Testes

```bash
pytest
```
