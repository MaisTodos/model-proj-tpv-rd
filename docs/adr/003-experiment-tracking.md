# 0003 - Abstração do Tracking de Experimentos (MLflow)

- **Status**: Aceito
- **Data**: 2026-10-01

## Contexto

O treinamento de modelos exige o rastreamento rigoroso de parâmetros, métricas e artefatos. A biblioteca padrão de mercado e que será adotada é o MLflow. No entanto, expor as APIs nativas do MLflow (`mlflow.set_tracking_uri`, `mlflow.start_run`, etc.) diretamente no código de modelagem (camada analítica) acopla a lógica do negócio à infraestrutura, dificultando testes e possíveis migrações futuras (ex: para AWS SageMaker Experiments).

## Decisão

Criar uma camada de abstração  para o tracking de experimentos.
A infraestrutura (MLOps) será responsável por instanciar a conexão, configurar URIs e gerir o ciclo de vida da *run* (`start_run`/`end_run`). 

O Cientista de Dados receberá uma interface simplificada (ex: `TrackerClient`) com métodos padronizados:
- `log_param(key, value)`
- `log_metric(key, value)`
- `log_model(model, artifact_path)`

## Consequências

- **(+) Desacoplamento:** O código de *data science* desconhece URIs, credenciais e a ferramenta exata de tracking subjacente.
- **(+) Padronização:** O pipeline central (Core) garante que *tags* obrigatórias (ex: versão do código, ambiente, `trace_id` do Datadog) sejam sempre registradas junto com a *run*.
- **(+) Transição Suave:** Facilidade em substituir o backend (ex: de MLflow para Datadog LLMObs ou W&B) alterando apenas o *Adapter* central.
- **(-)** Necessidade de manter uma classe de embrulho (*wrapper*) para os métodos que desejarmos suportar do MLflow.

## Alternativas consideradas

- **Uso direto do pacote `mlflow` nos scripts de treino:** Rejeitado. Isso obrigaria os cientistas de dados a gerir configurações de infraestrutura (URIs de tracking, gestão do context manager `with mlflow.start_run()`) e quebraria o isolamento proposto no ADR-002.