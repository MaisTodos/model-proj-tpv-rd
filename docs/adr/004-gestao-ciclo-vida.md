# 0004 - Gestão do Ciclo de Vida de Modelos (Model Registry)

- **Status**: Aceito
- **Data**: 2026-10-01

## Contexto

Após o treinamento e rastreamento de experimentos com MLflow (ADR 0003), é necessário um repositório central para gerenciar o ciclo de vida dos modelos (estágios como `Staging`, `Production`, `Archived`), versionamento e aprovações, permitindo que a aplicação de inferência baixe o artefato correto de forma determinística. As opções principais são o SageMaker Model Registry e o MLflow Model Registry.

## Decisão

Adotar o **MLflow Model Registry** com armazenamento físico de artefatos no **AWS S3** via `boto3`.
A interface de abstração do MLOps fornecerá o método `register_model()`. A aplicação de inferência utilizará a API do MLflow para resolver a URI do modelo em produção (`models:/<nome_do_modelo>/Production`) e fará o download direto do S3.

## Consequências

- **(+)** Consistência: Mantém a mesma interface (MLflow) para tracking e registry, reduzindo a carga cognitiva.
- **(+)** Portabilidade: A lógica de resolução de versão fica agnóstica à AWS.
- **(-)** Requer infraestrutura: O servidor do MLflow precisa de um backend database e permissões de IAM adequadas para o bucket S3 dos artefatos.

## Alternativas consideradas

- **SageMaker Model Registry:** Mais nativo na AWS, mas exigiria o uso intenso da API do Boto3 para aprovação/rejeição de modelos, o que fragmenta a experiência do cientista (MLflow para treino, AWS para registry). Descartado em favor da unificação.