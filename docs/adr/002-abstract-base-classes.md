# 0002 - Separação de Responsabilidades via Classes Base Abstratas (ABCs)

- **Status**: Proposto
- **Data**: 2026-10-01

## Contexto
O ambiente de desenvolvimento no AWS SageMaker será colaborativo e exige iterações rápidas dos cientistas de dados (definição de features, algoritmos, hyperparametros). Contudo, a infraestrutura da aplicação (logs via `ddtrace`, I/O com S3 via `boto3`, resiliência via `tenacity`) é rígida e padronizada. Precisamos de uma arquitetura que isole o código analítico do código de plataforma, permitindo que os cientistas alterem as regras de negócio do modelo sem risco de quebrar o motor de MLOps.

## Decisão
Adotar o padrão de projeto *Template Method* utilizando o módulo nativo `abc` do Python. A estrutura da aplicação será dividida em:
1. **Camada de Plataforma (Core/MLOps):** Controla o fluxo de execução, injeção de dependências, logging, telemetria e gestão de artefatos.
2. **Camada de Modelagem (Ciência de Dados):** Classes que herdam das interfaces da plataforma (ex: `BasePreprocessor`, `BaseModelTrainer`).

Os cientistas de dados devem estender essas classes e implementar obrigatoriamente os métodos definidos pelas assinaturas abstratas (ex: `process_features(df)`, `train()`, `get_metrics()`).

## Consequências
- **(+) Isolamento:** Protege o código de infraestrutura (logs, telemetria, I/O) contra alterações acidentais.
- **(+) Padronização e Testabilidade:** Garante que todos os modelos desenvolvidos a partir do template exponham os mesmos métodos, facilitando testes unitários e o deploy automatizado.
- **(-)** **Curva de aprendizagem:** Exige que cientistas de dados atuem no paradigma de Orientação a Objetos (OOP) ao invés de scripts processuais simples (Notebooks).
- **Mitigação:** O repositório fornecerá templates concretos de exemplo (ex: pipeline básica) para acelerar o desenvolvimento e servir de guia.

## Alternativas consideradas
- **Scripts YAML baseados em configuração:** Descartado por ser demasiado inflexível para tratamentos de dados complexos que exigem execução de código Python arbitrário (`numpy`, `pandas`/`narwhals`).
- **Jupyter Notebooks parametrizados:** Descartado devido à dificuldade de versionamento de código, baixa robustez para observabilidade avançada e complexidade no deploy em instâncias transacionais.