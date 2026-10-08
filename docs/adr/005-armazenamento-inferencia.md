# 0005 - Armazenamento e Monitoramento de Inferência (Drift)

- **Status**: Aceito
- **Data**: 2026-10-01

## Contexto

Modelos em produção sofrem degradação ao longo do tempo (Data Drift nas features de entrada, Concept Drift no alvo). É necessário capturar os dados de inferência em tempo real para análise assíncrona, sem impactar a latência da predição.

## Decisão

Implementar um padrão de **Shadow Logging no AWS S3**. 
O pipeline de inferência salvará os *payloads* de entrada e as predições em formato JSONL/Parquet particionado por data no S3. Utilizaremos o **AWS Athena** (via `PyAthena`) para consultar esses logs. A detecção de drift será feita por *jobs* agendados que comparam a distribuição dos dados de treino com a distribuição recente consultada via Athena.

## Consequências

- **(+)** Desempenho: Escrever logs no S3 não bloqueia a resposta da API de inferência.
- **(+)** Custo-benefício: O armazenamento no S3 e queries no Athena sob demanda são altamente escaláveis e baratos.
- **(-)** Complexidade assíncrona: Exige a configuração de tabelas no AWS Glue Data Catalog para que o Athena possa consultar os logs de forma estruturada.

## Alternativas consideradas

- **Envio de features direto para o Datadog:** Descartado. O Datadog é otimizado para séries temporais numéricas, não para distribuições estatísticas complexas de features de ML ou armazenamento de logs de dados brutos (alto custo).