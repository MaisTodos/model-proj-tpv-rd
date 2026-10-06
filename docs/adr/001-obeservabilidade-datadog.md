# 0001 - Observabilidade com Datadog nativo (não OTel Collector)

- **Status**: Aceito
- **Data**: 2026-10-01

## Contexto

A arquitetura de plataforma sugeria **OpenTelemetry Collector como hub trocável** ("app fala só
OTLP; o Collector decide o exporter - não acoplar ao Datadog"), visando portabilidade de vendor. A
primeira versão da instrumentação seguiu isso (OpenTelemetry + Sentry). O time optou por **Datadog
nativo**, por pragmatismo e alinhamento com o stack de observabilidade já usado na MaisTodos.

## Decisão

Usar o **Datadog de forma nativa**, via `ddtrace`:

- **APM**: `ddtrace-run` no entrypoint (instrumentação automática de FastAPI, `requests`, `boto3`).
- Config por ambiente (`DD_API_KEY`/`DD_SITE` do AWS Secret via ESO; `DD_ENV`, `DD_SERVICE`,
  `DD_LLMOBS_ENABLED`, `DD_LLMOBS_ML_APP`, `DD_LLMOBS_AGENTLESS_ENABLED`).
- Instrumentação **safe no-op**: se o `ddtrace` faltar nada quebra.

## Consequências

- **(-)** Acoplamento ao Datadog (menos portável).
- **Mitigação**: o que é backend-agnóstico foi mantido - o **wide-event por run** e o classificador
  de **`failure_category`** não dependem do Datadog e continuam sendo a fonte de verdade durável (ver
  `docs/observabilidade.md`).

## Alternativas consideradas

- **OTel Collector -> exporter Datadog**: mais portável, mais complexo de operar agora

