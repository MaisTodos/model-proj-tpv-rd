# 0006 - Coleta de Métricas de Infraestrutura em Treino/Inferência

- **Status**: Aceito
- **Data**: 2026-10-01

## Contexto

O `ddtrace` coleta tempos de execução e exceções em nível de código (APM). Contudo, durante o treinamento de modelos ou inferência, ocorrem gargalos de infraestrutura (CPU, RAM, GPU, I/O de disco) no ambiente do SageMaker que o APM não enxerga, dificultando o diagnóstico de falhas de hardware ou lentidão térmica.

## Decisão

Padronizar a inclusão do **Datadog Agent** nas imagens Docker customizadas do SageMaker ou via *Lifecycle Configurations*. O Agent enviará métricas de host (CPU, memória) e as vinculará aos traces da aplicação utilizando as tags padrão (`DD_ENV`, `DD_SERVICE`).

## Consequências

- **(+)** Observabilidade Full-Stack: Correlação direta entre um aumento na latência da predição e um pico de CPU no contêiner.
- **(+)** Alinhamento: Segue estritamente o ADR-001 de centralizar a observabilidade no Datadog nativo.
- **(-)** Manutenção de Imagens: Requer construção e manutenção de imagens Docker contendo a instalação do Datadog Agent, inviabilizando o uso de imagens AWS puras sem prévia modificação.

## Alternativas consideradas

- **CloudWatch Metrics nativas do SageMaker:** Descartado. Embora sejam úteis, forçariam o time a usar dashboards separados (Datadog para código, CloudWatch para hardware), quebrando a centralização proposta na arquitetura.