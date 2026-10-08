FROM python:3.11-slim

# Configurações do ambiente Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

# Diretório de trabalho padrão no SageMaker
WORKDIR /app

# Instalação de dependências do sistema operativo
# O pacote 'curl' é mantido para suportar a injeção do Datadog Agent via scripts de inicialização (ADR-006)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalação das bibliotecas Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Cópia do código-fonte e dos manifestos do projeto
COPY src/ ./src/
COPY project.yml .

# Entrypoint protegido pelo ddtrace-run para observabilidade nativa (ADR-001)
# O argumento --step será passado pelo SageMaker ou CLI (ex: docker run <image> --step training)
ENTRYPOINT ["ddtrace-run", "python", "src/main.py"]