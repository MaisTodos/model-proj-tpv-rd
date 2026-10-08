"""
Logs estruturados com structlog, no formato que o Datadog parseia sozinho.

Uma linha = um objeto JSON com os nomes reservados do Datadog (`message`,
`status`, `logger.name`, `timestamp`, `ddsource`). Os campos do evento ficam no
topo (`{"message": "journey.node", "status": "info", "wa_id": ..., "node": ...}`).

- Log  sob `ddtrace-run` com DD_LOGS_INJECTION=1 o
  ddtrace prepende o próprio processor (`dd.trace_id`, `dd.span_id`...); aqui só
  descartamos os valores vazios que ele injeta quando não há span ativo.

Uso: `logger = get_logger("orchestrator")`; `logger.info("journey.inbound"`;
`with bind_context(run_id=...): ...`.
"""
import logging
import sys

import structlog
from dotenv import load_dotenv
from structlog.contextvars import bound_contextvars as bind_context

# Carrega .env (dev local) antes de qualquer client ler variáveis de ambiente.
# Não-op se o arquivo não existir (containers/produção injetam o ambiente diretamente).
load_dotenv()

_DD_CORRELATION_KEYS = ("dd.trace_id", "dd.span_id", "dd.service", "dd.version", "dd.env")
_QUIET_LIBS = ("httpx", "httpcore", "urllib3", "botocore", "boto3")



def _datadog_keys(_logger, _method, event_dict):
    """Renomeia para os nomes reservados do Datadog e descarta correlação vazia."""
    event_dict["message"] = event_dict.pop("event", "")
    event_dict["status"] = event_dict.pop("level", "info")
    if "logger" in event_dict:
        event_dict["logger.name"] = event_dict.pop("logger")
    if "exception" in event_dict:
        event_dict["error.stack"] = event_dict.pop("exception")
    event_dict["ddsource"] = "python"
    for key in _DD_CORRELATION_KEYS:
        if event_dict.get(key) in ("0", "", None) and key in event_dict:
            del event_dict[key]
    return event_dict


# Chain compartilhada: nossos logs (structlog) e os da stdlib.
_SHARED = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.format_exc_info,
    _datadog_keys,
]


class JsonFormatter(structlog.stdlib.ProcessorFormatter):
    """Formatter dos handlers da stdlib (root e uvicorn via --log-config)."""

    def __init__(self) -> None:
        super().__init__(
            foreign_pre_chain=_SHARED,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(ensure_ascii=False, default=str),
            ],
        )


class HealthCheckLogFilter(logging.Filter):
    """Descarta os access logs de health check (ruído — sem valor de observabilidade)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


def silence_health_check_logs() -> None:
    """Aplica o filtro de health check no access log do uvicorn."""
    logging.getLogger("uvicorn.access").addFilter(HealthCheckLogFilter())


def configure_logging(level: str = "INFO") -> None:
    """Um handler JSON no root para tudo; libs ruidosas só a partir de WARNING."""
    root = logging.getLogger()
    if not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.setLevel(level.upper())
    for name in _QUIET_LIBS:
        logging.getLogger(name).setLevel(logging.WARNING)

    structlog.configure(
        processors=_SHARED + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


configure_logging()

get_logger = structlog.get_logger
