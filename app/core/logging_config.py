import logging
import os
import sys

import structlog
from structlog.types import EventDict, Processor


def add_log_level(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Добавляет уровень логирования в event_dict."""
    if method_name == 'warn':
        event_dict['level'] = 'warning'
    else:
        event_dict['level'] = method_name
    return event_dict


def setup_logging(log_level: str | None = None, log_format: str | None = None) -> None:
    """Настраивает структурированное логирование через structlog.

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Формат логов ('json' для production, 'console' для development)

    """
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

    if log_format is None:
        log_format = os.getenv('LOG_FORMAT', 'json').lower()

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_log_level,
        structlog.processors.TimeStamper(fmt='iso'),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_format == 'console':
        processors.extend(
            [
                structlog.dev.ConsoleRenderer(colors=True),
            ]
        )
    else:
        processors.extend(
            [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ]
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format='%(message)s',
        stream=sys.stdout,
        level=getattr(logging, log_level, logging.INFO),
    )

    uvicorn_logger = logging.getLogger('uvicorn')
    uvicorn_logger.setLevel(getattr(logging, log_level, logging.INFO))

    uvicorn_access = logging.getLogger('uvicorn.access')
    uvicorn_access.setLevel(getattr(logging, log_level, logging.INFO))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Получает логгер с контекстом модуля.

    Args:
        name: Имя модуля (обычно __name__)

    Returns:
        Настроенный логгер structlog

    """
    return structlog.get_logger(name)
