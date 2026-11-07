import logging
from logging import Logger as _LoggingLogger
from logging import LogRecord
from typing import override

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_STANDARD_FIELDS = frozenset(
    vars(LogRecord("", 0, "", 0, "", None, None)).keys() | {"message", "asctime"},
)


class _ExtraFormatter(logging.Formatter):
    @override
    def format(self, record: LogRecord) -> str:
        text = super().format(record)
        extra = {k: v for k, v in vars(record).items() if k not in _STANDARD_FIELDS}

        if not extra:
            return text

        return text + " " + " ".join(f"{key}={value}" for key, value in extra.items())


def get_logger(name: str) -> _LoggingLogger:
    return logging.getLogger(name)


def initialise_logging(*, level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_ExtraFormatter(_FORMAT))
    logging.basicConfig(level=level, handlers=[handler])
