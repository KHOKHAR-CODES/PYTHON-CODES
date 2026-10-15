import json
import logging
import sys
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar('request_id', default='-')


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            'timestamp': self.formatTime(record, datefmt='%Y-%m-%dT%H:%M:%S%z'),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'request_id': request_id_var.get(),
        }
        for key in ('endpoint', 'duration_ms', 'retrieved_chunks', 'selected_chunks', 'latency_ms', 'error_code'):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = 'INFO') -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
