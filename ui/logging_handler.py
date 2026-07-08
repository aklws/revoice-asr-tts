from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from app.core.logging import add_log_sink, remove_log_sink


class QtLogEmitter(QObject):
    message = Signal(str)


class QtLogHandler:
    def __init__(self, emitter: QtLogEmitter) -> None:
        self.emitter = emitter
        self._sink_id: int | None = None

    def emit(self, message: str) -> None:
        self.emitter.message.emit(message.rstrip())

    def attach(self) -> None:
        if self._sink_id is None:
            self._sink_id = add_log_sink(self.emit)

    def detach(self) -> None:
        if self._sink_id is None:
            return
        remove_log_sink(self._sink_id)
        self._sink_id = None
