from collections.abc import Callable
from typing import TypeAlias

EventCallback: TypeAlias = Callable[..., None]


class Emitter:
    def __init__(self):
        self.listeners: dict[str, list[EventCallback]] = {}

    def on(self, event: str, callback: EventCallback) -> None:
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append(callback)

    def off(self, event: str, callback: EventCallback) -> None:
        if event in self.listeners:
            self.listeners[event].remove(callback)

    def emit(self, event: str, *args, **kwargs) -> None:
        if event in self.listeners:
            for callback in self.listeners[event]:
                callback(*args, **kwargs)
