import datetime
from dataclasses import dataclass
from typing import Callable


def log_to_file(file_name: str, *args):
    pass
    # with open(file_name, "a") as f:
    #     print(*args, file=f)


@dataclass
class Logger:
    info: Callable
    debug: Callable
    error: Callable
    warning: Callable
    warn: Callable
    file: str = None

    def __post_init__(self):
        self.file = (
            self.file
            or f"logs/log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        )

    def log_to_file(self, *args):
        log_to_file(self.file, *args)


logger = Logger(
    info=print,
    debug=print,
    error=print,
    warning=print,
    warn=print,
)
