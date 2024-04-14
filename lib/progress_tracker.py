from logging import Logger
import logging


class ProgressTracker:
    def track(i: int, cost: float) -> None:
        pass


class LoggingProgressTracker(ProgressTracker):
    print_period: int
    logger: Logger

    def __init__(self, print_period) -> None:
        self.print_period = print_period
        self.logger = logging.getLogger("LoggingProgressTracker")

    def track(self, i: int, cost: float) -> None:
        if i % self.print_period == 0:
            self.logger.info("Iteration # [ %s ] cost is: %s", i, cost)


class NoopProgressTracker(ProgressTracker):

    def track(i: int, cost: float) -> None:
        pass


NOOP_PROGRESS_TRACKER = NoopProgressTracker()
