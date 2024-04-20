from logging import Logger
import logging
import math


class ProgressTracker:

    def track(self, i: int, cost: float) -> None:
        pass

    def track_gradient_check(self, i: int, of: int) -> None:
        pass


class LoggingProgressTracker(ProgressTracker):
    __print_period: int
    __logger: Logger

    def __init__(self, print_period) -> None:
        self.__print_period = print_period
        self.__logger = logging.getLogger("LoggingProgressTracker")

    def track(self, i: int, cost: float) -> None:
        if i % self.__print_period == 0:
            self.__logger.info("Iteration # [ %s ] cost is: %s", i, cost)

    def track_gradient_check(self, i: int, of: int) -> None:
        if i % (of // 20) == 0:
            progress = i / of * 100
            self.__logger.info(
                "Gradient check # [ %s ]: [ %s ] %% completed", i, math.ceil(progress)
            )


class NoopProgressTracker(ProgressTracker):

    def track(self, i: int, cost: float) -> None:
        pass

    def track_gradient_check(self, i: int, of: int) -> None:
        pass


NOOP_PROGRESS_TRACKER = NoopProgressTracker()
