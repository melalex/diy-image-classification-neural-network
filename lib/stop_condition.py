class StopCondition:
    def test(self, t: int, cost: float):
        pass

class IterCountStopCondition(StopCondition):
    max_iter_count: int

    def __init__(self, max_iter_count) -> None:
        self.max_iter_count = max_iter_count

    def test(self, t: int, cost: float):
        return self.max_iter_count >= t

class ApproximationStopCondition(StopCondition):
    accuracy: float

    def __init__(self, max_iter_count) -> None:
        self.max_iter_count = max_iter_count

    def test(self, t: int, cost: float):
        return self.accuracy >= cost
