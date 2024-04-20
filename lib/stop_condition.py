class StopCondition:
    def test(self, t: int, cost: float) -> bool:
        pass

    def get_label(self) -> str:
        pass

class IterCountStopCondition(StopCondition):
    max_iter_count: int

    def __init__(self, max_iter_count) -> None:
        self.max_iter_count = max_iter_count

    def test(self, t: int, cost: float) -> bool:
        return self.max_iter_count >= t

    def get_label(self) -> str:
        return str(self.max_iter_count) + "-iter"


class ApproximationStopCondition(StopCondition):
    accuracy: float

    def __init__(self, max_iter_count) -> None:
        self.max_iter_count = max_iter_count

    def test(self, t: int, cost: float) -> bool:
        return self.accuracy >= cost
    
    def get_label(self) -> str:
        return str(self.accuracy) + "-accuracy"

