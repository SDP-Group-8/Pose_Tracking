from abc import ABC, abstractmethod

class Score(ABC):
    @abstractmethod
    def compute_score():
        pass
