from scipy.stats import logistic

class ScoreNormalizer:
    @staticmethod
    def convert(score: float):
        return 1 - logistic.cdf(score, scale=2, loc=2)
