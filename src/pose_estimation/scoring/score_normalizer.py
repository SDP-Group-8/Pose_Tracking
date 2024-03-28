from scipy.stats import logistic

class ScoreNormalizer:
    @staticmethod
    def convert(score: float, loc: float, scale: float = 2):
        return 1 - logistic.cdf(score, scale, loc)
