import math
import json
import numpy as np

class PoissonModel:
    def __init__(self, max_goals=6, rho=0.0):
        self.max_goals = max_goals
        self.rho = rho

    def poisson_pmf(self, k: int, lam: float) -> float:
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return (lam ** k) * math.exp(-lam) / math.factorial(k)

    def _dixon_coles_tau(self, x: int, y: int, home_lambda: float, away_lambda: float) -> float:
        if self.rho == 0.0:
            return 1.0

        if x == 0 and y == 0:
            return 1.0 - (home_lambda * away_lambda * self.rho)
        elif x == 1 and y == 0:
            return 1.0 + (away_lambda * self.rho)
        elif x == 0 and y == 1:
            return 1.0 + (home_lambda * self.rho)
        elif x == 1 and y == 1:
            return 1.0 - self.rho
        else:
            return 1.0

    def generate_score_matrix(self, home_lambda: float, away_lambda: float) -> np.ndarray:
        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))

        home_probs = [self.poisson_pmf(i, home_lambda) for i in range(self.max_goals + 1)]
        away_probs = [self.poisson_pmf(j, away_lambda) for j in range(self.max_goals + 1)]

        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                tau = self._dixon_coles_tau(i, j, home_lambda, away_lambda)
                matrix[i, j] = max(0.0, home_probs[i] * away_probs[j] * tau)

        total_prob = matrix.sum()
        if total_prob > 0:
            matrix = matrix / total_prob

        return matrix

    def predict(self, home_lambda: float, away_lambda: float) -> dict:
        home_lambda = max(0.05, float(home_lambda))
        away_lambda = max(0.05, float(away_lambda))

        matrix = self.generate_score_matrix(home_lambda, away_lambda)

        home_win_prob = float(np.sum(np.tril(matrix, -1)))
        draw_prob = float(np.sum(np.diag(matrix)))
        away_win_prob = float(np.sum(np.triu(matrix, 1)))

        probabilities = {
            "H": round(home_win_prob, 3),
            "D": round(draw_prob, 3),
            "A": round(away_win_prob, 3)
        }
        predicted_outcome = max(probabilities, key=probabilities.get)

        btts_prob = float(np.sum(matrix[1:, 1:]))

        goal_grid = np.fromfunction(lambda i, j: i + j, matrix.shape, dtype=int)
        over_0_5 = float(np.sum(matrix[goal_grid > 0.5]))
        over_1_5 = float(np.sum(matrix[goal_grid > 1.5]))
        over_2_5 = float(np.sum(matrix[goal_grid > 2.5]))
        over_3_5 = float(np.sum(matrix[goal_grid > 3.5]))

        flat_indices = np.argsort(matrix.ravel())[::-1]
        top_scorelines = []
        for idx in flat_indices[:5]:
            i, j = np.unravel_index(idx, matrix.shape)
            top_scorelines.append({
                "score": f"{i}-{j}",
                "probability": round(float(matrix[i, j]), 3)
            })

        most_likely_score = top_scorelines[0]["score"]

        return {
            "expected_goals": {
                "home": round(home_lambda, 2),
                "away": round(away_lambda, 2)
            },
            "probabilities": {
                "home_win": round(home_win_prob, 3),
                "draw": round(draw_prob, 3),
                "away_win": round(away_win_prob, 3)
            },
            "predicted_outcome": predicted_outcome,
            "most_likely_score": most_likely_score,
            "top_scorelines": top_scorelines,
            "btts_probability": round(btts_prob, 3),
            "over_under": {
                "over_0_5": round(over_0_5, 3),
                "over_1_5": round(over_1_5, 3),
                "over_2_5": round(over_2_5, 3),
                "over_3_5": round(over_3_5, 3)
            }
        }


if __name__ == "__main__":
    model = PoissonModel(max_goals=6)
    res = model.predict(home_lambda=1.85, away_lambda=1.10)
    print("Poisson Model Test Output:")
    print(json.dumps(res, indent=2))