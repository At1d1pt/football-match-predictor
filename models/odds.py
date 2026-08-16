import json

def prob_to_decimal_odds(prob: float) -> float:
    if prob <= 0:
        return 0.0
    return round(1.0 / prob, 3)


def decimal_odds_to_prob(odds: float) -> float:
    if odds <= 1.0:
        return 0.0
    return round(1.0 / odds, 4)


def calculate_ev(p_model: float, bookie_odds: float) -> float:
    if bookie_odds <= 1.0 or p_model <= 0:
        return -1.0
    return round((p_model * bookie_odds) - 1.0, 4)


def calculate_edge(p_model: float, bookie_odds: float) -> float:
    if bookie_odds <= 1.0:
        return 0.0
    return round(p_model - (1.0 / bookie_odds), 4)


def kelly_criterion(p_model: float, bookie_odds: float, fraction: float = 0.25) -> float:
    if bookie_odds <= 1.0 or p_model <= 0:
        return 0.0

    b = bookie_odds - 1.0
    q = 1.0 - p_model
    full_kelly = (b * p_model - q) / b

    if full_kelly <= 0:
        return 0.0

    return round(min(full_kelly * fraction, 0.10), 4)


def compare_market(model_probs: dict, market_odds: dict, kelly_fraction: float = 0.25, min_ev: float = 0.02) -> dict:
    comparison = {}
    value_bets = []

    for outcome, p_model in model_probs.items():
        if outcome in market_odds:
            odds_bookie = float(market_odds[outcome])
            fair = prob_to_decimal_odds(p_model)
            ev = calculate_ev(p_model, odds_bookie)
            edge = calculate_edge(p_model, odds_bookie)
            stake = kelly_criterion(p_model, odds_bookie, fraction=kelly_fraction)
            is_value = ev >= min_ev

            comparison[outcome] = {
                "model_probability": round(p_model, 3),
                "fair_odds": fair,
                "market_odds": odds_bookie,
                "implied_probability": round(1.0 / odds_bookie, 3) if odds_bookie > 0 else 0,
                "expected_value_pct": round(ev * 100, 2),
                "edge_pct": round(edge * 100, 2),
                "recommended_stake_pct": round(stake * 100, 2),
                "is_value_bet": is_value
            }

            if is_value:
                value_bets.append({
                    "outcome": outcome,
                    "market_odds": odds_bookie,
                    "fair_odds": fair,
                    "ev_pct": round(ev * 100, 2),
                    "recommended_stake_pct": round(stake * 100, 2)
                })

    return {
        "outcomes": comparison,
        "value_bets": value_bets,
        "has_value_bet": len(value_bets) > 0
    }


if __name__ == "__main__":
    model_probs = {"H": 0.548, "D": 0.228, "A": 0.224}
    sample_market_odds = {"H": 2.05, "D": 3.50, "A": 4.00}

    print(json.dumps(compare_market(model_probs, sample_market_odds), indent=2))