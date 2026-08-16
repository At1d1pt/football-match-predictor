from urllib3.packages import ssl_match_hostname
import os
import requests as req

from joblib import load
from dotenv import load_dotenv

from features.builder import build_match_features, get_team_profile
from features.elo import latest_elo
from models.poisson import PoissonModel
from models.odds import prob_to_decimal_odds, compare_market

import pandas as pd

load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

classifier = load("models/model_xgboost.pkl")
home_model = load("models/model_home_goals.pkl")
away_model = load("models/model_away_goals.pkl")
feature_columns = load("models/feature_columns.pkl")

df = pd.read_csv("data/raw/compiled.csv")
elo_df = pd.read_csv("data/processed/elo_history.csv")
market_df = pd.read_csv("data/processed/market_values.csv")
xg_df = pd.read_csv("data/raw/xg.csv")

poisson_calculator = PoissonModel(max_goals=6)

def goals_to_result(home, away, threshold=0.25):
    diff = home - away

    if abs(diff) < threshold:
        return "D"
    elif diff > 0:
        return "H"
    else:
        return "A"

def fetch_market_odds(home_team: str, away_team: str) -> dict | None:
    if not ODDS_API_KEY or ODDS_API_KEY == "your_odds_api_key_here":
        return None

    try:
        url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "uk",
            "markets": "h2h",
            "oddsFormat": "decimal"
        }
        data = req.get(url, params=params, timeout=5).json()

        if not isinstance(data, list):
            return None
    except Exception:
        return None

    for match in data:
        h = match.get("home_team", "")
        a = match.get("away_team", "")

        if home_team.lower() in h.lower() and away_team.lower() in a.lower():
            result = {}

            for bookmaker in match.get("bookmakers", [])[:1]:  # Use first bookmaker only
                for market in bookmaker.get("markets", []):
                    key = market["key"]

                    if key == "h2h":
                        for o in market["outcomes"]:
                            if o["name"] == "Draw":
                                result["D"] = o["price"]
                            elif o["name"] == h:
                                result["H"] = o["price"]
                            else:
                                result["A"] = o["price"]

                    elif key == "totals":
                        for o in market["outcomes"]:
                            if o["name"] == "Over" and o.get("point") == 2.5:
                                result["over_2_5"] = o["price"]
                            elif o["name"] == "Under" and o.get("point") == 2.5:
                                result["under_2_5"] = o["price"]

                    elif key == "btts":
                        for o in market["outcomes"]:
                            if o["name"] == "Yes":
                                result["btts_yes"] = o["price"]
                            elif o["name"] == "No":
                                result["btts_no"] = o["price"]

            return result if result else None

    return None

def predict(home, away, date):
    features = build_match_features(home, away, date, df, elo_df, market_df, xg_df)
    X = pd.DataFrame([features])
    X = X[feature_columns]

    classifier_prediction = classifier.predict(X)[0]
    classifier_probabilities = classifier.predict_proba(X)[0]

    home_goals = float(home_model.predict(X)[0])
    away_goals = float(away_model.predict(X)[0])

    goal_prediction = goals_to_result(home_goals, away_goals, 0.25)
    poisson_res = poisson_calculator.predict(home_goals, away_goals)

    reverse_map = {
        0: "H",
        1: "D",
        2: "A"
    }

    p = poisson_res["probabilities"]
    ou = poisson_res["over_under"]
    btts = poisson_res["btts_probability"]

    fair_odds = {
        "match_result": {
            "H": prob_to_decimal_odds(p["home_win"]),
            "D": prob_to_decimal_odds(p["draw"]),
            "A": prob_to_decimal_odds(p["away_win"])
        },
        "over_under": {
            "over_0_5":  prob_to_decimal_odds(ou["over_0_5"]),
            "under_0_5": prob_to_decimal_odds(1.0 - ou["over_0_5"]),
            "over_1_5":  prob_to_decimal_odds(ou["over_1_5"]),
            "under_1_5": prob_to_decimal_odds(1.0 - ou["over_1_5"]),
            "over_2_5":  prob_to_decimal_odds(ou["over_2_5"]),
            "under_2_5": prob_to_decimal_odds(1.0 - ou["over_2_5"]),
            "over_3_5":  prob_to_decimal_odds(ou["over_3_5"]),
            "under_3_5": prob_to_decimal_odds(1.0 - ou["over_3_5"])
        },
        "btts": {
            "yes": prob_to_decimal_odds(btts),
            "no":  prob_to_decimal_odds(1.0 - btts)
        }
    }

    market_odds = fetch_market_odds(home, away)

    odds_analysis = None
    if market_odds:
        match_result_analysis = compare_market(
            {"H": p["home_win"], "D": p["draw"], "A": p["away_win"]},
            {k: market_odds[k] for k in ["H", "D", "A"] if k in market_odds}
        )
        ou_analysis = compare_market(
            {"over_2_5": ou["over_2_5"], "under_2_5": 1.0 - ou["over_2_5"]},
            {k: market_odds[k] for k in ["over_2_5", "under_2_5"] if k in market_odds}
        )
        btts_analysis = compare_market(
            {"yes": btts, "no": 1.0 - btts},
            {k: market_odds[k] for k in ["btts_yes", "btts_no"] if k in market_odds}
        )
        odds_analysis = {
            "match_result": match_result_analysis,
            "over_under_2_5": ou_analysis,
            "btts": btts_analysis,
            "all_value_bets": (
                match_result_analysis["value_bets"] +
                ou_analysis["value_bets"] +
                btts_analysis["value_bets"]
            )
        }

    home_stats = get_team_profile(home, date, df, elo_df, xg_df)
    away_stats = get_team_profile(away, date, df, elo_df, xg_df)

    return {
        "home_team": home,
        "away_team": away,

        "classifier": {
            "prediction": reverse_map[classifier_prediction],
            "home_win_probability": round(float(classifier_probabilities[0]), 3),
            "draw_probability": round(float(classifier_probabilities[1]), 3),
            "away_win_probability": round(float(classifier_probabilities[2]), 3)
        },

        "goal_model": {
            "home_goals": round(home_goals, 2),
            "away_goals": round(away_goals, 2),
            "prediction": goal_prediction
        },

        "poisson_model": poisson_res,

        "fair_odds": fair_odds,
        "market_odds": market_odds,
        "odds_analysis": odds_analysis,

        "home_stats": home_stats,
        "away_stats": away_stats,
    }
    
def get_elo_map():
    elo_map = {}

    for team in elo_df["home_team"].dropna().unique():
        elo = latest_elo(team, pd.Timestamp.today(), elo_df)

        if elo is not None:
            elo_map[team] = round(float(elo), 2)

    for team in elo_df["away_team"].dropna().unique():
        if team not in elo_map:
            elo = latest_elo(team, pd.Timestamp.today(), elo_df)

            if elo is not None:
                elo_map[team] = round(float(elo), 2)

    return elo_map