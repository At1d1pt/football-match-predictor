import os
import pandas as pd
import time

from dotenv import load_dotenv

from .calculations import *
from .league import get_league_positions, initialize_table, update_table, get_team_conceded, get_team_goals, get_total_goals

import requests

load_dotenv()

def build_features():
    df = pd.read_csv("data/raw/compiled.csv")
    elo_df = pd.read_csv("data/processed/elo_history.csv")
    market_df = pd.read_csv("data/processed/market_values.csv")
    xg_df = pd.read_csv("data/raw/xg.csv")
    
    df = df[df["season"] >= "2014"]

    df = df.merge(
        elo_df,
        on=["date", "home_team", "away_team"],
        how="left"
    )

    xg_df = xg_df.drop(columns=["season"])

    df = df.merge(
        xg_df,
        on=["date", "home_team", "away_team"],
        how="left"
    )

    market_lookup = (market_df.set_index(["season", "team"])["market_value"].to_dict())

    training_rows = []
    league_table = initialize_table()
    current_season = None

    for _, match in df.iterrows():
        if match["season"] != current_season:
            current_season = match["season"]
            league_table = initialize_table()


        total_league_goals = get_total_goals(league_table)
        positions = get_league_positions(league_table)
        home_position = positions.get(match["home_team"], 20)
        away_position = positions.get(match["away_team"], 20)
        position_diff = away_position - home_position

        home_total_goals_scored = get_team_goals(match["home_team"], league_table)
        away_total_goals_scored = get_team_goals(match["away_team"], league_table)

        home_total_conceded = get_team_conceded(match["home_team"], league_table)
        away_total_conceded = get_team_conceded(match["away_team"], league_table)        

        home_form = calculate_form(match["home_team"], df, match["date"])
        away_form = calculate_form(match["away_team"], df, match["date"])

        home_avg_goals = calculate_average_goals(match["home_team"], df, match["date"])
        away_avg_goals = calculate_average_goals(match["away_team"], df, match["date"])

        home_avg_conceded = calculate_average_conceded(match["home_team"], df, match["date"])
        away_avg_conceded = calculate_average_conceded(match["away_team"], df, match["date"])

        home_avg_shots = average_shots(match["home_team"], df, match["date"])
        away_avg_shots = average_shots(match["away_team"], df, match["date"])
        home_avg_shots_on_target = average_shots_on_target(match["home_team"], df, match["date"])
        away_avg_shots_on_target = average_shots_on_target(match["away_team"], df, match["date"])

        home_avg_corners = average_corners(match["home_team"], df, match["date"])
        away_avg_corners = average_corners(match["away_team"], df, match["date"])

        home_avg_yellow_cards = average_yellow_cards(match["home_team"], df, match["date"])
        away_avg_yellow_cards = average_yellow_cards(match["away_team"], df, match["date"])

        home_h2h, away_h2h, draw_h2h = head_to_head_form(match["home_team"], match["away_team"], df, match["date"])

        home_avg_xg = average_xg(match["home_team"], df, match["date"])
        away_avg_xg = average_xg(match["away_team"], df, match["date"])

        home_avg_xga = average_xga(match["home_team"], df, match["date"])
        away_avg_xga = average_xga(match["away_team"], df, match["date"])

        xg_difference = home_avg_xg - away_avg_xg
        xga_difference = home_avg_xga - away_avg_xga

        home_xg_conversion = home_avg_goals / home_avg_xg if home_avg_xg > 0 else 0
        away_xg_conversion = away_avg_goals / away_avg_xg if away_avg_xg > 0 else 0

        home_xga_prevention = home_avg_conceded / home_avg_xga if home_avg_xga > 0 else 0
        away_xga_prevention = away_avg_conceded / away_avg_xga if away_avg_xga > 0 else 0

        home_xg_form = home_avg_xg - home_avg_xga
        away_xg_form = away_avg_xg - away_avg_xga

        home_home_avg_xg = average_home_xg(match["home_team"], df, match["date"])
        away_away_avg_xg = average_away_xg(match["away_team"], df, match["date"])

        home_home_avg_xga = average_home_xga(match["home_team"], df, match["date"])
        away_away_avg_xga = average_away_xga(match["away_team"], df, match["date"])

        home_avg_xgd = home_avg_xg - home_avg_xga
        away_avg_xgd = away_avg_xg - away_avg_xga

        xgd_difference = home_avg_xgd - away_avg_xgd

        home_market_value = market_lookup.get((season_from_date(match["date"]), match["home_team"]), 0)
        away_market_value = market_lookup.get((season_from_date(match["date"]), match["away_team"]), 0)

        result = match["result"]

        training_rows.append({
            "date": match["date"],

            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "home_goals": match["home_goals"],
            "away_goals": match["away_goals"],

            "home_xg": match["home_xg"],
            "away_xg": match["away_xg"],

            #"home_league_position": home_position,
            #"away_league_position": away_position,
            "position_diff": position_diff,

            #"home_form": home_form,
            #"away_form": away_form,
            "form_difference": home_form - away_form,

            "home_home_form": calculate_home_form(match["home_team"], df, match["date"]),
            "away_away_form": calculate_away_form(match["away_team"], df, match["date"]),

            "home_avg_goals": home_avg_goals,
            "away_avg_goals": away_avg_goals,
            "home_avg_conceded": home_avg_conceded,
            "away_avg_conceded": away_avg_conceded,
            
            "home_avg_shots": home_avg_shots,
            "away_avg_shots": away_avg_shots,
            "home_avg_shots_on_target": home_avg_shots_on_target,
            "away_avg_shots_on_target": away_avg_shots_on_target,
            
            "home_avg_corners": home_avg_corners,
            "away_avg_corners": away_avg_corners,
            "corners_difference": home_avg_corners - away_avg_corners,
            
            "home_avg_yellow_cards": home_avg_yellow_cards,
            "away_avg_yellow_cards": away_avg_yellow_cards,
            "yellow_cards_difference": home_avg_yellow_cards - away_avg_yellow_cards,
            
            "result": result,
            
            #"home_elo": match["home_elo"],
            #"away_elo": match["away_elo"],
            "elo_difference": match["elo_diff"],
            #"abs_elo_difference": match["abs_elo_diff"],
            
            "shots_difference": home_avg_shots - away_avg_shots,
            "shots_on_target_difference": home_avg_shots_on_target - away_avg_shots_on_target,
            
            "home_conversion_rate": (home_avg_goals / max(home_avg_shots_on_target, 1)),
            "away_conversion_rate": (away_avg_goals / max(away_avg_shots_on_target, 1)),

            "home_rest_days": calculate_rest_days(match["home_team"], df, match["date"]),
            "away_rest_days": calculate_rest_days(match["away_team"], df, match["date"]),

            "home_head_to_head": home_h2h,
            "away_head_to_head": away_h2h,
            "draw_head_to_head": draw_h2h,

            "home_home_attack": get_home_attack_stats(match["home_team"], df, match["date"]),
            "home_home_defense": get_home_defense_stats(match["home_team"], df, match["date"]),

            "away_away_attack": get_away_attack_stats(match["away_team"], df, match["date"]),
            "away_away_defense": get_away_defense_stats(match["away_team"], df, match["date"]),

            "home_avg_xgd": home_avg_xgd,
            "away_avg_xgd": away_avg_xgd,
            "xgd_diff": xgd_difference,

            #"home_xg_conversion": home_xg_conversion,
            #"away_xg_conversion": away_xg_conversion,

            #"home_xga_prevention": home_xga_prevention,
            #"away_xga_prevention": away_xga_prevention,

            "home_xg_form": home_xg_form,
            "away_xg_form": away_xg_form,

            "home_home_avg_xg": home_home_avg_xg,
            "away_away_avg_xg": away_away_avg_xg,

            "home_home_avg_xga": home_home_avg_xga,
            "away_away_avg_xga": away_away_avg_xga,

            "market_value_difference": home_market_value - away_market_value,
            "market_value_ratio": home_market_value / max(away_market_value, 1)
        })

        update_table(league_table, match)

    training_df = pd.DataFrame(training_rows)
    training_df.to_csv("data/processed/training_data.csv", index=False)

def latest_elo(team, date, elo_df):
    history = elo_df[
        (
            (elo_df["home_team"] == team) |
            (elo_df["away_team"] == team)
        ) &
        (elo_df["date"] < date)
    ]

    if history.empty:
        return 1500.0

    latest = history.iloc[-1]

    if latest["home_team"] == team:
        return latest["home_elo"]

    return latest["away_elo"]

def build_match_features(home_team, away_team, date, df, elo_df, market_df, xg_df):
    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)

    from app.cache import get_standings
    table = get_standings() or []
    if not table:
        print("[builder] standings cache empty; using defaults")

    home_standings = 20
    away_standings = 20
    total_league_goals = 0
    home_total_goals_scored = 0
    away_total_goals_scored = 0

    for team in table:
        total_league_goals += team["goalsFor"]
        norm_name = normalize_team_name(team["team"]["name"])

        if norm_name == home_team:
            home_standings = team["position"]
            home_total_goals_scored = team["goalsFor"]

        if norm_name == away_team:
            away_standings = team["position"]
            away_total_goals_scored = team["goalsFor"]

    position_diff = away_standings - home_standings

    home_form = calculate_form(home_team, df, date)
    away_form = calculate_form(away_team, df, date)

    home_elo = latest_elo(home_team, date, elo_df)
    away_elo = latest_elo(away_team, date, elo_df)

    elo_difference = home_elo - away_elo

    home_avg_goals = calculate_average_goals(home_team, df, date)
    away_avg_goals = calculate_average_goals(away_team, df, date)

    home_avg_conceded = calculate_average_conceded(home_team, df, date)
    away_avg_conceded = calculate_average_conceded(away_team, df, date)

    home_avg_shots = average_shots(home_team, df, date)
    away_avg_shots = average_shots(away_team, df, date)

    home_avg_shots_on_target = average_shots_on_target(home_team, df, date)
    away_avg_shots_on_target = average_shots_on_target(away_team, df, date)

    home_avg_corners = average_corners(home_team, df, date)
    away_avg_corners = average_corners(away_team, df, date)

    home_avg_yellow_cards = average_yellow_cards(home_team, df, date)
    away_avg_yellow_cards = average_yellow_cards(away_team, df, date)

    home_h2h, away_h2h, draw_h2h = head_to_head_form(home_team, away_team, df, date)

    home_avg_xg = average_xg(home_team, xg_df, date)
    away_avg_xg = average_xg(away_team, xg_df, date)

    home_avg_xga = average_xga(home_team, xg_df, date)
    away_avg_xga = average_xga(away_team, xg_df, date)

    home_avg_xgd = home_avg_xg - home_avg_xga
    away_avg_xgd = away_avg_xg - away_avg_xga

    market_lookup = (
        market_df
        .set_index(["season", "team"])["market_value"]
        .to_dict()
    )

    season = season_from_date(date)

    home_market = market_lookup.get((season, home_team),0)

    away_market = market_lookup.get((season, away_team),0)

    return {

        "elo_difference": elo_difference,
        "position_diff": position_diff,
        "market_value_difference": home_market - away_market,
        "market_value_ratio": home_market / max(away_market, 1),
        "form_difference": home_form - away_form,

        "away_away_form": calculate_away_form(away_team, df, date),

        "xgd_diff": home_avg_xgd - away_avg_xgd,

        "home_home_avg_xg": average_home_xg( home_team, xg_df, date),

        "away_xg_form": away_avg_xgd,

        "home_head_to_head": home_h2h,

        "away_avg_goals": away_avg_goals,

        "home_home_attack": get_home_attack_stats(home_team, df, date),

        "home_home_form": calculate_home_form(home_team, df, date),

        "home_home_defense": get_home_defense_stats(home_team, df, date),

        "away_rest_days": calculate_rest_days(away_team, df, date),

        "corners_difference": home_avg_corners - away_avg_corners,

        "home_rest_days": calculate_rest_days(home_team, df, date),

        "away_avg_xgd": away_avg_xgd,

        "home_avg_xgd": home_avg_xgd,

        "away_away_avg_xg": average_away_xg(away_team, xg_df, date),

        "away_away_attack": get_away_attack_stats(away_team, df, date),

        "home_conversion_rate": home_avg_goals / max(home_avg_shots_on_target, 1),

        "away_away_defense": get_away_defense_stats(away_team, df, date),

        "draw_head_to_head": draw_h2h,

        "away_avg_corners": away_avg_corners,

        "home_avg_yellow_cards": home_avg_yellow_cards,

        "home_home_avg_xga": average_home_xga(home_team, xg_df, date),

        "home_avg_shots": home_avg_shots,

        "away_avg_shots": away_avg_shots,

        "home_avg_shots_on_target": home_avg_shots_on_target,

        "away_conversion_rate": away_avg_goals / max(away_avg_shots_on_target, 1),

        "away_avg_shots_on_target": away_avg_shots_on_target,

        "home_avg_conceded": home_avg_conceded,

        "home_avg_goals": home_avg_goals,

        "home_xg_form": home_avg_xgd,

        "shots_difference": home_avg_shots - away_avg_shots,

        "shots_on_target_difference": home_avg_shots_on_target - away_avg_shots_on_target,

        "home_avg_corners": home_avg_corners,

        "away_away_avg_xga": average_away_xga(away_team, xg_df, date),

        "away_avg_conceded": away_avg_conceded,

        "away_head_to_head": away_h2h,

        "away_avg_yellow_cards": away_avg_yellow_cards,

        "yellow_cards_difference": home_avg_yellow_cards - away_avg_yellow_cards
    }

def get_team_profile(team, date, df, elo_df, xg_df):
    """
    Returns a rich team profile dict for the given team ahead of the given date.
    Includes last-5 match results as structured objects, elo, xg, goals, etc.
    """
    team = normalize_team_name(team)

    last_five = get_last_five_matches(team, df, date)

    form_matches = []
    for _, match in last_five.iterrows():
        is_home = match["home_team"] == team
        opponent = match["away_team"] if is_home else match["home_team"]
        goals_for = int(match["home_goals"]) if is_home else int(match["away_goals"])
        goals_against = int(match["away_goals"]) if is_home else int(match["home_goals"])

        if goals_for > goals_against:
            result = "W"
        elif goals_for < goals_against:
            result = "L"
        else:
            result = "D"

        form_matches.append({
            "date": str(match["date"]),
            "opponent": normalize_team_name(opponent),
            "venue": "H" if is_home else "A",
            "goals_for": goals_for,
            "goals_against": goals_against,
            "score": f"{goals_for}–{goals_against}",
            "result": result,
        })

    current_elo = latest_elo(team, date, elo_df)

    previous_matches = elo_df[
        ((elo_df["home_team"] == team) | (elo_df["away_team"] == team)) &
        (elo_df["date"] < date)
    ].sort_values("date")

    elo_trend = 0.0
    if len(previous_matches) >= 2:
        old = previous_matches.iloc[-5] if len(previous_matches) >= 5 else previous_matches.iloc[0]
        old_elo = old["home_elo"] if old["home_team"] == team else old["away_elo"]
        elo_trend = round(float(current_elo) - float(old_elo), 1)

    avg_goals     = round(calculate_average_goals(team, df, date), 2)
    avg_conceded  = round(calculate_average_conceded(team, df, date), 2)
    avg_shots     = round(average_shots(team, df, date), 2)
    avg_shots_ot  = round(average_shots_on_target(team, df, date), 2)
    avg_corners   = round(average_corners(team, df, date), 2)
    avg_yellows   = round(average_yellow_cards(team, df, date), 2)
    rest_days     = calculate_rest_days(team, df, date)
    avg_xg        = round(average_xg(team, xg_df, date), 2)
    avg_xga       = round(average_xga(team, xg_df, date), 2)

    form_points = calculate_form(team, df, date)   
    form_chips  = [m["result"] for m in form_matches]

    return {
        "elo":            round(float(current_elo), 1),
        "elo_trend":      elo_trend,
        "form":           form_chips,           
        "form_points":    form_points,          
        "last_five":      form_matches,         
        "avg_goals":      avg_goals,
        "avg_conceded":   avg_conceded,
        "avg_shots":      avg_shots,
        "avg_shots_on_target": avg_shots_ot,
        "avg_corners":    avg_corners,
        "avg_yellows":    avg_yellows,
        "avg_xg":         avg_xg,
        "avg_xga":        avg_xga,
        "avg_xgd":        round(avg_xg - avg_xga, 2),
        "rest_days":      rest_days,
    }


if __name__ == "__main__":
    start = time.time()
    build_features()
    end = time.time()

    print(f"Time taken to build features: {end-start}s")
