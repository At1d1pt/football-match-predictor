import pandas as pd

def load_matches(path="data/raw/compiled.csv"):
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df

def initialize_ratings(df):
    teams = pd.concat([
        df["home_team"],
        df["away_team"]
    ]).unique()

    ratings = {}

    for team in teams:
        ratings[team] = 1500

    return ratings

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_ratings(rating_a, rating_b, home, away, k=32):
    expected_a = expected_score(rating_a, rating_b)
    expected_b = expected_score(rating_b, rating_a)

    if home > away:
        home_actual = 1
        away_actual = 0
    elif home < away:
        home_actual = 0
        away_actual = 1
    else:
        home_actual = 0.5
        away_actual = 0.5

    new_rating_a = rating_a + k * (home_actual - expected_a)
    new_rating_b = rating_b + k * (away_actual - expected_b)

    return new_rating_a, new_rating_b

def build_elo_history(df):
    ratings = initialize_ratings(df)
    elo_history = []

    for _, match in df.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_goals = match["home_goals"]
        away_goals = match["away_goals"]

        home_rating = ratings[home_team]
        away_rating = ratings[away_team]

        new_home_rating, new_away_rating = update_ratings(home_rating, away_rating, home_goals, away_goals)

        ratings[home_team] = new_home_rating
        ratings[away_team] = new_away_rating

        elo_history.append({
            "date": match["date"],
            "home_team": home_team,
            "away_team": away_team,
            "home_elo": new_home_rating,
            "away_elo": new_away_rating,
            "home_elo_before": home_rating,
            "away_elo_before": away_rating,
            "home_elo_after": ratings[home_team],
            "away_elo_after": ratings[away_team],
            "elo_diff": new_home_rating - new_away_rating,
            "abs_elo_diff": abs(new_home_rating - new_away_rating),
        })

    return pd.DataFrame(elo_history)

def latest_elo(team, date, elo_df):
    date = pd.to_datetime(date)
    elo_df = elo_df.copy()
    elo_df["date"] = pd.to_datetime(elo_df["date"])

    matches = elo_df[
        (elo_df["date"] < date) &
        (
            (elo_df["home_team"] == team) |
            (elo_df["away_team"] == team)
        )
    ].sort_values("date")

    if matches.empty:
        return 1500.0

    last_match = matches.iloc[-1]

    if last_match["home_team"] == team:
        return last_match["home_elo"]
    else:
        return last_match["away_elo"]

def main():
    df = load_matches()
    elo_df = build_elo_history(df)
    elo_df.to_csv("data/processed/elo_history.csv", index=False)

    print("Elo history generated successfully!")

if __name__ == "__main__":
    main()