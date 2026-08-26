import os
import time
import pandas as pd
from understatapi import UnderstatClient

from .calculations import normalize_team_name

LEAGUE = "EPL"
SEASONS = [str(y) for y in range(2014, 2027)]

OUTPUT_PATH = "data/raw/xG.csv"
REQUEST_DELAY_SECONDS = 2

def fetch_season(client, season):
    matches = client.league(league=LEAGUE).get_match_data(season=season)

    rows = []
    for match in matches:
        if not match["isResult"]:
            continue

        rows.append({
            "season": int(season),
            "date": match["datetime"][:10],
            "home_team": normalize_team_name(match["h"]["title"]),
            "away_team": normalize_team_name(match["a"]["title"]),
            "home_xg": float(match["xG"]["h"]),
            "away_xg": float(match["xG"]["a"]),
            "home_xga": float(match["xG"]["a"]),
            "away_xga": float(match["xG"]["h"])
        })

    return pd.DataFrame(rows)


def main():
    client = UnderstatClient()
    all_matches = []

    for season in SEASONS:
        print(f"Fetching {season}/{int(season)+1}")

        df = fetch_season(client, season)
        all_matches.append(df)
        time.sleep(REQUEST_DELAY_SECONDS)

    understat = pd.concat(all_matches, ignore_index=True)
    understat.to_csv(OUTPUT_PATH, index=False)

    print(understat.head())
    print(f"\nSaved {len(understat)} matches.")

if __name__ == "__main__":
    main()