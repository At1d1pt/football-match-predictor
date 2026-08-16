"""

This file helps in converting the raw CSV into a single, processed CSV, which helps in training the model.

"""

import pandas as pd
import os
from glob import glob

def process_data():
    files = glob("data/raw/pl_*.csv")
    #print(files)
    all_dfs = []

    for file in files:
        df = pd.read_csv(file)
        filename = os.path.basename(file)
        season = filename.split("_")[1].split(".")[0]
        df["season"] = season
        columns = [
            "season",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
            "HS",
            "AS",
            "HST",
            "AST",
            "HC",
            "AC",
            "HY",
            "AY"
        ]

        df = df[columns]
        df.rename(columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "FTR": "result",
            "HS": "home_shots",
            "AS": "away_shots",
            "HST": "home_shots_on_target",
            "AST": "away_shots_on_target",
            "HC": "home_corners",
            "AC": "away_corners",
            "HY": "home_yellows",
            "AY": "away_yellows"
        }, inplace=True)

        #print(df)
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)

    final_df["date"] = pd.to_datetime(final_df["date"], dayfirst=True)
    final_df.sort_values("date", inplace=True)

    final_df.to_csv("data/raw/compiled.csv", index=False)

if __name__ == "__main__":
    process_data()