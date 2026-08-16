import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

from .calculations import normalize_team_name

BASE_URL = "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1"

HEADERS = {
    "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def get_market_values(season):
    url = BASE_URL+f"?saison_id={season}"
    r = requests.get(url,headers=HEADERS)
    
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find_all("table", class_="items")[0]
    rows = table.find_all("tr")[2:]

    market_values = []

    for row in rows:
        cols = row.find_all("td")

        team = normalize_team_name(cols[1].get_text(strip=True))
        value = cols[-1].get_text(strip=True)

        market_values.append({
            "season": season,
            "team": team,
            "market_value": parse_market_value(value)
        })

    return pd.DataFrame(market_values)

def parse_market_value(value):
    value = value.replace("€", "").strip()

    if value.endswith("bn"):
        return float(value[:-2]) * 1000000

    elif value.endswith("m"):
        return float(value[:-1]) * 1000

    elif value.endswith("k"):
        return float(value[:-1])

    return value/1000

def build_market_value_dataset(start=2014, end=2026):
    all_data = []

    for season in range(start, end + 1):
        print(f"Downloading {season}/{season + 1}")

        df = get_market_values(season)
        all_data.append(df)

        time.sleep(2)

    market_df = pd.concat(all_data, ignore_index=True)
    market_df.to_csv(
        "data/processed/market_values.csv",
        index=False
    )

    return market_df

if __name__ == "__main__":
    build_market_value_dataset()