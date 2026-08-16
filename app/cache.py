import os
import requests

from dotenv import load_dotenv

load_dotenv()

LEAGUE_STANDINGS_URL = "https://api.football-data.org/v4/competitions/PL/standings"

_standings_cache: list[dict] = []


def get_standings() -> list[dict]:
    return _standings_cache


def fetch_and_cache_standings() -> None:
    global _standings_cache

    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    if not api_key:
        print("[cache] FOOTBALL_DATA_API_KEY not set; standings cache will stay empty")
        return

    try:
        response = requests.get(
            LEAGUE_STANDINGS_URL,
            headers={"X-Auth-Token": api_key},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        table = data.get("standings", [{}])[0].get("table", [])
        _standings_cache = table
        print(f"[cache] loaded {len(_standings_cache)} teams into league standings cache")
    except Exception as err:
        print(f"[cache] failed to load standings: {err}")
