import pandas as pd

from datetime import datetime

def season_from_date(date):
    if date is None or (isinstance(date, float) and pd.isna(date)):
        return None
    if isinstance(date, str):
        try:
            date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            date = datetime.strptime(date, "%d-%m-%Y")
    elif isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()

    if date.month >= 8:
        return date.year

    return date.year - 1

def get_last_five_matches(team, df, date):
    previous_matches = df[df["date"] < date]
    team_matches = previous_matches[(previous_matches["home_team"] == team) | (previous_matches["away_team"] == team)]
    last_five = team_matches.tail(5)
    return last_five

def calculate_form(team, df, date):
    last_five = get_last_five_matches(team, df, date)
    points = 0

    for _, match in last_five.iterrows():
        if match["home_team"] == team:
            if match["result"] == "H":
                points += 3
            elif match["result"] == "D":
                points += 1

        else:
            if match["result"] == "A":
                points += 3
            elif match["result"] == "D":
                points += 1

    return points

def calculate_average_goals(team, df, date):
    last_five = get_last_five_matches(team, df, date)
    total_goals = 0

    for _, match in last_five.iterrows():
        if match["home_team"] == team:
            total_goals += match["home_goals"]
        else:
            total_goals += match["away_goals"]

    average_goals = total_goals / len(last_five) if len(last_five) > 0 else 0
    return average_goals

def calculate_average_conceded(team, df, date):
    last_five = get_last_five_matches(team, df, date)
    total_conceded = 0

    for _, match in last_five.iterrows():
        if match["home_team"] == team:
            total_conceded += match["away_goals"]
        else:
            total_conceded += match["home_goals"]

    average_conceded = total_conceded / len(last_five) if len(last_five) > 0 else 0
    return average_conceded

def average_shots(team, df, date):
    last_five = get_last_five_matches(team, df, date)
    total_shots = 0

    for _, match in last_five.iterrows():
        if match["home_team"] == team:
            total_shots += match["home_shots"]
        else:
            total_shots += match["away_shots"]

    average_shots = total_shots / len(last_five) if len(last_five) > 0 else 0
    return average_shots

def average_shots_on_target(team, df, date):
    last_five = get_last_five_matches(team, df, date)
    total_shots_on_target = 0

    for _, match in last_five.iterrows():
        if match["home_team"] == team:
            total_shots_on_target += match["home_shots_on_target"]
        else:
            total_shots_on_target += match["away_shots_on_target"]

    average_shots_on_target = total_shots_on_target / len(last_five) if len(last_five) > 0 else 0
    return average_shots_on_target

def average_corners(team, df, date):
    last_five = get_last_five_matches(team, df, date)
    total_corners = 0

    for _, match in last_five.iterrows():
        if match["home_team"] == team:
            total_corners += match["home_corners"]
        else:
            total_corners += match["away_corners"]

    average_corners = total_corners / len(last_five) if len(last_five) > 0 else 0
    return average_corners

def average_yellow_cards(team, df, date):
    last_five = get_last_five_matches(team, df, date)
    total_yellow_cards = 0

    for _, match in last_five.iterrows():
        if match["home_team"] == team:
            total_yellow_cards += match["home_yellows"]
        else:
            total_yellow_cards += match["away_yellows"]

    average_yellow_cards = total_yellow_cards / len(last_five) if len(last_five) > 0 else 0
    return average_yellow_cards

def calculate_rest_days(team, df, date):
    previous_matches = df[df["date"] < date]
    team_matches = previous_matches[(previous_matches["home_team"] == team) | (previous_matches["away_team"] == team)]
    last_match = team_matches.tail(1)

    if last_match.empty:
        return 7  # 7 days of rest if no previous match is found

    last_match_date = pd.to_datetime(last_match.iloc[0]["date"], dayfirst=True)
    current_date = pd.to_datetime(date, dayfirst=True)
    rest_days = (current_date - last_match_date).days

    return rest_days

def calculate_home_form(team, df, date):
    previous_matches = df[df["date"] < date]

    home_matches = previous_matches[previous_matches["home_team"] == team]

    last_five = home_matches.tail(5)

    points = 0

    for _, match in last_five.iterrows():
        if match["home_goals"] > match["away_goals"]:
            points += 3
        elif match["home_goals"] == match["away_goals"]:
            points += 1

    return points

def calculate_away_form(team, df, date):
    previous_matches = df[df["date"] < date]

    away_matches = previous_matches[previous_matches["away_team"] == team]

    last_five = away_matches.tail(5)

    points = 0

    for _, match in last_five.iterrows():
        if match["away_goals"] > match["home_goals"]:
            points += 3
        elif match["away_goals"] == match["home_goals"]:
            points += 1

    return points


def calculate_head_to_head(home_team, away_team, df, date):
    previous_matches = df[df["date"] < date]
    head_to_head_matches = previous_matches[
        ((previous_matches["home_team"] == home_team) & (previous_matches["away_team"] == away_team)) |
        ((previous_matches["home_team"] == away_team) & (previous_matches["away_team"] == home_team))
    ]
    return head_to_head_matches

def head_to_head_form(home_team, away_team, df, date):
    h2h = calculate_head_to_head(home_team, away_team, df, date)
    home = 0
    away = 0
    draw = 0
    total = len(h2h)

    if total == 0:
        return 0,0,0

    for _, match in h2h.iterrows():
        if match["away_goals"] < match["home_goals"]:
            home += 1
        elif match["away_goals"] > match["home_goals"]:
            away += 1
        else:
            draw += 1
        
    return home/total, away/total, draw/total

def get_home_attack_stats(team, df, date, games=5):
    previous_matches = df[df["date"] < date]

    home_games = previous_matches[previous_matches["home_team"] == team].sort_values("date").tail(games)

    if home_games.empty:
        return 0

    avg_scored = home_games["home_goals"].mean()

    return avg_scored

def get_home_defense_stats(team, df, date, games=5):
    previous_matches = df[df["date"] < date]

    home_games = previous_matches[previous_matches["home_team"] == team].sort_values("date").tail(games)

    if home_games.empty:
        return 0

    avg_conceded = home_games["away_goals"].mean()

    return avg_conceded

def get_away_attack_stats(team, df, date, games=5):
    previous_matches = df[df["date"] < date]

    away_games = previous_matches[previous_matches["away_team"] == team].sort_values("date").tail(games)

    if away_games.empty:
        return 0

    avg_scored = away_games["away_goals"].mean()

    return avg_scored

def get_away_defense_stats(team, df, date, games=5):
    previous_matches = df[df["date"] < date]

    away_games = previous_matches[previous_matches["away_team"] == team].sort_values("date").tail(games)

    if away_games.empty:
        return 0

    avg_conceded = away_games["home_goals"].mean()

    return avg_conceded

def average_xg(team, df, date):
    last_five = get_last_five_matches(team, df, date)

    total_xg = 0
    games = 0

    for _, match in last_five.iterrows():
        if match["home_team"] == team:
            total_xg += match["home_xg"]
        else:
            total_xg += match["away_xg"]

        games += 1

    return total_xg / games if games > 0 else 0

def average_xga(team, df, date):
    last_five = get_last_five_matches(team, df, date)
    total_xga = 0
    games = 0

    for _, match in last_five.iterrows():
        if match["home_team"] == team:
            total_xga += match["away_xg"]
        else:
            total_xga += match["home_xg"]

        games += 1

    return total_xga / games if games > 0 else 0

def average_home_xg(team, df, date):
    home_matches = df[
        (df["home_team"] == team) &
        (pd.to_datetime(df["date"], dayfirst=True) < pd.to_datetime(date, dayfirst=True))
    ].sort_values("date", ascending=False)

    total_xg = 0
    games = 0

    for _, match in home_matches.iterrows():
        total_xg += match["home_xg"]
        games += 1

        if games == 5:
            break

    return total_xg / games if games > 0 else 0

def average_away_xg(team, df, date):
    away_matches = df[
        (df["away_team"] == team) &
        (pd.to_datetime(df["date"], dayfirst=True) < pd.to_datetime(date, dayfirst=True))
    ].sort_values("date", ascending=False)

    total_xg = 0
    games = 0

    for _, match in away_matches.iterrows():
        total_xg += match["away_xg"]
        games += 1

        if games == 5:
            break

    return total_xg / games if games > 0 else 0


def average_home_xga(team, df, date):
    home_matches = df[
        (df["home_team"] == team) &
        (pd.to_datetime(df["date"], dayfirst=True) < pd.to_datetime(date, dayfirst=True))
    ].sort_values("date", ascending=False)

    total_xga = 0
    games = 0

    for _, match in home_matches.iterrows():
        total_xga += match["home_xga"]
        games += 1

        if games == 5:
            break

    return total_xga / games if games > 0 else 0

def average_away_xga(team, df, date):
    away_matches = df[
        (df["away_team"] == team) &
        (pd.to_datetime(df["date"], dayfirst=True) < pd.to_datetime(date, dayfirst=True))
    ].sort_values("date", ascending=False)

    total_xga = 0
    games = 0

    for _, match in away_matches.iterrows():
        total_xga += match["away_xga"]
        games += 1

        if games == 5:
            break

    return total_xga / games if games > 0 else 0

def normalize_team_name(team):
    if not isinstance(team, str):
        return team

    team = team.strip()

    mapping = {
        "Arsenal FC": "Arsenal",
        "Arsenal": "Arsenal",
        "Chelsea FC": "Chelsea",
        "Chelsea": "Chelsea",
        "Liverpool FC": "Liverpool",
        "Liverpool": "Liverpool",
        "Manchester City FC": "Man City",
        "Manchester City": "Man City",
        "Man City": "Man City",
        "Manchester United FC": "Man United",
        "Manchester United": "Man United",
        "Man United": "Man United",
        "Tottenham Hotspur FC": "Tottenham",
        "Tottenham Hotspur": "Tottenham",
        "Tottenham": "Tottenham",
        "Brighton & Hove Albion FC": "Brighton",
        "Brighton & Hove Albion": "Brighton",
        "Brighton": "Brighton",
        "Newcastle United FC": "Newcastle",
        "Newcastle United": "Newcastle",
        "Newcastle": "Newcastle",
        "Crystal Palace FC": "Crystal Palace",
        "Crystal Palace": "Crystal Palace",
        "AFC Bournemouth": "Bournemouth",
        "Bournemouth FC": "Bournemouth",
        "Bournemouth": "Bournemouth",
        "Aston Villa FC": "Aston Villa",
        "Aston Villa": "Aston Villa",
        "Brentford FC": "Brentford",
        "Brentford": "Brentford",
        "Nottingham Forest FC": "Nott'm Forest",
        "Nottingham Forest": "Nott'm Forest",
        "Nott'm Forest": "Nott'm Forest",
        "Everton FC": "Everton",
        "Everton": "Everton",
        "Fulham FC": "Fulham",
        "Fulham": "Fulham",
        "Leeds United FC": "Leeds",
        "Leeds United": "Leeds",
        "Leeds": "Leeds",
        "Sunderland AFC": "Sunderland",
        "Sunderland": "Sunderland",
        "Coventry City": "Coventry",
        "Ipswich Town FC": "Ipswich",
        "Ipswich Town": "Ipswich",
        "Ipswich": "Ipswich",
        "Hull City": "Hull",
        "Burnley FC": "Burnley",
        "Burnley": "Burnley",
        "Sheffield United FC": "Sheffield United",
        "Sheffield United": "Sheffield United",
        "Luton Town FC": "Luton",
        "Luton Town": "Luton",
        "Luton": "Luton",
        "Norwich City FC": "Norwich",
        "Norwich City": "Norwich",
        "Norwich": "Norwich",
        "Watford FC": "Watford",
        "Watford": "Watford",
        "West Bromwich Albion FC": "West Brom",
        "West Bromwich Albion": "West Brom",
        "West Brom": "West Brom",
        "Cardiff City FC": "Cardiff",
        "Cardiff City": "Cardiff",
        "Swansea City FC": "Swansea",
        "Swansea City": "Swansea",
        "Queens Park Rangers FC": "QPR",
        "Queens Park Rangers": "QPR",
        "QPR": "QPR",
        "Wigan Athletic": "Wigan",
        "Wigan": "Wigan",
        "Blackburn Rovers FC": "Blackburn",
        "Blackburn Rovers": "Blackburn",
        "Blackburn": "Blackburn",
        "Bolton Wanderers": "Bolton",
        "Bolton": "Bolton",
        "Reading FC": "Reading",
        "Reading": "Reading",
        "Middlesbrough FC": "Middlesbrough",
        "Middlesbrough": "Middlesbrough",
        "Stoke City FC": "Stoke",
        "Stoke City": "Stoke",
        "Stoke": "Stoke",
        "West Ham United FC": "West Ham",
        "West Ham United": "West Ham",
        "West Ham": "West Ham",
        "Wolverhampton Wanderers FC": "Wolves",
        "Wolverhampton Wanderers": "Wolves",
        "Wolves": "Wolves",
        "Leicester City FC": "Leicester",
        "Leicester City": "Leicester",
        "Leicester": "Leicester",
        "Southampton FC": "Southampton",
        "Southampton": "Southampton",
        "Portsmouth FC": "Portsmouth",
        "Portsmouth": "Portsmouth",
        "Birmingham City FC": "Birmingham",
        "Birmingham City": "Birmingham",
        "Blackpool FC": "Blackpool",
        "Blackpool": "Blackpool",
        "Huddersfield Town FC": "Huddersfield",
        "Huddersfield Town": "Huddersfield",
    }

    if team in mapping:
        return mapping[team]

    clean_team = team.replace(" FC", "").replace(" AFC", "").strip()
    return mapping.get(clean_team, clean_team)
