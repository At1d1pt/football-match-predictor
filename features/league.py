from collections import defaultdict

def initialize_table():
    return defaultdict(lambda: {
        "points": 0,
        "goal_difference": 0,
        "goals_scored": 0,
        "matches_played": 0,
        "won": 0,
        "lost": 0,
        "draw": 0
    })

def get_league_positions(table):
    standings = sorted(
        table.items(),
        key=lambda x: (
            x[1]["points"],
            x[1]["goal_difference"],
            x[1]["goals_scored"]
        ),
        reverse=True
    )

    return {
        team: position + 1
        for position, (team, _) in enumerate(standings)
    }

def get_total_goals(table):
    goals = 0
    for team in table:
        goals += table[team]["goals_scored"]

    return goals

def get_team_goals(team, table):
    return table[team]["goals_scored"]

def get_team_conceded(team, table):
    return abs(table[team]["goals_scored"] - table[team]["goal_difference"])

def update_table(table, match):
    home = match["home_team"]
    away = match["away_team"]

    home_goals = match["home_goals"]
    away_goals = match["away_goals"]

    table[home]["matches_played"] += 1
    table[away]["matches_played"] += 1

    table[home]["goals_scored"] += home_goals
    table[away]["goals_scored"] += away_goals

    table[home]["goal_difference"] += home_goals - away_goals
    table[away]["goal_difference"] += away_goals - home_goals

    if home_goals > away_goals:
        table[home]["points"] += 3

    elif away_goals > home_goals:
        table[away]["points"] += 3

    else:
        table[home]["points"] += 1
        table[away]["points"] += 1