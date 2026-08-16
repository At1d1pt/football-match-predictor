# Golazo — Premier League Match Prediction API

Golazo is a FastAPI backend for Premier League match predictions. It combines a trained XGBoost result classifier, two XGBoost Poisson goal regressors, historical match statistics, Elo ratings, expected-goals data, and team market values. Optional integrations add current league standings and live bookmaker odds.

## Features

- Match-result probabilities: home win (`H`), draw (`D`), away win (`A`)
- Expected home and away goals, plus a Poisson scoreline distribution
- BTTS and over/under probabilities, and model-derived decimal fair odds
- Optional market-odds comparison, expected value, and Kelly-based stake suggestion
- Historical team form and performance profiles
- Latest Elo ratings for all teams in the data

## Project structure

```text
app/                 FastAPI routes, prediction orchestration, standings cache
features/            Feature engineering, Elo, xG and market-value collectors
models/              Serialized models, feature columns, Poisson and odds helpers
data/raw/            Historical PL matches, compiled dataset, xG dataset
data/processed/      Elo history, market values, engineered training dataset
tests/api.py         Interactive local API request script
```

The API requires these committed runtime artifacts: `models/model_xgboost.pkl`, `models/model_home_goals.pkl`, `models/model_away_goals.pkl`, `models/feature_columns.pkl`, `data/raw/compiled.csv`, `data/raw/xG.csv`, `data/processed/elo_history.csv`, and `data/processed/market_values.csv`.

## Setup

Requirements: Python 3.10+ recommended. Run commands from the repository root; the application loads data and models using relative paths.

```bash
git clone https://github.com/At1d1pt/football-match-predictor.git
cd football-match-predictor

pip install -r requirements.txt
```

Change `.env.example` to `.env` and edit the file:

```env
FOOTBALL_DATA_API_KEY=your_football_data_api_key_here
ODDS_API_KEY=your_odds_api_key_here
```

- `FOOTBALL_DATA_API_KEY` ([football-data.org](https://www.football-data.org/)) loads the Premier League table once at startup.
- `ODDS_API_KEY` ([The Odds API](https://the-odds-api.com/)) enables live market odds and value-bet analysis.

Neither key is needed for local model predictions. Without the football-data.org key, standings-derived features use defaults. Without the odds key (or when no match is found), `market_odds` and `odds_analysis` are `null`.

Start the server:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger documentation.

## API

### `GET /`

```json
{"message":"running"}
```

### `POST /predict`

Request body:

```json
{
  "home": "Arsenal",
  "away": "Chelsea",
  "date": "2026-08-22"
}
```

Example:

```bash
curl -X POST -H "Content-Type: application/json" -d "{\"home\":\"Arsenal\",\"away\":\"Coventry City\",\"date\":\"2026-08-22\"}" http://127.0.0.1:8000/predict
```

Response shape (numeric values vary by fixture):

```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "classifier": {"prediction":"H","home_win_probability":0.0,"draw_probability":0.0,"away_win_probability":0.0},
  "goal_model": {"home_goals":0.0,"away_goals":0.0,"prediction":"H"},
  "poisson_model": {
    "expected_goals":{"home":0.0,"away":0.0},
    "probabilities":{"home_win":0.0,"draw":0.0,"away_win":0.0},
    "predicted_outcome":"H", "most_likely_score":"0-0", "top_scorelines":[],
    "btts_probability":0.0,
    "over_under":{"over_0_5":0.0,"over_1_5":0.0,"over_2_5":0.0,"over_3_5":0.0}
  },
  "fair_odds": {}, "market_odds": null, "odds_analysis": null,
  "home_stats": {}, "away_stats": {}
}
```

All three fields are required strings. Use names represented in the historical datasets; the code normalizes several team-name variants. Use dates in `YYYY-MM-DD`, before the fixture being evaluated, so historical form can be calculated.

### `GET /elo`

Returns a JSON object mapping every historical team to its latest Elo rating:

```json
{"Arsenal":1500.0,"Chelsea":1500.0}
```

The values above are illustrative.

## Model and data notes

Features include form, goals/concessions, shots, shots on target, corners, yellow cards, rest days, head-to-head form, home/away attack and defence statistics, Elo differences, xG/xGA measures, current position difference, and market-value comparisons. The goal-model outputs are passed to a Poisson model that considers scorelines from 0–6 goals per team.

The bundled assets are sufficient to run the API. To regenerate portions of the pipeline:

```bash
python data/processor.py       

python -m features.elo        
python -m features.market
python -m features.builder

python models/_xgboost.py
python models/goals_model.py
```

`features/xg.py` and `features/market.py` are data collectors. Their dependencies, `understatapi` and `beautifulsoup4`, are included in `requirements.txt`. `models/_xgboost.py` trains and saves the outcome classifier and its feature-column list; `models/tune_xg.py` performs a randomized hyperparameter search before evaluating its best classifier.

## Testing

```bash
python -m tests.api
```

You can run this test file to test the API locally.

## Classifier Confusion Matrix and Feature Importance

|             | Predicted Home | Predicted Draw | Predicted Away |
|-------------|----------------|----------------|----------------|
| Actual Home | 130            | 9              | 23             |
| Actual Draw | 55             | 10             | 39             |
| Actual Away | 35             | 5              | 74             |

```
                   Feature  Importance
            elo_difference    0.081395
   market_value_difference    0.039708
             position_diff    0.039186
        market_value_ratio    0.031739
           form_difference    0.028065
            away_away_form    0.026047
                  xgd_diff    0.025206
              away_xg_form    0.023080
         home_head_to_head    0.022958
            home_home_form    0.022833
          home_home_avg_xg    0.022529
            away_avg_goals    0.022422
          home_home_attack    0.022322
        corners_difference    0.021931
         home_home_defense    0.021815
            away_rest_days    0.021595
         draw_head_to_head    0.021110
            home_rest_days    0.020879
          away_away_attack    0.020825
      home_conversion_rate    0.020779
              away_avg_xgd    0.020646
         away_away_defense    0.020636
            home_avg_shots    0.020428
     home_avg_yellow_cards    0.020199
              home_avg_xgd    0.020196
         home_home_avg_xga    0.020174
          away_away_avg_xg    0.020151
            away_avg_shots    0.019901
            home_avg_goals    0.019826
          away_avg_corners    0.019817
      away_conversion_rate    0.019791
         home_avg_conceded    0.019355
shots_on_target_difference    0.019230
  home_avg_shots_on_target    0.019116
         away_away_avg_xga    0.019113
          home_avg_corners    0.019091
          shots_difference    0.018832
         away_head_to_head    0.018514
              home_xg_form    0.018489
  away_avg_shots_on_target    0.017745
         away_avg_conceded    0.017700
     away_avg_yellow_cards    0.017590
   yellow_cards_difference    0.017038
```

## Limitations

- **No automatic matchday updates.** The historical datasets and trained model artifacts are not automatically updated when new Premier League matches are played. After each matchday, new match data must be obtained and uploaded to `data/raw/`, then processed manually through the data and feature-generation workflow above. Retrain and replace model artifacts as needed for predictions to reflect the added data.
- Standings are fetched only at startup; invalid or unavailable credentials leave the cache empty.
- Live odds use the first bookmaker returned by The Odds API.
- Predictions are statistical estimates, not financial or betting advice.

## License

This project is released under the [MIT License](LICENSE).
