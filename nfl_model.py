import math

import numpy as np
import pandas as pd


GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
SIGMA = 13.5
ELO_PER_POINT = 25.0
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def moneyline_probability(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    probabilities = np.where(
        values < 0,
        -values / (-values + 100.0),
        100.0 / (values + 100.0),
    )
    return pd.Series(probabilities, index=values.index)


def load_games(first_season: int = 2003) -> pd.DataFrame:
    games = pd.read_csv(GAMES_URL)
    games = games[games["season"] >= first_season].copy()
    games = games.rename(
        columns={
            "home_team": "home",
            "away_team": "away",
            "home_moneyline": "home_ml",
            "away_moneyline": "away_ml",
        }
    )
    games["home"] = games["home"].replace(TEAM_FIX)
    games["away"] = games["away"].replace(TEAM_FIX)
    games["neutral"] = games["location"].eq("Neutral")
    games["played"] = games["result"].notna()
    games = games.sort_values(["season", "gameday", "week"]).reset_index(drop=True)

    home_market = moneyline_probability(games["home_ml"])
    away_market = moneyline_probability(games["away_ml"])
    market_total = home_market + away_market
    games["market_home_win"] = home_market / market_total
    return games


def add_elo_predictions(
    games: pd.DataFrame,
    k_factor: float = 20.0,
    home_field_elo: float = 48.0,
    offseason_reversion: float = 1 / 3,
) -> tuple[pd.DataFrame, dict[str, float]]:
    ratings: dict[str, float] = {}
    previous_season = None
    predictions = []

    for game in games.itertuples(index=False):
        if previous_season is not None and game.season != previous_season:
            ratings = {
                team: 1500.0 + (rating - 1500.0) * (1.0 - offseason_reversion)
                for team, rating in ratings.items()
            }
        previous_season = game.season

        home_rating = ratings.setdefault(game.home, 1500.0)
        away_rating = ratings.setdefault(game.away, 1500.0)
        home_advantage = 0.0 if game.neutral else home_field_elo
        elo_difference = home_rating + home_advantage - away_rating
        predicted_margin = elo_difference / ELO_PER_POINT
        home_win_probability = normal_cdf(predicted_margin / SIGMA)
        predictions.append((home_rating, away_rating, predicted_margin, home_win_probability))

        if game.played:
            actual_margin = float(game.result)
            winner_difference = elo_difference if actual_margin > 0 else -elo_difference
            multiplier = (
                math.log(abs(actual_margin) + 1.0)
                * 2.2
                / (winner_difference * 0.001 + 2.2)
            )
            outcome = 1.0 if actual_margin > 0 else (0.0 if actual_margin < 0 else 0.5)
            shift = k_factor * multiplier * (outcome - home_win_probability)
            ratings[game.home] = home_rating + shift
            ratings[game.away] = away_rating - shift

    result = games.copy()
    result[["home_elo", "away_elo", "model_margin", "elo_home_win"]] = predictions
    return result, ratings


def adjusted_prediction(model_margin: float, home_adjustment: float, away_adjustment: float):
    margin = model_margin + home_adjustment - away_adjustment
    return margin, normal_cdf(margin / SIGMA)
