# NFL Win Probability Lab

A small Streamlit dashboard for private NFL analysis. The first version displays upcoming
games, Elo win probabilities, market-implied probabilities, projected margins, current Elo
ratings, and simple point-adjustment scenarios.

## Run locally

1. Create and activate a Python virtual environment.
2. Install the packages in `requirements.txt`.
3. Run `streamlit run app.py`.

The app downloads the latest games file from nflverse and caches the calculated predictions
for six hours.
