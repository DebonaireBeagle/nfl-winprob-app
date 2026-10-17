import pandas as pd
import streamlit as st

from nfl_model import add_elo_predictions, adjusted_prediction, load_games


st.set_page_config(page_title="NFL Win Probability Lab", page_icon="🏈", layout="wide")


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def build_predictions():
    games = load_games()
    return add_elo_predictions(games)


def percent(value):
    return "—" if pd.isna(value) else f"{value:.1%}"


st.title("🏈 NFL Win Probability Lab")
st.caption("A private analysis dashboard using an Elo model and nflverse game data.")

try:
    with st.spinner("Loading the latest NFL data…"):
        games, ratings = build_predictions()
except Exception as exc:
    st.error("The NFL data could not be loaded. Please try again in a few minutes.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

upcoming = games[~games["played"]].copy()
if upcoming.empty:
    st.info("No upcoming games are currently listed in the data source.")
    st.stop()

current_season = int(upcoming["season"].max())
season_games = upcoming[upcoming["season"] == current_season]
available_weeks = sorted(int(week) for week in season_games["week"].dropna().unique())

with st.sidebar:
    st.header("Controls")
    selected_week = st.selectbox("Week", available_weeks, index=0)
    st.caption("Data refreshes automatically every six hours.")
    if st.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

week_games = season_games[season_games["week"] == selected_week].copy()
week_games["matchup"] = week_games["away"] + " @ " + week_games["home"]

st.subheader(f"{current_season} · Week {selected_week}")
table = week_games[
    ["gameday", "matchup", "elo_home_win", "market_home_win", "model_margin"]
].rename(
    columns={
        "gameday": "Date",
        "matchup": "Matchup",
        "elo_home_win": "Elo home win",
        "market_home_win": "Market home win",
        "model_margin": "Home margin",
    }
)
st.dataframe(
    table.style.format(
        {
            "Elo home win": "{:.1%}",
            "Market home win": lambda value: percent(value),
            "Home margin": "{:+.1f}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

st.divider()
st.subheader("What-if analysis")
selected_matchup = st.selectbox("Choose a matchup", week_games["matchup"].tolist())
game = week_games.loc[week_games["matchup"] == selected_matchup].iloc[0]

left, right = st.columns(2)
with left:
    away_adjustment = st.number_input(
        f"{game['away']} adjustment (points)", min_value=-14.0, max_value=14.0, step=0.5
    )
with right:
    home_adjustment = st.number_input(
        f"{game['home']} adjustment (points)", min_value=-14.0, max_value=14.0, step=0.5
    )

adjusted_margin, adjusted_home_win = adjusted_prediction(
    float(game["model_margin"]), home_adjustment, away_adjustment
)
metric_one, metric_two, metric_three = st.columns(3)
metric_one.metric("Home win probability", f"{adjusted_home_win:.1%}")
metric_two.metric("Projected home margin", f"{adjusted_margin:+.1f}")
metric_three.metric("Market home probability", percent(game["market_home_win"]))
st.caption("Positive adjustments strengthen a team; negative adjustments weaken it.")

with st.expander("Current Elo ratings"):
    rating_table = pd.DataFrame(
        sorted(ratings.items(), key=lambda item: item[1], reverse=True),
        columns=["Team", "Elo"],
    )
    st.dataframe(rating_table.style.format({"Elo": "{:.0f}"}), hide_index=True)

st.caption("For analysis and entertainment only. Predictions are estimates, not guarantees.")
