"""Historical free-odds ingester (improvement #2).

Reads a historical MLB odds dataset, de-vigs the opening + closing moneylines,
joins the game outcome, and emits training rows for the market-microstructure
model. Data-agnostic: any free dataset (e.g. sportsbookreviewsonline season
files) maps onto the documented flat schema below.

Documented schema (one row per game; column names case-insensitive, common
aliases accepted):
    date, home_team, away_team,
    home_open, away_open, home_close, away_close,   # American moneylines
    home_score, away_score

Drop a CSV/XLSX with these columns in storage/hist_odds/ and run
`python -m mlb_value_bot.griffbet hist-train --file <path>`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlb_value_bot.analysis.ev_calculator import devigged_market_probs
from mlb_value_bot.constants import normalize_team
from mlb_value_bot.utils import get_logger

log = get_logger("griffbet.hist_odds")

_REQUIRED = ["date", "home_team", "away_team", "home_open", "away_open",
             "home_close", "away_close", "home_score", "away_score"]

# Accepted column aliases -> canonical name.
_ALIASES = {
    "home": "home_team", "away": "away_team", "visitor": "away_team", "visitor_team": "away_team",
    "home_ml_open": "home_open", "away_ml_open": "away_open",
    "home_ml_close": "home_close", "away_ml_close": "away_close",
    "home_open_ml": "home_open", "away_open_ml": "away_open",
    "home_close_ml": "home_close", "away_close_ml": "away_close",
    "home_runs": "home_score", "away_runs": "away_score",
    "home_final": "home_score", "away_final": "away_score", "game_date": "date",
}


def load_history(path: str | Path) -> pd.DataFrame:
    """Load a historical odds file (CSV or XLSX) and normalize to the schema."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in _ALIASES.items() if k in df.columns})
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Historical odds file is missing columns {missing}. "
            f"Required: {_REQUIRED}. Got: {list(df.columns)}"
        )
    df["home_team"] = df["home_team"].map(lambda x: normalize_team(str(x)))
    df["away_team"] = df["away_team"].map(lambda x: normalize_team(str(x)))
    return df


def build_market_rows(df: pd.DataFrame, devig_method: str = "power") -> pd.DataFrame:
    """De-vig open + close, join outcome; emit market-microstructure rows.

    Columns: date, devig_open_home, devig_close_home, line_move (close-open home
    prob), fav_dog (close home prob - 0.5), home_won. Rows with bad odds, missing
    scores, or ties are dropped.
    """
    out = []
    for _, r in df.iterrows():
        try:
            open_h, _ = devigged_market_probs(int(r["home_open"]), int(r["away_open"]), devig_method)
            close_h, _ = devigged_market_probs(int(r["home_close"]), int(r["away_close"]), devig_method)
        except (ValueError, TypeError):
            continue
        hs, as_ = r["home_score"], r["away_score"]
        if pd.isna(hs) or pd.isna(as_) or int(hs) == int(as_):
            continue
        out.append({
            "date": str(r["date"]),
            "devig_open_home": open_h,
            "devig_close_home": close_h,
            "line_move": close_h - open_h,
            "fav_dog": close_h - 0.5,
            "home_won": int(int(hs) > int(as_)),
        })
    res = pd.DataFrame(out)
    return res.sort_values("date").reset_index(drop=True) if not res.empty else res
