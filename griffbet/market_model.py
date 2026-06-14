"""Market-microstructure model (improvement #2).

Trained on the historical odds dataset, it asks the review's central question on
the one signal a free odds dataset gives cheaply: does LINE MOVEMENT (open ->
close) predict the outcome BEYOND the closing line itself?

    logit(p_home) = logit(devig_close_home) + Σ βᵢ · featureᵢ
    features = [line_move (close-open home prob), fav_dog (close home prob - .5)]

The de-vigged CLOSE is the offset (the sharp benchmark); β learns the residual.
If, out-of-sample, the model beats the close baseline, line movement carries
exploitable info (reverse-line-movement / steam). If β shrinks to 0 (or it
loses OOS), the close is efficient on this signal -- the honest default.

Reuses the residual engine's regularized logistic-with-offset fitter, so it's
the same cold-start-safe, transparent machinery, just on market features.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from mlb_value_bot.griffbet.residual_model import _logit, _neg_ll
from mlb_value_bot.utils import get_logger

log = get_logger("griffbet.market_model")

FEATURES = ["line_move", "fav_dog"]


@dataclass
class MarketModel:
    features: list[str]
    beta: list[float]
    mean: list[float]
    std: list[float]
    l2: float
    n_train: int
    oos: dict | None = None
    notes: list[str] = field(default_factory=list)

    def predict_home_prob(self, row: dict) -> float:
        x = np.array([float(row.get(f, 0.0)) for f in self.features])
        std = np.where(np.array(self.std) <= 0, 1.0, np.array(self.std))
        xs = (x - np.array(self.mean)) / std
        z = _logit(float(row["devig_close_home"])) + float(xs @ np.array(self.beta))
        return float(1.0 / (1.0 + np.exp(-z)))

    def to_dict(self) -> dict:
        return {"features": self.features, "beta": self.beta, "mean": self.mean,
                "std": self.std, "l2": self.l2, "n_train": self.n_train,
                "oos": self.oos, "notes": self.notes}


def fit_market_model(df, l2: float) -> MarketModel:
    """Fit the close-anchored residual on [line_move, fav_dog]."""
    if df.empty:
        return MarketModel(FEATURES, [0.0] * len(FEATURES), [0.0] * len(FEATURES),
                           [1.0] * len(FEATURES), l2, 0, notes=["no data"])
    X_raw = df[FEATURES].to_numpy(dtype=float)
    mean, std = X_raw.mean(axis=0), X_raw.std(axis=0)
    std_safe = np.where(std <= 0, 1.0, std)
    X = (X_raw - mean) / std_safe
    offset = np.array([_logit(m) for m in df["devig_close_home"].to_numpy(dtype=float)])
    y = df["home_won"].to_numpy(dtype=float)
    res = minimize(_neg_ll, np.zeros(X.shape[1]), args=(X, offset, y, l2),
                   jac=True, method="L-BFGS-B")
    return MarketModel(FEATURES, res.x.tolist(), mean.tolist(), std.tolist(),
                       float(l2), int(len(df)), notes=[f"converged={res.success}"])


def _eval(model: MarketModel, df) -> dict:
    """Model vs the close baseline (the de-vigged close itself), on held-out."""
    from mlb_value_bot.griffbet.referee import brier_score, log_loss
    if df.empty:
        return {"n": 0}
    probs, close, y = [], [], []
    for _, r in df.iterrows():
        probs.append(model.predict_home_prob({f: r[f] for f in FEATURES} | {"devig_close_home": r["devig_close_home"]}))
        close.append(float(r["devig_close_home"]))
        y.append(int(r["home_won"]))
    return {"n": len(y),
            "model_brier": brier_score(probs, y), "model_log_loss": log_loss(probs, y),
            "close_brier": brier_score(close, y), "close_log_loss": log_loss(close, y)}


def time_split_eval(df, l2: float, train_frac: float = 0.7) -> dict:
    """Train on the earliest games, test on the rest. Does line movement beat
    the closing line out-of-sample?"""
    if len(df) < 50:
        return {"sufficient": False, "n": int(len(df)),
                "note": "need >= 50 games for a market-microstructure split"}
    cut = max(1, int(len(df) * train_frac))
    model = fit_market_model(df.iloc[:cut], l2)
    ev = _eval(model, df.iloc[cut:])
    ev["sufficient"] = True
    ev["n_train"] = cut
    ev["beats_close_log_loss"] = (
        ev.get("model_log_loss") is not None and ev.get("close_log_loss") is not None
        and ev["model_log_loss"] < ev["close_log_loss"]
    )
    ev["coefficients"] = dict(zip(model.features, [round(b, 4) for b in model.beta]))
    return ev


def train_from_history(df_market, l2: float) -> tuple[MarketModel, dict]:
    """Fit on all rows + compute the OOS verdict. Returns (model, oos)."""
    model = fit_market_model(df_market, l2)
    oos = time_split_eval(df_market, l2)
    model.oos = oos
    return model, oos


def model_path():
    from mlb_value_bot.utils import STORAGE_DIR
    return STORAGE_DIR / "griff_market_model.json"


def save_model(model: MarketModel) -> None:
    from mlb_value_bot.utils import ensure_dirs
    ensure_dirs()
    with model_path().open("w", encoding="utf-8") as fh:
        json.dump(model.to_dict(), fh, indent=2)
