"""GriffBet residual market-error engine (the #2 rebuild).

Instead of BiffBet's hand-weighted blend (final = blend*model + (1-blend)*market),
GriffBet now models the MARKET'S ERROR directly:

    logit(p_home) = logit(market_devig_home) + Σ βᵢ · featureᵢ

The de-vigged market is a FIXED OFFSET (the sharp prior); the model learns
shrunken coefficients on the transparent features (the seven component deltas)
plus market-context features. Properties that make this the right first model
for the "accumulate live, free, slow" path:

  * Cold-start-safe: heavy L2 drives β -> 0 with little data, so p ≈ market
    (trust the market until we've actually learned something).
  * Still transparent: the coefficients say WHICH features predict market
    mispricing -- a learned model, not a black box.
  * No new dependency: a regularized logistic-with-offset fit by scipy.optimize.

Training rows are EXTRACTED from already-stored history (BiffBet + GriffBet
reasoning_json): each analyzed game carries its component deltas + market prob,
and graded games carry the outcome. The set grows daily.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mlb_value_bot.utils import get_logger

log = get_logger("griffbet.residual")

# Fixed feature order. Component weighted-deltas (home-favoring) + market context.
COMPONENT_FEATURES = ["starter", "bullpen", "bullpen_fatigue", "lineup", "park", "home_field", "form"]
CONTEXT_FEATURES = ["sharp_minus_square", "dispersion", "data_confidence"]
FEATURES = COMPONENT_FEATURES + CONTEXT_FEATURES

_EPS = 1e-12


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _reasoning(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def feature_vector(reasoning: dict) -> dict | None:
    """Extract the FEATURES dict + market_devig_home from one game's reasoning.
    Returns None if the market devig prob is missing (can't anchor)."""
    ma = reasoning.get("market_anchor") or {}
    market = ma.get("market_devig_home_prob")
    if market is None:
        return None
    comps = {c["name"]: c for c in (reasoning.get("components") or [])}
    feat = {name: float(comps.get(name, {}).get("weighted_delta", 0.0) or 0.0)
            for name in COMPONENT_FEATURES}
    mi = reasoning.get("market_intel") or {}
    sms = mi.get("sharp_minus_square_pp")
    disp = mi.get("dispersion_pp")
    feat["sharp_minus_square"] = (float(sms) / 100.0) if sms is not None else 0.0
    feat["dispersion"] = (float(disp) / 100.0) if disp is not None else 0.0
    feat["data_confidence"] = float(ma.get("data_confidence", 0.0) or 0.0) / 100.0
    feat["_market_devig_home"] = float(market)
    return feat


def _home_won(result: str, recommended_side: str) -> int | None:
    if result not in ("win", "loss"):
        return None
    home_pick = recommended_side == "home"
    won = result == "win"
    return int(won == home_pick)  # home won iff (home pick & win) or (away pick & loss)


def extract_training_data(extra_frames: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    """Mine (features…, market_devig_home, home_won) from stored history.

    Reads BiffBet's and GriffBet's recommendation stores read-only, dedupes by
    (date, game_id), keeps only GRADED games (win/loss). The component deltas are
    the same model features GriffBet computes live, so a row's provenance doesn't
    matter -- it's a (features, market, outcome) tuple.
    """
    from mlb_value_bot.tracking import recommendations as biff
    from mlb_value_bot.griffbet import tracking as griff

    frames = []
    for src in (biff.to_dataframe(), griff.to_dataframe(), *(extra_frames or [])):
        if src is not None and not src.empty:
            frames.append(src)
    if not frames:
        return pd.DataFrame(columns=[*FEATURES, "market_devig_home", "home_won", "date"])

    seen: set[tuple] = set()
    rows: list[dict] = []
    # Prefer the longest-history source first (BiffBet), then fill gaps.
    for df in frames:
        for _, r in df.iterrows():
            key = (r.get("date"), int(r.get("game_id")))
            if key in seen:
                continue
            won = _home_won(r.get("result"), r.get("recommended_side"))
            if won is None:
                continue
            feat = feature_vector(_reasoning(r.get("reasoning_json")))
            if feat is None:
                continue
            seen.add(key)
            row = {name: feat[name] for name in FEATURES}
            row["market_devig_home"] = feat["_market_devig_home"]
            row["home_won"] = won
            row["date"] = r.get("date")
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("date").reset_index(drop=True) if not out.empty else out


# --- the model ---------------------------------------------------------------
@dataclass
class ResidualModel:
    features: list[str]
    beta: list[float]
    mean: list[float]
    std: list[float]
    l2: float
    n_train: int
    trained_at: str | None = None
    notes: list[str] = field(default_factory=list)
    oos: dict | None = None   # stored time-split verdict (beats_market_log_loss, …)

    def to_dict(self) -> dict:
        return {
            "features": self.features, "beta": self.beta, "mean": self.mean,
            "std": self.std, "l2": self.l2, "n_train": self.n_train,
            "trained_at": self.trained_at, "notes": self.notes, "oos": self.oos,
            "market_offset": True,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResidualModel":
        return cls(features=d["features"], beta=d["beta"], mean=d["mean"], std=d["std"],
                   l2=d["l2"], n_train=d["n_train"], trained_at=d.get("trained_at"),
                   notes=d.get("notes", []), oos=d.get("oos"))

    def predict_home_prob(self, feature_dict: dict, market_devig_home: float) -> float:
        """p_home = sigmoid(logit(market) + standardized(x)·β). Falls back to the
        market exactly when β is all-zero (cold start)."""
        x = np.array([float(feature_dict.get(f, 0.0)) for f in self.features], dtype=float)
        std = np.array(self.std, dtype=float)
        std = np.where(std <= 0, 1.0, std)
        xs = (x - np.array(self.mean, dtype=float)) / std
        z = _logit(market_devig_home) + float(xs @ np.array(self.beta, dtype=float))
        return float(1.0 / (1.0 + np.exp(-z)))


def _neg_ll(beta, X, offset, y, l2):
    z = offset + X @ beta
    p = 1.0 / (1.0 + np.exp(-z))
    ll = -np.sum(y * np.log(p + _EPS) + (1 - y) * np.log(1 - p + _EPS)) + l2 * np.sum(beta ** 2)
    grad = X.T @ (p - y) + 2.0 * l2 * beta
    return ll, grad


def fit_residual_model(df: pd.DataFrame, l2: float, trained_at: str | None = None) -> ResidualModel:
    """Fit the regularized logistic-with-market-offset on extracted rows.

    Features are standardized so L2 shrinkage applies evenly. The market
    log-odds enters as a fixed offset, so β learns the RESIDUAL (market error).
    With few rows + strong l2, β stays near 0 (≈ trust the market).
    """
    if df.empty:
        return ResidualModel(FEATURES, [0.0] * len(FEATURES), [0.0] * len(FEATURES),
                             [1.0] * len(FEATURES), l2, 0, trained_at, ["no training data"])
    X_raw = df[FEATURES].to_numpy(dtype=float)
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    std_safe = np.where(std <= 0, 1.0, std)
    X = (X_raw - mean) / std_safe
    offset = np.array([_logit(m) for m in df["market_devig_home"].to_numpy(dtype=float)])
    y = df["home_won"].to_numpy(dtype=float)

    res = minimize(_neg_ll, np.zeros(X.shape[1]), args=(X, offset, y, l2),
                   jac=True, method="L-BFGS-B")
    beta = res.x.tolist()
    return ResidualModel(FEATURES, beta, mean.tolist(), std.tolist(), float(l2),
                         int(len(df)), trained_at,
                         [f"converged={res.success}", f"l2={l2}"])


# --- persistence -------------------------------------------------------------
def model_path():
    from mlb_value_bot.utils import STORAGE_DIR
    return STORAGE_DIR / "griff_residual_model.json"


def save_model(model: ResidualModel) -> None:
    from mlb_value_bot.utils import ensure_dirs
    ensure_dirs()
    with model_path().open("w", encoding="utf-8") as fh:
        json.dump(model.to_dict(), fh, indent=2)


def load_model() -> ResidualModel | None:
    p = model_path()
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as fh:
            return ResidualModel.from_dict(json.load(fh))
    except (json.JSONDecodeError, KeyError) as exc:
        log.warning("residual model load failed (%s)", exc)
        return None


# --- out-of-sample validation + ablation -------------------------------------
def _eval_on(model: ResidualModel, df: pd.DataFrame) -> dict:
    """Brier + log loss of the model AND the market baseline on a (held-out) set,
    using the home-side prob vs the home_won outcome."""
    from mlb_value_bot.griffbet.referee import brier_score, log_loss
    if df.empty:
        return {"n": 0}
    probs, mkt, y = [], [], []
    for _, r in df.iterrows():
        feat = {f: r[f] for f in FEATURES}
        m = float(r["market_devig_home"])
        probs.append(model.predict_home_prob(feat, m))
        mkt.append(m)
        y.append(int(r["home_won"]))
    return {
        "n": len(y),
        "model_brier": brier_score(probs, y), "model_log_loss": log_loss(probs, y),
        "market_brier": brier_score(mkt, y), "market_log_loss": log_loss(mkt, y),
    }


def time_split_eval(df: pd.DataFrame, l2: float, train_frac: float = 0.7) -> dict:
    """Train on the earliest `train_frac` of games (by date), test on the rest.
    Reports whether the model beats the market baseline OUT OF SAMPLE. The whole
    point: a model that only wins in-sample is noise."""
    if len(df) < 10:
        return {"sufficient": False, "n": int(len(df)),
                "note": "need >= 10 graded games for a time split"}
    cut = max(1, int(len(df) * train_frac))
    train, test = df.iloc[:cut], df.iloc[cut:]
    if test.empty:
        return {"sufficient": False, "n": int(len(df)), "note": "empty test split"}
    model = fit_residual_model(train, l2)
    ev = _eval_on(model, test)
    ev["sufficient"] = True
    ev["n_train"] = int(len(train))
    ev["beats_market_log_loss"] = (
        ev.get("model_log_loss") is not None and ev.get("market_log_loss") is not None
        and ev["model_log_loss"] < ev["market_log_loss"]
    )
    return ev


def ablation(df: pd.DataFrame, l2: float, train_frac: float = 0.7) -> list[dict]:
    """Per-feature incremental value: zero each feature, retrain, measure the
    OOS log-loss change vs the full model. Positive delta = the feature HELPS
    out of sample (removing it hurt). Negative = it's likely noise."""
    base = time_split_eval(df, l2, train_frac)
    if not base.get("sufficient") or base.get("model_log_loss") is None:
        return []
    cut = max(1, int(len(df) * train_frac))
    train, test = df.iloc[:cut], df.iloc[cut:]
    out = []
    for f in FEATURES:
        ablated = df.copy()
        ablated[f] = 0.0
        m = fit_residual_model(ablated.iloc[:cut], l2)
        ev = _eval_on(m, ablated.iloc[cut:])
        if ev.get("model_log_loss") is None:
            continue
        out.append({
            "feature": f,
            "oos_log_loss_without": ev["model_log_loss"],
            "delta_vs_full": round(ev["model_log_loss"] - base["model_log_loss"], 4),
        })
    out.sort(key=lambda r: r["delta_vs_full"], reverse=True)
    return out


# --- train + persist convenience ---------------------------------------------
def train_and_save(config: dict, trained_at: str | None = None) -> dict:
    """Extract stored history, fit on all of it, compute the time-split OOS
    verdict + ablation, persist the model. Returns a report dict for the CLI."""
    res_cfg = config.get("model", {}).get("residual", {})
    l2 = float(res_cfg.get("l2", 50.0))
    df = extract_training_data()
    model = fit_residual_model(df, l2, trained_at=trained_at)
    oos = time_split_eval(df, l2) if not df.empty else {"sufficient": False, "n": 0}
    model.oos = oos
    save_model(model)
    return {
        "n_train": model.n_train,
        "l2": l2,
        "oos": oos,
        "ablation": ablation(df, l2) if df is not None and len(df) >= 10 else [],
        "coefficients": dict(zip(model.features, [round(b, 4) for b in model.beta])),
    }


def is_ready(model: "ResidualModel | None", config: dict) -> tuple[bool, str]:
    """Warmup gate: is the residual model trusted enough to drive live picks?

    Requires n_train >= min_train_games, AND (when require_beats_market) a stored
    time-split verdict that the model beat the market baseline OUT OF SAMPLE.
    Until then GriffBet stays on its hand-weighted blend (no data-starved bets).
    """
    res_cfg = config.get("model", {}).get("residual", {})
    min_train = int(res_cfg.get("min_train_games", 300))
    require_beats = bool(res_cfg.get("require_beats_market", True))
    if model is None:
        return False, "no trained model"
    if model.n_train < min_train:
        return False, f"warmup: n_train {model.n_train} < {min_train}"
    if require_beats:
        oos = model.oos or {}
        if not oos.get("sufficient"):
            return False, "warmup: no out-of-sample verdict yet"
        if not oos.get("beats_market_log_loss"):
            return False, "held back: does not beat market log loss out-of-sample"
    return True, f"residual active (n_train={model.n_train})"
