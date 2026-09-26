"""RSI (Recursive Self-Improvement) -- the engine's learning loop.

Analytics half (this package): mines rsi_scored_games (every scored game
across engines, picks AND passes, minus the 20% holdout) for segment-level
trends, tracks them as rsi_proposals through the watch -> pending lifecycle,
scores approved challengers against the champion (shadow_stats), and checks
active model versions for rollback conditions. It REPLACES the
.claude/skills/self-improve retro sidecar; bucket definitions now live in
rsi/segments.py and tracking/performance.py delegates to them.

Nothing here changes the live model: the engine only ever reads
rsi_model_versions(status='active') and rsi_proposals(status='approved'),
and every transition beyond watch/pending is Jim's button press on the site.

Config lives in rsi/config_rsi.yaml (load_rsi_config), independent of
config.yaml / config_football.yaml -- the same pattern as football/__init__.
"""
from __future__ import annotations

from pathlib import Path

RSI_CONFIG_PATH = Path(__file__).resolve().parent / "config_rsi.yaml"


def load_rsi_config() -> dict:
    """Load the RSI loop's own config. Never reads config.yaml."""
    from mlb_value_bot.utils import load_config

    return load_config(str(RSI_CONFIG_PATH))
