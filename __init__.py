"""mlb_value_bot — a transparent, terminal-based +EV bet finder for MLB.

The package is intentionally organized so each layer has one job:

  data/      external I/O (The Odds API, MLB Stats API, pybaseball) + caching
  analysis/  pure-ish computation (metrics -> win probability -> EV)
  tracking/  durable record of recommendations, results, and performance
  backtest/  re-run the model over historical games

Design goal: measure whether the model has a real edge, not to "pick winners".
"""

__version__ = "0.1.0"
