"""Grade GriffBet's open bets against final scores (parallels tracking/results)."""
from __future__ import annotations

from dataclasses import dataclass, field

from mlb_value_bot.data.mlb_client import MLBClient
from mlb_value_bot.griffbet import tracking as gtrack
from mlb_value_bot.utils import get_logger

log = get_logger("griffbet.results")


@dataclass
class GriffGradingSummary:
    date: str
    graded: int = 0
    wins: int = 0
    losses: int = 0
    voids: int = 0
    pending: int = 0
    staked: float = 0.0
    profit_loss: float = 0.0
    bets: list = field(default_factory=list)


def grade_date(game_date: str, mlb_client: MLBClient | None = None) -> GriffGradingSummary:
    mlb = mlb_client or MLBClient()
    open_bets = gtrack.get_open_for_date(game_date)
    summary = GriffGradingSummary(date=game_date)
    if not open_bets:
        return summary

    results_by_id = {r.game_id: r for r in mlb.get_results(game_date)}
    for bet in open_bets:
        game = results_by_id.get(bet["game_id"])
        if game is None:
            summary.pending += 1
            continue
        if not game.is_final:
            if game.status in {"Postponed", "Cancelled", "Canceled", "Suspended"}:
                gtrack.update_result(bet["id"], "void", 0.0)
                summary.voids += 1
            else:
                summary.pending += 1
            continue
        bet_team = bet["home_team"] if bet["recommended_side"] == "home" else bet["away_team"]
        winner = game.winner
        if winner is None:
            summary.pending += 1
            continue
        stake = float(bet["kelly_stake"])
        decimal_odds = float(bet["decimal_odds"])
        if winner == bet_team:
            pl = stake * (decimal_odds - 1.0)
            gtrack.update_result(bet["id"], "win", pl)
            summary.wins += 1
        else:
            pl = -stake
            gtrack.update_result(bet["id"], "loss", pl)
            summary.losses += 1
        summary.graded += 1
        summary.staked += stake
        summary.profit_loss += pl

    log.info("GriffBet graded %s: %d settled (%dW-%dL), %d void, %d pending, P/L %.4f",
             game_date, summary.graded, summary.wins, summary.losses,
             summary.voids, summary.pending, summary.profit_loss)
    return summary


def grade_all_open(before: str, mlb_client: MLBClient | None = None) -> list[GriffGradingSummary]:
    """Self-healing sweep: grade every past date with open GriffBet bets."""
    mlb = mlb_client or MLBClient()
    dates = gtrack.get_open_dates(before=before)
    return [grade_date(d, mlb_client=mlb) for d in dates]
