"""Grade open recommendations against final scores from the MLB Stats API."""
from __future__ import annotations

from dataclasses import dataclass

from mlb_value_bot.data.mlb_client import MLBClient
from mlb_value_bot.tracking import recommendations as recs
from mlb_value_bot.utils import get_logger

log = get_logger("tracking.results")


@dataclass
class GradedBet:
    rec_id: int
    matchup: str
    side: str
    result: str
    profit_loss: float


@dataclass
class GradingSummary:
    date: str
    graded: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0            # totals only: final total exactly on the line (stake back)
    voids: int = 0
    pending: int = 0           # still not final (in progress / no result yet)
    staked: float = 0.0        # total bankroll fraction staked on graded bets
    profit_loss: float = 0.0   # net bankroll fraction
    bets: list[GradedBet] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.bets is None:
            self.bets = []

    @property
    def roi(self) -> float:
        return self.profit_loss / self.staked if self.staked > 0 else 0.0


def grade_date(game_date: str, mlb_client: MLBClient | None = None) -> GradingSummary:
    """Fetch final scores for `game_date` and settle all pending bets on it."""
    mlb = mlb_client or MLBClient()
    open_bets = recs.get_open_for_date(game_date)
    summary = GradingSummary(date=game_date)
    if not open_bets:
        log.info("No open bets for %s", game_date)
        return summary

    results_by_id = {r.game_id: r for r in mlb.get_results(game_date)}

    for bet in open_bets:
        game = results_by_id.get(bet["game_id"])
        matchup = f'{bet["away_team"]} @ {bet["home_team"]}'

        if game is None:
            summary.pending += 1
            continue

        # Postponed/cancelled/suspended -> void the bet (stake returned).
        if not game.is_final:
            if game.status in {"Postponed", "Cancelled", "Canceled", "Suspended"}:
                recs.update_result(bet["id"], "void", 0.0)
                summary.voids += 1
                summary.bets.append(GradedBet(bet["id"], matchup, bet["recommended_side"], "void", 0.0))
            else:
                summary.pending += 1
            continue

        bet_team = bet["home_team"] if bet["recommended_side"] == "home" else bet["away_team"]
        winner = game.winner
        stake = float(bet["kelly_stake"])
        decimal_odds = float(bet["decimal_odds"])

        if winner is None:
            summary.pending += 1
            continue
        if winner == bet_team:
            pl = stake * (decimal_odds - 1.0)
            recs.update_result(bet["id"], "win", pl)
            summary.wins += 1
            result_str = "win"
        else:
            pl = -stake
            recs.update_result(bet["id"], "loss", pl)
            summary.losses += 1
            result_str = "loss"

        summary.graded += 1
        summary.staked += stake
        summary.profit_loss += pl
        summary.bets.append(GradedBet(bet["id"], matchup, bet["recommended_side"], result_str, pl))

    log.info(
        "Graded %s: %d settled (%dW-%dL), %d void, %d pending, P/L %.4f",
        game_date, summary.graded, summary.wins, summary.losses, summary.voids, summary.pending, summary.profit_loss,
    )
    return summary


def grade_totals_date(game_date: str, mlb_client: MLBClient | None = None) -> GradingSummary:
    """Settle pending TOTALS (over/under) bets for a date against the final total.

    Line-aware grading: final total runs vs the line we BET (`opening_line`,
    falling back to `market_total`). Over wins when total > line, under when
    total < line; an exact integer-line match is a PUSH (stake refunded). P/L is
    in bankroll-fraction units like the moneyline (PAPER while paper_only -- it
    lives in its own table, so it never mixes with the real-money ML bankroll).
    """
    from mlb_value_bot.tracking import totals_recommendations as totals

    mlb = mlb_client or MLBClient()
    open_bets = totals.get_open_for_date(game_date)
    summary = GradingSummary(date=game_date)
    if not open_bets:
        log.info("No open totals bets for %s", game_date)
        return summary

    results_by_id = {r.game_id: r for r in mlb.get_results(game_date)}

    for bet in open_bets:
        game = results_by_id.get(bet["game_id"])
        matchup = f'{bet["away_team"]} @ {bet["home_team"]}'
        side = bet["pick_side"]

        if game is None:
            summary.pending += 1
            continue
        if not game.is_final:
            if game.status in {"Postponed", "Cancelled", "Canceled", "Suspended"}:
                totals.update_result(bet["id"], "void", 0.0)
                summary.voids += 1
                summary.bets.append(GradedBet(bet["id"], matchup, side, "void", 0.0))
            else:
                summary.pending += 1
            continue
        if game.home_score is None or game.away_score is None:
            summary.pending += 1
            continue

        total_runs = game.home_score + game.away_score
        line = bet["opening_line"] if bet["opening_line"] is not None else bet["market_total"]
        if line is None:
            summary.pending += 1
            continue

        stake = float(bet["kelly_stake"])
        decimal_odds = float(bet["decimal_odds"])

        if total_runs == line:                       # exact -> push (stake back)
            totals.update_result(bet["id"], "push", 0.0)
            summary.pushes += 1                      # a push is NOT a void (and never a loss)
            summary.bets.append(GradedBet(bet["id"], matchup, side, "push", 0.0))
            continue

        went_over = total_runs > line
        won = (side == "over" and went_over) or (side == "under" and not went_over)
        if won:
            pl = stake * (decimal_odds - 1.0)
            totals.update_result(bet["id"], "win", pl)
            summary.wins += 1
            result_str = "win"
        else:
            pl = -stake
            totals.update_result(bet["id"], "loss", pl)
            summary.losses += 1
            result_str = "loss"

        summary.graded += 1
        summary.staked += stake
        summary.profit_loss += pl
        summary.bets.append(GradedBet(bet["id"], matchup, f"{side} {line}", result_str, pl))

    log.info(
        "Graded totals %s: %d settled (%dW-%dL-%dP), %d void, %d pending, P/L %.4f",
        game_date, summary.graded, summary.wins, summary.losses, summary.pushes,
        summary.voids, summary.pending, summary.profit_loss,
    )
    return summary


def grade_all_open_totals(before: str, mlb_client: MLBClient | None = None) -> list[GradingSummary]:
    """Grade every past date (< `before`) with pending totals bets (self-healing
    backfill, same pattern as the moneyline grade_all_open)."""
    from mlb_value_bot.tracking import totals_recommendations as totals

    mlb = mlb_client or MLBClient()
    dates = totals.get_open_dates(before=before)
    if not dates:
        log.info("No past dates with open totals bets.")
        return []
    if len(dates) > 1:
        log.info("Open totals bets on %d dates (%s..%s) - sweeping all.", len(dates), dates[0], dates[-1])
    return [grade_totals_date(d, mlb_client=mlb) for d in dates]


def grade_all_open(before: str, mlb_client: MLBClient | None = None) -> list[GradingSummary]:
    """Grade EVERY past date (< `before`) that still has pending bets.

    Self-healing backfill: grading only yesterday orphans any bet whose result
    wasn't captured on its one chance (pipeline run failed, game suspended past
    the grading run, rows created before grading shipped). Sweeping all open
    dates means a missed grade is retried on every subsequent run until the
    MLB API can settle it.
    """
    mlb = mlb_client or MLBClient()
    dates = recs.get_open_dates(before=before)
    if not dates:
        log.info("No past dates with open bets.")
        return []
    if len(dates) > 1:
        log.info("Open bets on %d dates (%s..%s) — sweeping all.", len(dates), dates[0], dates[-1])
    return [grade_date(d, mlb_client=mlb) for d in dates]
