"""MLB Stats API + pybaseball access.

The MLB Stats API (statsapi.mlb.com) is free, key-less, and authoritative for
schedule, probable pitchers, lineups, and final scores. pybaseball is wrapped
in analysis/* (pitcher_metrics, team_metrics) — this module owns the schedule
and results side of things plus a small player-id helper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_cls

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mlb_value_bot.constants import normalize_team
from mlb_value_bot.utils import get_logger, load_config

log = get_logger("data.mlb_client")

# Statuses that mean "don't bet / don't grade as a real game".
NON_PLAYED_STATES = {"Postponed", "Cancelled", "Canceled", "Suspended"}


@dataclass
class ProbablePitcher:
    player_id: int | None
    name: str | None


@dataclass
class ScheduledGame:
    game_id: int
    game_date: str               # YYYY-MM-DD (local game date as returned)
    status: str
    home_team: str               # canonical
    away_team: str               # canonical
    home_pitcher: ProbablePitcher
    away_pitcher: ProbablePitcher
    venue: str | None = None
    game_datetime: str | None = None   # ISO UTC first-pitch time (MLB API `gameDate`)
    # MLB team IDs (needed for follow-up calls like recent-games / boxscores).
    # 0 = couldn't resolve (degrades to "no bullpen fatigue data" on this game).
    home_team_id: int = 0
    away_team_id: int = 0

    @property
    def is_playable(self) -> bool:
        return self.status not in NON_PLAYED_STATES

    @property
    def has_both_pitchers(self) -> bool:
        return bool(self.home_pitcher.player_id) and bool(self.away_pitcher.player_id)


@dataclass
class GameResult:
    game_id: int
    status: str
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None

    @property
    def is_final(self) -> bool:
        return self.status in {"Final", "Game Over", "Completed Early"}

    @property
    def winner(self) -> str | None:
        """Canonical name of the winning team, or None if not decided."""
        if not self.is_final or self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return self.home_team
        if self.away_score > self.home_score:
            return self.away_team
        return None  # tie (shouldn't happen in MLB)


class MLBClient:
    """Resilient client for the MLB Stats API."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        mlb_cfg = self.config.get("mlb_api", {})
        self.base_url = mlb_cfg.get("base_url", "https://statsapi.mlb.com/api")
        self.timeout = float(mlb_cfg.get("request_timeout_seconds", 20))
        self.session = requests.Session()

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params or {}, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"MLB API error {resp.status_code} for {url}: {resp.text[:200]}")
        return resp.json()

    # -- Schedule / probable pitchers ----------------------------------------
    def get_schedule(self, game_date: str | date_cls) -> list[ScheduledGame]:
        """Return scheduled games for a date with probable pitchers hydrated.

        `game_date` may be a date or 'YYYY-MM-DD' string.
        """
        date_str = game_date.isoformat() if isinstance(game_date, date_cls) else str(game_date)
        data = self._get(
            "/v1/schedule",
            {
                "sportId": 1,
                "date": date_str,
                "hydrate": "probablePitcher,linescore,team,venue",
            },
        )
        games: list[ScheduledGame] = []
        for day in data.get("dates", []):
            for g in day.get("games", []):
                games.append(self._parse_scheduled(g, date_str))
        log.info("Schedule %s: %d games", date_str, len(games))
        return games

    def _parse_scheduled(self, g: dict, date_str: str) -> ScheduledGame:
        teams = g.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        return ScheduledGame(
            game_id=int(g.get("gamePk")),
            game_date=g.get("officialDate", date_str),
            status=g.get("status", {}).get("detailedState", "Unknown"),
            home_team=normalize_team(home.get("team", {}).get("name")) or "",
            away_team=normalize_team(away.get("team", {}).get("name")) or "",
            home_pitcher=self._parse_pitcher(home.get("probablePitcher")),
            away_pitcher=self._parse_pitcher(away.get("probablePitcher")),
            venue=g.get("venue", {}).get("name"),
            game_datetime=g.get("gameDate"),
            home_team_id=int(home.get("team", {}).get("id") or 0),
            away_team_id=int(away.get("team", {}).get("id") or 0),
        )

    @staticmethod
    def _parse_pitcher(pp: dict | None) -> ProbablePitcher:
        if not pp:
            return ProbablePitcher(player_id=None, name=None)
        return ProbablePitcher(player_id=pp.get("id"), name=pp.get("fullName"))

    # -- Results --------------------------------------------------------------
    def get_results(self, game_date: str | date_cls) -> list[GameResult]:
        """Return final (or in-progress) scores for a date."""
        date_str = game_date.isoformat() if isinstance(game_date, date_cls) else str(game_date)
        data = self._get(
            "/v1/schedule",
            {"sportId": 1, "date": date_str, "hydrate": "linescore,team"},
        )
        results: list[GameResult] = []
        for day in data.get("dates", []):
            for g in day.get("games", []):
                results.append(self._parse_result(g))
        return results

    def _parse_result(self, g: dict) -> GameResult:
        teams = g.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        return GameResult(
            game_id=int(g.get("gamePk")),
            status=g.get("status", {}).get("detailedState", "Unknown"),
            home_team=normalize_team(home.get("team", {}).get("name")) or "",
            away_team=normalize_team(away.get("team", {}).get("name")) or "",
            home_score=home.get("score"),
            away_score=away.get("score"),
        )

    def get_result_for_game(self, game_id: int, game_date: str | date_cls) -> GameResult | None:
        """Convenience: find a single game's result by id on a known date."""
        for r in self.get_results(game_date):
            if r.game_id == int(game_id):
                return r
        return None

    # -- Standings ------------------------------------------------------------
    def get_standings(self, season: int) -> dict[str, dict[str, float]]:
        """Return {canonical_team: {wins, losses, games, winpct}} for a season.

        Used as the model's base-rate input (regressed toward .500 early). Reads
        both leagues' regular-season standings.
        """
        data = self._get(
            "/v1/standings",
            {"leagueId": "103,104", "season": int(season), "standingsTypes": "regularSeason"},
        )
        out: dict[str, dict[str, float]] = {}
        for record in data.get("records", []):
            for tr in record.get("teamRecords", []):
                name = normalize_team(tr.get("team", {}).get("name"))
                if not name:
                    continue
                wins = int(tr.get("wins", 0))
                losses = int(tr.get("losses", 0))
                games = wins + losses
                winpct = wins / games if games > 0 else 0.5
                out[name] = {
                    "wins": float(wins),
                    "losses": float(losses),
                    "games": float(games),
                    "winpct": float(winpct),
                }
        log.info("Standings %s: %d teams", season, len(out))
        return out

    # -- Team offense / bullpen (automatable proxies, no FanGraphs) ------------
    @staticmethod
    def _parse_ip(ip: object) -> float:
        """MLB innings strings ('72.1' = 72 1/3) -> decimal innings."""
        try:
            whole, _, frac = str(ip).partition(".")
            return int(whole) + (int(frac) / 3.0 if frac else 0.0)
        except (ValueError, AttributeError):
            return 0.0

    def get_team_hitting(self, season: int) -> dict[str, dict[str, float]]:
        """{canonical_team: {obp, slg, ops}} from MLB Stats API team hitting.

        Used as an automatable stand-in for FanGraphs wRC+ (which is Cloudflare-
        blocked). Returns {} on failure so callers degrade gracefully.
        """
        try:
            data = self._get(
                "/v1/teams/stats",
                {"sportId": 1, "season": int(season), "group": "hitting", "stats": "season"},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("team hitting fetch failed (%s)", exc)
            return {}
        out: dict[str, dict[str, float]] = {}
        for grp in data.get("stats", []):
            for sp in grp.get("splits", []):
                team = normalize_team(sp.get("team", {}).get("name"))
                st = sp.get("stat", {})
                if not team:
                    continue
                try:
                    out[team] = {
                        "obp": float(st["obp"]),
                        "slg": float(st["slg"]),
                        "ops": float(st["ops"]),
                    }
                except (KeyError, TypeError, ValueError):
                    continue
        return out

    # -- Per-pitcher reliever stats (for leverage-arm ranking) -----------------
    def get_per_pitcher_reliever_stats(
        self, season: int
    ) -> dict[str, list[dict]]:
        """Per-team list of relievers with their season stats.

        Same source as `get_team_bullpen_era` (one all-pitchers call) but we
        keep the per-pitcher rows instead of aggregating to a team ERA. Used to
        identify each team's high-leverage arms (lowest-ERA relievers with
        enough sample). Returns {} on failure.

        Output shape: {canonical_team: [{player_id, name, era, ip, appearances}, ...]}
        sorted by ERA ascending (best arms first).
        """
        try:
            data = self._get(
                "/v1/stats",
                {"stats": "season", "group": "pitching", "season": int(season),
                 "sportId": 1, "gameType": "R", "playerPool": "all", "limit": 3000},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("per-pitcher reliever stats fetch failed (%s)", exc)
            return {}

        by_team: dict[str, list[dict]] = {}
        for grp in data.get("stats", []):
            for sp in grp.get("splits", []):
                team = normalize_team(sp.get("team", {}).get("name"))
                st = sp.get("stat", {})
                player = sp.get("player") or {}
                g = st.get("gamesPlayed") or 0
                gs = st.get("gamesStarted") or 0
                if not team or not g or gs / g >= 0.5:  # starters excluded
                    continue
                try:
                    era = float(st.get("era"))
                except (TypeError, ValueError):
                    continue
                ip = self._parse_ip(st.get("inningsPitched"))
                if ip <= 0:
                    continue
                by_team.setdefault(team, []).append({
                    "player_id": int(player.get("id")) if player.get("id") else None,
                    "name": player.get("fullName") or "?",
                    "era": era,
                    "ip": ip,
                    "appearances": int(g),
                })
        # Sort each team's relievers by ERA ascending (best first).
        for team in by_team:
            by_team[team].sort(key=lambda r: r["era"])
        return by_team

    # -- Recent game pitching logs (for bullpen fatigue) ----------------------
    def get_recent_games_for_team(
        self,
        team_id: int,
        end_date: date_cls,
        days_back: int = 5,
    ) -> list[dict]:
        """Team's recently-completed games over a sliding window.

        Returns a list of {gamePk, date, status} sorted newest-first, filtered
        to games that are actually final (so the boxscore is meaningful). We
        pull a wider window than we strictly need (default 5d to find 3 games)
        because off-days and rainouts mean "yesterday's game" isn't always
        literally yesterday.
        """
        from datetime import timedelta
        start_date = end_date - timedelta(days=days_back)
        try:
            data = self._get(
                "/v1/schedule",
                {
                    "sportId": 1,
                    "teamId": int(team_id),
                    "startDate": start_date.isoformat(),
                    "endDate": (end_date).isoformat(),
                    "gameType": "R",
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("recent games fetch failed for team %s (%s)", team_id, exc)
            return []
        games: list[dict] = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                status = g.get("status", {}).get("detailedState", "")
                if status not in {"Final", "Game Over", "Completed Early"}:
                    continue
                games.append({
                    "gamePk": int(g.get("gamePk")),
                    "date": d.get("date"),
                    "status": status,
                })
        games.sort(key=lambda x: x["date"], reverse=True)
        return games

    def get_pitching_log(self, game_pk: int) -> dict[str, list[dict]]:
        """Per-game pitching log: who pitched, in order, with pitches thrown.

        Returns {"home": [...], "away": [...]} where each entry is
        {player_id, name, pitches, ip, is_starter}. The starter is the FIRST
        pitcher in MLB's `teams.<side>.pitchers` order-of-appearance list.
        Final/in-progress games only -- partial data is returned as-is. {} on
        failure (transient API errors should not break a slate).
        """
        try:
            data = self._get(f"/v1/game/{int(game_pk)}/boxscore")
        except Exception as exc:  # noqa: BLE001
            log.warning("boxscore fetch failed for game %s (%s)", game_pk, exc)
            return {}

        out: dict[str, list[dict]] = {"home": [], "away": []}
        for side in ("home", "away"):
            team = data.get("teams", {}).get(side, {}) or {}
            order = team.get("pitchers") or []  # order of appearance
            players = team.get("players") or {}
            for i, pid in enumerate(order):
                p = players.get(f"ID{pid}") or {}
                person = p.get("person") or {}
                stats = (p.get("stats") or {}).get("pitching") or {}
                try:
                    pitches = int(stats.get("pitchesThrown") or 0)
                except (TypeError, ValueError):
                    pitches = 0
                ip = self._parse_ip(stats.get("inningsPitched"))
                out[side].append({
                    "player_id": int(person.get("id")) if person.get("id") else None,
                    "name": person.get("fullName") or "?",
                    "pitches": pitches,
                    "ip": ip,
                    "is_starter": i == 0,
                })
        return out

    def get_team_bullpen_era(self, season: int) -> dict[str, float]:
        """{canonical_team: IP-weighted reliever ERA}, relievers := GS/G < 0.5.

        One call to the all-pitchers season endpoint (playerPool=all), aggregated
        per team. Automatable stand-in for FanGraphs bullpen FIP. {} on failure.
        """
        try:
            data = self._get(
                "/v1/stats",
                {"stats": "season", "group": "pitching", "season": int(season),
                 "sportId": 1, "gameType": "R", "playerPool": "all", "limit": 3000},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("bullpen stats fetch failed (%s)", exc)
            return {}
        ip_sum: dict[str, float] = {}
        era_ip: dict[str, float] = {}
        for grp in data.get("stats", []):
            for sp in grp.get("splits", []):
                team = normalize_team(sp.get("team", {}).get("name"))
                st = sp.get("stat", {})
                g = st.get("gamesPlayed") or 0
                gs = st.get("gamesStarted") or 0
                if not team or not g or gs / g >= 0.5:  # starters excluded
                    continue
                innings = self._parse_ip(st.get("inningsPitched"))
                try:
                    era = float(st.get("era"))
                except (TypeError, ValueError):
                    continue
                if innings <= 0:
                    continue
                ip_sum[team] = ip_sum.get(team, 0.0) + innings
                era_ip[team] = era_ip.get(team, 0.0) + era * innings
        return {t: era_ip[t] / ip_sum[t] for t in ip_sum if ip_sum[t] > 0}
