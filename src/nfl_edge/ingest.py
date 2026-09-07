"""Data ingestion: nflreadpy -> local parquet cache.

Pulls schedules (with closing spread/total/moneyline lines and results) and
play-by-play, caches them to data/raw/ so downstream steps are fast and
offline-reproducible.

CLI:  python -m nfl_edge.ingest            # uses config defaults
      python scripts/build_dataset.py      # same, with friendlier output
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import nflreadpy as nfl

# repo layout: src/nfl_edge/ingest.py -> repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

# Columns we actually use from the (wide) schedule table.
SCHEDULE_COLS = [
    "game_id", "season", "game_type", "week", "gameday", "gametime",
    "away_team", "home_team", "away_score", "home_score",
    "result", "total", "overtime", "location", "div_game",
    "spread_line", "away_spread_odds", "home_spread_odds",
    "total_line", "under_odds", "over_odds",
    "away_moneyline", "home_moneyline",
    "away_rest", "home_rest", "roof", "surface", "temp", "wind",
    "away_qb_name", "home_qb_name",
]


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_schedules(seasons: range | list[int]) -> pl.DataFrame:
    """Per-game schedule + closing lines + results.

    NOTE: the single line per game here is the CLOSING consensus line. There is
    no opening line or per-book entry line, so historical CLV is not derivable
    from this table -- CLV is a live-ops metric (see README).
    """
    seasons = list(seasons)
    df = nfl.load_schedules(seasons=seasons)
    keep = [c for c in SCHEDULE_COLS if c in df.columns]
    return df.select(keep)


def load_pbp(seasons: range | list[int]) -> pl.DataFrame:
    """Play-by-play (EPA, success, cpoe, etc.) -- feature foundation for phase 2."""
    return nfl.load_pbp(seasons=list(seasons))


def cache_schedules(seasons: range | list[int]) -> Path:
    _ensure_dirs()
    df = load_schedules(seasons)
    out = RAW_DIR / "schedules.parquet"
    df.write_parquet(out)
    _sanity_schedules(df)
    print(f"[schedules] {df.height} games x {df.width} cols -> {out}")
    return out


def cache_pbp(seasons: range | list[int]) -> Path:
    _ensure_dirs()
    df = load_pbp(seasons)
    out = RAW_DIR / "pbp.parquet"
    df.write_parquet(out)
    print(f"[pbp] {df.height} plays x {df.width} cols -> {out}")
    return out


def _sanity_schedules(df: pl.DataFrame) -> None:
    """Fail loudly if the data's semantics drift from what the code assumes."""
    played = df.filter(pl.col("result").is_not_null())
    if played.height == 0:
        return
    ok_result = played.select(
        (pl.col("result") == (pl.col("home_score") - pl.col("away_score"))).all()
    ).item()
    ok_total = played.select(
        (pl.col("total") == (pl.col("home_score") + pl.col("away_score"))).all()
    ).item()
    assert ok_result, "result != home_score - away_score (semantics drift!)"
    assert ok_total, "total != home_score + away_score (semantics drift!)"


def read_cached_schedules() -> pl.DataFrame:
    return pl.read_parquet(RAW_DIR / "schedules.parquet")


# --------------------------------------------------------------------------
# Player stats + rosters (for the props model). Future seasons 404 until they
# exist, so we load season-by-season and skip what isn't published yet.
# --------------------------------------------------------------------------
PLAYER_COLS = [
    "player_id", "player_name", "player_display_name", "position",
    "position_group", "season", "week", "season_type", "team", "opponent_team",
    "attempts", "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
]


def _load_available(loader, seasons: list[int], label: str):
    """Call an nflreadpy loader per season, skipping seasons that 404."""
    frames, got = [], []
    for yr in seasons:
        try:
            frames.append(loader(seasons=[yr]))
            got.append(yr)
        except Exception as e:  # noqa: BLE001 -- future season not published yet
            print(f"  [{label}] {yr} unavailable, skipping ({type(e).__name__})")
    if not frames:
        raise RuntimeError(f"no {label} seasons available")
    print(f"  [{label}] loaded seasons {got[0]}-{got[-1]}")
    return pl.concat(frames, how="diagonal_relaxed")


def cache_player_stats(seasons: range | list[int]) -> Path:
    _ensure_dirs()
    df = _load_available(nfl.load_player_stats, list(seasons), "player_stats")
    df = df.select([c for c in PLAYER_COLS if c in df.columns])
    out = RAW_DIR / "player_stats.parquet"
    df.write_parquet(out)
    print(f"[player_stats] {df.height} rows -> {out}")
    return out


def cache_rosters(season: int) -> Path:
    _ensure_dirs()
    df = nfl.load_rosters(seasons=[season])
    out = RAW_DIR / f"rosters_{season}.parquet"
    df.write_parquet(out)
    print(f"[rosters] {df.height} players ({season}) -> {out}")
    return out


if __name__ == "__main__":
    # Defaults chosen from the coverage audit:
    #   schedules 1999+ (spread/total complete), pbp 2015+ (modeling window).
    cache_schedules(range(1999, 2026))
    # pbp is large; comment out if you only need the backtest harness first.
    # cache_pbp(range(2015, 2026))
