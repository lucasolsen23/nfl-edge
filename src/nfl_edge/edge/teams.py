"""NFL team name <-> abbreviation, to match Kalshi tickers with book team names.

Kalshi encodes teams as nflverse-style abbreviations in the market ticker suffix
(e.g. ...-KC, ...-NYG). The Odds API uses full names ("Kansas City Chiefs").
"""

from __future__ import annotations

# full book name -> nflverse abbreviation
NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

# a few abbreviation aliases seen across data sources
ABBR_ALIASES = {"WSH": "WAS", "JAC": "JAX", "LA": "LAR", "OAK": "LV", "SD": "LAC", "STL": "LAR"}


ABBR_TO_NAME = {v: k for k, v in NAME_TO_ABBR.items()}


def name(ab: str) -> str:
    """Abbreviation -> full team name (falls back to the abbr itself)."""
    return ABBR_TO_NAME.get(norm_abbr(ab), ab)


def short_name(ab: str) -> str:
    """Abbreviation -> city/nickname short label, e.g. 'Tampa Bay'."""
    full = ABBR_TO_NAME.get(norm_abbr(ab))
    return " ".join(full.split()[:-1]) if full else ab


def abbr(name: str) -> str | None:
    """Map a book team name to an abbreviation (best effort)."""
    if name in NAME_TO_ABBR:
        return NAME_TO_ABBR[name]
    # fall back: match on last word (mascot) if a city prefix differs
    for full, ab in NAME_TO_ABBR.items():
        if full.split()[-1] == name.split()[-1]:
            return ab
    return None


def norm_abbr(a: str) -> str:
    return ABBR_ALIASES.get(a.upper(), a.upper())
