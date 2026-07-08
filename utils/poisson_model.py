#!/usr/bin/env python3
"""Poisson Distribution Model for Football Match Prediction.

Calculates match outcome probabilities (1X2), goal totals (Over/Under),
BTTS, and exact score distributions from expected goals (xG) inputs.

This module implements the "Poisson + Kelly" engine that computes model
probabilities for any match and compares them against bookmaker odds to
detect value edges (>12.5% threshold for S1 screening).

Usage:
    from utils.poisson_model import PoissonDistribution

    model = PoissonDistribution(home_xg=1.65, away_xg=0.80)
    outcome = model.outcome()            # {'home': 0.52, 'draw': 0.25, 'away': 0.23}
    totals  = model.totals(line=2.5)     # {'over': 0.48, 'under': 0.52}
    btts    = model.btts()               # {'yes': 0.42, 'no': 0.58}
    scores  = model.top_scores(n=6)      # [(1-0, 0.12), (0-0, 0.10), ...]
    ev      = model.value_edge(odds={'home': 1.55, 'draw': 4.0, 'away': 6.5})
"""

from math import exp, factorial, floor
from typing import Optional


class PoissonDistribution:
    """Poisson distribution model for football match outcomes.

    Uses independent Poisson distributions for home and away goals
    to compute joint probabilities for all match results.
    """

    def __init__(self, home_xg: float, away_xg: float, max_goals: int = 10):
        """
        Args:
            home_xg: Expected goals for home team.
            away_xg: Expected goals for away team.
            max_goals: Maximum goals to consider per team (default 10).
        """
        self.home_xg = home_xg
        self.away_xg = away_xg
        self.max_goals = max_goals

        # Pre-compute score matrix for efficiency
        self._score_matrix = self._build_score_matrix()

    # ─── Core Poisson calculations ───

    @staticmethod
    def pmf(k: int, lam: float) -> float:
        """Poisson probability mass function: P(X = k | λ)."""
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return exp(-lam) * lam ** k / factorial(k)

    def _build_score_matrix(self) -> dict:
        """Build 2D score probability matrix P(home=i, away=j)."""
        return {
            (i, j): self.pmf(i, self.home_xg) * self.pmf(j, self.away_xg)
            for i in range(self.max_goals + 1)
            for j in range(self.max_goals + 1)
        }

    def margin_probs(self) -> dict:
        """P(home_goals - away_goals = m) for each margin m."""
        out: dict[int, float] = {}
        for (i, j), p in self._score_matrix.items():
            m = i - j
            out[m] = out.get(m, 0.0) + p
        return out

    def total_probs(self) -> dict:
        """P(home_goals + away_goals = t) for each total t."""
        out: dict[int, float] = {}
        for (i, j), p in self._score_matrix.items():
            t = i + j
            out[t] = out.get(t, 0.0) + p
        return out

    # ─── Match outcome ───

    def outcome(self) -> dict[str, float]:
        """1X2 probabilities: home win, draw, away win.

        Returns:
            {'home': <P(home win)>, 'draw': <P(draw)>, 'away': <P(away win)>}
        """
        m = self.margin_probs()
        home = sum(p for k, p in m.items() if k > 0)
        draw = m.get(0, 0.0)
        away = sum(p for k, p in m.items() if k < 0)
        return {'home': home, 'draw': draw, 'away': away}

    def totals(self, line: float = 2.5) -> dict[str, float]:
        """Over/Under probability for a given goal line.

        Args:
            line: Goal line (e.g. 2.5 for O/U 2.5).

        Returns:
            {'over': <P(>line)>, 'under': <P(<line)>},
            'push': <P(=line)> for half-lines (0.25, 0.75) — zero for .5 lines.
        """
        t = self.total_probs()
        over = sum(p for k, p in t.items() if k > line)
        under = sum(p for k, p in t.items() if k < line)
        push = sum(p for k, p in t.items() if k == line) if line == int(line) else 0.0
        return {'over': over, 'under': under, 'push': push}

    def btts(self) -> dict[str, float]:
        """Both Teams To Score probability.

        Returns:
            {'yes': <P(both score)>, 'no': <P(at least one fails to score)>}
        """
        yes = sum(p for (i, j), p in self._score_matrix.items() if i >= 1 and j >= 1)
        return {'yes': yes, 'no': 1.0 - yes}

    def goal_distribution(self) -> dict[str, float]:
        """Full probability distribution of total goals 0..max_goals*2.

        Returns dict mapping goal count (str) to probability.
        """
        t = self.total_probs()
        return {str(k): v for k, v in sorted(t.items())}

    def home_goals_dist(self) -> dict[str, float]:
        """Home team goals distribution."""
        out: dict[str, float] = {}
        for (i, _), p in self._score_matrix.items():
            out[str(i)] = out.get(str(i), 0.0) + p
        return {k: round(v, 6) for k, v in sorted(out.items())}

    def away_goals_dist(self) -> dict[str, float]:
        """Away team goals distribution."""
        out: dict[str, float] = {}
        for (_, j), p in self._score_matrix.items():
            out[str(j)] = out.get(str(j), 0.0) + p
        return {k: round(v, 6) for k, v in sorted(out.items())}

    def top_scores(self, n: int = 6) -> list[tuple[str, float]]:
        """Most likely exact scores, sorted by probability descending.

        Returns:
            List of [(score_str, probability), ...]
        """
        sorted_scores = sorted(self._score_matrix.items(), key=lambda kv: kv[1], reverse=True)
        return [
            (f"{i}-{j}", round(p, 6))
            for (i, j), p in sorted_scores[:n]
        ]

    # ─── Value detection (S1 screening) ───

    @staticmethod
    def no_vig(odds: dict[str, float]) -> tuple[dict[str, float], float]:
        """Remove bookmaker margin (vig) from decimal odds.

        Args:
            odds: {'home': odds_home, 'draw': odds_draw, 'away': odds_away}

        Returns:
            (fair_probs, margin) where fair_probs sum to 1.0.
        """
        raw = {k: 1.0 / v for k, v in odds.items()}
        s = sum(raw.values())
        fair = {k: v / s for k, v in raw.items()}
        margin = s - 1.0
        return fair, margin

    def value_edge(
        self,
        odds: dict[str, float],
        edge_threshold: float = 0.125,
    ) -> dict:
        """Detect value edges between model and bookmaker odds.

        This is the S1 screening engine: finds matches where
        model probability differs from implied probability by > threshold.

        Args:
            odds: Bookmaker decimal odds {'home': float, 'draw': float, 'away': float}.
            edge_threshold: Minimum edge to flag (default 12.5%).

        Returns:
            Dict with per-outcome edges, and 'flagged' list of high-value opportunities.
        """
        model = self.outcome()
        fair, margin = self.no_vig(odds)

        edges = {}
        for k in ("home", "draw", "away"):
            edge = model[k] - fair[k]
            ev = model[k] * odds[k] - 1.0  # expected value per unit
            edges[k] = {
                'model_prob': round(model[k], 4),
                'fair_prob': round(fair[k], 4),
                'implied_prob': round(1.0 / odds[k], 4),
                'edge': round(edge, 4),
                'ev_per_unit': round(ev, 4),
                'flag': abs(edge) >= edge_threshold,
            }

        # Flag outcomes exceeding the edge threshold
        flagged = [k for k, v in edges.items() if v['flag']]

        return {
            'market_margin': round(margin, 4),
            'edges': edges,
            'flagged': flagged,
            'has_value': len(flagged) > 0,
            'max_edge': max(abs(e['edge']) for e in edges.values()),
        }

    def over_under_value(
        self,
        line: float,
        odds_over: float,
        odds_under: float,
        edge_threshold: float = 0.125,
    ) -> dict:
        """Detect value edges for Over/Under markets.

        Handles .5 lines (no push), integer lines (push possible),
        and .25/.75 quarter lines (split-stake) — matching real betting
        market lines like 2.5, 3, 2.25, 2.75.

        Args:
            line: Goal line (e.g. 2.5, 3, 2.25).
            odds_over: Bookmaker decimal odds for Over.
            odds_under: Bookmaker decimal odds for Under.
            edge_threshold: Minimum edge to flag (default 12.5%).

        Returns:
            Dict with model vs market comparison for O/U line.
        """
        # Determine if quarter line (e.g. 2.25, 2.75)
        is_quarter = abs(round(line * 2) - line * 2) > 1e-9

        if is_quarter:
            # Quarter lines split stake across two adjacent .5 lines
            # e.g. 2.25 → 50% stake on 2.0, 50% on 2.5
            lower = floor(line) - 0.0   # 2.25 → 2.0
            upper = floor(line) + 0.5   # 2.25 → 2.5
            # But the lower bound for a quarter line like 2.25
            # actually splits between integer and next half:
            # 2.25 = (2.0 + 2.5) / 2  → 2.0 is lower, 2.5 is upper
            lower_hl = floor(line)
            upper_hl = floor(line) + 0.5
            m_lower = self.totals(lower_hl)  # 2.0 → has push
            m_upper = self.totals(upper_hl)  # 2.5 → no push

            # For Over 2.25: half goes to Over 2.0, half to Over 2.5
            model_over = 0.5 * m_lower['over'] + 0.5 * m_upper['over']
            # For Under 2.25: half goes to Under 2.0, half to Under 2.5
            model_under = 0.5 * m_lower['under'] + 0.5 * m_upper['under']
        else:
            m = self.totals(line)
            model_over = m['over']
            model_under = m['under']

        vig = 1.0 / odds_over + 1.0 / odds_under
        fair_over = (1.0 / odds_over) / vig
        fair_under = (1.0 / odds_under) / vig
        margin = vig - 1.0

        edge_over = model_over - fair_over
        edge_under = model_under - fair_under

        flagged_over = abs(edge_over) >= edge_threshold
        flagged_under = abs(edge_under) >= edge_threshold

        return {
            'line': line,
            'quarter_line': is_quarter,
            'market_margin': round(margin, 4),
            'over': {
                'model_prob': round(model_over, 4),
                'fair_prob': round(fair_over, 4),
                'edge': round(edge_over, 4),
                'ev': round(model_over * odds_over - 1.0, 4),
                'flag': flagged_over,
            },
            'under': {
                'model_prob': round(model_under, 4),
                'fair_prob': round(fair_under, 4),
                'edge': round(edge_under, 4),
                'ev': round(model_under * odds_under - 1.0, 4),
                'flag': flagged_under,
            },
            'flagged': [k for k, v in [('over', flagged_over), ('under', flagged_under)] if v],
        }

    def summary(self) -> str:
        """Generate a human-readable summary of all Poisson calculations."""
        outcome = self.outcome()
        totals_25 = self.totals(2.5)
        btts_val = self.btts()
        top = self.top_scores(6)

        lines = [
            f"Poisson Model (xG {self.home_xg:.2f} vs {self.away_xg:.2f}):",
            f"  1X2  | Home: {outcome['home']*100:5.1f}%  Draw: {outcome['draw']*100:5.1f}%  Away: {outcome['away']*100:5.1f}%",
            f"  O/U 2.5 | Over: {totals_25['over']*100:5.1f}%  Under: {totals_25['under']*100:5.1f}%  Push: {totals_25['push']*100:5.1f}%",
            f"  BTTS   | Yes:  {btts_val['yes']*100:5.1f}%  No: {btts_val['no']*100:5.1f}%",
            f"  Top scores:",
        ]
        for score, prob in top:
            lines.append(f"    {score}: {prob*100:5.1f}%")

        return "\n".join(lines)


# ─── CLI entry point ───

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Poisson football model")
    parser.add_argument("--home-xg", type=float, required=True)
    parser.add_argument("--away-xg", type=float, required=True)
    args = parser.parse_args()

    model = PoissonDistribution(args.home_xg, args.away_xg)
    print(model.summary())