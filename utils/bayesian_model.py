#!/usr/bin/env python3
"""Bayesian Credible Interval for Football Match Outcomes.

Posterior distribution over match parameters given historical data.
Uses MCMC (or analytical normal approximation) to produce uncertainty
estimates around xG predictions and win probabilities.

Integrates with:
- Poisson distribution for goal modeling
- Beta priors for team strength
- Normal priors for home advantage

Usage:
    bayes = BayesianFootballModel(historical_data_df, method='mcmc')
    result = bayes.predict('曼城', '利物浦', alpha=0.95)
    print(result.summary())
    # {'home_win_prob': 0.52, 'home_win_ci': (0.44, 0.60),
    #  'xg_home': 1.65, 'xg_home_ci': (1.20, 2.10), ...}
"""

from __future__ import annotations
import math
import hashlib
import numpy as np
from scipy.stats import beta, normal, nbinom
from scipy.special import logit, expit
from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ─── Data class ───

class MatchRecord:
    """Single match result for Bayesian training."""
    def __init__(self, date, home_team, away_team,
                 home_goals, away_goals,
                 league="", is_friendly=False):
        self.date = date
        self.home_team = home_team
        self.away_team = away_team
        self.home_goals = home_goals
        self.away_goals = away_goals
        self.league = league
        self.is_friendly = is_friendly


# ─── Bayesian Model ───

class BayesianFootballModel:
    """Bayesian model for football match outcomes.

    Parameters:
        - Team strength μ_t ~ Normal(0, σ²) with informative priors
        - Attack λ_t ~ Gamma(α, β) for expected goals
        - Home advantage h ~ Normal(0, σ²_h)
        - Goal count ~ Poisson(λ) or Negative Binomial(λ, θ)

    Inference methods:
        - 'mcmc': Full MCMC sampling (slow but exact)
        - 'laplace': Laplace approximation (fast, good for most cases)
        - 'empirical': Empirical Bayesian (plug-in estimates)

    References:
        - Carlin & Louis, "Bayesian Methods for Data Analysis"
        - Dixon & Coles (1997) — but with Bayesian priors
    """

    def __init__(self, method: str = "laplace",
                 n_samples: int = 5000,
                 n_prior_strength: float = 50.0,
                 n_prior_goals: float = 5.0,
                 prior_home_adv: float = 0.25):
        """
        Args:
            method: Inference method.
            n_samples: MCMC samples (if method='mcmc').
            n_prior_strength: Pseudo-count for team strength prior.
            n_prior_goals: Pseudo-count for goals prior.
            prior_home_adv: Prior mean for home advantage.
        """
        self.method = method
        self.n_samples = n_samples
        self.n_prior_strength = n_prior_strength
        self.n_prior_goals = n_prior_goals
        self.prior_home_adv = prior_home_adv

        self.teams = []
        self.strengths = {}        # team → mean strength
        self.strength_uncertainty = {}  # team → std
        self.attack_rates = {}     # team → mean xG rate
        self.attack_uncertainty = {}  # team → std
        self.defence_rates = {}    # team → mean conceded rate
        self.home_adv_samples = []
        self.xg_posteriors = {}    # (home, away) → samples
        self.win_prob_posteriors = {}  # (home, away) → samples
        self._history = []

    # ─── Training ───

    def add_match(self, match: MatchRecord):
        self._history.append(match)

    def add_matches(self, matches: List[MatchRecord]):
        self._history.extend(matches)

    def fit(self, verbose: bool = True):
        if len(self._history) < 10:
            raise ValueError("Need at least 10 matches for Bayesian inference")

        cutoff = datetime.now()
        # Simple empirical Bayesian for now (upgrade to MCMC later)
        self.teams = list(set(
            t for m in self._history for t in [m.home_team, m.away_team]
        ))

        # Collect goals
        scored = {t: [] for t in self.teams}
        conceded = {t: [] for t in self.teams}
        for m in self._history:
            scored[m.home_team].append(m.home_goals)
            scored[m.away_team].append(m.away_goals)
            conceded[m.away_team].append(m.home_goals)
            conceded[m.home_team].append(m.away_goals)

        # Posterior estimates (Gamma conjugate for Poisson)
        for t in self.teams:
            alpha = self.n_prior_goals + sum(scored[t])
            beta_param = self.n_prior_strength + len(scored[t])
            self.attack_rates[t] = alpha / beta_param
            self.attack_uncertainty[t] = math.sqrt(alpha) / beta_param

            a = self.n_prior_goals + sum(conceded[t])
            b = self.n_prior_strength + len(conceded[t])
            self.defence_rates[t] = a / b
            self.defence_uncertainty[t] = math.sqrt(a) / b

            self.strengths[t] = self.attack_rates[t] - self.defence_rates[t]
            self.strength_uncertainty[t] = math.sqrt(
                self.attack_uncertainty[t]**2 + self.defence_uncertainty[t]**2
            )

        # Home advantage posterior
        home_diffs = []
        for m in self._history:
            home_diffs.append(m.home_goals - m.away_goals)
        h_mean = np.mean(home_diffs) if home_diffs else 0.25
        h_std = np.std(home_diffs) / math.sqrt(len(home_diffs)) if len(home_diffs) > 1 else 0.1
        self.home_adv_mean = h_mean
        self.home_adv_std = h_std

        if verbose:
            print(f"[fit] {len(self._history)} matches, {len(self.teams)} teams")
            print(f"  Home adv: {h_mean:.3f} ± {h_std:.3f}")

        return self

    # ─── Prediction ───

    def predict(
        self,
        home: str,
        away: str,
        alpha: float = 0.95,
    ) -> dict:
        """Predict match with Bayesian credible intervals.

        Args:
            home: Home team name
            away: Away team name
            alpha: Credible interval level (e.g. 0.95 for 95% CI)

        Returns:
            Dict with point estimates and credible intervals.
        """
        # Generate MCMC-like samples (simplified version)
        lam_h_samples, lam_a_samples = self._generate_samples(home, away)
        win_samples = []
        xg_home_samples = []
        xg_away_samples = []

        for lh, la in zip(lam_h_samples, lam_a_samples):
            # 1X2 via Poisson
            home_win = sum(1 for _ in range(1000)
                          if np.random.poisson(lh) > np.random.poisson(la)) / 1000.0
            win_samples.append(home_win)
            xg_home_samples.append(lh)
            xg_away_samples.append(la)

        # Compute credible intervals
        p_home = np.mean(win_samples)
        ci_home = float(np.percentile(win_samples, [50 - alpha*50, 50 + alpha*50]))

        return {
            "home": home,
            "away": away,
            "p_home_win": round(float(p_home), 4),
            "p_home_win_ci": (
                round(float(np.percentile(win_samples, 50 - alpha*50)), 4),
                round(float(np.percentile(win_samples, 50 + alpha*50)), 4)
            ),
            "xg_home": round(float(np.mean(xg_home_samples)), 3),
            "xg_home_ci": (
                round(float(np.percentile(xg_home_samples, 50 - alpha*50)), 3),
                round(float(np.percentile(xg_home_samples, 50 + alpha*50)), 3)
            ),
            "xg_away": round(float(np.mean(xg_away_samples)), 3),
            "xg_away_ci": (
                round(float(np.percentile(xg_away_samples, 50 - alpha*50)), 3),
                round(float(np.percentile(xg_away_samples, 50 + alpha*50)), 3)
            ),
            "alpha": alpha,
            "confidence_level": f"{alpha*100:.0f}% CI",
        }

    def _generate_samples(self, home: str, away: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate correlated xG samples for home and away."""
        home_str = self.strengths.get(home, 0.0)
        away_str = self.strengths.get(away, 0.0)
        home_str_unc = self.strength_uncertainty.get(home, 0.1)
        away_str_unc = self.strength_uncertainty.get(away, 0.1)

        # Sample team strengths from posteriors
        h_strengths = np.random.normal(home_str, home_str_unc, self.n_samples)
        a_strengths = np.random.normal(away_str, away_str_unc, self.n_samples)

        # Convert to xG with home advantage
        h_xgs = np.exp(h_strengths + self.home_adv_mean)
        a_xgs = np.exp(a_strengths - self.home_adv_mean)

        return h_xgs, a_xgs

    def summary(self) -> str:
        lines = ["Bayesian Football Model Summary"]
        lines.append(f"  Teams: {len(self.teams)}")
        lines.append(f"  Matches: {len(self._history)}")
        lines.append(f"  Home advantage: {self.home_adv_mean:.3f} ± {self.home_adv_std:.3f}")
        lines.append("  Team strengths (attack - defence):")
        for t in sorted(self.teams, key=lambda x: self.strengths[x], reverse=True)[:10]:
            lines.append(f"    {t:12s}: {self.strengths[t]:+.3f} ± {self.strength_uncertainty[t]:.3f}")
        return "\n".join(lines)


# ─── CLI ───

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bayesian football model")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--method", default="laplace")
    parser.add_argument("--alpha", type=float, default=0.95)
    args = parser.parse_args()

    model = BayesianFootballModel(method=args.method)

    # Demo data
    for d, h, a, hs, as_ in [
        ("2024-03-10", "曼城", "利物浦", 2, 1),
        ("2024-03-03", "曼城", "切尔西", 3, 0),
        ("2024-02-25", "利物浦", "阿森纳", 1, 1),
        ("2024-02-18", "阿森纳", "曼城", 0, 2),
        ("2024-02-10", "利物浦", "切尔西", 2, 0),
        ("2024-02-03", "切尔西", "利物浦", 1, 3),
        ("2024-01-28", "曼城", "阿森纳", 1, 0),
        ("2024-01-20", "利物浦", "曼城", 1, 1),
        ("2024-01-15", "阿森纳", "切尔西", 2, 1),
        ("2024-01-08", "切尔西", "阿森纳", 0, 2),
        ("2024-01-01", "曼城", "利物浦", 2, 2),
        ("2023-12-25", "阿森纳", "曼城", 1, 1),
        ("2023-12-20", "利物浦", "切尔西", 0, 0),
        ("2023-12-15", "曼城", "切尔西", 3, 1),
        ("2023-12-10", "利物浦", "阿森纳", 2, 1),
    ]:
        model.add_match(MatchRecord(d, h, a, hs, as_))

    model.fit()
    print(model.summary())
    print()
    result = model.predict(args.home, args.away, alpha=args.alpha)
    for k, v in result.items():
        if isinstance(v, tuple):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")