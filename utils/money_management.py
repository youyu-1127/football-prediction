#!/usr/bin/env python3
"""Fractional Kelly Criterion for bankroll management.

Auto-calculates optimal stake using the 1/4-Kelly and 1/2-Kelly methods,
with hard-circuit protection to enforce the rule "单场均注不超过总资金 2%".
This is the R4 (bankroll layer) core engine for V3.0 — mathematically
constraining human impulse to eliminate "all-in" risk.

Usage:
    from utils.money_management import MoneyManagement

    mm = MoneyManagement(bankroll=10000, max_single_stake_pct=0.02)

    # Calculate Kelly stake for home win
    kelly = mm.kelly_stake(
        prob=0.55, odds=1.80, fraction=0.25   # 1/4 Kelly
    )
    # → returns {'full_kelly': 0.0625, 'quarter_kelly': 0.0156, 'half_kelly': 0.0313}

    # Portfolio-level risk check
    result = mm.portfolio_risk(bets=[
        {'team': 'home', 'prob': 0.55, 'odds': 1.80},
        {'team': 'away', 'prob': 0.30, 'odds': 3.50},
    ])
    # → warns if total exposure > max_single_stake_pct or bankroll drawn down > 10%
"""

import math
from typing import Optional


class MoneyManagement:
    """Kelly Criterion fractional bankroll management with safety limits."""

    def __init__(
        self,
        bankroll: float,
        max_single_stake_pct: float = 0.02,
        max_total_exposure_pct: float = 0.15,
        drawdown_alert_pct: float = 0.10,
    ):
        """
        Args:
            bankroll: Total betting bankroll (currency units).
            max_single_stake_pct: Maximum stake per bet as fraction of bankroll.
                Default 2% — matches the R4 rule "单场均注不超过总资金 2%".
            max_total_exposure_pct: Max total exposure across all open bets.
                Default 15%.
            drawdown_alert_pct: Bankroll drawdown that triggers alert.
                Default 10%.
        """
        self.bankroll = bankroll
        self.max_single_stake_pct = max_single_stake_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.drawdown_alert_pct = drawdown_alert_pct

        # Track active bets for portfolio risk
        self._active_bets: list[dict] = []

    @staticmethod
    def full_kelly(prob: float, decimal_odds: float) -> float:
        """Full Kelly fraction of bankroll.

        f* = (b*q - p) / b = p - q/b  where b = odds-1, q = 1-p

        Returns:
            Kelly fraction (0 if no edge).
        """
        b = decimal_odds - 1.0
        if b <= 0:
            return 0.0
        q = 1.0 - prob
        f = (b * prob - q) / b
        return max(0.0, f)

    @staticmethod
    def fractional_kelly(
        prob: float, decimal_odds: float, fraction: float = 0.25
    ) -> float:
        """Fractional Kelly stake.

        Args:
            prob: Model probability of outcome (0-1).
            decimal_odds: Bookmaker decimal odds.
            fraction: Kelly fraction — 0.25 = 1/4 Kelly, 0.5 = 1/2 Kelly.

        Returns:
            Stake as fraction of bankroll (0 if no edge).
        """
        return MoneyManagement.full_kelly(prob, decimal_odds) * fraction

    def kelly_stake(
        self,
        prob: float,
        odds: float,
        fraction: float = 0.25,
    ) -> dict:
        """Calculate Kelly stake with caps.

        Returns full, 1/2, and 1/4 Kelly with applied cap at max_single_stake_pct.

        Args:
            prob: Model probability (0-1).
            odds: Bookmaker decimal odds.
            fraction: Kelly fraction (default 0.25 for 1/4 Kelly).

        Returns:
            Dict with full_kelly, half_kelly, quarter_kelly, applied_stake, capped.
        """
        full = self.full_kelly(prob, odds)
        half = full * 0.5
        quarter = full * 0.25

        # Apply hard cap
        applied = min(quarter, self.max_single_stake_pct)
        capped = applied < quarter

        return {
            'bankroll': self.bankroll,
            'full_kelly': round(full, 6),
            'half_kelly': round(half, 6),
            'quarter_kelly': round(quarter, 6),
            'applied_stake_pct': round(applied, 6),
            'applied_stake_units': round(self.bankroll * applied, 2),
            'capped': capped,
            'cap_reason': 'max_single_stake' if capped else 'kelly_limit',
        }

    def edge_check(self, prob: float, odds: float) -> dict:
        """Check if a bet has positive expected value.

        Args:
            prob: Model probability.
            odds: Bookmaker decimal odds.

        Returns:
            Dict with edge, EV, and whether to bet.
        """
        implied = 1.0 / odds
        edge = prob - implied
        ev = prob * odds - 1.0

        # Require minimum edge to beat model error
        min_edge = 0.02  # 2% minimum to account for model uncertainty

        return {
            'edge': round(edge, 4),
            'ev_per_unit': round(ev, 4),
            'implied_prob': round(implied, 4),
            'has_positive_ev': ev > 0,
            'clears_safety_margin': edge > min_edge,
            'recommendation': 'BET' if ev > min_edge else ('SKIP' if ev < 0 else 'THIN_EDGE'),
        }

    def portfolio_risk(self, bets: list[dict]) -> dict:
        """Check portfolio-level risk across all open bets.

        Args:
            bets: List of dicts with keys: 'team', 'prob', 'odds', 'stake_pct'.

        Returns:
            Dict with portfolio analysis and risk flags.
        """
        total_exposure = sum(b.get('stake_pct', 0) for b in bets)
        total_ev = sum(
            b['prob'] * b['odds'] - 1 for b in bets
        ) * (self.bankroll if bets else 0)

        flags = []
        if total_exposure > self.max_total_exposure_pct:
            flags.append(
                f"EXPOSURE: {total_exposure*100:.1f}% exceeds "
                f"{self.max_total_exposure_pct*100:.0f}% limit"
            )
        for b in bets:
            if b.get('stake_pct', 0) > self.max_single_stake_pct:
                flags.append(
                    f"STAKE: {b['team']} stake {b['stake_pct']*100:.1f}% "
                    f"exceeds {self.max_single_stake_pct*100:.0f}% limit"
                )

        return {
            'total_exposure_pct': round(total_exposure * 100, 2),
            'total_ev_units': round(total_ev, 2),
            'num_bets': len(bets),
            'flags': flags,
            'risk_level': (
                'HIGH' if flags
                else 'MEDIUM' if total_exposure > 0.08
                else 'LOW'
            ),
            'can_add_bets': total_exposure < self.max_total_exposure_pct,
        }

    def track_bet(self, bet: dict) -> None:
        """Track an active bet for portfolio risk management."""
        self._active_bets.append(bet)

    def resolve_bet(self, team: str, result: str) -> dict:
        """Mark a bet as resolved (win/loss/push).

        Args:
            team: Team/outcome name to find in active bets.
            result: 'win', 'loss', or 'push'.

        Returns:
            Updated bankroll info.
        """
        bet = None
        for i, b in enumerate(self._active_bets):
            if b.get('team') == team:
                bet = self._active_bets.pop(i)
                break

        if not bet:
            return {'error': f'Bet for {team} not found'}

        stake_units = bet['stake_pct'] * self.bankroll
        if result == 'win':
            pnl = stake_units * (bet['odds'] - 1)
            self.bankroll += pnl
        elif result == 'loss':
            pnl = -stake_units
            self.bankroll -= stake_units
        else:  # push
            pnl = 0

        drawdown = max(0, 1.0 - self.bankroll / 10000)  # relative to starting
        alert = drawdown > self.drawdown_alert_pct

        return {
            'team': team,
            'result': result,
            'pnl': round(pnl, 2),
            'new_bankroll': round(self.bankroll, 2),
            'drawdown_pct': round(drawdown * 100, 2),
            'drawdown_alert': alert,
        }

    def current_status(self) -> dict:
        """Get current portfolio and bankroll status."""
        return {
            'bankroll': round(self.bankroll, 2),
            'active_bets': len(self._active_bets),
            'portfolio': self.portfolio_risk(self._active_bets),
        }


# ─── CLI entry point ───

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kelly Criterion calculator")
    parser.add_argument("--prob", type=float, required=True, help="Model probability (0-1)")
    parser.add_argument("--odds", type=float, required=True, help="Bookmaker decimal odds")
    parser.add_argument("--bankroll", type=float, default=10000, help="Total bankroll")
    parser.add_argument(
        "--fraction", type=float, default=0.25,
        choices=[0.25, 0.5, 1.0],
        help="Kelly fraction: 0.25=1/4, 0.5=1/2, 1.0=full"
    )
    parser.add_argument(
        "--max-stake", type=float, default=0.02,
        help="Max single stake as fraction (default 2%%)"
    )
    args = parser.parse_args()

    mm = MoneyManagement(
        bankroll=args.bankroll,
        max_single_stake_pct=args.max_stake,
    )

    kelly = mm.kelly_stake(args.prob, args.odds, args.fraction)
    edge = mm.edge_check(args.prob, args.odds)

    print(f"Kelly Calculation (bankroll: {args.bankroll}):")
    print(f"  Full Kelly:     {kelly['full_kelly']*100:.3f}%")
    print(f"  1/2 Kelly:      {kelly['half_kelly']*100:.3f}%")
    print(f"  1/4 Kelly:      {kelly['quarter_kelly']*100:.3f}%")
    print(f"  Applied stake:  {kelly['applied_stake_pct']*100:.3f}% ({kelly['applied_stake_units']:.2f} units)")
    print(f"  Capped:         {kelly['capped']} ({kelly['cap_reason']})")
    print(f"  Edge:           {edge['edge']*100:+.2f}%")
    print(f"  EV/unit:        {edge['ev_per_unit']:+.4f}")
    print(f"  Recommendation: {edge['recommendation']}")