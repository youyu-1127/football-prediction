#!/usr/bin/env python3
"""ELO Rating System for Football Teams.

Computes dynamic ELO ratings from match results and generates "theory handicaps"
(theoretical Asian handicap lines). Used in S1 screening: when the bookmaker's
opening line deviates from the ELO theory line by >0.5 goals, the system flags
a potential anomaly ("可能有鬼").

This module is the independent pricing anchor for the S1 screening layer —
it prices matches purely from historical performance data, with no bookmaker
data influencing the rating.

Usage:
    from utils.team_strength import TeamStrength

    ts = TeamStrength()

    # Add historical results (can batch-load from a CSV or DB)
    ts.update_result('曼城', '利物浦', 2, 1)   # 曼城 home, 2-1
    ts.update_result('皇马', '拜仁', 1, 1)     # 皇马 home, 1-1

    # Get current ELO ratings
    rating_home = ts.get_rating('曼城')          # e.g. 1850
    rating_away = ts.get_rating('利物浦')        # e.g. 1820

    # Theory handicap from ELO
    theory_handicap = ts.theory_handicap('曼城', '利物浦')
    # → {'home_elo': 1850, 'away_elo': 1820, 'elo_diff': 30, 'theory_handicap': 0.25}

    # S1 screening alert
    alert = ts.s1_screen(
        bookmaker_line=-0.75,
        home_team='曼城',
        away_team='利物浦'
    )
    # → {'deviation': 0.50, 'flagged': True, 'reason': '...'
"""

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

# Default base ELO values for well-known teams (fallback when no history exists)
DEFAULT_ELO = {
    '曼城': 2100, '皇马': 2080, '拜仁': 2070, '巴萨': 2050, '阿森纳': 2050,
    '兵工厂': 2050, '利物浦': 2030, '法国': 2030, '巴黎圣曼': 2020,
    '大巴黎': 2020, '英格兰': 2020, '马竞': 2000, '巴西': 2000,
    '切尔西': 1980, '国际米兰': 1980, 'ac米兰': 1980, '阿根廷': 1980,
    '尤文图斯': 1970, '德国': 1970, '多特蒙德': 1960, '曼联': 1950,
    '意大利': 1950, '荷兰': 1940, '不伦瑞克': 1930, '比利时': 1930,
    '热刺': 1920, '里昂': 1920, '本菲卡': 1920, '阿贾克斯': 1920,
    '葡萄牙': 1920, '那不勒斯': 1900, '波尔图': 1900, '克罗地亚': 1900,
    '摩纳哥': 1880, '罗马': 1880, '拉齐奥': 1860, '日本': 1830,
    '瑞典': 1820, '塞尔维亚': 1810, '智利': 1810, '美国': 1790,
    '韩国': 1800, '乌兹别克': 1780, '墨西哥': 1850, '哥伦比亚': 1840,
    '丹麦': 1850, '乌拉圭': 1890, '秘鲁': 1700, '加拿大': 1750,
    '捷克': 1750, '美国': 1790, '巴拿马': 1630, '刚果金': 1620,
    '克罗地亚': 1900, '加纳': 1680, '科特迪瓦': 1700, '塞内加尔': 1730,
    '尼日利亚': 1700, '摩洛哥': 1750, '突尼斯': 1700, '埃及': 1710,
    '阿尔及利亚': 1700, '南非': 1600, '伊朗': 1680, '沙特': 1650,
    '韩国': 1800, '日本': 1830, '澳大利亚': 1770, '伊拉克': 1640,
    '中国': 1650, '阿联酋': 1620, '卡塔尔': 1640, '泰国': 1500,
    '印尼': 1400, '越南': 1450, '朝鲜': 1550, '印度': 1300,
    '土耳其': 1790, '俄罗斯': 1700, '波兰': 1770, '乌克兰': 1750,
    '瑞士': 1800, '奥地利': 1730, '挪威': 1740, '希腊': 1720,
    '以色列': 1660, '芬兰': 1680, '冰岛': 1690, '匈牙利': 1720,
    '罗马尼亚': 1730, '保加利亚': 1700, '捷克': 1750, '斯洛伐克': 1710,
    '斯洛文尼亚': 1720, '克罗地亚': 1900, '塞尔维亚': 1810,
    '北马其顿': 1735, '黑山': 1300, '波黑': 1380, '阿尔巴尼亚': 1258,
    '爱沙尼亚': 1435, '拉脱维亚': 1530, '立陶宛': 1294, '白俄罗斯': 1620,
}

# Default expected goals for league contexts (xG priors)
DEFAULT_XG_PRIOR = {
    'home': 1.50, 'away': 1.10,  # general league averages
    'premier_league': (1.65, 1.15),
    'la_liga': (1.55, 1.12),
    'bundesliga': (1.70, 1.20),
    'serie_a': (1.45, 1.08),
    'ligue_1': (1.50, 1.10),
    'world_cup': (1.25, 1.15),
    'friendly': (1.35, 1.25),
}


@dataclass
class TeamProfile:
    """Stores ELO rating and match statistics for a single team."""
    name: str
    rating: float = 1500.0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0
    matches: int = 0
    home_xg: float = 1.50
    away_xg: float = 1.10
    xg_ratio: float = 1.0

    def win_pct(self) -> float:
        return self.wins / self.matches if self.matches > 0 else 0.5

    def goal_diff(self) -> int:
        return self.goals_scored - self.goals_conceded


class TeamStrength:
    """Dynamic ELO rating system with theory handicap and S1 screening.

    ELO calculation follows the standard football-adapted formula:
      - Expected score based on ELO difference
      - K-factor adapts to match importance and margin
      - Home advantage baked into the expected score
    """

    def __init__(
        self,
        k_factor: float = 30.0,
        home_advantage: float = 100.0,
        default_elo: Optional[dict] = None,
        elo_db_path: Optional[str] = None,
    ):
        """
        Args:
            k_factor: Rating change multiplier. 30 is standard for football.
            home_advantage: ELO points added to home team rating.
                +100 ≈ +0.35 goals expected advantage.
            default_elo: Dict mapping team name → base ELO. Overrides embedded defaults.
            elo_db_path: Path to a JSON file with persistent ELO data.
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}
        self.profiles: dict[str, TeamProfile] = {}

        # Load default ELO values
        self.ratings.update(DEFAULT_ELO)
        if default_elo:
            self.ratings.update(default_elo)

        # Load from persistent file if available
        if elo_db_path and os.path.exists(elo_db_path):
            with open(elo_db_path, 'r') as f:
                data = json.load(f)
                self.ratings.update(data.get('ratings', {}))
                for name, prof in data.get('profiles', {}).items():
                    self.profiles[name] = TeamProfile(**prof)

        self.elo_db_path = elo_db_path

    def _get_expected_score(self, rating_a: float, rating_b: float,
                            is_home: bool = True) -> float:
        """Expected score for team A vs team B using ELO logistic formula.

        E_A = 1 / (1 + 10^((R_B - R_A +/- HA) / 400))

        Args:
            rating_a: Rating of team A.
            rating_b: Rating of team B.
            is_home: Whether team A is playing at home.

        Returns:
            Expected score (0-1) for team A.
        """
        effective_diff = (rating_b - rating_a) - (self.home_advantage if is_home else 0)
        return 1.0 / (1.0 + math.pow(10.0, effective_diff / 400.0))

    def update_result(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        importance: float = 1.0,
    ) -> dict:
        """Update ELO ratings after a match result.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            home_goals: Home team goals scored.
            away_goals: Away team goals scored.
            importance: Match importance multiplier (1.0 = league, 1.5 = knockout, 0.7 = friendly).

        Returns:
            Dict with rating changes and new ratings.
        """
        r_home = self.ratings.get(home_team, 1500.0)
        r_away = self.ratings.get(away_team, 1500.0)

        # Expected scores
        e_home = self._get_expected_score(r_home, r_away, is_home=True)
        e_away = 1.0 - e_home

        # Actual scores
        s_home = 1 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0)
        s_away = 1 - s_home

        # Margin-adjusted K-factor (larger margin → slightly higher K)
        margin = abs(home_goals - away_goals)
        k = self.k_factor * importance * (1.0 + 0.05 * min(margin, 3))

        # Rating updates
        dr_home = k * (s_home - e_home)
        dr_away = k * (s_away - e_away)

        self.ratings[home_team] = r_home + dr_home
        self.ratings[away_team] = r_away + dr_away

        # Update profiles
        self._update_profile(home_team, home_goals, away_goals, win=(home_goals > away_goals),
                             draw=(home_goals == away_goals), is_home=True)
        self._update_profile(away_team, away_goals, home_goals, win=(away_goals > home_goals),
                             draw=(home_goals == away_goals), is_home=False)

        return {
            'home_team': home_team,
            'away_team': away_team,
            'score': f'{home_goals}-{away_goals}',
            'r_home_before': round(r_home, 1),
            'r_away_before': round(r_away, 1),
            'r_home_after': round(self.ratings[home_team], 1),
            'r_away_after': round(self.ratings[away_team], 1),
            'delta_home': round(dr_home, 2),
            'delta_away': round(dr_away, 2),
        }

    def _update_profile(self, team: str, goals_for: int, goals_against: int,
                        win: bool = False, draw: bool = False, is_home: bool = True):
        """Update team profile statistics."""
        if team not in self.profiles:
            self.profiles[team] = TeamProfile(name=team, rating=self.ratings.get(team, 1500.0))

        p = self.profiles[team]
        p.matches += 1
        p.goals_scored += goals_for
        p.goals_conceded += goals_against
        if win:
            p.wins += 1
        elif draw:
            p.draws += 1
        else:
            p.losses += 1
        p.rating = self.ratings.get(team, 1500.0)

        # Estimate home/away xG from recent profile
        if p.matches > 0:
            avg_total_goals = (p.goals_scored + p.goals_conceded) / p.matches
            goal_diff_pct = (p.goals_scored - p.goals_conceded) / max(p.goals_conceded, 1)
            if is_home:
                p.home_xg = max(0.3, 1.50 * (1 + 0.1 * goal_diff_pct))
                p.away_xg = max(0.3, 1.10 * (1 - 0.05 * goal_diff_pct))
            else:
                p.home_xg = max(0.3, 1.50 * (1 + 0.05 * goal_diff_pct))
                p.away_xg = max(0.3, 1.10 * (1 + 0.1 * goal_diff_pct))

            p.xg_ratio = p.home_xg / max(p.away_xg, 0.01)

    def get_rating(self, team: str) -> float:
        """Get current ELO rating for a team."""
        return self.ratings.get(team, 1500.0)

    def get_profiles(self, team: str) -> Optional[TeamProfile]:
        """Get full profile for a team."""
        return self.profiles.get(team)

    def theory_handicap(self, home_team: str, away_team: str) -> dict:
        """Calculate theory Asian handicap from ELO difference.

        ELO difference → expected goal difference → Asian handicap.
        Formula: theory_handicap ≈ (elo_diff - 100) / 250
        This maps: 0 ELO diff → ~0 handicap, +250 ELO → ~+1 handicap.

        Args:
            home_team: Home team name.
            away_team: Away team name.

        Returns:
            Dict with elo ratings, difference, and theory handicap.
        """
        r_home = self.ratings.get(home_team, 1500.0)
        r_away = self.ratings.get(away_team, 1500.0)
        elo_diff = r_home - r_away

        # Convert ELO difference to expected goal difference
        # Standard mapping: 200 ELO ≈ 0.7 goals ≈ 0.75 Asian handicap
        expected_gd = elo_diff / 280.0  # calibrated to typical football data

        # Theory handicap (negative = away favorite in Asian terms)
        theory_hc = expected_gd

        # Map to standard handicap increments
        standard_lines = [-2.0, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25,
                          0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]
        nearest = min(standard_lines, key=lambda x: abs(x - theory_hc))

        return {
            'home_elo': round(r_home, 1),
            'away_elo': round(r_away, 1),
            'elo_diff': round(elo_diff, 1),
            'theory_handicap': round(theory_hc, 2),
            'standard_line': nearest,
            'expected_goals_home': round(max(0.3, 1.10 + elo_diff / 560.0), 2),
            'expected_goals_away': round(max(0.3, 1.10 - elo_diff / 560.0), 2),
        }

    def s1_screen(
        self,
        bookmaker_line: float,
        home_team: str,
        away_team: str,
    ) -> dict:
        """S1 screening: compare bookmaker line vs ELO theory line.

        If deviation > 0.5 goals, flag as potential anomaly ("可能有鬼").

        Args:
            bookmaker_line: Bookmaker Asian handicap (negative = away favorite).
            home_team: Home team name.
            away_team: Away team name.

        Returns:
            Dict with deviation, flag status, and analysis.
        """
        theory = self.theory_handicap(home_team, away_team)
        deviation = bookmaker_line - theory['theory_handicap']
        abs_deviation = abs(deviation)

        flagged = abs_deviation > 0.5
        severity = (
            'CRITICAL' if abs_deviation > 1.0
            else 'HIGH' if abs_deviation > 0.75
            else 'WARNING' if abs_deviation > 0.5
            else 'OK'
        )

        # Interpretation
        if deviation > 0:
            interpretation = (
                f"机构盘口比理论盘口更{'深' if bookmaker_line > 0 else '浅'}"
                f"（机构看好{'主队' if bookmaker_line < 0 else '客队'}超过{abs_deviation:.1f}球）"
            )
        else:
            interpretation = (
                f"机构盘口比理论盘口更{'浅' if bookmaker_line > 0 else '深'}"
                f"（机构低估{'主队' if bookmaker_line > 0 else '客队'}{abs_deviation:.1f}球）"
            )

        return {
            'home_team': home_team,
            'away_team': away_team,
            'bookmaker_line': bookmaker_line,
            'theory_handicap': theory['theory_handicap'],
            'elo_diff': theory['elo_diff'],
            'deviation': round(deviation, 2),
            'abs_deviation': round(abs_deviation, 2),
            'flagged': flagged,
            'severity': severity,
            'interpretation': interpretation,
            'expected_goals': {
                'home': theory['expected_goals_home'],
                'away': theory['expected_goals_away'],
            },
        }

    def team_comparison(self, team_a: str, team_b: str) -> dict:
        """Compare two teams across all metrics."""
        r_a = self.get_rating(team_a)
        r_b = self.get_rating(team_b)
        p_a = self.get_profiles(team_a)
        p_b = self.get_profiles(team_b)

        return {
            'team_a': {
                'name': team_a,
                'elo': round(r_a, 1),
                'matches': p_a.matches if p_a else 0,
                'win_pct': round(p_a.win_pct() * 100, 1) if p_a else 'N/A',
                'goal_diff': p_a.goal_diff() if p_a else 0,
            },
            'team_b': {
                'name': team_b,
                'elo': round(r_b, 1),
                'matches': p_b.matches if p_b else 0,
                'win_pct': round(p_b.win_pct() * 100, 1) if p_b else 'N/A',
                'goal_diff': p_b.goal_diff() if p_b else 0,
            },
            'elo_diff': round(r_a - r_b, 1),
        }

    def save(self, path: Optional[str] = None) -> None:
        """Persist ratings and profiles to a JSON file."""
        save_path = path or self.elo_db_path
        if not save_path:
            return

        data = {
            'ratings': {k: round(v, 1) for k, v in self.ratings.items()},
            'profiles': {k: asdict(v) for k, v in self.profiles.items()},
        }
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Load ratings and profiles from a JSON file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self.ratings.update(data.get('ratings', {}))
            for name, prof in data.get('profiles', {}).items():
                self.profiles[name] = TeamProfile(**prof)


# ─── CLI entry point ───

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ELO rating system for football")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # match command
    p_match = sub.add_parser("match", help="Record a match result")
    p_match.add_argument("home_team")
    p_match.add_argument("away_team")
    p_match.add_argument("home_goals", type=int)
    p_match.add_argument("away_goals", type=int)
    p_match.add_argument("--importance", type=float, default=1.0)

    # handicap command
    p_hc = sub.add_parser("handicap", help="Calculate theory handicap")
    p_hc.add_argument("home_team")
    p_hc.add_argument("away_team")

    # screen command
    p_sc = sub.add_parser("screen", help="S1 screening: bookmaker vs theory")
    p_sc.add_argument("bookmaker_line", type=float)
    p_sc.add_argument("home_team")
    p_sc.add_argument("away_team")

    # rating command
    p_ra = sub.add_parser("rating", help="Get team rating")
    p_ra.add_argument("team")

    args = parser.parse_args()
    ts = TeamStrength()

    if args.cmd == "match":
        result = ts.update_result(
            args.home_team, args.away_team,
            args.home_goals, args.away_goals,
            args.importance
        )
        print(f"Result: {result['home_team']} {result['score']} {result['away_team']}")
        print(f"ELO: {result['home_team']} {result['r_home_before']}→{result['r_home_after']} "
              f"({result['delta_home']:+.1f}) | "
              f"{result['away_team']} {result['r_away_before']}→{result['r_away_after']} "
              f"({result['delta_away']:+.1f})")

    elif args.cmd == "handicap":
        hc = ts.theory_handicap(args.home_team, args.away_team)
        print(f"Theory Handicap: {args.home_team} vs {args.away_team}")
        print(f"  ELO: {hc['home_elo']} vs {hc['away_elo']} (diff: {hc['elo_diff']:+.1f})")
        print(f"  Theory HC: {hc['theory_handicap']:+.2f} (standard: {hc['standard_line']:+.2f})")
        print(f"  Expected goals: {hc['expected_goals_home']} vs {hc['expected_goals_away']}")

    elif args.cmd == "screen":
        alert = ts.s1_screen(args.bookmaker_line, args.home_team, args.away_team)
        print(f"S1 Screening: {args.home_team} vs {args.away_team}")
        print(f"  Bookmaker line:  {args.bookmaker_line:+.2f}")
        print(f"  Theory line:     {alert['theory_handicap']:+.2f}")
        print(f"  Deviation:       {alert['deviation']:+.2f}")
        print(f"  Severity:        {alert['severity']}")
        print(f"  Flagged:         {alert['flagged']}")
        print(f"  Interpretation:  {alert['interpretation']}")

    elif args.cmd == "rating":
        rating = ts.get_rating(args.team)
        print(f"{args.team} ELO rating: {rating:.1f}")