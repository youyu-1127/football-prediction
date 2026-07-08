#!/usr/bin/env python3
"""
🧮 机构量化系统 v1.0
====================
独立的机构量化分析模块，四大引擎完整实现。

使用方式:
  from utils.institution_quant import InstitutionQuant
  
  iq = InstitutionQuant()
  result = iq.analyze("曼城", "利物浦", odds={"home": 1.55, "draw": 4.0, "away": 6.5})
"""

import math
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field


# ======================== ELO 数据库 ========================

ELO_DATABASE = {
    '曼城': 2100, 'man city': 2100,
    '皇马': 2080, 'real madrid': 2080,
    '拜仁': 2070, 'bayern': 2070,
    '巴萨': 2050, 'barcelona': 2050, '埃尔切': 2050,
    '伯恩利': 2050, '西班牙': 2050,
    '阿森纳': 2050, '兵工厂': 2050, '枪手': 2050,
    '利物浦': 2030, '法国': 2030, '英格兰': 2020,
    '切尔西': 1980, '蓝军': 1980,
    '国际米兰': 1980, '国米': 1980,
    '巴黎圣曼': 2020, '大巴黎': 2020, 'psg': 2020,
    '尤文图斯': 1970, '尤文': 1970, '斑马': 1970,
    '多特蒙德': 1960, '多特': 1960,
    '曼联': 1950,
    '那不勒斯': 1920, '那不': 1920,
    '热刺': 1920,
    '马竞': 2000,
    '本菲卡': 1920,
    '里昂': 1920,
    '阿贾克斯': 1920,
    '比利时': 1930,
    '葡萄牙': 1920,
    '克罗地亚': 1900,
    '勒沃库森': 1895, '药厂': 1895,
    '波尔图': 1900,
    '巴西': 2000,
    '阿根廷': 1980,
    '意大利': 1950,
    '德国': 1970,
    '荷兰': 1940,
    '哥伦比亚': 1840,
    '乌拉圭': 1890,
    '墨西哥': 1850,
    '美国': 1790,
    '日本': 1830,
    '韩国': 1800,
    '澳大利亚': 1770,
    '尼日利亚': 1700,
    '塞内加尔': 1730,
    '摩洛哥': 1750,
    '伊朗': 1680,
    '埃及': 1710,
    '沙特': 1650,
    '挪威': 1740,
    '瑞典': 1820,
    '丹麦': 1850,
    '瑞士': 1800,
    '奥地利': 1730,
    '苏格兰': 1710,
    '希腊': 1720,
    '土耳其': 1790,
    '乌克兰': 1750,
    '捷克': 1750,
    '波兰': 1770,
    '塞尔维亚': 1810,
    '罗马尼亚': 1730,
    '俄罗斯': 1700,
    '智利': 1840,
    '秘鲁': 1700,
    '巴拉圭': 1680,
    '厄瓜多尔': 1720,
    '加拿大': 1700,
    '哥斯达黎加': 1650,
    '洪都拉斯': 1600,
    '巴拿马': 1630,
    '刚果': 1620,
    '刚果金': 1620,
    '加纳': 1680,
    '科特迪瓦': 1700,
    '突尼斯': 1700,
    '阿尔及利亚': 1700,
    '马里': 1600,
    '喀麦隆': 1690,
    '南非': 1600,
    '埃塞俄比亚': 1500,
    '肯尼亚': 1500,
    '乌干达': 1500,
    '赞比亚': 1500,
    '津巴布韦': 1500,
    '布基纳法索': 1580,
    '贝宁': 1600,
    '尼日尔': 1500,
    '几内亚': 1500,
    '智利大学': 1520,
}


def get_elo(team_name: str) -> float:
    """查找球队 ELO 评分"""
    key = team_name.lower().strip()
    if key in ELO_DATABASE:
        return ELO_DATABASE[key]
    for db_key, elo in ELO_DATABASE.items():
        if db_key in key or key in db_key:
            return elo
    return 1500


def elo_to_xg(home_elo: float, away_elo: float) -> Tuple[float, float]:
    """ELO 评分 → 预期进球 (xG)"""
    elo_diff = home_elo - away_elo
    xg_diff = elo_diff / 400.0
    home_adv = 0.0
    hxg = round(max(0.4, 1.35 + xg_diff + home_adv), 2)
    axg = round(max(0.4, 1.35 - xg_diff - home_adv), 2)
    return (hxg, axg)


# ======================== [1] xG 引擎 ========================

class XGEngine:
    """xG 引擎: ELO → xG → 泊松概率"""
    
    @staticmethod
    def poisson_pmf(k: float, lam: float) -> float:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    
    @staticmethod
    def score_matrix(hxg: float, axg: float, mg: int = 10) -> Dict[Tuple[int, int], float]:
        return {
            (i, j): XGEngine.poisson_pmf(i, hxg) * XGEngine.poisson_pmf(j, axg)
            for i in range(mg + 1)
            for j in range(mg + 1)
        }
    
    @staticmethod
    def margin_probs(hxg: float, axg: float, mg: int = 10) -> Dict[int, float]:
        out = {}
        for (i, j), p in XGEngine.score_matrix(hxg, axg, mg).items():
            out[i - j] = out.get(i - j, 0.0) + p
        return out
    
    @staticmethod
    def total_probs(hxg: float, axg: float, mg: int = 10) -> Dict[int, float]:
        out = {}
        for (i, j), p in XGEngine.score_matrix(hxg, axg, mg).items():
            out[i + j] = out.get(i + j, 0.0) + p
        return out
    
    @staticmethod
    def outcome_probs(hxg: float, axg: float, mg: int = 10) -> Tuple[float, float, float]:
        m = XGEngine.margin_probs(hxg, axg, mg)
        return (
            sum(p for k, p in m.items() if k > 0),
            m.get(0, 0.0),
            sum(p for k, p in m.items() if k < 0)
        )
    
    @staticmethod
    def totals(hxg: float, axg: float, line: float, mg: int = 10) -> Tuple[float, float]:
        t = XGEngine.total_probs(hxg, axg, mg)
        return (
            sum(p for k, p in t.items() if k > line),
            sum(p for k, p in t.items() if k < line)
        )
    
    @staticmethod
    def btts(hxg: float, axg: float, mg: int = 10) -> Tuple[float, float]:
        yes = sum(
            p for (i, j), p in XGEngine.score_matrix(hxg, axg, mg).items()
            if i >= 1 and j >= 1
        )
        return (yes, 1 - yes)
    
    @staticmethod
    def top_scores(hxg: float, axg: float, n: int = 6, mg: int = 10) -> List[Tuple[Tuple[int, int], float]]:
        return sorted(
            XGEngine.score_matrix(hxg, axg, mg).items(),
            key=lambda kv: kv[1],
            reverse=True
        )[:n]
    
    def get_xg(self, home_team: str, away_team: str) -> Tuple[float, float]:
        elo_h = get_elo(home_team)
        elo_a = get_elo(away_team)
        return elo_to_xg(elo_h, elo_a)
    
    def get_all_probs(self, home_team: str, away_team: str) -> Dict:
        hxg, axg = self.get_xg(home_team, away_team)
        h, d, w = self.outcome_probs(hxg, axg)
        o25, u25 = self.totals(hxg, axg, 2.5)
        by, bn = self.btts(hxg, axg)
        scores = self.top_scores(hxg, axg, 6)
        return {
            'hxg': hxg, 'axg': axg,
            'home_prob': round(h, 4), 'draw_prob': round(d, 4), 'away_prob': round(w, 4),
            'over_25': round(o25, 4), 'under_25': round(u25, 4),
            'btts_yes': round(by, 4), 'top_scores': scores,
        }


# ======================== [2] Leg 引擎 ========================

class LegEngine:
    """Leg 引擎: 概率定位 + 置信区间"""
    
    @staticmethod
    def calculate_theory_odds(probs: Dict[str, float]) -> Dict[str, float]:
        return {k: round(1.0 / v, 2) if v > 0.01 else 99.9 for k, v in probs.items()}
    
    @staticmethod
    def calculate_confidence_interval(probs: Dict[str, float], alpha: float = 0.95) -> Dict:
        ci = {}
        for k, p in probs.items():
            se = math.sqrt(p * (1 - p) / 1000)
            margin = 1.96 * se
            ci[k] = {
                'prob': p,
                'ci_lower': round(max(0, p - margin), 4),
                'ci_upper': round(min(1, p + margin), 4),
                'se': round(se, 4)
            }
        return ci
    
    @staticmethod
    def identify_league_class(home_elo: float, away_elo: float) -> str:
        diff = abs(home_elo - away_elo)
        if diff > 300: return '顶级联赛'
        elif diff > 200: return '强队对决'
        elif diff > 100: return '中上对决'
        else: return '势均力敌'
    
    @staticmethod
    def classify_match_strength(home_elo: float, away_elo: float) -> Dict:
        diff = home_elo - away_elo
        if diff > 150: return {'type': '强弱分明', 'favor': 'home', 'margin': '大'}
        elif diff > 50: return {'type': '主队稍强', 'favor': 'home', 'margin': '中'}
        elif diff > -50: return {'type': '势均力敌', 'favor': 'draw', 'margin': '小'}
        elif diff > -150: return {'type': '客队稍强', 'favor': 'away', 'margin': '中'}
        else: return {'type': '强弱分明', 'favor': 'away', 'margin': '大'}


# ======================== [3] 漏洞引擎 ========================

class LeakEngine:
    """漏洞引擎: EV/Kelly 检测市场偏差"""
    
    @staticmethod
    def remove_vig(odds: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        raw = {k: 1.0 / v for k, v in odds.items()}
        s = sum(raw.values())
        fair = {k: v / s for k, v in raw.items()}
        return fair, s - 1.0
    
    @staticmethod
    def calc_ev(prob: float, odds: float) -> float:
        return prob * odds - 1.0
    
    @staticmethod
    def detect_leaks(
        model_probs: Dict[str, float],
        market_odds: Dict[str, float],
        threshold: float = 0.02
    ) -> Dict:
        fair_odds, margin = LeakEngine.remove_vig(market_odds)
        leaks = {}
        for k in ('home', 'draw', 'away'):
            m = model_probs[k]
            mo = market_odds[k]
            fo = fair_odds[k]
            ev = LeakEngine.calc_ev(m, mo)
            if abs(ev) > threshold:
                leaks[k] = {
                    'model_prob': round(m, 4),
                    'market_odd': mo,
                    'fair_odd': round(fo, 2),
                    'edge': round((m - fo), 4),
                    'ev': round(ev, 4),
                    'direction': '+EV' if ev > 0 else '-EV',
                }
        has_value = any(l['direction'] == '+EV' for l in leaks.values())
        return {
            'margin': round(margin * 100, 1),
            'fair_odds': {k: round(v, 2) for k, v in fair_odds.items()},
            'leaks': leaks,
            'has_value': has_value,
            'summary': '有正EV机会' if has_value else '无明显漏洞',
        }


# ======================== [4] Kelly 引擎 ========================

class KellyEngine:
    """Kelly 引擎: 仓位管理"""
    
    @staticmethod
    def kelly_stake(prob: float, odds: float, fraction: float = 0.25, max_stake_pct: float = 0.02) -> float:
        b = odds - 1.0
        if b <= 0 or prob <= 0:
            return 0.0
        f = (b * prob - (1 - prob)) / b
        f = max(0.0, f) * fraction
        return round(min(f, max_stake_pct), 4)
    
    @staticmethod
    def kelly_all_outcomes(
        probs: Dict[str, float],
        odds: Dict[str, float],
        fraction: float = 0.25,
        bankroll: float = 10000
    ) -> Dict:
        results = {}
        for k in ('home', 'draw', 'away'):
            stake_pct = KellyEngine.kelly_stake(probs[k], odds[k], fraction)
            ev = LeakEngine.calc_ev(probs[k], odds[k])
            results[k] = {
                'prob': round(probs[k], 4),
                'odd': odds[k],
                'stake_pct': stake_pct,
                'stake_units': round(stake_pct * bankroll, 0),
                'ev': round(ev, 4),
                'recommended': ev > 0.02,
            }
        return results


# ======================== [5] 信号引擎 ========================

class SignalEngine:
    """信号引擎: 让球/大小球分析"""
    
    @staticmethod
    def asian_handicap(hxg: float, axg: float, line: float, side: str, odds: float) -> Dict:
        mp = XGEngine.margin_probs(hxg, axg)
        sign = 1 if side == "home" else -1
        win = push = lose = 0.0
        for margin, prob in mp.items():
            net = margin * sign + line
            if net > 0.01: win += prob
            elif net < -0.01: lose += prob
            else: push += prob
        ev = win * (odds - 1) - lose * 1.0
        return {
            'line': line, 'side': side,
            'win_prob': round(win, 4), 'push_prob': round(push, 4), 'lose_prob': round(lose, 4),
            'ev': round(ev, 4), 'kelly': round(KellyEngine.kelly_stake(win, odds), 4),
            'recommendation': '推荐' if ev > 0.02 else ('避免' if ev < -0.02 else '中性'),
        }
    
    @staticmethod
    def total_goals(hxg: float, axg: float, line: float, side: str, odds: float) -> Dict:
        tp = XGEngine.total_probs(hxg, axg)
        if side == 'over':
            over_prob = sum(p for k, p in tp.items() if k > line)
        else:
            over_prob = 1 - sum(p for k, p in tp.items() if k > line)
        ev = LeakEngine.calc_ev(over_prob, odds)
        return {
            'line': line, 'side': side,
            'prob': round(over_prob, 4), 'ev': round(ev, 4),
            'kelly': round(KellyEngine.kelly_stake(over_prob, odds), 4),
            'recommendation': '推荐' if ev > 0.02 else ('避免' if ev < -0.02 else '中性'),
        }


# ======================== 主类 ========================

@dataclass
class QuantResult:
    """机构量化结果"""
    home_team: str
    away_team: str
    match_type: str
    elo_home: float
    elo_away: float
    xg: Tuple[float, float]
    probs: Dict
    leg_info: Dict
    market_leak: Optional[Dict] = None
    kelly: Optional[Dict] = None
    asian_handicap: Optional[Dict] = None
    total_goals: Optional[Dict] = None
    conclusion: str = ""


class InstitutionQuant:
    """机构量化系统 v1.0 - 独立模块"""
    
    def __init__(self):
        self.xg_engine = XGEngine()
        self.leg_engine = LegEngine()
        self.leak_engine = LeakEngine()
        self.kelly_engine = KellyEngine()
        self.signal_engine = SignalEngine()
    
    def analyze(
        self, home_team: str, away_team: str,
        odds: Optional[Dict[str, float]] = None,
        asian_line: Optional[float] = None, asian_side: Optional[str] = None, asian_odds: Optional[float] = None,
        total_line: Optional[float] = None, total_side: Optional[str] = None, total_odds: Optional[float] = None,
    ) -> QuantResult:
        """一键分析: 完整机构量化流程"""
        
        # Step 1: xG 引擎
        xg = self.xg_engine.get_all_probs(home_team, away_team)
        
        # Step 2: Leg 引擎
        probs = {'home': xg['home_prob'], 'draw': xg['draw_prob'], 'away': xg['away_prob']}
        theory = self.leg_engine.calculate_theory_odds(probs)
        ci = self.leg_engine.calculate_confidence_interval(probs)
        strength = self.leg_engine.classify_match_strength(get_elo(home_team), get_elo(away_team))
        leg_info = {
            'model_probs': probs, 'theory_odds': theory,
            'confidence_interval': ci, 'match_strength': strength,
            'league_class': self.leg_engine.identify_league_class(get_elo(home_team), get_elo(away_team)),
        }
        
        # Step 3-5: 漏洞/Kelly/信号
        market_leak = self.leak_engine.detect_leaks(probs, odds) if odds else None
        kelly = self.kelly_engine.kelly_all_outcomes(probs, odds) if odds else None
        asian_handicap = self.signal_engine.asian_handicap(xg['hxg'], xg['axg'], asian_line, asian_side, asian_odds) if asian_line else None
        total_goals = self.signal_engine.total_goals(xg['hxg'], xg['axg'], total_line, total_side, total_odds) if total_line else None
        
        conclusion = self._generate_conclusion(probs, odds, market_leak, kelly)
        
        return QuantResult(
            home_team, away_team, strength['type'],
            get_elo(home_team), get_elo(away_team),
            (xg['hxg'], xg['axg']), xg, leg_info,
            market_leak, kelly, asian_handicap, total_goals, conclusion,
        )
    
    def _generate_conclusion(self, probs, odds, market_leak, kelly) -> str:
        if odds is None:
            return "仅计算模型概率，无赔率数据"
        best_dir = max(probs, key=probs.get)
        label_map = {'home': '主胜', 'draw': '平局', 'away': '客胜'}
        label = label_map[best_dir]
        ev_detail = market_leak['leaks'] if market_leak else {}
        pos_ev = [k for k, v in ev_detail.items() if v['direction'] == '+EV']
        kelly_recs = []
        if kelly:
            for k, v in kelly.items():
                if v['recommended']:
                    kelly_recs.append(f"{label_map[k]} Kelly {v['stake_pct']*100:.1f}%")
        if pos_ev:
            return f"✅ 方向: {label} | 正EV: {', '.join(f'{label_map[k]}+EV' for k in pos_ev)}"
        elif kelly_recs:
            return f"⚠️ 方向: {label} | Kelly推荐: {', '.join(kelly_recs)}"
        else:
            return f"❌ 无明显 +EV 机会 | 方向: {label} (模型概率 {probs[best_dir]*100:.1f}%)"
    
    def quick_xg(self, home_team: str, away_team: str) -> Tuple[float, float]:
        return self.xg_engine.get_xg(home_team, away_team)
    
    def quick_probs(self, home_team: str, away_team: str) -> Dict:
        return self.xg_engine.get_all_probs(home_team, away_team)


# ======================== 便捷函数 ========================

def institution_quant_analyze(home: str, away: str, odds: Optional[Dict] = None, **kwargs) -> QuantResult:
    """便捷入口"""
    return InstitutionQuant().analyze(home, away, odds=odds, **kwargs)


if __name__ == "__main__":
    iq = InstitutionQuant()
    hxg, axg = iq.quick_xg("曼城", "利物浦")
    print(f"曼城 xG: {hxg}, 利物浦 xG: {axg}")
    probs = iq.quick_probs("曼城", "利物浦")
    print(f"概率: {probs}")
    result = iq.analyze("曼城", "利物浦", odds={"home": 1.55, "draw": 4.0, "away": 6.5})
    print(f"\n结论: {result.conclusion}")
    print(f"ELO: 曼城 {result.elo_home} vs 利物浦 {result.elo_away}")
    print(f"市场 margin: {result.market_leak['margin']}%")
    print(f"有正EV: {result.market_leak['has_value']}")
    if result.market_leak:
        for k, v in result.market_leak['leaks'].items():
            print(f"  {k}: EV={v['ev']:+.4f}, {v['direction']}")
