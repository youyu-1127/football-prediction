#!/usr/bin/env python3
"""
🏆 足球预测系统 V3.0 回测引擎
================================

整合 5 大系统对历史数据进行回测验证:
1. Poisson 泊松模型
2. Dixon-Coles 双泊松模型  
3. ELO 评分系统
4. Kelly 仓位管理
5. Bayesian 可信区间

数据来源: football-data.co.uk (英超、西甲、德甲等主流联赛)

Usage:
  python3 backtest_v3.py --league EPL --season 2024  # 历史CSV回测
  python3 backtest_v3.py --demo                        # 演示模式
  python3 backtest_v3.py --live "曼城 vs 利物浦"       # 实时预测
"""

import os
import sys
import math
import json
import argparse
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

# 导入我们的核心算法
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.poisson_model import PoissonDistribution
from utils.dc_model import DixonColesModel, MatchRecord
from utils.team_strength import TeamStrength
from utils.money_management import MoneyManagement

# ─── 配置 ───

LEAGUE_URLS = {
    "EPL": "eng-premier-league",
    "L1": "fra-ligue-1",
    "BL1": "ger-bundesliga",
    "SA": "ita-serie-a",
    "P1": "esp-primera-division",
    "E1": "eng-premier-league",
}

FOOTBALL_DATA_URLS = {
    "EPL": "https://www.football-data.co.uk/data.php?league=E1&season=2024",
    "E2": "https://www.football-data.co.uk/data.php?league=E2&season=2024",
    "L1": "https://www.football-data.co.uk/data.php?league=L1&season=2024",
    "BL1": "https://www.football-data.co.uk/data.php?league=D1&season=2024",
    "SA": "https://www.football-data.co.uk/data.php?league=I1&season=2024",
    "P1": "https://www.football-data.co.uk/data.php?league=ES1&season=2024",
}


# ─── 1. Poisson 系统 ───

def poisson_system(home_xg, away_xg):
    """Poisson 单泊松模型"""
    model = PoissonDistribution(home_xg, away_xg)
    outcome = model.outcome()
    totals = model.totals(2.5)
    btts = model.btts()
    top = model.top_scores(6)
    return {
        "system": "Poisson",
        "xg_home": home_xg, "xg_away": away_xg,
        "home": outcome["home"], "draw": outcome["draw"], "away": outcome["away"],
        "over_25": totals["over"], "under_25": totals["under"],
        "btts_yes": btts["yes"],
        "top_scores": top,
    }


# ─── 2. Dixon-Coles 系统 ───

def dc_system(home, away, history=None):
    """Dixon-Coles 双泊松模型"""
    model = DixonColesModel(half_life_days=730)
    if history:
        for m in history:
            model.add_match(m)
    model.fit(verbose=False)
    return model.predict(home, away)


# ─── 3. ELO 系统 ───

def elo_system(home_team, away_team, bookmaker_line=0.0):
    """ELO 评分 + 理论盘口 + S1 初筛"""
    ts = TeamStrength()
    r_home = ts.get_rating(home_team)
    r_away = ts.get_rating(away_team)
    theory = ts.theory_handicap(home_team, away_team)
    s1 = ts.s1_screen(bookmaker_line, home_team, away_team)
    return {
        "system": "ELO",
        "elo_home": round(r_home, 0),
        "elo_away": round(r_away, 0),
        "elo_diff": round(theory["elo_diff"], 1),
        "theory_handicap": theory["theory_handicap"],
        "s1_deviation": s1["deviation"],
        "s1_flagged": s1["flagged"],
    }


# ─── 4. Kelly 仓位 ───

def kelly_system(prob, odds, bankroll=10000):
    """Kelly 仓位管理"""
    mm = MoneyManagement(bankroll, max_single_stake_pct=0.02)
    k = mm.kelly_stake(prob, odds, 0.25)  # 1/4 Kelly
    e = mm.edge_check(prob, odds)
    return {
        "system": "Kelly",
        "stake_pct": k["applied_stake_pct"],
        "stake_units": k["applied_stake_units"],
        "edge": e["edge"],
        "ev_per_unit": e["ev_per_unit"],
        "recommendation": e["recommendation"],
    }


# ─── 5. Bayesian 系统 ───

def bayesian_system(home, away, history=None, alpha=0.95):
    """Bayesian 可信区间"""
    from utils.bayesian_model import BayesianFootballModel, MatchRecord as BMatchRecord
    model = BayesianFootballModel(method="empirical")
    if history:
        for m in history:
            model.add_match(BMatchRecord(m.date, m.home_team, m.away_team,
                                         m.home_goals, m.away_goals))
    model.fit(verbose=False)
    result = model.predict(home, away, alpha=alpha)
    return result


# ─── 数据获取 ───

def fetch_historical_data(league="EPL", season="2024"):
    """从 football-data.co.uk 获取历史数据"""
    key = league.upper()
    if key not in FOOTBALL_DATA_URLS:
        print(f"⚠️ 联赛 {key} 不在支持列表: {list(FOOTBALL_DATA_URLS.keys())}")
        return []

    url = FOOTBALL_DATA_URLS[key]
    try:
        import requests
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            lines = resp.text.strip().split('\n')
            reader = __import__('csv').DictReader(lines)
            return list(reader)
    except Exception as e:
        print(f"⚠️ 抓取失败: {e}")
    return []


def parse_historical_data(rows):
    """解析 CSV 数据为 MatchRecord"""
    matches = []
    for r in rows:
        try:
            date = r.get('Date', '')
            home = r.get('HomeTeam', '').strip()
            away = r.get('AwayTeam', '').strip()
            hgoals = int(r.get('FTHG', 0))
            agoals = int(r.get('FTAG', 0))
            if date and home and away:
                matches.append(MatchRecord(date, home, away, hgoals, agoals))
        except (ValueError, TypeError):
            continue
    return matches


# ─── 回测引擎 ───

def run_backtest(data, league="EPL", n_max=100):
    """对历史数据运行回测"""
    print(f"📊 开始回测: {len(data)} 场比赛")
    print(f"   联赛: {league}, 模型: Poisson + Dixon-Coles + ELO + Kelly\n")

    # 收集历史数据训练 ELO
    all_matches = parse_historical_data(data)
    model = DixonColesModel(half_life_days=730)
    ts = TeamStrength()

    for m in all_matches:
        model.add_match(m)
        ts.update_result(m.home_team, m.away_team, m.home_goals, m.away_goals)

    model.fit(verbose=False)

    results = []
    for i, row in enumerate(data[:n_max]):
        try:
            home = row.get('HomeTeam', '').strip()
            away = row.get('AwayTeam', '').strip()
            hgoals = int(row.get('FTHG', 0))
            agoals = int(row.get('FTAG', 0))
            date = row.get('Date', '')
            odds_home = float(row.get('B365H', 0) or 0)
            odds_draw = float(row.get('B365D', 0) or 0)
            odds_away = float(row.get('B365A', 0) or 0)

            if not home or not away or hgoals == agoals == 0:
                continue

            # ELO xG
            elo_h = ts.get_rating(home)
            elo_a = ts.get_rating(away)
            xg_h, xg_a = _elo_to_xg(elo_h, elo_a)

            # 1. Poisson
            p_model = PoissonDistribution(xg_h, xg_a)
            p_outcome = p_model.outcome()

            # 2. DC (sample for speed)
            dc_result = dc_system(home, away, all_matches[:min(50, len(all_matches))])

            # 3. ELO
            e_result = elo_system(home, away, -0.5)

            # 4. Kelly
            k_home = kelly_system(p_outcome['home'], odds_home)
            k_away = kelly_system(p_outcome['away'], odds_away)

            # Actual result
            actual = 'home' if hgoals > agoals else ('draw' if hgoals == agoals else 'away')

            # 5. Value edge
            edge = p_model.value_edge({'home': odds_home, 'draw': odds_draw, 'away': odds_away})

            results.append({
                'date': date,
                'home': home,
                'away': away,
                'hxg': xg_h, 'axg': xg_a,
                'p_home': p_outcome['home'], 'p_draw': p_outcome['draw'], 'p_away': p_outcome['away'],
                'actual': actual,
                'hgoals': hgoals, 'agoals': agoals,
                'odds_home': odds_home, 'odds_draw': odds_draw, 'odds_away': odds_away,
                'k_home_stake': k_home['stake_pct'],
                'k_away_stake': k_away['stake_pct'],
                'edge_flagged': edge['has_value'],
                's1_flagged': e_result['s1_flagged'],
            })

            if (i + 1) % 20 == 0:
                print(f"   已处理 {i+1}/{min(n_max, len(data))} 场...")

        except Exception:
            continue

    return results, all_matches


def _elo_to_xg(elo_h, elo_a):
    """ELO 转 xG (简化公式)"""
    diff = (elo_h - elo_a) / 280.0
    league_avg = 1.3
    xg_h = league_avg * math.exp(diff)
    xg_a = league_avg * math.exp(-diff)
    return round(xg_h, 2), round(xg_a, 2)


# ─── 统计汇总 ───

def generate_summary(results):
    """生成回测总结"""
    total = len(results)
    if total == 0:
        return {}

    # 1X2 预测
    correct = 0
    correct_top3 = 0
    for r in results:
        probs = {'home': r['p_home'], 'draw': r['p_draw'], 'away': r['p_away']}
        pred = max(probs, key=probs.get)
        if pred == r['actual']:
            correct += 1
        # 实际结果在前三
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top3 = [p for p, _ in sorted_probs[:3]]
        if r['actual'] in top3:
            correct_top3 += 1

    # 进球数
    correct_goals = 0
    for r in results:
        total_goals = r['hgoals'] + r['agoals']
        # 如果模型概率最高的进球区间匹配实际
        correct_goals += 1  # 简化

    # EV 统计
    ev_total = 0
    ev_count = 0
    for r in results:
        if r['odds_home'] > 0 and r['k_home_stake'] > 0:
            ev_total += (r['p_home'] * r['odds_home'] - 1) * r['k_home_stake']
            ev_count += 1

    avg_ev = ev_total / ev_count if ev_count > 0 else 0

    # Kelly 收益模拟
    bankroll = 10000
    bet_count = 0
    total_profit = 0
    for r in results:
        if r['odds_home'] > 0:
            stake = bankroll * r['k_home_stake']
            if stake > 0:
                bet_count += 1
                if r['actual'] == 'home':
                    profit = stake * (r['odds_home'] - 1)
                else:
                    profit = -stake
                total_profit += profit
                bankroll += profit

    roi = total_profit / 10000 * 100 if bet_count > 0 else 0

    # 边检测统计
    edge_detected = sum(1 for r in results if r['edge_flagged'])
    s1_flagged = sum(1 for r in results if r['s1_flagged'])

    # 打印结果
    print("\n" + "=" * 70)
    print("  📊 回测总结")
    print("=" * 70)
    print(f"  总比赛数: {total}")
    print(f"  1X2 准确率: {correct/total*100:.1f}%")
    print(f"  实际结果在前三: {correct_top3/total*100:.1f}%")
    print(f"  平均 EV: {avg_ev:+.4f}")
    print(f"  Kelly 模拟收益: ¥{total_profit:.0f} (ROI: {roi:+.1f}%)")
    print(f"  投注次数: {bet_count}")
    print(f"  价值边检测: {edge_detected} 场 ({edge_detected/total*100:.1f}%)")
    print(f"  S1 异常报警: {s1_flagged} 场")
    print("=" * 70)

    return {
        'total': total,
        'accuracy_1x2': round(correct/total*100, 1),
        'accuracy_top3': round(correct_top3/total*100, 1),
        'avg_ev': round(avg_ev, 4),
        'kelly_roi': round(roi, 1),
        'kelly_profit': round(total_profit, 0),
        'bet_count': bet_count,
        'edge_detected': edge_detected,
        's1_flagged': s1_flagged,
    }


# ─── 实时预测 ───

def run_live_prediction(home, away, odds_home=None, odds_draw=None, odds_away=None):
    """实时预测单场比赛"""
    print(f"🔮 预测: {home} vs {away}\n")

    # 训练 ELO
    model = DixonColesModel(half_life_days=730)
    ts = TeamStrength()

    # Demo data for training
    demo_matches = [
        MatchRecord("2024-03-10", "曼城", "利物浦", 2, 1),
        MatchRecord("2024-03-03", "曼城", "切尔西", 3, 0),
        MatchRecord("2024-02-25", "利物浦", "阿森纳", 1, 1),
        MatchRecord("2024-02-18", "阿森纳", "曼城", 0, 2),
        MatchRecord("2024-02-10", "利物浦", "切尔西", 2, 0),
        MatchRecord("2024-02-03", "切尔西", "利物浦", 1, 3),
        MatchRecord("2024-01-28", "曼城", "阿森纳", 1, 0),
        MatchRecord("2024-01-20", "利物浦", "曼城", 1, 1),
        MatchRecord("2024-01-15", "阿森纳", "切尔西", 2, 1),
        MatchRecord("2024-01-08", "切尔西", "阿森纳", 0, 2),
        MatchRecord("2024-01-01", "曼城", "利物浦", 2, 2),
        MatchRecord("2023-12-25", "阿森纳", "曼城", 1, 1),
        MatchRecord("2023-12-20", "利物浦", "切尔西", 0, 0),
        MatchRecord("2023-12-15", "曼城", "切尔西", 3, 1),
        MatchRecord("2023-12-10", "利物浦", "阿森纳", 2, 1),
    ]
    for m in demo_matches:
        model.add_match(m)
        ts.update_result(m.home_team, m.away_team, m.home_goals, m.away_goals)
    model.fit(verbose=False)

    # ELO → xG
    elo_h = ts.get_rating(home)
    elo_a = ts.get_rating(away)
    xg_h, xg_a = _elo_to_xg(elo_h, elo_a)

    print(f"📈 1. ELO 评分:")
    print(f"   {home}: {elo_h:.0f}")
    print(f"   {away}: {elo_a:.0f}")
    theory = ts.theory_handicap(home, away)
    print(f"   理论盘口: {theory['theory_handicap']:+.2f}")
    print()

    # Poisson
    print(f"📊 2. Poisson 模型:")
    p_model = PoissonDistribution(xg_h, xg_a)
    outcome = p_model.outcome()
    print(f"   1X2: Home {outcome['home']*100:.1f}% | Draw {outcome['draw']*100:.1f}% | Away {outcome['away']*100:.1f}%")
    print(f"   O/U 2.5: Over {p_model.totals(2.5)['over']*100:.1f}% | Under {p_model.totals(2.5)['under']*100:.1f}%")
    print(f"   BTTS: Yes {p_model.btts()['yes']*100:.1f}% | No {p_model.btts()['no']*100:.1f}%")
    print()

    # Value Edge
    if odds_home and odds_draw and odds_away:
        print(f"🔍 3. 价值边检测:")
        edge = p_model.value_edge({'home': odds_home, 'draw': odds_draw, 'away': odds_away})
        print(f"   市场 margin: {edge['market_margin']*100:.1f}%")
        if edge['has_value']:
            print(f"   ⚠️ 价值边: {edge['flagged']}")
            for k in edge['flagged']:
                v = edge['edges'][k]
                print(f"     {k}: Edge {v['edge']*100:+.1f}%, EV {v['ev_per_unit']:+.1%}")
        else:
            print(f"   ✅ 无显著价值边")
        print()

    # Kelly
    print(f"💰 4. Kelly 仓位:")
    mm = MoneyManagement(10000, max_single_stake_pct=0.02)
    if odds_home:
        k = mm.kelly_stake(outcome['home'], odds_home, 0.25)
        print(f"   Home: ¥{k['applied_stake_units']:.0f} ({k['applied_stake_pct']*100:.2f}%)")
    if odds_away:
        k = mm.kelly_stake(outcome['away'], odds_away, 0.25)
        print(f"   Away: ¥{k['applied_stake_units']:.0f} ({k['applied_stake_pct']*100:.2f}%)")
    print()

    # Top scores
    print(f"🎯 最可能比分:")
    for sc, p in p_model.top_scores(6):
        print(f"   {sc}: {p*100:.1f}%")


# ─── 演示模式 ───

def run_demo():
    """演示模式: 使用模拟数据"""
    print("🎬 演示模式: 使用模拟比赛数据\n")

    # 模拟 20 场比赛
    teams = ["曼城", "利物浦", "阿森纳", "切尔西", "热刺", "纽卡斯尔",
             "布莱顿", "维拉", "狼队", "埃弗顿"]
    import random
    random.seed(42)

    dates = []
    for month in range(1, 10):
        for day in range(3, 28, 5):
            dates.append(f"2024-{month:02d}-{day:02d}")

    matches = []
    for d in dates:
        for i in range(len(teams)):
            for j in range(i+1, min(i+3, len(teams))):
                matches.append({
                    'Date': d,
                    'HomeTeam': teams[i],
                    'AwayTeam': teams[j],
                    'FTHG': random.randint(0, 4),
                    'FTAG': random.randint(0, 4),
                    'B365H': round(random.uniform(1.3, 2.5), 2),
                    'B365D': round(random.uniform(3.5, 4.5), 2),
                    'B365A': round(random.uniform(2.5, 5.0), 2),
                })

    print(f"📊 模拟 {len(matches)} 场比赛\n")
    results, _ = run_backtest(matches, "Demo", min(50, len(matches)))
    summary = generate_summary(results)

    # 实时预测示例
    print("\n🔮 实时预测示例:")
    run_live_prediction("曼城", "利物浦", 1.55, 4.0, 6.5)


# ─── CLI ───

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Prediction V3.0 Backtest")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 历史回测
    p1 = sub.add_parser("历史", help="对历史 CSV 数据回测")
    p1.add_argument("--league", default="EPL", help="联赛代码 (EPL/L1/BL1/SA/P1)")
    p1.add_argument("--season", default="2024", help="赛季")
    p1.add_argument("--n-max", type=int, default=100, help="最大比赛数")

    # 实时预测
    p2 = sub.add_parser("实时", help="实时预测单场")
    p2.add_argument("match", help="对阵 (如 '曼城 vs 利物浦')")
    p2.add_argument("--odds", nargs=3, type=float, help="赔率 (Home Draw Away)")

    # 演示
    p3 = sub.add_parser("演示", help="运行演示模式")

    args = parser.parse_args()

    if args.cmd == "历史":
        data = fetch_historical_data(args.league, args.season)
        if not data:
            print("⚠️ 无法获取历史数据，运行演示模式...")
            run_demo()
        else:
            print(f"📥 获取到 {len(data)} 场比赛")
            results, _ = run_backtest(data, args.league, args.n_max)
            generate_summary(results)

    elif args.cmd == "实时":
        parts = args.match.split(" vs ")
        if len(parts) != 2:
            print("❌ 格式: 实时 '曼城 vs 利物浦'")
            sys.exit(1)
        run_live_prediction(parts[0], parts[1], *args.odds if args.odds else None)

    elif args.cmd == "演示":
        run_demo()