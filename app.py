#!/usr/bin/env python3
"""
🏆 Football Prediction Dashboard — V3.0 Web Interface

Interactive dashboard combining all 4 core algorithms:
- Poisson Distribution (单泊松)
- Dixon-Coles Double-Poisson (双泊松 + 低分修正)
- Kelly Criterion (凯利仓位管理)
- ELO Rating (动态评分 + 理论盘口)
- Bayesian Credible Interval (贝叶斯可信区间)

Usage:
    pip install streamlit numpy scipy
    streamlit run app.py --server.port 8502
"""

import streamlit as st
import numpy as np
from datetime import datetime

from utils.poisson_model import PoissonDistribution
from utils.dc_model import DixonColesModel, MatchRecord
from utils.money_management import MoneyManagement
from utils.team_strength import TeamStrength
from utils.bayesian_model import BayesianFootballModel

# ─── Page Config ───

st.set_page_config(
    page_title="⚽ Football Prediction V3.0",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚽ Football Prediction System V3.0")
st.caption("Poisson + Dixon-Coles + Kelly + ELO + Bayesian")

# ─── Sidebar ───

st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox(
    "Prediction Model",
    ["Poisson (Baseline)", "Dixon-Coles (Recommended)", "Bayesian (Uncertainty)"]
)
kelly_fraction = st.sidebar.selectbox(
    "Kelly Fraction",
    ["1/4 Kelly (Conservative)", "1/2 Kelly (Recommended)", "Full Kelly (Aggressive)"]
)
confidence = st.sidebar.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)

# ─── Match Input ───

st.header("📋 Match Input")
col1, col2 = st.columns(2)
with col1:
    home_team = st.text_input("Home Team", value="曼城")
    home_xg = st.number_input("Home xG", value=1.65, min_value=0.1, step=0.05)
with col2:
    away_team = st.text_input("Away Team", value="利物浦")
    away_xg = st.number_input("Away xG", value=0.80, min_value=0.1, step=0.05)

st.subheader("Bookmaker Odds (Decimal)")
odds_col1, odds_col2, odds_col3 = st.columns(3)
with odds_col1:
    odds_home = st.number_input("Home (1)", value=1.55, min_value=1.01, step=0.01)
with odds_col2:
    odds_draw = st.number_input("Draw (X)", value=4.00, min_value=1.01, step=0.01)
with odds_col3:
    odds_away = st.number_input("Away (2)", value=6.50, min_value=1.01, step=0.01)

st.subheader("Over/Under Odds")
o1, o2 = st.columns(2)
with o1:
    odds_over = st.number_input("Over 2.5", value=1.90, min_value=1.01, step=0.01)
with o2:
    odds_under = st.number_input("Under 2.5", value=1.90, min_value=1.01, step=0.01)

st.subheader("Bankroll")
bankroll = st.number_input("Total Bankroll (¥)", value=10000, min_value=100, step=100)

# ─── Run Button ───

if st.button("🚀 Run Prediction", type="primary"):
    # ─── 1. Poisson Model ───

    st.header("📊 1. Poisson Distribution Model")
    p_model = PoissonDistribution(home_xg, away_xg)
    outcome = p_model.outcome()
    totals = p_model.totals(2.5)
    btts = p_model.btts()
    top_scores = p_model.top_scores(6)

    # 1X2
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Home Win", f"{outcome['home']*100:.1f}%")
    with c2:
        st.metric("Draw", f"{outcome['draw']*100:.1f}%")
    with c3:
        st.metric("Away Win", f"{outcome['away']*100:.1f}%")

    # O/U
    st.divider()
    st.subheader("Over/Under 2.5")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Over 2.5", f"{totals['over']*100:.1f}%")
    with c2:
        st.metric("Under 2.5", f"{totals['under']*100:.1f}%")

    # BTTS
    st.subheader("BTTS")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("BTTS Yes", f"{btts['yes']*100:.1f}%")
    with c2:
        st.metric("BTTS No", f"{btts['no']*100:.1f}%")

    # Top Scores
    st.subheader("Top 6 Scorelines")
    score_table = np.array([[sc[0], f"{sc[1]*100:.1f}%"] for sc in top_scores])
    st.table(score_table)

    # ─── 2. Value Edge Detection ───

    st.header("🔍 2. Value Edge Detection (S1 Screening)")
    edge = p_model.value_edge(
        {"home": odds_home, "draw": odds_draw, "away": odds_away},
        edge_threshold=0.125
    )

    if edge["has_value"]:
        st.warning(f"⚠️ Value detected! Edges in: {edge['flagged']}")
        for k in edge["flagged"]:
            v = edge["edges"][k]
            st.info(f"**{k}**: Edge {v['edge']*100:+.1f}%, EV {v['ev_per_unit']:+.1%}")
    else:
        st.success("✅ No significant value edges detected")

    st.write(f"**Market Margin:** {edge['market_margin']*100:.1f}%")

    # ─── 3. Kelly Calculator ───

    st.header("💰 3. Kelly Criterion (Bankroll Management)")
    frac_map = {"1/4 Kelly (Conservative)": 0.25, "1/2 Kelly (Recommended)": 0.5, "Full Kelly (Aggressive)": 1.0}
    frac = frac_map[kelly_fraction]

    mm = MoneyManagement(bankroll=bankroll, max_single_stake_pct=0.02)

    st.subheader("Home Win")
    k_home = mm.kelly_stake(outcome["home"], odds_home, frac)
    e_home = mm.edge_check(outcome["home"], odds_home)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Stake (% of bankroll)", f"{k_home['applied_stake_pct']*100:.2f}%")
    with c2:
        st.metric("Stake (¥)", f"{k_home['applied_stake_units']:.2f}")
    with c3:
        st.metric(f"Recommendation", e_home["recommendation"])

    st.subheader("Away Win")
    k_away = mm.kelly_stake(outcome["away"], odds_away, frac)
    e_away = mm.edge_check(outcome["away"], odds_away)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Stake (% of bankroll)", f"{k_away['applied_stake_pct']*100:.2f}%")
    with c2:
        st.metric("Stake (¥)", f"{k_away['applied_stake_units']:.2f}")
    with c3:
        st.metric(f"Recommendation", e_away["recommendation"])

    # Portfolio
    bets = []
    if k_home["applied_stake_pct"] > 0:
        bets.append({"team": home_team, "prob": outcome["home"], "odds": odds_home, "stake_pct": k_home["applied_stake_pct"]})
    if k_away["applied_stake_pct"] > 0:
        bets.append({"team": away_team, "prob": outcome["away"], "odds": odds_away, "stake_pct": k_away["applied_stake_pct"]})

    if bets:
        st.subheader("Portfolio Risk")
        pr = mm.portfolio_risk(bets)
        st.metric("Total Exposure", f"{pr['total_exposure_pct']:.1f}%")
        st.metric("Risk Level", pr["risk_level"])

    # ─── 4. ELO Comparison ───

    st.header("📈 4. ELO Rating & Theory Handicap")
    ts = TeamStrength()

    # Quick ratings
    r_home = ts.get_rating(home_team)
    r_away = ts.get_rating(away_team)
    c1, c2 = st.columns(2)
    with c1:
        st.metric(f"{home_team} ELO", f"{r_home:.0f}")
    with c2:
        st.metric(f"{away_team} ELO", f"{r_away:.0f}")

    # Theory handicap
    theory = ts.theory_handicap(home_team, away_team)
    st.subheader("Theory Handicap")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("ELO Diff", f"{theory['elo_diff']:+.1f}")
    with c2:
        st.metric("Theory HC", f"{theory['theory_handicap']:+.2f}")
    with c3:
        st.metric("Standard Line", theory["standard_line"])

    # S1 Screening
    st.subheader("S1 Screening (Bookmaker vs Theory)")
    bm_line = -0.75  # default, user can change
    s1 = ts.s1_screen(bm_line, home_team, away_team)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Bookmaker Line", f"{bm_line:+.2f}")
    with c2:
        st.metric("Theory Line", f"{s1['theory_handicap']:+.2f}")
    with c3:
        severity = "⚠️" if s1["flagged"] else "✅"
        st.metric("Deviation", f"{s1['deviation']:+.2f} {severity}")

    # ─── 5. Dixon-Coles (if selected) ───

    if model_choice == "Dixon-Coles (Recommended)":
        st.header("🎲 5. Dixon-Coles Double-Poisson")
        dc_model = DixonColesModel(half_life_days=730)
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
            dc_model.add_match(MatchRecord(d, h, a, hs, as_))
        dc_model.fit(verbose=False)
        dc_result = dc_model.predict(home_team, away_team)
        st.write(dc_result.summary())

    # ─── 6. Bayesian (if selected) ───

    if model_choice == "Bayesian (Uncertainty)":
        st.header("🔮 6. Bayesian Credible Intervals")
        bayes = BayesianFootballModel(method="empirical")
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
            bayes.add_match(MatchRecord(d, h, a, hs, as_))
        bayes.fit(verbose=False)
        result = bayes.predict(home_team, away_team, alpha=confidence)
        for k, v in result.items():
            if isinstance(v, tuple):
                st.metric(k, f"{v[0]:.3f} — {v[1]:.3f} ({result['confidence_level']})")
            else:
                st.metric(k, v)

    # ─── Summary ───

    st.divider()
    st.header("📋 Final Summary")
    st.write("**Poisson 1X2:** Home {:.0f}% | Draw {:.0f}% | Away {:.0f}%".format(
        outcome["home"]*100, outcome["draw"]*100, outcome["away"]*100))
    st.write("**Dixon-Coles (if available):** See above")
    st.write("**Bayesian (if available):** See above")
    st.write(f"**Kelly Stake (1X2):** Home ¥{k_home['applied_stake_units']:.0f} | Away ¥{k_away['applied_stake_units']:.0f}")
    st.write(f"**Portfolio Risk:** {mm.portfolio_risk(bets)['risk_level']}")