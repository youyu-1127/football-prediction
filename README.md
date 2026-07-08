# ⚽ Football Prediction System V3.0

> **6 大核心算法 · 可直接集成 · 5 套系统隔离跑**

## 🏛️ 核心算法模块

| 模块 | 文件 | 说明 |
|------|------|------|
| **Poisson 泊松模型** | `utils/poisson_model.py` | 1X2/O-U/BTTS 概率 + S1 初筛价值检测 |
| **Dixon-Coles 双泊松** | `utils/dc_model.py` | 低分修正 + 时间衰减 + 环境修正 + 负二项 |
| **Bayesian 可信区间** | `utils/bayesian_model.py` | 后验不确定性估计 |
| **Kelly 仓位管理** | `utils/money_management.py` | 1/2/1/4 Kelly + 2% 熔断 + 组合风控 |
| **ELO 评分系统** | `utils/team_strength.py` | 动态评分 + 理论盘口 + S1 初筛报警 |

## 🚀 快速开始

```bash
# 安装依赖
pip install numpy scipy

# 使用 Poisson 模型
python3 -c "
from utils.poisson_model import PoissonDistribution
m = PoissonDistribution(1.65, 0.80)
print(m.summary())
v = m.value_edge({'home':1.55,'draw':4.0,'away':6.5})
print('Value edges:', v['flagged'])
"

# 使用 Dixon-Coles 模型
python3 -c "
from utils.dc_model import DixonColesModel, MatchRecord
model = DixonColesModel(half_life_days=730)
model.add_match(MatchRecord('2024-03-10', '曼城', '利物浦', 2, 1))
# ... 灌更多历史数据
model.fit()
print(model.predict('曼城', '利物浦').summary())
"

# 使用 ELO 初筛
python3 -c "
from utils.team_strength import TeamStrength
ts = TeamStrength()
ts.update_result('曼城', '利物浦', 2, 1)
print(ts.theory_handicap('曼城', '利物浦'))
print(ts.s1_screen(-0.75, '曼城', '利物浦'))
"

# 使用 Kelly 算仓
python3 -c "
from utils.money_management import MoneyManagement
mm = MoneyManagement(bankroll=10000, max_single_stake_pct=0.02)
k = mm.kelly_stake(prob=0.55, odds=1.80, fraction=0.25)
print(k)
"
```

## Streamlit 交互式 Dashboard

```bash
pip install streamlit
streamlit run app.py
```

## 📊 算法对比

```
Poisson:       独立双泊松, 快速, 适合初筛
Dixon-Coles:   双泊松 + 低分修正 + 环境修正, 更精确
Bayesian:      后验可信区间, 量化不确定性
ELO:           动态评分 + 理论盘口, 初筛"可能有鬼"
Kelly:         1/4 Kelly 仓位, 2% 硬性熔断
```

## 📁 参考项目

- [turingism/worldcup-predictor](https://github.com/turingism/worldcup-predictor) — Dixon-Coles 双泊松
- [AmirMotefaker/ai-football-prediction-engine](https://github.com/AmirMotefaker/ai-football-prediction-engine-world-cup-2026) — 环境修正
- [tdfarrell/betting-stake-optimizer](https://github.com/tdfarrell/betting-stake-optimizer) — Kelly 计算器

## 📜 License

MIT