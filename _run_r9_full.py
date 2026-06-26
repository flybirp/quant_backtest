"""Run R9 backtest on full pool."""
import sys, time, yaml, os, json
sys.path.insert(0, '.')
from backend.main import _config_from_dict
from backend.backtest_engine import run_backtest

codes = [f.replace('.csv','') for f in sorted(os.listdir('/Users/flybirp/Documents/mainland_data_2014')) if f.endswith('.csv')]
print(f'Full pool: {len(codes)} stocks', flush=True)

with open('strategies/rule/恐慌错杀-狙击_r9.yaml') as f:
    raw = yaml.safe_load(f)
raw['stock_pool'] = codes
c = _config_from_dict(raw)

t0 = time.time()
r = run_backtest(c, '2014-01-01', '2025-12-31')
t = time.time() - t0

d = r.__dict__
print(f'R9全量: EV={d["expected_value"]:.2f}% Win={d["win_rate"]:.1f}% Trade={d["total_trades"]} Time={t:.0f}s', flush=True)

json.dump({
    'trades': [x.__dict__ if hasattr(x, '__dict__') else x for x in r.trades],
    'equity_curve': r.equity_curve,
    'summary': {k: v for k, v in d.items() if k not in ('trades', 'equity_curve', 'annual_returns', 'monthly_returns')}
}, open('results/恐慌错杀-狙击_r9_全量.json', 'w'), default=str)
print('Saved.', flush=True)
