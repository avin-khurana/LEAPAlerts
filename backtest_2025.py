#!/usr/bin/env python3
"""
LEAPS Backtest 2025
Runs the same signal logic as scanner.py but:
  - Entry dates filtered to calendar year 2025
  - Forward horizons (3M/6M/12M) included only when that exit date is <= today
  - Upper loop bound set to len(df)-1 so entries near year-end aren't silently dropped
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

from scanner import (
    TICKERS, RISK_FREE_RATE, IV_RANK_MAX, PROFIT_TARGET,
    fix_cols, calc_indicators, bs_greeks,
    profit_target_strategy,
    _CSS, _SORT_JS,
)

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance not installed — run: pip install yfinance")


def backtest_ticker_2025(ticker):
    """Signal logic identical to scanner.backtest_ticker, but:
       - only emits entries whose entry_date falls in 2025
       - only emits a horizon row when i+fwd < len(df)  (no future exits)
    """
    trades = []
    try:
        df  = fix_cols(yf.download(ticker, period='5y', progress=False, auto_adjust=True))
        if len(df) < 460:
            return trades
        ind = calc_indicators(df)
        c, rv = ind['c'], ind['rv']

        # walk-forward IV rank
        ivr_arr = np.full(len(rv), np.nan)
        for i in range(252, len(rv)):
            sub = rv.iloc[max(0, i - 252):i + 1].dropna()
            if len(sub) < 30:
                continue
            lo, hi = float(sub.min()), float(sub.max())
            ivr_arr[i] = 0 if hi == lo else (float(sub.iloc[-1]) - lo) / (hi - lo) * 100

        # Full range — no -252 cutoff; we check per-horizon below
        for i in range(210, len(df) - 1):
            entry_date = str(df.index[i].date())
            if not entry_date.startswith('2025'):
                continue                       # only 2025 entries

            price  = float(c.iloc[i])
            rsi_v  = float(ind['rsi'].iloc[i])
            s200   = float(ind['sma200'].iloc[i])
            hc     = float(ind['hist'].iloc[i])
            hp     = float(ind['hist'].iloc[i - 1])
            vol_i  = float(ind['v'].iloc[i])
            v50_i  = float(ind['vol50'].iloc[i])
            ivr_i  = ivr_arr[i]

            if any(np.isnan([rsi_v, s200, hc, hp, v50_i, ivr_i])) or v50_i == 0:
                continue
            if not (price > s200 and 38 <= rsi_v <= 62 and
                    hc > 0 > hp and vol_i > v50_i * 1.15 and ivr_i < IV_RANK_MAX):
                continue

            curr_iv = float(rv.iloc[i]) if not np.isnan(float(rv.iloc[i])) else 0.30
            K = round(price / 5) * 5

            for fwd in [63, 126, 252]:
                if i + fwd >= len(df):          # exit not yet reached
                    continue
                T_entry = fwd / 252
                g_in = bs_greeks(price, K, T_entry, RISK_FREE_RATE, curr_iv)
                if g_in is None or g_in['price'] <= 0:
                    continue
                ep  = g_in['price']
                xp  = float(c.iloc[i + fwd])
                xiv = float(rv.iloc[i + fwd]) if not np.isnan(float(rv.iloc[i + fwd])) else curr_iv
                if fwd == 252:
                    xprem = max(xp - K, 0.0)
                else:
                    T_rem = (252 - fwd) / 252
                    gx    = bs_greeks(xp, K, T_rem, RISK_FREE_RATE, xiv)
                    xprem = gx['price'] if gx else max(xp - K, 0.0)

                trades.append(dict(
                    ticker=ticker,
                    entry_date=entry_date,
                    entry_price=round(price, 2),
                    strike=K,
                    entry_iv=round(curr_iv * 100, 1),
                    iv_rank_entry=round(ivr_i, 1),
                    entry_prem=round(ep, 2),
                    horizon_days=fwd,
                    exit_price=round(xp, 2),
                    exit_prem=round(xprem, 2),
                    stock_ret=round((xp - price) / price * 100, 1),
                    pnl_pct=round((xprem - ep) / ep * 100, 1),
                    win=(xprem - ep) / ep > 0,
                ))
    except Exception as e:
        print(f"    error {ticker}: {e}")
    return trades


def stats(sub):
    if sub.empty:
        return {}
    n   = len(sub)
    wr  = sub['win'].mean() * 100
    av  = sub['pnl_pct'].mean()
    sh  = av / sub['pnl_pct'].std() if sub['pnl_pct'].std() > 0 else 0
    bu  = (sub['pnl_pct'] < -80).sum()
    return dict(n=n, wr=wr, av=av, sh=sh, bu=bu, bu_pct=bu / n * 100,
                bst=sub['pnl_pct'].max(), wst=sub['pnl_pct'].min(),
                ast=sub['stock_ret'].mean(), md=sub['pnl_pct'].median())


def generate_2025_html(all_trades, run_date):
    """Custom backtest HTML that shows every horizon (3M/6M/12M) in the trades table."""
    if not all_trades:
        return "<html><body style='background:#0d1117;color:#c9d1d9'>No 2025 backtest data.</body></html>"

    df = pd.DataFrame(all_trades)

    def sc(v):
        c = "pos" if v > 0 else "neg"
        return f'<span class="{c}">{v:+.1f}%</span>'

    # ── summary stat cards ──────────────────────────────────────────────────
    def stat_card(label, s, color):
        if not s:
            return ""
        return f"""
<div style="background:#161b22;border:1px solid #30363d;border-left:4px solid {color};
     border-radius:8px;padding:20px;flex:1;min-width:220px">
  <div style="color:#8b949e;font-size:11px;text-transform:uppercase;
       letter-spacing:1px;margin-bottom:12px">{label}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px">
    <div><div style="color:#8b949e;font-size:11px">Signals</div>
         <div style="font-size:20px;font-weight:bold">{s['n']}</div></div>
    <div><div style="color:#8b949e;font-size:11px">Win Rate</div>
         <div style="font-size:20px;font-weight:bold;
              color:{'#3fb950' if s['wr']>=60 else '#d29922'}">{s['wr']:.0f}%</div></div>
    <div><div style="color:#8b949e;font-size:11px">Avg P&L</div>{sc(s['av'])}</div>
    <div><div style="color:#8b949e;font-size:11px">Sharpe</div>
         <div style="color:{'#3fb950' if s['sh']>=1 else '#d29922'}">{s['sh']:.2f}</div></div>
    <div><div style="color:#8b949e;font-size:11px">Best</div>{sc(s['bst'])}</div>
    <div><div style="color:#8b949e;font-size:11px">Worst</div>{sc(s['wst'])}</div>
    <div><div style="color:#8b949e;font-size:11px">Blow-ups (&lt;-80%)</div>
         <div style="color:{'#d29922' if s['bu_pct']>0 else '#3fb950'}">{s['bu_pct']:.1f}%</div></div>
    <div><div style="color:#8b949e;font-size:11px">Avg Stock</div>{sc(s['ast'])}</div>
  </div>
</div>"""

    s3  = stats(df[df['horizon_days'] == 63])
    s6  = stats(df[df['horizon_days'] == 126])
    s12 = stats(df[df['horizon_days'] == 252])

    pt_df = profit_target_strategy(df)
    spt   = stats(pt_df.rename(columns={'final_pnl': 'pnl_pct'})) if not pt_df.empty else {}

    cards = (stat_card("3-Month Hold (63d)", s3, "#00b0ff") +
             stat_card("6-Month Hold (126d)", s6, "#ffd700") +
             stat_card("12-Month Hold (252d)", s12, "#ff9800") +
             stat_card(f"Profit-Target ({PROFIT_TARGET:.0f}%+ exit at 3M)", spt, "#00c853"))

    # ── per-ticker table (best available horizon per ticker) ─────────────────
    tkr_rows = ""
    for tkr in sorted(df['ticker'].unique()):
        for fwd, lbl in [(63, '3M'), (126, '6M'), (252, '12M')]:
            sub = df[(df['ticker'] == tkr) & (df['horizon_days'] == fwd)]
            if sub.empty:
                continue
            wr  = sub['win'].mean()
            ap  = sub['pnl_pct'].mean()
            ast = sub['stock_ret'].mean()
            wr_c = "#3fb950" if wr >= 0.6 else "#d29922"
            ap_c = "#3fb950" if ap > 0 else "#f85149"
            ast_c = "#3fb950" if ast > 0 else "#f85149"
            entries = ", ".join(sorted(sub['entry_date'].unique()))
            tkr_rows += f"""<tr>
              <td style="font-weight:bold">{tkr}</td>
              <td>{lbl}</td>
              <td>{len(sub)}</td>
              <td style="color:{wr_c}">{wr*100:.0f}%</td>
              <td style="color:{ap_c}">{ap:+.1f}%</td>
              <td style="color:{ast_c}">{ast:+.1f}%</td>
              <td style="font-size:12px;color:#8b949e">{entries}</td>
            </tr>"""

    # ── all trades table (every row, every horizon) ──────────────────────────
    horizon_label = {63: '3M', 126: '6M', 252: '12M'}
    horizon_color = {63: '#00b0ff', 126: '#ffd700', 252: '#ff9800'}
    trade_rows = ""
    for _, row in df.sort_values(['ticker', 'entry_date', 'horizon_days']).iterrows():
        pc  = "#3fb950" if row['pnl_pct'] > 0 else "#f85149"
        sc2 = "#3fb950" if row['stock_ret'] > 0 else "#f85149"
        hc  = horizon_color[row['horizon_days']]
        hl  = horizon_label[row['horizon_days']]
        trade_rows += f"""<tr>
          <td style="font-weight:bold">{row['ticker']}</td>
          <td>{row['entry_date']}</td>
          <td><span style="background:{hc};color:#000;padding:2px 8px;
               border-radius:10px;font-size:11px;font-weight:bold">{hl}</span></td>
          <td>${row['entry_price']:,.2f}</td>
          <td>${row['strike']:,.0f}</td>
          <td>{row['entry_iv']:.0f}%</td>
          <td>{row['iv_rank_entry']:.0f}</td>
          <td>${row['entry_prem']:.2f}</td>
          <td>${row['exit_price']:,.2f}</td>
          <td style="color:{sc2}">{row['stock_ret']:+.1f}%</td>
          <td style="color:{pc}">{row['pnl_pct']:+.1f}%</td>
          <td>{'✓' if row['win'] else '✗'}</td>
        </tr>"""

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LEAPS Backtest 2025</title>
<style>{_CSS}
.cards{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px}}
.disc{{background:#1c1007;border:1px solid #d29922;border-radius:8px;padding:16px;
       color:#d29922;font-size:13px;line-height:1.7;margin-bottom:24px}}
</style></head><body>
<h1>LEAPS Backtest — 2025 Signals &nbsp;<small style="color:#8b949e;font-size:14px">{run_date}</small></h1>

<div class="disc">
⚠ <strong>Assumptions:</strong> IV proxied by 30-day realized vol (real premiums cost more → real returns
are lower than shown). ATM strikes only. No bid-ask spread (add ~5-10% friction per trade).
Walk-forward: no lookahead bias. 12M horizon data only available for signals entered before ~May 2025.
Past performance does not predict future results.
</div>

<h2>Strategy Comparison — All Horizons</h2>
<div class="cards">{cards}</div>

<h2>Per-Ticker Breakdown (all horizons)</h2>
<div class="sec"><table id="tkrt">
  <thead><tr>
    <th onclick="sortTable('tkrt',0)">Ticker</th>
    <th onclick="sortTable('tkrt',1)">Horizon</th>
    <th onclick="sortTable('tkrt',2)">Signals</th>
    <th onclick="sortTable('tkrt',3)">Win Rate</th>
    <th onclick="sortTable('tkrt',4)">Avg Opt P&L</th>
    <th onclick="sortTable('tkrt',5)">Avg Stock Ret</th>
    <th>Entry Dates</th>
  </tr></thead>
  <tbody>{tkr_rows}</tbody>
</table></div>

<h2>All Trades — Every Horizon (sorted by Ticker → Date → Horizon)</h2>
<div class="sec"><table id="all">
  <thead><tr>
    <th onclick="sortTable('all',0)">Ticker</th>
    <th onclick="sortTable('all',1)">Entry Date</th>
    <th onclick="sortTable('all',2)">Horizon</th>
    <th onclick="sortTable('all',3)">Entry Price</th>
    <th onclick="sortTable('all',4)">Strike</th>
    <th onclick="sortTable('all',5)">Entry IV</th>
    <th onclick="sortTable('all',6)">IV Rank</th>
    <th onclick="sortTable('all',7)">Premium</th>
    <th onclick="sortTable('all',8)">Exit Price</th>
    <th onclick="sortTable('all',9)">Stock Ret</th>
    <th onclick="sortTable('all',10)">Opt P&L</th>
    <th>Win</th>
  </tr></thead>
  <tbody>{trade_rows}</tbody>
</table></div>

<div class="foot">LEAPS Scanner v3.0 &nbsp;|&nbsp; {run_date} &nbsp;|&nbsp;
2025 Walk-Forward Backtest &nbsp;|&nbsp; Not financial advice.</div>
<script>{_SORT_JS}</script></body></html>"""


def main():
    run_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"\n{'═'*72}")
    print(f"   LEAPS BACKTEST — 2025 ENTRIES ONLY  —  {run_date}")
    print(f"{'═'*72}\n  Downloading 5-year history and scanning for 2025 signals...\n")

    all_trades = []
    for t in TICKERS:
        print(f"    {t:<6}...", end='', flush=True)
        trades = backtest_ticker_2025(t)
        all_trades.extend(trades)
        counts = {fwd: sum(1 for x in trades if x['horizon_days'] == fwd) for fwd in [63, 126, 252]}
        unique = len(set(tr['entry_date'] for tr in trades))
        print(f" {unique} signals  (3M={counts[63]} 6M={counts[126]} 12M={counts[252]})")

    if not all_trades:
        print("\n  No 2025 signals found — criteria were not met for any ticker this year.")
        sys.exit(0)

    df = pd.DataFrame(all_trades)
    unique_signals = df.drop_duplicates(subset=['ticker', 'entry_date'])
    print(f"\n  Total unique 2025 signals : {len(unique_signals)}")
    print(f"  Entry date range          : {df['entry_date'].min()} → {df['entry_date'].max()}")

    print(f"\n{'═'*72}")
    print("  2025 BACKTEST SUMMARY")
    print(f"{'═'*72}")
    for fwd, lbl in [(63, '3M'), (126, '6M'), (252, '12M')]:
        sub = df[df['horizon_days'] == fwd]
        if sub.empty:
            print(f"  {lbl}: no completed exits yet")
            continue
        s = stats(sub)
        print(f"  {lbl}: n={s['n']}  WR={s['wr']:.0f}%  Avg={s['av']:+.1f}%  "
              f"Sharpe={s['sh']:.2f}  Best={s['bst']:+.0f}%  Worst={s['wst']:+.0f}%")

    pt = profit_target_strategy(df)
    if not pt.empty:
        ppt = pt.rename(columns={'final_pnl': 'pnl_pct'})
        s   = stats(ppt)
        n3  = (pt['exit_rule'] == 'exit_3M_profit').sum()
        print(f"  ProfitTarget: n={s['n']}  WR={s['wr']:.0f}%  Avg={s['av']:+.1f}%  "
              f"Sharpe={s['sh']:.2f}  ({n3} exited early at 3M)")

    print(f"\n  Per-ticker breakdown (12M where available):")
    sub12 = df[df['horizon_days'] == 252]
    if not sub12.empty:
        grp = (sub12.groupby('ticker')
               .agg(n=('win', 'count'), wr=('win', 'mean'), ap=('pnl_pct', 'mean'),
                    ast=('stock_ret', 'mean'))
               .round(1).sort_values('ap', ascending=False))
        for tkr, row in grp.iterrows():
            print(f"    {tkr:<6}  n={int(row['n'])}  WR={row['wr']*100:.0f}%  "
                  f"AvgOpt={row['ap']:+.1f}%  AvgStock={row['ast']:+.1f}%")

    # Save outputs
    html_title = f"{run_date} (2025 entries only)"
    html = generate_2025_html(all_trades, html_title)
    with open('leaps_backtest_2025.html', 'w') as f:
        f.write(html)
    df.to_csv('leaps_backtest_2025.csv', index=False)

    print(f"\n  Saved → leaps_backtest_2025.html")
    print(f"  Saved → leaps_backtest_2025.csv")
    print(f"{'═'*72}\n")


if __name__ == '__main__':
    main()
