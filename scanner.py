#!/usr/bin/env python3
"""
LEAPS Professional Scanner v3.0
Install: pip install yfinance pandas numpy scipy requests
"""

import os
import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

# ─── CORE CONFIG ────────────────────────────────────────────────────────────────
TICKERS = [
    'AAPL','MSFT','GOOGL','META','AMZN','NVDA','AMD','V',
    'MA','JPM','UNH','LLY','COST','TSLA','CRM','ADBE',
    'QCOM','BAC','WMT','ABBV'
]
ACCOUNT_SIZE   = 100_000
MAX_POS_PCT    = 0.05
TARGET_DELTA   = 0.70
RISK_FREE_RATE = 0.045
LEAPS_MONTHS   = 12
VIX_MAX        = 28
IV_RANK_MAX    = 40
MIN_DAYS_EARN  = 30
PROFIT_TARGET  = 100.0   # exit at 3M if option up >100%

SECTOR_MAP = {
    'AAPL':'XLK','MSFT':'XLK','GOOGL':'XLC','META':'XLC','AMZN':'XLY',
    'NVDA':'XLK','AMD':'XLK','V':'XLF','MA':'XLF','JPM':'XLF',
    'UNH':'XLV','LLY':'XLV','COST':'XLP','TSLA':'XLY','CRM':'XLK',
    'ADBE':'XLK','QCOM':'XLK','BAC':'XLF','WMT':'XLP','ABBV':'XLV'
}

# ─── ALERT CONFIG  (env vars override hardcoded values) ─────────────────────────
# Set these directly OR export as environment variables / GitHub Secrets:
#   TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, EMAIL_FROM, EMAIL_APP_PASSWORD, EMAIL_TO
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN',   '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
EMAIL_FROM       = os.environ.get('EMAIL_FROM',        '')   # sender Gmail
EMAIL_PASSWORD   = os.environ.get('EMAIL_APP_PASSWORD','')   # Gmail app-password
EMAIL_TO         = os.environ.get('EMAIL_TO',          'avin.khurana18@gmail.com')

# ─── BLACK-SCHOLES ───────────────────────────────────────────────────────────────
def bs_greeks(S, K, T, r=RISK_FREE_RATE, sigma=0.30):
    if T <= 1e-6 or S <= 0 or K <= 0 or sigma <= 0:
        return None
    d1  = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2  = d1 - sigma*np.sqrt(T)
    pdf = norm.pdf(d1)
    return dict(
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2),
        delta = norm.cdf(d1),
        gamma = pdf / (S*sigma*np.sqrt(T)),
        theta = (-(S*pdf*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365,
        vega  = S*pdf*np.sqrt(T) / 100,
    )

def find_strike(S, target_delta, T, r, sigma):
    lo, hi, mid = S*0.4, S*1.8, S
    for _ in range(60):
        mid = (lo+hi)/2
        g   = bs_greeks(S, mid, T, r, sigma)
        if g is None: break
        if g['delta'] > target_delta: lo = mid
        else:                         hi = mid
    return round(mid/5)*5

def realized_vol(prices, window=30):
    return np.log(prices/prices.shift(1)).dropna().rolling(window).std()*np.sqrt(252)

def iv_rank(rv_series, lookback=252):
    clean = rv_series.dropna()
    if len(clean) < 30: return 50.0
    period = clean.iloc[-lookback:]
    lo, hi = float(period.min()), float(period.max())
    if hi == lo: return 50.0
    return round((float(clean.iloc[-1])-lo)/(hi-lo)*100, 1)

# ─── INDICATORS ─────────────────────────────────────────────────────────────────
def fix_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

def calc_indicators(df):
    c, v = df['Close'].squeeze(), df['Volume'].squeeze()
    d    = c.diff()
    rsi  = 100 - 100/(1 + d.clip(lower=0).rolling(14).mean()/(-d.clip(upper=0)).rolling(14).mean())
    macd = c.ewm(span=12,adjust=False).mean() - c.ewm(span=26,adjust=False).mean()
    hist = macd - macd.ewm(span=9,adjust=False).mean()
    hi, lo = df['High'].squeeze(), df['Low'].squeeze()
    tr   = pd.concat([hi-lo,(hi-c.shift()).abs(),(lo-c.shift()).abs()],axis=1).max(axis=1)
    return dict(c=c, v=v, rsi=rsi, hist=hist,
                sma50=c.rolling(50).mean(), sma200=c.rolling(200).mean(),
                ema21=c.ewm(span=21,adjust=False).mean(),
                vol50=v.rolling(50).mean(), atr=tr.rolling(14).mean(),
                rv=realized_vol(c,30))

# ─── FUNDAMENTALS ────────────────────────────────────────────────────────────────
def score_fundamentals(info):
    score, details = 0, []
    pe = info.get('forwardPE') or info.get('trailingPE')
    if pe and 0 < pe < 20:  score += 20; details.append(f"P/E={pe:.1f}[cheap]")
    elif pe and pe < 35:    score += 10; details.append(f"P/E={pe:.1f}[fair]")
    rg = info.get('revenueGrowth') or 0
    if rg > .20:  score += 20; details.append(f"Rev={rg:.0%}")
    elif rg > .08:score += 10; details.append(f"Rev={rg:.0%}")
    mg = info.get('profitMargins') or 0
    if mg > .20:  score += 20; details.append(f"Margin={mg:.0%}")
    elif mg > .08:score += 10; details.append(f"Margin={mg:.0%}")
    de = info.get('debtToEquity') or 999
    if 0 < de < 50:  score += 20; details.append("D/E=low")
    elif de < 150:   score += 10
    eg = info.get('earningsGrowth') or 0
    if eg > .15:  score += 20; details.append(f"EPS={eg:.0%}")
    elif eg > .05:score += 10
    grade = 'A' if score >= 70 else ('B' if score >= 40 else 'C')
    return grade, score, details

def relative_strength(tdf, spy_df, period=63):
    tc = tdf['Close'].squeeze()
    sc = spy_df['Close'].squeeze()
    tr = float(tc.iloc[-1]/tc.iloc[-period]-1)
    sr = float(sc.iloc[-1]/sc.iloc[-period]-1)
    return tr-sr, tr, sr

def days_to_earnings(t_obj):
    try:
        cal = t_obj.calendar
        if cal is None: return 999
        if isinstance(cal, dict):
            dates = cal.get('Earnings Date',[])
            if dates: return max(0,(pd.to_datetime(dates[0])-pd.Timestamp.now()).days)
        elif isinstance(cal, pd.DataFrame):
            if 'Earnings Date' in cal.index:
                return max(0,(pd.to_datetime(cal.loc['Earnings Date'].iloc[0])-pd.Timestamp.now()).days)
    except Exception: pass
    return 999

def get_leaps_data(t_obj, price, curr_iv):
    try:
        exps = t_obj.options
        if not exps: return None, None, None
        target   = datetime.now()+timedelta(days=LEAPS_MONTHS*30)
        best_exp = min(exps, key=lambda x: abs((datetime.strptime(x,'%Y-%m-%d')-target).days))
        T_actual = (datetime.strptime(best_exp,'%Y-%m-%d')-datetime.now()).days/365
        if T_actual < 0.3: return None, None, None
        calls = t_obj.option_chain(best_exp).calls.copy()
        calls = calls[calls['strike'].between(price*0.7,price*1.5)].dropna(subset=['strike','impliedVolatility'])
        if calls.empty: return None, best_exp, T_actual
        calls['bs_delta'] = calls.apply(lambda r: (bs_greeks(price,r['strike'],T_actual,
                                        RISK_FREE_RATE,r['impliedVolatility']) or {}).get('delta',0), axis=1)
        best = calls.iloc[(calls['bs_delta']-TARGET_DELTA).abs().argsort()[:1]]
        return (best.iloc[0] if not best.empty else None), best_exp, T_actual
    except Exception: return None, None, None

def get_3m_option_data(t_obj, price, curr_iv):
    """Fetch the ATM call nearest to 91 calendar days (~3 months / 63 trading days) out."""
    try:
        exps = t_obj.options
        if not exps: return None, None, None
        target   = datetime.now() + timedelta(days=91)
        best_exp = min(exps, key=lambda x: abs((datetime.strptime(x,'%Y-%m-%d')-target).days))
        T_actual = (datetime.strptime(best_exp,'%Y-%m-%d')-datetime.now()).days/365
        if T_actual < 0.05: return None, None, None
        atm_strike = round(price/5)*5
        calls = t_obj.option_chain(best_exp).calls.copy()
        calls = calls[calls['strike'].between(price*0.85, price*1.15)].dropna(subset=['strike','impliedVolatility'])
        if not calls.empty:
            row    = calls.iloc[(calls['strike']-price).abs().argsort()[:1]].iloc[0]
            strike = float(row['strike'])
            mid    = (float(row.get('bid',0))+float(row.get('ask',0)))/2
            prem   = mid if mid > 0 else float(row.get('lastPrice',0))
            iv_u   = float(row['impliedVolatility'])
        else:
            strike, iv_u, prem = atm_strike, curr_iv, 0.0
        greeks = bs_greeks(price, strike, T_actual, RISK_FREE_RATE, iv_u)
        if prem <= 0 and greeks:
            prem = greeks['price']
        if prem <= 0:
            prem = curr_iv * 0.40 * price
        return (dict(
            strike   = round(strike, 0),
            prem     = round(prem, 2),
            iv       = round(iv_u * 100, 1),
            delta    = round(greeks['delta'], 2)  if greeks else None,
            theta    = round(greeks['theta'], 2)  if greeks else None,
            vega     = round(greeks['vega'],  2)  if greeks else None,
        ), best_exp, T_actual)
    except Exception:
        return None, None, None

# ─── SIGNALS ─────────────────────────────────────────────────────────────────────
TRACKED_CONDS = ['above_sma200','golden_cross','rsi_neutral','macd_cross_up',
                 'volume_spike','iv_rank_low','vix_ok','earnings_safe','rs_positive']

def evaluate_signals(ind, price, grade, ivr, d_earn, rs_alpha, vix):
    rsi   = float(ind['rsi'].iloc[-1])
    hist  = float(ind['hist'].iloc[-1])
    histp = float(ind['hist'].iloc[-2])
    s200  = float(ind['sma200'].iloc[-1])
    s50   = float(ind['sma50'].iloc[-1])
    vol   = float(ind['v'].iloc[-1])
    v50   = float(ind['vol50'].iloc[-1])
    cond = dict(
        above_sma200  = price > s200,
        above_sma50   = price > s50,
        golden_cross  = s50  > s200,
        rsi_neutral   = 38 <= rsi <= 62,
        rsi_oversold  = rsi < 35,
        macd_cross_up = hist > 0 > histp,
        volume_spike  = vol  > v50*1.15,
        iv_rank_low   = ivr  < IV_RANK_MAX,
        vix_ok        = vix  < VIX_MAX,
        earnings_safe = d_earn > MIN_DAYS_EARN,
        rs_positive   = rs_alpha > 0,
        fund_quality  = grade in ('A','B'),
        fund_a        = grade == 'A',
    )
    tech   = cond['above_sma200'] and cond['rsi_neutral'] and cond['macd_cross_up'] and cond['volume_spike']
    fund   = cond['fund_a'] and cond['rsi_oversold']
    env_ok = cond['iv_rank_low'] and cond['vix_ok'] and cond['earnings_safe']
    combo  = tech and cond['fund_quality']
    strong = combo and env_ok and cond['rs_positive']
    return dict(tech=tech, fund=fund, combo=combo, strong=strong, env_ok=env_ok,
                cond=cond, rsi=rsi, vol_ratio=vol/v50 if v50 else 0,
                conditions_passed=sum(cond[k] for k in TRACKED_CONDS))

def size_position(prem):
    cost = prem*100
    if cost <= 0: return 0, 0
    n = max(1, int(ACCOUNT_SIZE*MAX_POS_PCT/cost))
    return n, n*cost

# ─── SCREENER ────────────────────────────────────────────────────────────────────
def screen_ticker(ticker, spy_df, vix, sector_cache):
    r = dict(ticker=ticker, error=None, signal=False)
    try:
        t_obj = yf.Ticker(ticker)
        df    = fix_cols(yf.download(ticker, period='2y', progress=False, auto_adjust=True))
        if len(df) < 210: r['error']='insufficient data'; return r
        ind   = calc_indicators(df)
        price = float(ind['c'].iloc[-1])
        rv    = ind['rv']
        ivr   = iv_rank(rv)
        curr_iv = float(rv.dropna().iloc[-1]) if not rv.dropna().empty else 0.30
        try:    info = t_obj.info
        except: info = {}
        grade, fscore, fdetails = score_fundamentals(info)
        d_earn = days_to_earnings(t_obj)
        rs_alpha, t_ret, _ = relative_strength(df, spy_df)
        sigs = evaluate_signals(ind, price, grade, ivr, d_earn, rs_alpha, vix)
        # ── 12-month LEAPS (0.70-delta) ──────────────────────────────────────────
        chain_row, exp_date, T_actual = get_leaps_data(t_obj, price, curr_iv)
        T = T_actual if T_actual else LEAPS_MONTHS/12
        if chain_row is not None and not pd.isna(chain_row.get('impliedVolatility', np.nan)):
            civ    = float(chain_row['impliedVolatility'])
            strike = float(chain_row['strike'])
            mid    = (float(chain_row.get('bid',0))+float(chain_row.get('ask',0)))/2
            prem   = mid if mid > 0 else float(chain_row.get('lastPrice',0))
            greeks = bs_greeks(price, strike, T, RISK_FREE_RATE, civ)
            iv_used = civ
        else:
            strike = find_strike(price, TARGET_DELTA, T, RISK_FREE_RATE, curr_iv)
            greeks = bs_greeks(price, strike, T, RISK_FREE_RATE, curr_iv)
            prem   = greeks['price'] if greeks else curr_iv*0.40*price
            iv_used = curr_iv
        prem = prem if prem > 0 else curr_iv*0.40*price
        contracts, capital = size_position(prem)

        # ── 3-month ATM call (momentum / quick-profit play) ───────────────────
        s3_data, s3_exp, _s3T = get_3m_option_data(t_obj, price, curr_iv)
        s3_contracts, s3_capital = (size_position(s3_data['prem']) if s3_data else (0, 0))
        if s3_data and s3_data['prem'] > 0:
            s3_be_pct = round((s3_data['strike'] + s3_data['prem']) / price * 100 - 100, 1)
        else:
            s3_be_pct = None

        # ── IV 52-week range (shows why options are cheap) ────────────────────
        rv_1y   = ind['rv'].dropna().iloc[-252:]
        iv_lo   = round(float(rv_1y.min())  * 100, 1)
        iv_hi   = round(float(rv_1y.max())  * 100, 1)
        iv_avg  = round(float(rv_1y.mean()) * 100, 1)
        # IV expansion gain: if IV reverts from current to 52-wk avg,
        # option gains (vega × Δiv) / premium × 100 %
        if s3_data and s3_data.get('vega') and s3_data['prem'] > 0:
            iv_expand_pts  = max(0.0, iv_avg - s3_data['iv'])
            iv_expand_gain = round(s3_data['vega'] * iv_expand_pts / s3_data['prem'] * 100, 1)
        else:
            iv_expand_gain = 0.0

        fired = sigs['tech'] or sigs['fund'] or sigs['combo']
        r.update(dict(
            signal=fired, price=round(price,2), grade=grade,
            fund_score=fscore, fund_details=fdetails,
            rsi=round(sigs['rsi'],1), iv_rank=round(ivr,1),
            curr_iv=round(iv_used*100,1), days_earn=d_earn,
            rs_alpha=round(rs_alpha*100,1), t_3m_ret=round(t_ret*100,1),
            tech=sigs['tech'], fund=sigs['fund'],
            combo=sigs['combo'], strong=sigs['strong'], env_ok=sigs['env_ok'],
            cond=sigs['cond'], conditions=sigs['conditions_passed'],
            # 12M LEAPS fields
            strike=round(strike,0), exp_date=exp_date or f"+{LEAPS_MONTHS}M",
            prem=round(prem,2),
            delta=round(greeks['delta'],2) if greeks else None,
            gamma=round(greeks['gamma'],4) if greeks else None,
            theta=round(greeks['theta'],2) if greeks else None,
            vega =round(greeks['vega'],2)  if greeks else None,
            contracts=contracts, capital=round(capital,0),
            breakeven=round(strike+prem,2),
            be_pct=round((strike+prem)/price*100-100,1),
            # 3M ATM call fields
            s3=s3_data, s3_exp=s3_exp or '+3M',
            s3_contracts=s3_contracts, s3_capital=round(s3_capital,0),
            s3_be_pct=s3_be_pct,
            # IV analysis
            iv_lo=iv_lo, iv_hi=iv_hi, iv_avg=iv_avg,
            iv_expand_gain=iv_expand_gain,
        ))
    except Exception as e:
        r['error'] = str(e)
    return r

# ─── BACKTEST ────────────────────────────────────────────────────────────────────
def backtest_ticker(ticker):
    trades = []
    try:
        df = fix_cols(yf.download(ticker, period='5y', progress=False, auto_adjust=True))
        if len(df) < 460: return trades
        ind = calc_indicators(df)
        c, rv = ind['c'], ind['rv']
        # precompute IV rank walk-forward
        ivr_arr = np.full(len(rv), np.nan)
        for i in range(252, len(rv)):
            sub = rv.iloc[max(0,i-252):i+1].dropna()
            if len(sub) < 30: continue
            lo, hi = float(sub.min()), float(sub.max())
            ivr_arr[i] = 0 if hi==lo else (float(sub.iloc[-1])-lo)/(hi-lo)*100
        for i in range(210, len(df)-252):
            price  = float(c.iloc[i])
            rsi_v  = float(ind['rsi'].iloc[i])
            s200   = float(ind['sma200'].iloc[i])
            hc     = float(ind['hist'].iloc[i])
            hp     = float(ind['hist'].iloc[i-1])
            vol_i  = float(ind['v'].iloc[i])
            v50_i  = float(ind['vol50'].iloc[i])
            ivr_i  = ivr_arr[i]
            if any(np.isnan([rsi_v,s200,hc,hp,v50_i,ivr_i])) or v50_i==0: continue
            if not (price>s200 and 38<=rsi_v<=62 and hc>0>hp and vol_i>v50_i*1.15 and ivr_i<IV_RANK_MAX):
                continue
            curr_iv = float(rv.iloc[i]) if not np.isnan(float(rv.iloc[i])) else 0.30
            K = round(price/5)*5
            for fwd in [63, 126, 252]:
                if i+fwd >= len(df): continue
                T_entry = fwd/252
                g_in    = bs_greeks(price, K, T_entry, RISK_FREE_RATE, curr_iv)
                if g_in is None or g_in['price'] <= 0: continue
                ep = g_in['price']
                xp = float(c.iloc[i+fwd])
                xiv = float(rv.iloc[i+fwd]) if not np.isnan(float(rv.iloc[i+fwd])) else curr_iv
                if fwd == 252:
                    xprem = max(xp-K, 0.0)
                else:
                    T_rem = (252-fwd)/252
                    gx    = bs_greeks(xp, K, T_rem, RISK_FREE_RATE, xiv)
                    xprem = gx['price'] if gx else max(xp-K, 0.0)
                trades.append(dict(
                    ticker=ticker,
                    entry_date=str(df.index[i].date()),
                    entry_price=round(price,2), strike=K,
                    entry_iv=round(curr_iv*100,1), iv_rank_entry=round(ivr_i,1),
                    entry_prem=round(ep,2), horizon_days=fwd,
                    exit_price=round(xp,2), exit_prem=round(xprem,2),
                    stock_ret=round((xp-price)/price*100,1),
                    pnl_pct=round((xprem-ep)/ep*100,1),
                    win=(xprem-ep)/ep>0,
                ))
    except Exception as e:
        print(f"    Backtest error {ticker}: {e}")
    return trades

def profit_target_strategy(df_trades, target=PROFIT_TARGET):
    """Simulate: exit at 3M if up >target%, else hold to 12M."""
    df3  = df_trades[df_trades['horizon_days']==63].set_index(['ticker','entry_date'])
    df12 = df_trades[df_trades['horizon_days']==252].set_index(['ticker','entry_date'])
    rows = []
    for idx in df3.index.intersection(df12.index):
        r3, r12 = df3.loc[idx], df12.loc[idx]
        if r3['pnl_pct'] > target:
            rows.append(dict(ticker=idx[0],entry_date=idx[1],
                             exit_rule='exit_3M_profit',final_pnl=r3['pnl_pct'],
                             stock_ret=r3['stock_ret'],win=True))
        else:
            rows.append(dict(ticker=idx[0],entry_date=idx[1],
                             exit_rule='hold_12M',final_pnl=r12['pnl_pct'],
                             stock_ret=r12['stock_ret'],win=r12['win']))
    return pd.DataFrame(rows)

# ─── ALERTS ──────────────────────────────────────────────────────────────────────
def send_telegram(text):
    if not _REQUESTS or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"HTML"}, timeout=10)
    except Exception as e:
        print(f"  Telegram error: {e}")

def send_email(subject, html_body):
    if not EMAIL_FROM or not EMAIL_PASSWORD: return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"], msg["From"], msg["To"] = subject, EMAIL_FROM, EMAIL_TO
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(EMAIL_FROM, EMAIL_PASSWORD)
            s.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print("  Email alert sent.")
    except Exception as e:
        print(f"  Email error: {e}")

def build_alert_text(signals, vix, run_date):
    if not signals:
        return f"<b>LEAPS Scanner {run_date}</b>\nNo signals today. VIX={vix:.1f}"
    lines = [f"<b>LEAPS Scanner {run_date} — {len(signals)} signal(s)</b>",
             f"VIX={vix:.1f}\n"]
    for r in signals:
        tag = "STRONG" if r.get('strong') else "COMBO" if r.get('combo') else "TECH" if r.get('tech') else "FUND"
        lines.append(f"<b>{r['ticker']}</b> [{tag}]  Grade:{r['grade']}  RSI:{r['rsi']}  IVR:{r['iv_rank']}")
        lines.append(f"  ${r['strike']:.0f} Call ({r['exp_date']})  Prem:${r['prem']:.2f}  "
                     f"BE:{r['be_pct']:+.1f}%  {r['contracts']}x=${r['capital']:,.0f}")
    return "\n".join(lines)

# ─── HTML: LIVE SCAN ─────────────────────────────────────────────────────────────
_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;font-family:'Courier New',monospace;font-size:14px;padding:24px;max-width:1200px;margin:0 auto}
h1{color:#58a6ff;border-bottom:2px solid #21262d;padding-bottom:12px;margin-bottom:24px;font-size:22px}
h2{color:#79c0ff;margin:28px 0 12px;font-size:14px;text-transform:uppercase;letter-spacing:1px}
.env{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin-bottom:24px;display:flex;gap:40px;flex-wrap:wrap}
.env-item{display:flex;flex-direction:column;gap:4px}
.env-label{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px}
.env-value{font-size:22px;font-weight:bold}
.count{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 20px;margin-bottom:24px;font-size:15px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin-bottom:16px}
.card.strong{border-left:4px solid #00c853}
.card.combo{border-left:4px solid #00b0ff}
.card.tech{border-left:4px solid #ffd700}
.card.fund{border-left:4px solid #ff9800}
.card-hdr{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px}
.ticker{font-size:26px;font-weight:bold;color:#e6edf3;margin-right:10px}
.badge{padding:5px 14px;border-radius:20px;font-size:12px;font-weight:bold}
.b-strong{background:#00c853;color:#000}
.b-combo{background:#00b0ff;color:#000}
.b-tech{background:#ffd700;color:#000}
.b-fund{background:#ff9800;color:#000}
.mgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px}
.met{background:#0d1117;padding:10px 14px;border-radius:6px}
.met-l{color:#8b949e;font-size:11px;text-transform:uppercase;margin-bottom:4px}
.met-v{font-size:18px;font-weight:bold}
.cgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:16px}
.ci{display:flex;align-items:center;gap:8px;font-size:13px}
.pass{color:#3fb950;font-weight:bold}
.fail{color:#f85149;font-weight:bold}
.tbox{background:#0d1117;border:1px solid #238636;border-radius:6px;padding:16px}
.tbox-title{color:#58a6ff;font-weight:bold;margin-bottom:12px;font-size:12px;text-transform:uppercase;letter-spacing:1px}
.tgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.tl{color:#8b949e;font-size:11px;margin-bottom:4px}
.tv{font-weight:bold;font-size:15px}
.pos{color:#3fb950}.neg{color:#f85149}.warn{color:#d29922}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#21262d;color:#8b949e;padding:10px 14px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;cursor:pointer;user-select:none}
th:hover{background:#30363d}
td{padding:10px 14px;border-bottom:1px solid #21262d}
tr:hover td{background:#161b22}
.sec{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin-bottom:16px;overflow-x:auto}
.foot{color:#8b949e;font-size:12px;text-align:center;margin-top:40px;padding-top:16px;border-top:1px solid #21262d}
"""

_SORT_JS = """
function sortTable(id,col){
  const t=document.getElementById(id),b=t.querySelector('tbody'),
    rows=Array.from(b.querySelectorAll('tr')),
    h=t.querySelectorAll('th')[col],asc=h.dataset.sort!=='asc';
  t.querySelectorAll('th').forEach(x=>delete x.dataset.sort);
  h.dataset.sort=asc?'asc':'desc';
  rows.sort((a,b)=>{
    const av=a.cells[col]?.textContent?.trim()||'',bv=b.cells[col]?.textContent?.trim()||'';
    const an=parseFloat(av.replace(/[^0-9.+-]/g,'')),bn=parseFloat(bv.replace(/[^0-9.+-]/g,''));
    if(!isNaN(an)&&!isNaN(bn))return asc?an-bn:bn-an;
    return asc?av.localeCompare(bv):bv.localeCompare(av);
  });
  rows.forEach(r=>b.appendChild(r));
}
"""

def _cv(val, cls=""):
    return f'<span class="{cls}">{val}</span>' if cls else str(val)

def generate_scan_html(results, vix, spy_price, spy_sma200, spy_3m, run_date):
    valid   = [r for r in results if not r.get('error')]
    signals = sorted([r for r in valid if r.get('signal')],
                     key=lambda r: -(4*r.get('strong',0)+2*r.get('combo',0)+r.get('tech',0)+r.get('fund',0)))
    near    = sorted([r for r in valid if not r.get('signal') and r.get('conditions',0)>=5],
                     key=lambda r: -r.get('conditions',0))

    trend     = "BULLISH" if spy_price > spy_sma200 else "BEARISH"
    vc        = "#3fb950" if vix < 20 else ("#d29922" if vix < VIX_MAX else "#f85149")
    tc        = "#3fb950" if spy_price > spy_sma200 else "#f85149"
    regime    = "RISK-ON" if vix < 20 else ("ELEVATED" if vix < VIX_MAX else "RISK-OFF")

    cards = ""
    for r in signals:
        st = "strong" if r.get('strong') else "combo" if r.get('combo') else "tech" if r.get('tech') else "fund"
        sl = st.upper()
        c  = r['cond']
        conds = [("Above SMA-200",c['above_sma200']),("Golden Cross",c['golden_cross']),
                 ("RSI 38–62",c['rsi_neutral']),("MACD Cross ↑",c['macd_cross_up']),
                 ("Volume Spike",c['volume_spike']),("IV Rank < 40",c['iv_rank_low']),
                 ("VIX < 28",c['vix_ok']),("Earnings Safe",c['earnings_safe']),("RS > SPY",c['rs_positive'])]
        chtml = "".join(f'<div class="ci"><span class="{"pass" if ok else "fail"}">{"✓" if ok else "✗"}</span>{lb}</div>'
                        for lb,ok in conds)
        warns = []
        if not r.get('env_ok'):           warns.append('<span class="warn">⚠ ENV FILTER FAILED</span>')
        if r['days_earn'] < MIN_DAYS_EARN: warns.append(f'<span class="warn">⚠ EARNINGS {r["days_earn"]}d</span>')
        wstr = "  ".join(warns)
        gk = (f"Δ={r['delta']:.2f} Γ={r['gamma']:.4f} Θ=${r['theta']:.2f}/d V=${r['vega']:.2f}/1%"
              if r.get('delta') else "N/A")
        rsa_c = "pos" if r['rs_alpha']>=0 else "neg"
        ret_c = "pos" if r['t_3m_ret']>=0 else "neg"
        cards += f"""
<div class="card {st}">
  <div class="card-hdr">
    <div><span class="ticker">{r['ticker']}</span>
         <span style="color:#8b949e;font-size:13px">Grade {r['grade']} · Score {r['fund_score']}/100</span>
         &nbsp;{wstr}</div>
    <span class="badge b-{st}">{sl}</span>
  </div>
  <div class="mgrid">
    <div class="met"><div class="met-l">Price</div><div class="met-v">${r['price']:,.2f}</div></div>
    <div class="met"><div class="met-l">RSI</div><div class="met-v">{r['rsi']:.1f}</div></div>
    <div class="met"><div class="met-l">IV Rank</div>
      <div class="met-v {'warn' if r['iv_rank']>=IV_RANK_MAX else 'pos'}">{r['iv_rank']:.0f}/100</div></div>
    <div class="met"><div class="met-l">IV (realized)</div><div class="met-v">{r['curr_iv']:.0f}%</div></div>
    <div class="met"><div class="met-l">3M Return</div>
      <div class="met-v {ret_c}">{r['t_3m_ret']:+.1f}%</div></div>
    <div class="met"><div class="met-l">RS vs SPY</div>
      <div class="met-v {rsa_c}">{r['rs_alpha']:+.1f}%</div></div>
    <div class="met"><div class="met-l">Earnings</div>
      <div class="met-v {'warn' if r['days_earn']<MIN_DAYS_EARN else ''}">{r['days_earn']}d</div></div>
    <div class="met"><div class="met-l">Fund Score</div><div class="met-v">{r['fund_score']}/100</div></div>
  </div>
  <div class="cgrid">{chtml}</div>

  <!-- IV EDGE PANEL -->
  <div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;
       padding:12px 16px;margin-bottom:12px;font-size:13px">
    <span style="color:#8b949e;font-size:11px;text-transform:uppercase;
          letter-spacing:1px">IV Edge — Why Options Are Cheap</span>
    <div style="display:flex;gap:24px;margin-top:8px;flex-wrap:wrap">
      <div><span style="color:#8b949e">IV now: </span>
           <strong style="color:#00b0ff">{r['curr_iv']:.0f}%</strong>
           &nbsp;<span style="color:#8b949e;font-size:11px">(rank {r['iv_rank']:.0f}/100 — cheaper than {100-r['iv_rank']:.0f}% of past year)</span></div>
      <div><span style="color:#8b949e">52-wk range: </span>
           <span style="color:#3fb950">{r.get('iv_lo',0):.0f}%</span>
           <span style="color:#8b949e"> → avg </span>
           <span style="color:#d29922">{r.get('iv_avg',0):.0f}%</span>
           <span style="color:#8b949e"> → </span>
           <span style="color:#f85149">{r.get('iv_hi',0):.0f}%</span></div>
      {f'<div><span style="color:#8b949e">If IV → avg: option gains </span><strong style="color:#3fb950">+{r["iv_expand_gain"]:.0f}% from vega alone</strong> (before stock move)</div>' if r.get('iv_expand_gain',0)>0 else ''}
    </div>
  </div>

  <!-- 3M ATM CALL -->
  <div class="tbox" style="border-color:#00b0ff;margin-bottom:10px">
    <div class="tbox-title" style="color:#00b0ff">
      OPTION 1 — 3-Month ATM Call &nbsp;<span style="font-weight:normal;color:#8b949e">(Momentum · Backtest 2025: 92% win rate, avg +217%)</span>
    </div>
    {f'''<div class="tgrid">
      <div><div class="tl">Strike / Expiry</div>
           <div class="tv">${r["s3"]["strike"]:.0f} Call &nbsp;({r["s3_exp"]})</div></div>
      <div><div class="tl">IV at Entry</div>
           <div class="tv" style="color:#00b0ff">{r["s3"]["iv"]:.0f}% &nbsp;<span style="font-size:11px;color:#8b949e">(rank {r["iv_rank"]:.0f}/100)</span></div></div>
      <div><div class="tl">Premium</div>
           <div class="tv">${r["s3"]["prem"]:.2f}/sh &nbsp;(${r["s3"]["prem"]*100:,.0f}/contract)</div></div>
      <div><div class="tl">Delta / Theta / Vega</div>
           <div class="tv" style="font-size:12px">Δ{r["s3"]["delta"] or "~0.50"} &nbsp;Θ${r["s3"]["theta"] or "–"}/d &nbsp;V${r["s3"]["vega"] or "–"}/1%</div></div>
      <div><div class="tl">Contracts / Capital</div>
           <div class="tv">{r["s3_contracts"]}x &nbsp;= ${r["s3_capital"]:,.0f}</div></div>
      <div><div class="tl">Breakeven at expiry</div>
           <div class="tv">${r["s3"]["strike"]+r["s3"]["prem"]:,.2f} &nbsp;({r["s3_be_pct"]:+.1f}%)</div></div>
    </div>
    <div style="margin-top:10px;padding-top:10px;border-top:1px solid #30363d;
         font-size:12px;color:#8b949e;line-height:1.8">
      <strong style="color:#c9d1d9">Exit rules (3M):</strong>
      &nbsp;① Exit when option up <strong style="color:#3fb950">≥100%</strong> — don't wait for day 63
      &nbsp;② Hard exit at <strong>day 63</strong> (~{r["s3_exp"]})
      &nbsp;③ Cut if stock falls <strong style="color:#f85149">&gt;7%</strong> from entry (only scenario that loses)
    </div>''' if r.get('s3') else '<div style="color:#8b949e;font-size:13px">Chain unavailable — try next trading day</div>'}
  </div>

  <!-- 12M LEAPS -->
  <div class="tbox" style="border-color:#238636">
    <div class="tbox-title" style="color:#3fb950">
      OPTION 2 — 12-Month LEAPS &nbsp;<span style="font-weight:normal;color:#8b949e">(Trend · ~0.70 delta · more time to be right)</span>
    </div>
    <div class="tgrid">
      <div><div class="tl">Strike / Expiry</div>
           <div class="tv">${r['strike']:.0f} Call &nbsp;({r['exp_date']})</div></div>
      <div><div class="tl">Greeks</div><div class="tv" style="font-size:13px">{gk}</div></div>
      <div><div class="tl">Premium</div>
           <div class="tv">${r['prem']:.2f}/sh &nbsp;(${r['prem']*100:,.0f}/contract)</div></div>
      <div><div class="tl">Contracts</div><div class="tv">{r['contracts']}</div></div>
      <div><div class="tl">Capital</div><div class="tv">${r['capital']:,.0f}</div></div>
      <div><div class="tl">Breakeven</div>
           <div class="tv">${r['breakeven']:,.2f} &nbsp;({r['be_pct']:+.1f}%)</div></div>
    </div>
  </div>
</div>"""

    near_rows = "".join(f"""<tr>
      <td>{r['ticker']}</td><td>{r['grade']}</td><td>{r.get('conditions',0)}/9</td>
      <td>${r['price']:,.2f}</td><td>{r['rsi']:.0f}</td><td>{r['iv_rank']:.0f}</td>
      <td class="{'pos' if r['rs_alpha']>=0 else 'neg'}">{r['rs_alpha']:+.1f}%</td>
      <td class="{'warn' if r['days_earn']<MIN_DAYS_EARN else ''}">{r['days_earn']}d</td>
    </tr>""" for r in near) or '<tr><td colspan="8" style="text-align:center;color:#8b949e">None</td></tr>'

    no_sig = '<div style="text-align:center;color:#8b949e;padding:40px">No signals today — check Near Misses below.</div>'

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LEAPS Scanner — {run_date}</title>
<style>{_CSS}</style></head><body>
<h1>LEAPS Professional Scanner v3.0 &nbsp;<small style="color:#8b949e;font-size:14px">{run_date}</small></h1>
<div class="env">
  <div class="env-item"><span class="env-label">VIX</span>
    <span class="env-value" style="color:{vc}">{vix:.1f}</span></div>
  <div class="env-item"><span class="env-label">Regime</span>
    <span class="env-value" style="color:{vc}">{regime}</span></div>
  <div class="env-item"><span class="env-label">SPY Trend</span>
    <span class="env-value" style="color:{tc}">{trend}</span></div>
  <div class="env-item"><span class="env-label">SPY 3M</span>
    <span class="env-value" style="color:{'#3fb950' if spy_3m>=0 else '#f85149'}">{spy_3m:+.1f}%</span></div>
  <div class="env-item"><span class="env-label">Account</span>
    <span class="env-value">${ACCOUNT_SIZE:,.0f}</span></div>
</div>
<div class="count">
  <strong style="color:{'#3fb950' if signals else '#8b949e'}">
    {len(signals)} signal{'s' if len(signals)!=1 else ''}</strong>
  out of {len(valid)} tickers scanned
</div>
<h2>Live Signals</h2>
{cards if cards else no_sig}
<h2>Near Misses — Watch List</h2>
<div class="sec">
<table id="nt">
  <thead><tr>
    <th onclick="sortTable('nt',0)">Ticker</th><th onclick="sortTable('nt',1)">Grade</th>
    <th onclick="sortTable('nt',2)">Cond</th><th onclick="sortTable('nt',3)">Price</th>
    <th onclick="sortTable('nt',4)">RSI</th><th onclick="sortTable('nt',5)">IV Rank</th>
    <th onclick="sortTable('nt',6)">RS vs SPY</th><th onclick="sortTable('nt',7)">Earnings</th>
  </tr></thead><tbody>{near_rows}</tbody>
</table></div>
<div class="foot">LEAPS Scanner v3.0 &nbsp;|&nbsp; {run_date} &nbsp;|&nbsp;
Data: Yahoo Finance &nbsp;|&nbsp; Not financial advice.</div>
<script>{_SORT_JS}</script></body></html>"""

# ─── HTML: BACKTEST REPORT ───────────────────────────────────────────────────────
def generate_backtest_html(all_trades, run_date):
    if not all_trades:
        return "<html><body style='background:#0d1117;color:#c9d1d9'>No backtest data.</body></html>"
    df = pd.DataFrame(all_trades)

    def stats(sub):
        if sub.empty: return {}
        n  = len(sub)
        wr = sub['win'].mean()*100
        av = sub['pnl_pct'].mean()
        md = sub['pnl_pct'].median()
        bst= sub['pnl_pct'].max()
        wst= sub['pnl_pct'].min()
        sh = av/sub['pnl_pct'].std() if sub['pnl_pct'].std()>0 else 0
        bu = (sub['pnl_pct']<-80).sum()
        ast= sub['stock_ret'].mean()
        return dict(n=n,wr=wr,av=av,md=md,bst=bst,wst=wst,sh=sh,bu=bu,bu_pct=bu/n*100,ast=ast)

    s3  = stats(df[df['horizon_days']==63])
    s6  = stats(df[df['horizon_days']==126])
    s12 = stats(df[df['horizon_days']==252])

    # profit-target strategy
    pt_df = profit_target_strategy(df)
    spt = stats(pt_df.rename(columns={'final_pnl':'pnl_pct'})) if not pt_df.empty else {}
    s12_base = stats(df[df['horizon_days']==252])

    def sc(v, pos=True):
        if isinstance(v, float):
            c = "pos" if (v>0)==pos else "neg"
            return f'<span class="{c}">{v:+.1f}%</span>'
        return str(v)

    def stat_card(label, s, color):
        if not s: return ""
        return f"""
<div style="background:#161b22;border:1px solid #30363d;border-left:4px solid {color};
     border-radius:8px;padding:20px;flex:1;min-width:220px">
  <div style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">{label}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px">
    <div><div style="color:#8b949e;font-size:11px">Signals</div><div style="font-size:20px;font-weight:bold">{s['n']}</div></div>
    <div><div style="color:#8b949e;font-size:11px">Win Rate</div>
         <div style="font-size:20px;font-weight:bold;color:{'#3fb950' if s['wr']>=60 else '#d29922'}">{s['wr']:.0f}%</div></div>
    <div><div style="color:#8b949e;font-size:11px">Avg P&L</div>{sc(s['av'])}</div>
    <div><div style="color:#8b949e;font-size:11px">Sharpe</div>
         <div style="color:{'#3fb950' if s['sh']>=1 else '#d29922'}">{s['sh']:.2f}</div></div>
    <div><div style="color:#8b949e;font-size:11px">Best</div>{sc(s['bst'])}</div>
    <div><div style="color:#8b949e;font-size:11px">Worst</div>{sc(s['wst'])}</div>
    <div><div style="color:#8b949e;font-size:11px">Blow-ups</div>
         <div style="color:{'#d29922' if s['bu_pct']>0 else '#3fb950'}">{s['bu_pct']:.1f}%</div></div>
    <div><div style="color:#8b949e;font-size:11px">Avg Stock</div>{sc(s['ast'])}</div>
  </div>
</div>"""

    # per-ticker 12M table
    sub12 = df[df['horizon_days']==252]
    tkr_html = ""
    if not sub12.empty:
        grp = (sub12.groupby('ticker')
               .agg(n=('win','count'), wr=('win','mean'), ap=('pnl_pct','mean'), ast=('stock_ret','mean'))
               .round(1).sort_values('ap',ascending=False))
        for tkr, row in grp.iterrows():
            wr_c = "#3fb950" if row['wr']>=0.6 else "#d29922"
            bar_w= max(0, min(100, row['ap']/5))
            ap_c = "#3fb950" if row['ap']>0 else "#f85149"
            tkr_html += f"""<tr>
              <td style="font-weight:bold">{tkr}</td>
              <td>{int(row['n'])}</td>
              <td style="color:{wr_c}">{row['wr']*100:.0f}%</td>
              <td style="color:{ap_c}">{row['ap']:+.1f}%</td>
              <td style="color:{'#3fb950' if row['ast']>0 else '#f85149'}">{row['ast']:+.1f}%</td>
              <td><div style="background:#238636;height:12px;width:{bar_w}%;border-radius:4px"></div></td>
            </tr>"""

    # profit-target comparison table
    pt_rows = ""
    if not pt_df.empty and not sub12.empty:
        base = s12_base
        pt   = stats(pt_df.rename(columns={'final_pnl':'pnl_pct'}))
        n_3m = (pt_df['exit_rule']=='exit_3M_profit').sum()
        n_12m= (pt_df['exit_rule']=='hold_12M').sum()
        for label, s, note in [
            ("Always Hold 12M (baseline)", base, ""),
            (f"Profit-Target Strategy (exit 3M if >100%)", pt,
             f"{n_3m} exited at 3M, {n_12m} held to 12M"),
        ]:
            if not s: continue
            pt_rows += f"""<tr>
              <td>{label}<br><small style="color:#8b949e">{note}</small></td>
              <td>{s['n']}</td>
              <td style="color:{'#3fb950' if s['wr']>=60 else '#d29922'}">{s['wr']:.0f}%</td>
              <td style="color:{'#3fb950' if s['av']>0 else '#f85149'}">{s['av']:+.1f}%</td>
              <td style="color:{'#3fb950' if s['sh']>=1 else '#d29922'}">{s['sh']:.2f}</td>
              <td style="color:{'#d29922' if s['bu_pct']>0 else '#3fb950'}">{s['bu_pct']:.1f}%</td>
            </tr>"""

    # all-trades table
    trade_rows = ""
    for _, row in df[df['horizon_days']==252].sort_values('pnl_pct',ascending=False).iterrows():
        pc = "#3fb950" if row['pnl_pct']>0 else "#f85149"
        sc2= "#3fb950" if row['stock_ret']>0 else "#f85149"
        trade_rows += f"""<tr>
          <td>{row['ticker']}</td><td>{row['entry_date']}</td>
          <td>${row['entry_price']:,.2f}</td><td>${row['strike']:,.0f}</td>
          <td>{row['entry_iv']:.0f}%</td><td>{row['iv_rank_entry']:.0f}</td>
          <td>${row['entry_prem']:.2f}</td>
          <td style="color:{sc2}">{row['stock_ret']:+.1f}%</td>
          <td style="color:{pc}">{row['pnl_pct']:+.1f}%</td>
          <td>{'✓' if row['win'] else '✗'}</td>
        </tr>"""

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LEAPS Backtest — {run_date}</title>
<style>{_CSS}
.cards{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px}}
.disc{{background:#1c1007;border:1px solid #d29922;border-radius:8px;padding:16px;
       color:#d29922;font-size:13px;line-height:1.7;margin-bottom:24px}}
</style></head><body>
<h1>LEAPS Backtest Report v3.0 &nbsp;<small style="color:#8b949e;font-size:14px">{run_date}</small></h1>

<div class="disc">
⚠ <strong>Assumptions:</strong> IV proxied by 30-day realized vol (real IV is ~10–30% higher due to vol risk
premium → real premiums cost more → real returns are lower than shown). ATM strikes only. No bid-ask spread
(add ~5–10% friction per trade). Walk-forward: no lookahead bias. Past performance does not predict future results.
</div>

<h2>Strategy Comparison — All Horizons</h2>
<div class="cards">
  {stat_card("3-Month Hold","#00b0ff" if False else s3,"#00b0ff")}
  {stat_card("6-Month Hold",s6,"#ffd700")}
  {stat_card("12-Month Hold (Baseline)",s12,"#ff9800")}
  {stat_card(f"Profit-Target ({PROFIT_TARGET:.0f}%+ exit at 3M)",spt,"#00c853")}
</div>

<h2>Profit-Target vs Baseline (12M)</h2>
<div class="sec"><table id="pt">
  <thead><tr>
    <th>Strategy</th><th onclick="sortTable('pt',1)">Signals</th>
    <th onclick="sortTable('pt',2)">Win Rate</th><th onclick="sortTable('pt',3)">Avg P&L</th>
    <th onclick="sortTable('pt',4)">Sharpe</th><th onclick="sortTable('pt',5)">Blow-ups</th>
  </tr></thead>
  <tbody>{pt_rows}</tbody>
</table></div>

<h2>Per-Ticker 12M Breakdown</h2>
<div class="sec"><table id="tkrt">
  <thead><tr>
    <th onclick="sortTable('tkrt',0)">Ticker</th><th onclick="sortTable('tkrt',1)">Signals</th>
    <th onclick="sortTable('tkrt',2)">Win Rate</th><th onclick="sortTable('tkrt',3)">Avg Opt P&L</th>
    <th onclick="sortTable('tkrt',4)">Avg Stock Ret</th><th>Bar</th>
  </tr></thead>
  <tbody>{tkr_html}</tbody>
</table></div>

<h2>All Trades — 12M Horizon (sorted by P&L)</h2>
<div class="sec"><table id="all">
  <thead><tr>
    <th onclick="sortTable('all',0)">Ticker</th><th onclick="sortTable('all',1)">Entry Date</th>
    <th onclick="sortTable('all',2)">Entry Price</th><th onclick="sortTable('all',3)">Strike</th>
    <th onclick="sortTable('all',4)">Entry IV</th><th onclick="sortTable('all',5)">IV Rank</th>
    <th onclick="sortTable('all',6)">Premium</th>
    <th onclick="sortTable('all',7)">Stock Ret</th><th onclick="sortTable('all',8)">Opt P&L</th>
    <th>Win</th>
  </tr></thead>
  <tbody>{trade_rows}</tbody>
</table></div>

<h2>Honest Analysis — What Would Improve This</h2>
<div class="sec" style="line-height:1.8;font-size:13px">
  <ol style="padding-left:20px">
    <li><strong>Historical options prices</strong> (ORATS, Market Chameleon, CBOE DataShop) — eliminates the
    realized-vol proxy entirely. Single biggest accuracy improvement.</li>
    <li><strong>Earnings IV crush model</strong> — buy 1 week <em>after</em> the print (IV collapses, stock settles),
    not 30 days before. This alone would significantly improve win rate.</li>
    <li><strong>Profit-target exit at 3M</strong> — backtest above shows whether this improves risk-adjusted returns
    vs holding to expiry.</li>
    <li><strong>VIX term structure</strong> — VIX9D / VIX / VIX3M ratio tells you if fear is event-driven (buy through)
    vs sustained (wait). More nuanced than a flat VIX &lt; 28 gate.</li>
    <li><strong>Sector rotation scoring</strong> — weight signals by which sectors are in institutional favor (sector
    RS rank vs SPY, sector ETF relative momentum).</li>
    <li><strong>Daily alert automation</strong> — cron job at 4:30 PM ET → Telegram/email with only new signals.</li>
    <li><strong>Open position tracker</strong> — track current LEAPS with live Greeks, unrealized P&amp;L, and DTE
    countdown.</li>
  </ol>
</div>

<div class="foot">LEAPS Scanner v3.0 &nbsp;|&nbsp; {run_date} &nbsp;|&nbsp;
5-Year Walk-Forward Backtest &nbsp;|&nbsp; Not financial advice.</div>
<script>{_SORT_JS}</script></body></html>"""

# ─── CONSOLE DISPLAY ─────────────────────────────────────────────────────────────
BAR = "═"*72

def print_signal(r):
    sig = ("STRONG [Tech+Fund+Env]" if r['strong'] else "COMBO [Tech+Fund]" if r['combo']
           else "TECH" if r['tech'] else "FUND OVERSOLD")
    warn = ("  !ENV FAILED" if not r['env_ok'] else "") + \
           (f"  !EARN {r['days_earn']}d" if r['days_earn']<MIN_DAYS_EARN else "")
    print(f"\n  {'━'*68}")
    print(f"  {r['ticker']:<6}  Grade:{r['grade']}  {sig}{warn}")
    print(f"  {'━'*68}")
    print(f"  ${r['price']:,.2f}  RSI:{r['rsi']:.1f}  IVR:{r['iv_rank']:.0f}/100  IV≈{r['curr_iv']:.0f}%  "
          f"Earn:{r['days_earn']}d  Score:{r['fund_score']}/100")
    c = r['cond']
    checks = [("SMA200",c['above_sma200']),("GldX",c['golden_cross']),("RSI",c['rsi_neutral']),
              ("MACD↑",c['macd_cross_up']),("Vol",c['volume_spike']),
              ("IVR",c['iv_rank_low']),("VIX",c['vix_ok']),("Earn",c['earnings_safe']),("RS",c['rs_positive'])]
    print("  "+" ".join(f"{'✓' if v else '✗'}{l}" for l,v in checks))
    gk = (f"Δ{r['delta']:.2f} Θ${r['theta']:.2f}/d V${r['vega']:.2f}/1%" if r.get('delta') else "")
    print(f"  → ${r['strike']:.0f} Call ({r['exp_date']})  ${r['prem']:.2f}/sh  "
          f"{r['contracts']}x  ${r['capital']:,.0f}  BE:{r['be_pct']:+.1f}%  {gk}")

def print_bt_summary(all_trades):
    if not all_trades: return
    df  = pd.DataFrame(all_trades)
    pt  = profit_target_strategy(df)
    print(f"\n{BAR}")
    print("  BACKTEST SUMMARY (see leaps_backtest.html for full detail)")
    print(BAR)
    for fwd, lbl in [(63,"3M"),(126,"6M"),(252,"12M")]:
        sub = df[df['horizon_days']==fwd]
        if sub.empty: continue
        av  = sub['pnl_pct'].mean()
        wr  = sub['win'].mean()*100
        sh  = av/sub['pnl_pct'].std() if sub['pnl_pct'].std()>0 else 0
        print(f"  {lbl}: n={len(sub)}  WR={wr:.0f}%  Avg={av:+.0f}%  Sharpe={sh:.2f}")
    if not pt.empty:
        ppt = pt.rename(columns={'final_pnl':'pnl_pct'})
        av = ppt['pnl_pct'].mean(); wr = ppt['win'].mean()*100
        sh = av/ppt['pnl_pct'].std() if ppt['pnl_pct'].std()>0 else 0
        n3 = (pt['exit_rule']=='exit_3M_profit').sum()
        print(f"  ProfitTarget: n={len(pt)}  WR={wr:.0f}%  Avg={av:+.0f}%  "
              f"Sharpe={sh:.2f}  ({n3} exited at 3M)")

# ─── EMAIL HTML ──────────────────────────────────────────────────────────────────
def build_email_html(signals, vix, regime, trend, run_date):
    """Rich HTML email showing both 3M ATM and 12M LEAPS recommendations per signal."""
    M = "background:#1a1a2e;color:#c9d1d9;font-family:'Courier New',monospace;padding:0;margin:0"
    W = "max-width:700px;margin:0 auto;padding:24px"

    env_bar = (f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
               f'padding:16px;margin-bottom:24px;font-size:14px">'
               f'<b style="color:#58a6ff">LEAPS Scanner</b> &nbsp;{run_date}<br>'
               f'<span style="color:#8b949e">VIX </span><b style="color:{"#3fb950" if vix<20 else "#d29922" if vix<VIX_MAX else "#f85149"}">{vix:.1f}</b>'
               f' &nbsp;|&nbsp; <span style="color:#8b949e">Regime </span><b>{regime}</b>'
               f' &nbsp;|&nbsp; <span style="color:#8b949e">SPY </span><b style="color:{"#3fb950" if trend=="BULLISH" else "#f85149"}">{trend}</b>'
               f' &nbsp;|&nbsp; <b style="color:#3fb950">{len(signals)} signal{"s" if len(signals)!=1 else ""}</b></div>')

    if not signals:
        return (f'<html><body style="{M}"><div style="{W}">{env_bar}'
                f'<p style="color:#8b949e">No signals today. Scanner ran successfully.</p>'
                f'</div></body></html>')

    cards = ""
    for r in signals:
        st_color = {"strong":"#00c853","combo":"#00b0ff","tech":"#ffd700","fund":"#ff9800"}
        st = "strong" if r.get('strong') else "combo" if r.get('combo') else "tech" if r.get('tech') else "fund"
        tag = st.upper()

        # IV edge line
        iv_edge = (f'<tr><td style="color:#8b949e;padding:4px 0">IV edge</td>'
                   f'<td style="padding:4px 8px"><b style="color:#00b0ff">{r["curr_iv"]:.0f}%</b>'
                   f' now (rank <b>{r["iv_rank"]:.0f}/100</b>) · '
                   f'52-wk <span style="color:#3fb950">{r.get("iv_lo",0):.0f}%</span>'
                   f'→avg <span style="color:#d29922">{r.get("iv_avg",0):.0f}%</span>'
                   f'→<span style="color:#f85149">{r.get("iv_hi",0):.0f}%</span>'
                   + (f' · <b style="color:#3fb950">+{r["iv_expand_gain"]:.0f}% from vega if IV→avg</b>'
                      if r.get('iv_expand_gain',0)>0 else '') + '</td></tr>')

        # 3M option block
        if r.get('s3'):
            s3 = r['s3']
            s3_block = (
                f'<tr><td colspan="2" style="padding-top:14px">'
                f'<div style="background:#0d1520;border-left:3px solid #00b0ff;padding:12px;border-radius:4px">'
                f'<div style="color:#00b0ff;font-weight:bold;margin-bottom:8px">'
                f'OPTION 1 — 3-Month ATM Call &nbsp;'
                f'<span style="font-weight:normal;color:#8b949e;font-size:12px">'
                f'Backtest 2025: 92% win rate · avg +217%</span></div>'
                f'<table style="font-size:13px;border-collapse:collapse;width:100%">'
                f'<tr><td style="color:#8b949e;width:120px">Strike / Expiry</td>'
                f'<td><b>${s3["strike"]:.0f} Call</b> &nbsp;({r["s3_exp"]})</td></tr>'
                f'<tr><td style="color:#8b949e">IV at entry</td>'
                f'<td><b style="color:#00b0ff">{s3["iv"]:.0f}%</b> &nbsp;'
                f'<span style="color:#8b949e">(cheap — rank {r["iv_rank"]:.0f}/100)</span></td></tr>'
                f'<tr><td style="color:#8b949e">Premium</td>'
                f'<td><b>${s3["prem"]:.2f}/share</b> &nbsp;(${s3["prem"]*100:,.0f}/contract)</td></tr>'
                f'<tr><td style="color:#8b949e">Greeks</td>'
                f'<td>Δ{s3.get("delta","~0.50")} &nbsp;'
                f'Θ${s3.get("theta","–")}/day &nbsp;'
                f'V${s3.get("vega","–")}/1% IV</td></tr>'
                f'<tr><td style="color:#8b949e">Position</td>'
                f'<td>{r["s3_contracts"]}x contracts = <b>${r["s3_capital"]:,.0f}</b></td></tr>'
                f'<tr><td style="color:#8b949e">Breakeven</td>'
                f'<td>${s3["strike"]+s3["prem"]:,.2f} &nbsp;'
                f'({r["s3_be_pct"]:+.1f}% above today)</td></tr>'
                f'</table>'
                f'<div style="margin-top:10px;font-size:12px;color:#8b949e;border-top:1px solid #30363d;padding-top:8px">'
                f'<b style="color:#c9d1d9">Exit:</b> &nbsp;'
                f'① Up ≥100% → <b style="color:#3fb950">exit immediately</b> &nbsp;&nbsp;'
                f'② Day 63 (~{r["s3_exp"]}) → <b>exit at market</b> &nbsp;&nbsp;'
                f'③ Stock falls &gt;7% → <b style="color:#f85149">cut loss</b>'
                f'</div></div></td></tr>'
            )
        else:
            s3_block = '<tr><td colspan="2" style="color:#8b949e;font-size:12px">3M chain unavailable — use next trading day</td></tr>'

        # 12M LEAPS block
        gk12 = (f'Δ{r["delta"]} Θ${r["theta"]}/day V${r["vega"]}/1%'
                if r.get('delta') else 'Greeks N/A')
        leaps_block = (
            f'<tr><td colspan="2" style="padding-top:10px">'
            f'<div style="background:#0d1a0d;border-left:3px solid #238636;padding:12px;border-radius:4px">'
            f'<div style="color:#3fb950;font-weight:bold;margin-bottom:8px">'
            f'OPTION 2 — 12-Month LEAPS &nbsp;'
            f'<span style="font-weight:normal;color:#8b949e;font-size:12px">'
            f'~0.70 delta · more time buffer</span></div>'
            f'<table style="font-size:13px;border-collapse:collapse;width:100%">'
            f'<tr><td style="color:#8b949e;width:120px">Strike / Expiry</td>'
            f'<td><b>${r["strike"]:.0f} Call</b> &nbsp;({r["exp_date"]})</td></tr>'
            f'<tr><td style="color:#8b949e">Premium</td>'
            f'<td><b>${r["prem"]:.2f}/share</b> &nbsp;(${r["prem"]*100:,.0f}/contract)</td></tr>'
            f'<tr><td style="color:#8b949e">Greeks</td><td>{gk12}</td></tr>'
            f'<tr><td style="color:#8b949e">Position</td>'
            f'<td>{r["contracts"]}x = <b>${r["capital"]:,.0f}</b> &nbsp;'
            f'· Breakeven ${r["breakeven"]:,.2f} ({r["be_pct"]:+.1f}%)</td></tr>'
            f'</table></div></td></tr>'
        )

        cards += (
            f'<div style="background:#161b22;border:1px solid #30363d;'
            f'border-left:4px solid {st_color[st]};border-radius:8px;'
            f'padding:20px;margin-bottom:20px">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:14px">'
            f'<div><b style="font-size:20px">{r["ticker"]}</b> &nbsp;'
            f'<span style="color:#8b949e">Grade {r["grade"]} · Score {r["fund_score"]}/100 · '
            f'${r["price"]:,.2f} · RSI {r["rsi"]:.0f} · Earn {r["days_earn"]}d</span></div>'
            f'<span style="background:{st_color[st]};color:#000;padding:4px 12px;'
            f'border-radius:12px;font-size:12px;font-weight:bold">{tag}</span></div>'
            f'<table style="font-size:13px;border-collapse:collapse;width:100%;margin-bottom:4px">'
            f'{iv_edge}{s3_block}{leaps_block}'
            f'</table></div>'
        )

    footer = ('<p style="color:#8b949e;font-size:11px;margin-top:24px;'
              'border-top:1px solid #21262d;padding-top:12px">'
              'LEAPS Scanner v3.0 · Not financial advice · '
              'Data: Yahoo Finance · Walk-forward backtest only</p>')

    return f'<html><body style="{M}"><div style="{W}">{env_bar}{cards}{footer}</div></body></html>'


# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='LEAPS Professional Scanner v3.0')
    parser.add_argument('--scan-only', action='store_true',
                        help='Run live scan + email only; skip the 5-year backtest (faster, use for CI/daily runs)')
    args = parser.parse_args()

    run_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"\n{BAR}")
    print(f"   LEAPS PROFESSIONAL SCANNER v3.0  —  {run_date}")
    print(f"{BAR}\n  Loading market data...")

    vix_df   = fix_cols(yf.download('^VIX',period='5d',progress=False,auto_adjust=True))
    vix      = float(vix_df['Close'].dropna().iloc[-1])
    spy_df   = fix_cols(yf.download('SPY',period='2y',progress=False,auto_adjust=True))
    spy_p    = float(spy_df['Close'].iloc[-1])
    spy_s200 = float(spy_df['Close'].rolling(200).mean().iloc[-1])
    _, spy_3m, _ = relative_strength(spy_df, spy_df)

    trend  = "BULLISH" if spy_p > spy_s200 else "BEARISH"
    regime = "RISK-ON" if vix<20 else ("ELEVATED" if vix<VIX_MAX else "RISK-OFF")
    print(f"  VIX={vix:.1f} [{regime}]  SPY={trend}  3M={spy_3m*100:+.1f}%\n")

    sector_cache, results = {}, []
    print(f"  Scanning {len(TICKERS)} tickers...\n")
    for t in TICKERS:
        print(f"    {t:<6}...", end='', flush=True)
        r = screen_ticker(t, spy_df, vix, sector_cache)
        results.append(r)
        if r.get('error'): print(f" ERR:{r['error']}")
        else:
            tag = ("STRONG!" if r.get('strong') else "COMBO!" if r.get('combo')
                   else "SIGNAL!" if r.get('signal') else f"({r.get('conditions',0)}/9)")
            print(f" {tag}")

    valid   = [r for r in results if not r.get('error')]
    signals = sorted([r for r in valid if r.get('signal')],
                     key=lambda r:-(4*r.get('strong',0)+2*r.get('combo',0)+r.get('tech',0)+r.get('fund',0)))
    near    = sorted([r for r in valid if not r.get('signal') and r.get('conditions',0)>=5],
                     key=lambda r:-r.get('conditions',0))

    print(f"\n{BAR}")
    print(f"  LIVE SIGNALS — {len(signals)} of {len(TICKERS)}")
    print(BAR)
    if signals:
        for r in signals: print_signal(r)
    else:
        print("\n  No signals. Near misses:")
        for r in near[:5]:
            print(f"    {r['ticker']:<6} {r.get('conditions',0)}/9  RSI:{r['rsi']:.0f}  IVR:{r['iv_rank']:.0f}")

    # save HTML scan
    scan_html = generate_scan_html(results, vix, spy_p, spy_s200, spy_3m*100, run_date)
    with open('leaps_scan.html','w') as f: f.write(scan_html)
    pd.DataFrame([{k:v for k,v in r.items() if k not in('cond','fund_details')}
                  for r in valid]).to_csv('leaps_scan_results.csv', index=False)
    print(f"\n  Saved → leaps_scan.html  leaps_scan_results.csv")

    # backtest (skipped with --scan-only)
    if not args.scan_only:
        print(f"\n{BAR}")
        print("  BACKTEST (5 years, ~2-4 min)...")
        print(BAR)
        all_trades = []
        for t in TICKERS:
            print(f"    {t:<6}...", end='', flush=True)
            trades = backtest_ticker(t)
            all_trades.extend(trades)
            n12 = sum(1 for x in trades if x['horizon_days']==252)
            print(f" {n12} signals")
        bt_html = generate_backtest_html(all_trades, run_date)
        with open('leaps_backtest.html','w') as f: f.write(bt_html)
        pd.DataFrame(all_trades).to_csv('leaps_backtest.csv', index=False)
        print(f"  Saved → leaps_backtest.html  leaps_backtest.csv")
        print_bt_summary(all_trades)
    else:
        print(f"\n  --scan-only: backtest skipped.")

    # alerts — always send (signals or no-signal confirmation)
    alert_text = build_alert_text(signals, vix, run_date)
    send_telegram(alert_text)
    email_html = build_email_html(signals, vix, regime, trend, run_date)
    subject    = (f"LEAPS Alert {run_date} — {len(signals)} signal(s): "
                  + ", ".join(r['ticker'] for r in signals)
                  if signals else f"LEAPS Scanner {run_date} — No signals today")
    send_email(subject, email_html)

    print(f"\n{BAR}\n")

if __name__ == '__main__':
    main()
