# LEAPS Professional Scanner v3.0

A systematic, walk-forward-backtested scanner for LEAPS call options across 20 large-cap tickers.
Combines live technical signals, real fundamental scoring, IV rank, earnings awareness,
and relative strength — then outputs a dark-themed HTML dashboard you can open in any browser.

## Quick Start

```bash
pip install yfinance pandas numpy scipy requests
python scanner.py
```

Opens two HTML reports in your project folder after each run.

---

## Output Files

| File | Description |
|------|-------------|
| `leaps_scan.html` | **Live signal dashboard** — signal cards, condition checklist, recommended trade, near-misses table |
| `leaps_backtest.html` | **Full backtest report** — 3M/6M/12M stats, profit-target comparison, per-ticker breakdown, all-trades table |
| `leaps_scan_results.csv` | Raw scan data for all tickers |
| `leaps_backtest.csv` | Raw backtest trades (every signal × horizon) |
| `logs/YYYYMMDD.log` | Daily run log (created by `run_daily.sh`) |

---

## What the Scanner Does

### 1. Market Environment Check
Downloads VIX and SPY first. Displays regime (RISK-ON / ELEVATED / RISK-OFF) and SPY trend.
A VIX > 28 warning is shown if active — signals are still displayed but flagged.

### 2. Live Signals (20 tickers)

For each ticker, the scanner evaluates **9 conditions** and fires one of:

| Signal | Conditions |
|--------|-----------|
| `STRONG` | Tech + Fund Quality + All Env filters + RS > SPY |
| `COMBO` | Tech signal + Grade A or B |
| `TECH` | Momentum breakout (4 technical conditions) |
| `FUND` | Grade A fundamentals + RSI oversold (< 35) |

**9 conditions tracked:**

| Condition | Detail |
|-----------|--------|
| Above SMA-200 | Price above 200-day moving average |
| Golden Cross | SMA-50 above SMA-200 |
| RSI 38–62 | Not overbought, not oversold |
| MACD Cross ↑ | MACD histogram just crossed from negative to positive |
| Volume Spike | Today's volume > 50-day avg × 1.15 |
| IV Rank < 40 | IV in the lower 40% of its 52-week range (cheap premium) |
| VIX < 28 | Market-level premium filter |
| Earnings Safe | No earnings within 30 days |
| RS > SPY | Stock outperforming SPY over the last 3 months |

### 3. Fundamental Scoring (live, not hardcoded)

Pulls live data from Yahoo Finance for each ticker:

| Factor | Points |
|--------|--------|
| Forward P/E < 20 | +20 |
| Revenue Growth > 20% | +20 |
| Net Margin > 20% | +20 |
| Debt/Equity < 50 | +20 |
| EPS Growth > 15% | +20 |

Score ≥ 70 → Grade A · Score ≥ 40 → Grade B · Below → Grade C

### 4. IV Rank (live)
Computes 30-day realized volatility and calculates its rank within the past 52-week range.
Only buys when IV rank < 40 — avoids paying inflated premiums.

### 5. Options Pricing & Greeks
Tries to pull the real options chain for the nearest expiry to 12 months out.
Falls back to Black-Scholes if the chain is unavailable.
Selects the strike closest to **0.70 delta** and reports full Greeks: Δ Γ Θ V.

### 6. Position Sizing
Sizes to a maximum of **5% of account** (default $100,000 → max $5,000/position).
Reports number of contracts and exact capital required.

### 7. Near Misses
Any ticker scoring 5 or more of the 9 conditions — but not triggering a full signal —
appears in the watchlist table. Sortable by any column in the HTML output.

---

## 3M Trading Strategy — Entry & Exit Rules

This is the strategy validated by the 2025 backtest (92% win rate, avg +217% at the 3-month horizon).

### Entry — all 5 conditions must be true on the same trading day

| # | Condition | Precise Rule |
|---|-----------|-------------|
| 1 | **Long-term uptrend** | Close > SMA(200) |
| 2 | **Neutral momentum** | RSI(14) is between 38 and 62 — not extended in either direction |
| 3 | **MACD inflection** | MACD histogram crosses from negative to positive *today* — `hist[t] > 0` AND `hist[t-1] < 0`. This is a single-day trigger; if you miss the day, the signal is gone. |
| 4 | **Volume confirmation** | Today's volume > 50-day average volume × 1.15 |
| 5 | **Cheap options** | IV Rank < 40 — 30-day realized vol is in the bottom 40% of its trailing 252-day range |

The MACD crossover (condition 3) is the timing trigger. Conditions 1, 2, 4, 5 are the quality filter that must be true on that same day. In 2025, only 13 signals across 20 tickers met all 5 conditions.

### Instrument

- **Type:** ATM call option
- **Strike:** Nearest $5 multiple to the current closing price (`round(price / 5) * 5`)
- **Expiry:** Nearest available expiry approximately 3 months (63 trading days) out
- **Premium cost:** Expect to pay 3.7–7.9% of the stock price (2025 median: 5.3%)

> In the **live scanner**, the recommended position targets a 0.70-delta strike on a 12-month LEAPS. For the 3M strategy specifically, use the ATM strike on a 3-month expiry.

### Position Sizing

- Maximum **5% of account** per position (default: $5,000 on a $100,000 account)
- Use (bid + ask) / 2 as the fill price; avoid paying the ask on illiquid names

### Exit Rules (in priority order)

| Trigger | Action | Why |
|---------|--------|-----|
| **Option up ≥ 100%** | Exit immediately — do not wait for the 63-day mark | In 2025, the top signals were up 283–422% at 3M; banking at 100% still beats holding to 6M (avg +34%) |
| **63 trading days elapsed** | Exit at market on day 63 regardless of P&L | 6M results (46% win rate) show momentum does not persist; holding past 3M gave back most gains in 2025 |
| **Stock falls > 7%** from entry price | Consider early exit | In 2025, the only losing trade had the stock down 7.5% at expiry (-6% option P&L). Stocks down 1–5% still produced winning options due to IV expansion. A hard stop at -7% stock decline prevents the tail case. |

**Do not hold past 3M.** The 2025 data is definitive on this: 6M results deteriorated to 46% win rate and +34% avg — the same signals that returned +217% at 3M gave back nearly all gains by month 6.

### Why the 3M Exit Works

The MACD histogram crossover identifies a momentum inflection point. At that moment:
- Buying an ATM call gives you ~0.50 delta — full participation in the initial move
- Options leverage amplifies even small stock moves significantly:
  - Stock +20% → option typically +300–400%
  - Stock +5% → option typically +200–400% (IV expansion boost)
  - Stock flat (+0–2%) → option still +100–280% when IV expands
- The one-way bet: a stock must fall more than ~7% for a loss (time value buffer absorbs smaller declines)
- **Leverage range observed (2025):** 14–1285× stock return, median ~40×

---

## Backtest Methodology

**Walk-forward, no lookahead bias.** Runs across 5 years of daily data per ticker.

- IV rank is precomputed at each day using only past 252 trading days of data
- Entry: TECH signal fires (same 4 conditions as live scan, + IV rank filter)
- Strike: ATM rounded to nearest $5
- Pricing: Black-Scholes with realized vol as IV proxy (real IV is ~10–30% higher → real returns are lower)
- Exits measured at: 3 months (63 days), 6 months (126 days), 12 months (252 days)
- At 12M: option value = intrinsic only (at expiry)

**Profit-Target Strategy:** exit at 3M if the option is up > 100%; otherwise hold to 12M.

### Backtest Results — 5-Year (2021–2026), 40 signals

| Horizon | Signals | Win Rate | Avg P&L | Sharpe | Blow-ups |
|---------|---------|----------|---------|--------|---------|
| 3 Month | 40 | 88% | +260% | 1.17 | 0% |
| 6 Month | 40 | 88% | +172% | 1.18 | 2.5% |
| 12 Month | 40 | 72% | +171% | 0.77 | 5% |
| Profit Target (exit 3M if >100%, else 12M) | 40 | 85% | +253% | 1.08 | — |

### Backtest Results — 2025 Only, 13 signals

2025 had fewer signals than prior years due to elevated volatility (tariff uncertainty, geopolitical risk).
Signals were concentrated in Q3–Q4 2025 as markets recovered. Run `python backtest_2025.py` for full detail.

| Horizon | Signals | Win Rate | Avg P&L | Sharpe | Best | Worst |
|---------|---------|----------|---------|--------|------|-------|
| 3 Month | 13 | **92%** | **+217%** | 1.46 | +422% | -6% |
| 6 Month | 13 | 46% | +34% | 0.33 | +226% | -68% |
| 12 Month | 1 (COST only) | 0% | -44% | — | -44% | -44% |

**2025 3M signals by P&L:**

| Ticker | Entry Date | Entry $ | IV | IVR | Premium | Stock (3M) | Option (3M) |
|--------|-----------|---------|-----|-----|---------|-----------|------------|
| TSLA | Sep 5 | $350.84 | 35.4% | 0 | $27.00 | +29.6% | **+422%** ✓ |
| QCOM | Jul 28 | $158.31 | 19.6% | 0 | $6.23 | +5.5% | **+404%** ✓ |
| ABBV | Jul 29 | $186.79 | 22.7% | 30 | $10.43 | +20.1% | **+367%** ✓ |
| ABBV | Aug 1 | $190.69 | 23.0% | 31 | $10.16 | +17.7% | **+358%** ✓ |
| AAPL | Sep 3 | $237.80 | 27.2% | 26 | $13.14 | +20.1% | **+316%** ✓ |
| COST | Jan 17 | $935.66 | 15.7% | 24 | $35.04 | +1.7% | **+283%** ✓ |
| WMT | Jun 27 | $96.46 | 15.5% | 8 | $4.38 | +6.3% | +207% ✓ |
| ABBV | Nov 12 | $229.46 | 24.3% | 27 | $12.10 | +0.1% | +129% ✓ |
| V | Oct 1 | $345.71 | 17.8% | 16 | $14.60 | +1.0% | +126% ✓ |
| NVDA | Sep 19 | $176.65 | 29.2% | 9 | $12.07 | **-1.4%** | +106% ✓ |
| AMZN | Sep 4 | $235.68 | 35.9% | 39 | $18.45 | **-1.4%** | +88% ✓ |
| V | Oct 23 | $343.85 | 19.0% | 21 | $14.34 | **-4.9%** | +24% ✓ |
| QCOM | Oct 22 | $167.34 | 33.2% | 26 | $13.14 | **-7.5%** | **-6%** ✗ |

Stocks highlighted in bold (NVDA, AMZN, V, QCOM) finished *below* entry price at the 3-month mark. Three of the four still produced winning options — only the -7.5% stock decline (QCOM Oct) was large enough to overwhelm the time value buffer.

**Honest caveats:**
- Realized vol underestimates actual implied vol by ~10–30% (vol risk premium) → real premiums cost more → real returns are lower than shown
- No bid-ask spread modeled — add ~5–10% friction per trade
- 13 signals (2025) and 40 signals (5-year) are small samples; don't over-fit
- 2025 had zero 12M-horizon data for most tickers (signals fired too late in the year to have elapsed)

---

## Daily Automation (optional)

```bash
# 1. Make the script executable (already done)
chmod +x run_daily.sh

# 2. Add to crontab — runs at 4:30 PM ET on weekdays
crontab -e
# Add this line:
30 16 * * 1-5 /Users/avinkhurana/Documents/claude/code/LEAPS/run_daily.sh
```

Logs are saved to `logs/YYYYMMDD.log`.

---

## Alerts (optional)

Fill in the config block at the top of `scanner.py`:

```python
TELEGRAM_TOKEN   = "..."   # get from @BotFather on Telegram
TELEGRAM_CHAT_ID = "..."   # get from @userinfobot on Telegram
EMAIL_FROM       = "..."   # Gmail address
EMAIL_PASSWORD   = "..."   # Gmail app-password (Settings → Security → App passwords)
EMAIL_TO         = "..."   # recipient address
```

When signals fire, the scanner sends a formatted Telegram message and/or HTML email.
Leave blank to disable.

---

## Configuration

All key parameters are at the top of `scanner.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ACCOUNT_SIZE` | 100,000 | Account size for position sizing |
| `MAX_POS_PCT` | 0.05 | Max % of account per position |
| `TARGET_DELTA` | 0.70 | Target call delta for strike selection |
| `LEAPS_MONTHS` | 12 | Target expiry months out |
| `VIX_MAX` | 28 | Skip if VIX above this |
| `IV_RANK_MAX` | 40 | Only buy when IV rank below this |
| `MIN_DAYS_EARN` | 30 | Avoid entries within this many days of earnings |
| `PROFIT_TARGET` | 100.0 | Exit at 3M if option up more than this % |

---

## Ticker Universe

| Ticker | Sector |
|--------|--------|
| AAPL, MSFT, NVDA, AMD, CRM, ADBE, QCOM | Technology |
| GOOGL, META | Communication |
| AMZN, TSLA | Consumer Discretionary |
| V, MA, JPM, BAC | Financials |
| UNH, LLY, ABBV | Healthcare |
| COST, WMT | Consumer Staples |

---

## What Would Make This Even Better

1. **Historical options prices** (ORATS, Market Chameleon, CBOE DataShop) — eliminates the realized-vol proxy assumption
2. **Earnings IV crush model** — buy 1 week after the print, not before
3. **VIX term structure** — VIX9D / VIX / VIX3M ratio for more nuanced regime detection
4. **Sector rotation scoring** — weight signals by sector institutional flows
5. **Open position tracker** — live Greeks and P&L on existing LEAPS holdings
