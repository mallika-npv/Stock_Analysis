"""
Nifty 50 - descriptive cycle analysis (no prediction)
-----------------------------------------------------
Answers: "What did the market actually do over these years, and what repeats?"

  1. Overview          - total/annual returns, volatility, worst drawdown
  2. Bull/bear phases  - swings of >= REVERSAL (default 10%) from peak/trough
  3. Drawdowns         - every fall >= 5%: depth, days to bottom, days to recover
  4. Trend state       - simple, transparent 50/200-day moving-average regimes
  5. Volatility        - calm vs turbulent periods, and how long they last
  6. Calendar          - year x month heatmap, month-of-year, weekday effects
  7. Return structure  - streaks, big-day concentration, rolling 1-year returns

Setup:   pip install pandas numpy matplotlib
Data:    put your yearly niftyindices CSVs in the folder set below
Outputs: ./Nifty_op/  (PNG charts + CSV tables)
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ SETTINGS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "nifty_manual")   # or: r"C:\path\to\nifty_manual"
REVERSAL = 0.10        # a bull/bear phase flips after a 10% move against it
MIN_DRAWDOWN = 0.05    # list drawdowns of at least 5%
os.chdir(BASE_DIR)
OUT_DIR = "Nifty_op"
os.makedirs("output", exist_ok=True)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ------------------------------------------------------------------ 0. LOAD
def load_manual(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {folder}")
    parts = []
    for fp in files:
        d = pd.read_csv(fp)
        d.columns = [str(x).strip().title() for x in d.columns]
        if "Date" not in d.columns or "Close" not in d.columns:
            print(f"  SKIPPED {os.path.basename(fp)} (no Date/Close column)")
            continue
        d["Date"] = pd.to_datetime(d["Date"].astype(str).str.strip(),
                                   dayfirst=True, errors="coerce")
        for col in ["Open", "High", "Low", "Close"]:
            if col not in d.columns:
                d[col] = d["Close"]
            d[col] = pd.to_numeric(d[col].astype(str).str.replace(",", "").str.strip(),
                                   errors="coerce")
        d = d[["Date", "Open", "High", "Low", "Close"]].dropna(subset=["Date", "Close"])
        print(f"  {os.path.basename(fp):45s}{len(d):4d} rows  "
              f"{d['Date'].min().date()} -> {d['Date'].max().date()}")
        parts.append(d)
    out = pd.concat(parts).drop_duplicates("Date").set_index("Date").sort_index()
    gaps = out.index.to_series().diff().dt.days
    for dt, g in gaps[gaps > 10].items():
        print(f"  WARNING: {int(g)}-day gap before {dt.date()} - a file may be missing")
    return out


print("Loading files...")
df = load_manual(DATA_FOLDER)
c = df["Close"]
ret = c.pct_change().dropna()
years_span = (c.index[-1] - c.index[0]).days / 365.25
print(f"\n{len(df)} trading days, {c.index[0].date()} -> {c.index[-1].date()} "
      f"({years_span:.1f} years)\n")


def pct(x, d=1):
    return f"{x * 100:+.{d}f}%"


# ------------------------------------------------------------- 1. OVERVIEW
total = c.iloc[-1] / c.iloc[0] - 1
cagr = (c.iloc[-1] / c.iloc[0]) ** (1 / years_span) - 1
ann_vol = ret.std() * np.sqrt(252)
underwater = c / c.cummax() - 1
print("=" * 70)
print("1. OVERVIEW")
print("=" * 70)
print(f"Start {c.iloc[0]:,.0f}  ->  End {c.iloc[-1]:,.0f}   Total {pct(total)}   CAGR {pct(cagr)}")
print(f"Annualised volatility {ann_vol * 100:.1f}%   |   Worst drawdown {pct(underwater.min())} "
      f"(bottom on {underwater.idxmin().date()})")
print(f"Currently {pct(underwater.iloc[-1])} below the all-time high in this data "
      f"({c.cummax().iloc[-1]:,.0f})")

# calendar-year returns (first/last years are partial)
ylast = c.groupby(c.index.year).last()
yprev = ylast.shift(1)
yprev.iloc[0] = c.iloc[0]
year_ret = (ylast / yprev - 1)
year_ret.index.name = "Year"
print("\nCalendar-year returns (first and last years are partial):")
for y, r in year_ret.items():
    tag = " (partial)" if y in (c.index[0].year, c.index[-1].year) else ""
    print(f"  {y}: {pct(r):>8}{tag}")


# ------------------------------------------------------- 2. BULL/BEAR PHASES
def zigzag(close, thr):
    v = close.values
    pivots, direction, hi, lo = [], 0, 0, 0
    for i in range(1, len(v)):
        if direction == 0:
            if v[i] >= v[lo] * (1 + thr):
                pivots.append(lo); direction = 1; hi = i
            elif v[i] <= v[hi] * (1 - thr):
                pivots.append(hi); direction = -1; lo = i
            else:
                hi = i if v[i] > v[hi] else hi
                lo = i if v[i] < v[lo] else lo
        elif direction == 1:
            if v[i] > v[hi]:
                hi = i
            elif v[i] <= v[hi] * (1 - thr):
                pivots.append(hi); direction = -1; lo = i
        else:
            if v[i] < v[lo]:
                lo = i
            elif v[i] >= v[lo] * (1 + thr):
                pivots.append(lo); direction = 1; hi = i
    return pivots


piv = zigzag(c, REVERSAL)
nodes = sorted(set([0] + piv + [len(c) - 1]))
phases = []
for a, b in zip(nodes[:-1], nodes[1:]):
    r = c.iloc[b] / c.iloc[a] - 1
    phases.append({
        "phase": "UP (bull)" if r > 0 else "DOWN (bear)",
        "start": c.index[a].date(), "end": c.index[b].date(),
        "start_level": round(c.iloc[a]), "end_level": round(c.iloc[b]),
        "return_%": round(r * 100, 1),
        "calendar_days": (c.index[b] - c.index[a]).days,
        "note": ("partial/ongoing" if (a == 0 or b == len(c) - 1) else ""),
    })
ph = pd.DataFrame(phases)
ph.to_csv(f"{OUT_DIR}/phases.csv", index=False)

print("\n" + "=" * 70)
print(f"2. BULL / BEAR PHASES  (a phase ends after a {REVERSAL * 100:.0f}% reversal)")
print("=" * 70)
print(ph.to_string(index=False))
full = ph[ph["note"] == ""]
for name in ["UP (bull)", "DOWN (bear)"]:
    s = full[full["phase"] == name]
    if len(s):
        print(f"  {name:12s} complete phases: {len(s)}  | avg return {s['return_%'].mean():+.1f}% "
              f"| avg length {s['calendar_days'].mean():.0f} days")

# ------------------------------------------------------------ 3. DRAWDOWNS
grp = (underwater == 0).cumsum()
episodes = []
for _, seg in underwater.groupby(grp):
    seg = seg[seg < 0]
    if seg.empty:
        continue
    start = seg.index[0]
    peak_date = c.index[c.index.get_loc(start) - 1]
    trough_date = seg.idxmin()
    after = underwater.loc[trough_date:]
    rec = after[after == 0]
    rec_date = rec.index[0] if len(rec) else None
    episodes.append({
        "peak_date": peak_date.date(), "trough_date": trough_date.date(),
        "recovery_date": rec_date.date() if rec_date is not None else "not yet",
        "depth_%": round(seg.min() * 100, 1),
        "days_peak_to_trough": (trough_date - peak_date).days,
        "days_trough_to_recovery": ((rec_date - trough_date).days if rec_date is not None else None),
    })
dd = pd.DataFrame(episodes)
dd_big = dd[dd["depth_%"] <= -MIN_DRAWDOWN * 100].sort_values("depth_%").reset_index(drop=True)
dd_big.to_csv(f"{OUT_DIR}/drawdowns.csv", index=False)
print("\n" + "=" * 70)
print(f"3. DRAWDOWNS of {MIN_DRAWDOWN * 100:.0f}% or more (deepest first)")
print("=" * 70)
print(dd_big.to_string(index=False))
if len(dd_big):
    print(f"\n  {len(dd_big)} such falls in {years_span:.1f} years  "
          f"(about one every {years_span / len(dd_big) * 12:.0f} months)")
    print(f"  Median depth {dd_big['depth_%'].median():.1f}%  |  median days to bottom "
          f"{dd_big['days_peak_to_trough'].median():.0f}  |  median days to recover "
          f"{dd_big['days_trough_to_recovery'].dropna().median():.0f}")

# ---------------------------------------------------- 4. MOVING-AVG TREND STATE
sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
valid = sma200.notna()
state = pd.Series(index=c.index, dtype=object)
state[valid] = "Mixed/Transition"
state[valid & (c > sma200) & (sma50 > sma200)] = "Bull (above 200DMA)"
state[valid & (c < sma200) & (sma50 < sma200)] = "Bear (below 200DMA)"
st = state.dropna()

run_id = (st != st.shift()).cumsum()
runs = []
for _, seg in st.groupby(run_id):
    i0, i1 = c.index.get_loc(seg.index[0]), c.index.get_loc(seg.index[-1])
    base = c.iloc[i0 - 1] if i0 > 0 else c.iloc[i0]
    runs.append({"state": seg.iloc[0], "start": seg.index[0].date(), "end": seg.index[-1].date(),
                 "trading_days": len(seg), "return_%": round((c.iloc[i1] / base - 1) * 100, 1)})
runs = pd.DataFrame(runs)
runs.to_csv(f"{OUT_DIR}/trend_state_runs.csv", index=False)

r_state = ret.reindex(st.index).dropna()
stat = pd.DataFrame({
    "pct_of_time": st.value_counts(normalize=True).mul(100).round(1),
    "n_episodes": runs.groupby("state").size(),
    "median_episode_days": runs.groupby("state")["trading_days"].median(),
    "avg_daily_ret_%": r_state.groupby(st).mean().mul(100).round(3),
    "ann_vol_%": r_state.groupby(st).std().mul(np.sqrt(252) * 100).round(1),
})
print("\n" + "=" * 70)
print("4. TREND STATE  (price vs 200-day average; 50 vs 200-day average)")
print("=" * 70)
print(stat.to_string())
print("\nEpisodes of 20+ trading days:")
print(runs[runs["trading_days"] >= 20].to_string(index=False))
flips = int((st.map({"Bull (above 200DMA)": 1, "Bear (below 200DMA)": -1,
                     "Mixed/Transition": 0}).diff().abs() > 0).sum())
print(f"\n  State changed {flips} times -> frequent flips = choppy market, signals unreliable then")

# ------------------------------------------------------------ 5. VOLATILITY
vol20 = ret.rolling(20).std() * np.sqrt(252)
hi_thr, lo_thr = vol20.quantile(0.75), vol20.quantile(0.25)
vol_state = pd.Series(np.where(vol20 > hi_thr, "High vol",
                      np.where(vol20 < lo_thr, "Low vol", "Normal")), index=vol20.index)[vol20.notna()]
vr = ret.reindex(vol_state.index)
vtab = pd.DataFrame({
    "pct_of_days": vol_state.value_counts(normalize=True).mul(100).round(1),
    "avg_daily_ret_%": vr.groupby(vol_state).mean().mul(100).round(3),
    "avg_ann_vol_%": vol20.reindex(vol_state.index).groupby(vol_state).mean().mul(100).round(1),
})
print("\n" + "=" * 70)
print("5. VOLATILITY REGIMES (20-day realised vol, top/bottom quartile)")
print("=" * 70)
print(vtab.to_string())
print(f"\n  Calmest 20-day stretch: {vol20.min() * 100:.1f}% annualised ({vol20.idxmin().date()})")
print(f"  Wildest 20-day stretch: {vol20.max() * 100:.1f}% annualised ({vol20.idxmax().date()})")
ac = ret.abs().autocorr(1)
print(f"  Volatility clustering (autocorr of |daily move|, lag 1): {ac:.2f} "
      f"-> {'big moves tend to come in bunches' if ac > 0.15 else 'weak clustering'}")

# ------------------------------------------------------------- 6. CALENDAR
mclose = c.groupby([c.index.year, c.index.month]).last()
mret = mclose.pct_change().dropna()
mret.index = pd.MultiIndex.from_tuples(mret.index, names=["Year", "Month"])
heat = mret.unstack("Month").reindex(columns=range(1, 13))
heat.columns = MONTHS
heat.to_csv(f"{OUT_DIR}/monthly_returns.csv")

month_stats = pd.DataFrame({
    "avg_%": mret.groupby(level="Month").mean().mul(100).round(2),
    "median_%": mret.groupby(level="Month").median().mul(100).round(2),
    "pct_positive": mret.gt(0).groupby(level="Month").mean().mul(100).round(0),
    "n_years": mret.groupby(level="Month").size(),
})
month_stats.index = [MONTHS[m - 1] for m in month_stats.index]

wd = ret.groupby(ret.index.dayofweek)
wd_stats = pd.DataFrame({
    "avg_%": wd.mean().mul(100).round(3),
    "pct_positive": wd.apply(lambda x: (x > 0).mean() * 100).round(0),
    "n_days": wd.size(),
})
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
wd_stats.index = [DAY_NAMES[i] for i in wd_stats.index]   # Sat/Sun = special sessions

print("\n" + "=" * 70)
print("6. CALENDAR PATTERNS")
print("=" * 70)
print("Month of year (only ~6-7 observations per month -> treat as anecdotes):")
print(month_stats.to_string())
print("\nDay of week:")
print(wd_stats.to_string())

# -------------------------------------------------- 7. RETURN STRUCTURE
sign = np.sign(ret)
rid = (sign != sign.shift()).cumsum()
streaks = sign.groupby(rid).agg(["first", "size"])
up_len = streaks[streaks["first"] > 0]["size"]
dn_len = streaks[streaks["first"] < 0]["size"]


def streak_end(rid_series, runs_df, longest):
    gid = runs_df["size"].idxmax()
    seg = rid_series[rid_series == gid].index
    return f"{seg[0].date()} -> {seg[-1].date()} ({longest} days)"


print("\n" + "=" * 70)
print("7. RETURN STRUCTURE")
print("=" * 70)
print(f"Up days {ret.gt(0).mean() * 100:.0f}% | average up day {pct(ret[ret > 0].mean(), 2)} "
      f"| average down day {pct(ret[ret < 0].mean(), 2)}")
print(f"Days with a move beyond +/-2%: {(ret.abs() > 0.02).sum()}  "
      f"({(ret.abs() > 0.02).mean() * 100:.1f}% of days)")
print(f"Longest winning streak : {streak_end(rid, up_len.to_frame('size'), up_len.max())}")
print(f"Longest losing streak  : {streak_end(rid, dn_len.to_frame('size'), dn_len.max())}")

order = ret.sort_values()
n = 10
without_best = (1 + ret.drop(order.index[-n:])).prod() - 1
without_worst = (1 + ret.drop(order.index[:n])).prod() - 1
print(f"\nTotal return                : {pct(total)}")
print(f"... without the {n} best days  : {pct(without_best)}")
print(f"... without the {n} worst days : {pct(without_worst)}")
print("  -> a handful of extreme days drive a large share of the outcome")
top = order.index[-n:]
n_in_fall = int(sum(underwater.loc[d] < -0.05 for d in top))
print(f"  Of the {n} best days, {n_in_fall} happened while the index was already "
      f">5% below its high (big rallies often start inside falls)")

roll12 = c.pct_change(252).dropna()
if len(roll12):
    print(f"\nRolling 1-year return: best {pct(roll12.max())} | worst {pct(roll12.min())} "
          f"| positive {roll12.gt(0).mean() * 100:.0f}% of the time")

# ------------------------------------------------------------ CHARTS
UP, DOWN = "#2e9e5b", "#d64545"
fig, ax = plt.subplots(4, 1, figsize=(14, 13), sharex=True,
                       gridspec_kw={"height_ratios": [3.2, 1.2, 1.2, 0.5]})

a = ax[0]
for a_i, b_i in zip(nodes[:-1], nodes[1:]):
    col = UP if c.iloc[b_i] > c.iloc[a_i] else DOWN
    a.axvspan(c.index[a_i], c.index[b_i], color=col, alpha=0.10, lw=0)
a.plot(c.index, c, color="black", lw=1.1, label="Nifty 50")
a.plot(sma50.index, sma50, color="tab:blue", lw=0.9, label="50-day avg")
a.plot(sma200.index, sma200, color="tab:purple", lw=1.1, label="200-day avg")
for a_i, b_i in zip(nodes[:-1], nodes[1:]):
    r = c.iloc[b_i] / c.iloc[a_i] - 1
    mid = c.index[a_i] + (c.index[b_i] - c.index[a_i]) / 2
    a.annotate(f"{r * 100:+.0f}%", (mid, c.iloc[a_i:b_i + 1].max()),
               ha="center", va="bottom", fontsize=8, color=UP if r > 0 else DOWN)
a.set_yscale("log")
a.set_title(f"Nifty 50: bull/bear phases (shaded, {REVERSAL * 100:.0f}% reversal rule), "
            "with 50/200-day averages")
a.legend(loc="upper left", ncol=3)

ax[1].fill_between(underwater.index, underwater * 100, 0, color=DOWN, alpha=0.5)
ax[1].set_ylabel("Drawdown %")
ax[1].set_title("Distance below previous peak", fontsize=10)

ax[2].plot(vol20.index, vol20 * 100, color="tab:orange", lw=1)
ax[2].axhline(vol20.median() * 100, color="gray", ls="--", lw=0.8)
ax[2].set_ylabel("Ann. vol %")
ax[2].set_title("20-day realised volatility (dashed = median)", fontsize=10)

scol = {"Bull (above 200DMA)": UP, "Bear (below 200DMA)": DOWN, "Mixed/Transition": "#e0a030"}
for s_name, col in scol.items():
    idx = st[st == s_name].index
    ax[3].vlines(idx, 0, 1, color=col, lw=1.2)
ax[3].set_yticks([])
ax[3].legend(handles=[Patch(color=v, label=k) for k, v in scol.items()],
             loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.4), fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cycles_overview.png", dpi=130)

fig2, bx = plt.subplots(2, 2, figsize=(15, 10))
# yearly returns
yr = year_ret
bx[0, 0].bar(yr.index.astype(str), yr * 100, color=[UP if v > 0 else DOWN for v in yr])
for i, v in enumerate(yr):
    bx[0, 0].text(i, v * 100 + (1 if v > 0 else -3), f"{v * 100:+.0f}%", ha="center", fontsize=9)
bx[0, 0].set_title("Calendar-year returns (first & last year partial)")
bx[0, 0].axhline(0, color="black", lw=0.6)

# heatmap
hv = heat.values * 100
vmax = np.nanmax(np.abs(hv))
im = bx[0, 1].imshow(hv, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
bx[0, 1].set_xticks(range(12)); bx[0, 1].set_xticklabels(MONTHS)
bx[0, 1].set_yticks(range(len(heat))); bx[0, 1].set_yticklabels(heat.index)
for i in range(hv.shape[0]):
    for j in range(hv.shape[1]):
        if not np.isnan(hv[i, j]):
            bx[0, 1].text(j, i, f"{hv[i, j]:.1f}", ha="center", va="center", fontsize=8)
bx[0, 1].set_title("Monthly returns % (year x month)")
plt.colorbar(im, ax=bx[0, 1], shrink=0.8)

# month of year
bx[1, 0].bar(month_stats.index, month_stats["avg_%"],
             color=[UP if v > 0 else DOWN for v in month_stats["avg_%"]])
for i, (m, row) in enumerate(month_stats.iterrows()):
    bx[1, 0].text(i, row["avg_%"] + (0.15 if row["avg_%"] > 0 else -0.45),
                  f"{int(row['pct_positive'])}%\nup", ha="center", fontsize=8)
bx[1, 0].axhline(0, color="black", lw=0.6)
bx[1, 0].set_title("Average return by month of year (labels = % of years positive)")

# rolling 12m
if len(roll12):
    bx[1, 1].plot(roll12.index, roll12 * 100, color="tab:blue")
    bx[1, 1].fill_between(roll12.index, roll12 * 100, 0, where=roll12 > 0, color=UP, alpha=0.25)
    bx[1, 1].fill_between(roll12.index, roll12 * 100, 0, where=roll12 < 0, color=DOWN, alpha=0.25)
    bx[1, 1].axhline(0, color="black", lw=0.6)
bx[1, 1].set_title("Rolling 1-year return %")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/calendar_and_returns.png", dpi=130)

print(f"\nSaved in ./{OUT_DIR}/: cycles_overview.png, calendar_and_returns.png,")
print("  phases.csv, drawdowns.csv, trend_state_runs.csv, monthly_returns.csv")
print("Descriptive analysis of past data only - not a forecast or investment advice.")
plt.show()
