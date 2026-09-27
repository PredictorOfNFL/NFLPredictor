"""
NFL Model Board builder (version 2)
===================================
Downloads the latest NFL data, runs the game and player models, and writes a
website file. What it builds:
  - Game predictions (spread, win %, total points) with a quarterback adjustment
  - Weather forecasts for this week's outdoor games
  - Player prop projections and over/under chances
  - Fantasy rankings, start/sit comparisons, and rising-usage players
  - A results tracker that grades every pick after the games
  - Player pages with game logs and usage trends

How to run (from Command Prompt, in the folder with this file):
    py nfl_site.py
Then double-click nfl_model_board.html to open it in your browser.
It also saves history.json in the same folder. Keep that file: it's how the
results tracker remembers what the site predicted before each game.

Optional: to fill in sportsbook prop lines automatically, get a free key at
https://the-odds-api.com and paste it into ODDS_API_KEY below.
(On GitHub, add it as a secret named ODDS_API_KEY instead.)
"""
import os, re, json, urllib.request, urllib.parse
import numpy as np, pandas as pd, nflreadpy as nfl
from scipy.stats import norm, poisson
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

# ------------------------------- SETTINGS -------------------------------
CURRENT_SEASON = 2026          # change to 2027 next year
FIRST_SEASON = 2015            # oldest season the game model learns from
PROP_START = 2018              # oldest season the player models learn from
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")   # or paste your key between the quotes
OUT_FILE = os.environ.get("OUTPUT_FILE", "nfl_model_board.html")
# Web address of your assistant server (Cloudflare Worker). On GitHub, add it as
# an Actions variable named ASSISTANT_URL. Leave "" to hide the assistant.
ASSISTANT_URL = os.environ.get("ASSISTANT_URL", "")
HISTORY_FILE = "history.json"
# Edge alerts: you get a GitHub notification (email / app) when the model finds
ALERT_PROP_EDGE = 0.08         # a prop at least 8% better than the best sportsbook odds
ALERT_SPREAD_EDGE = 3.5        # a spread at least 3.5 points away from Vegas
ALERT_TOTAL_EDGE = 4.0         # a total at least 4 points away from Vegas
# Prop line pulls per week. 1 fits the free Odds API plan. 2 (Friday + Sunday
# morning) lets the results tracker measure closing-line value, but needs a paid plan.
PROP_PULLS_PER_WEEK = 1
SIMULATIONS = 10000            # seasons simulated for playoff odds
ROLLING_GAMES, MARGIN_STD = 10, 13.5
seasons = list(range(FIRST_SEASON, CURRENT_SEASON + 1))
POS = ["QB", "RB", "WR", "TE"]

def nz(v, d=1):
    """Round a number for the website, or None if it's missing."""
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        return None
    return round(float(v), d)

def txt(v):
    return v if isinstance(v, str) else None

# Stadium locations (for weather) and roof type
STADIUMS = {  # stadium_id: (lat, lon, roof)  roof: out / dome / retract
    "ATL97": (33.7554, -84.4008, "retract"), "BAL00": (39.2780, -76.6227, "out"),
    "BOS00": (42.0909, -71.2643, "out"), "BUF00": (42.7738, -78.7870, "out"),
    "CAR00": (35.2258, -80.8528, "out"), "CHI98": (41.8623, -87.6167, "out"),
    "CIN00": (39.0955, -84.5161, "out"), "CLE00": (41.5061, -81.6995, "out"),
    "DAL00": (32.7473, -97.0945, "retract"), "DEN00": (39.7439, -105.0201, "out"),
    "DET00": (42.3400, -83.0456, "dome"), "GNB00": (44.5013, -88.0622, "out"),
    "HOU00": (29.6847, -95.4107, "retract"), "IND00": (39.7601, -86.1639, "retract"),
    "JAX00": (30.3239, -81.6373, "out"), "KAN00": (39.0489, -94.4839, "out"),
    "LAX01": (33.9535, -118.3392, "dome"), "LON00": (51.5560, -0.2796, "out"),
    "LON02": (51.6043, -0.0664, "retract"), "MAD01": (40.4531, -3.6883, "retract"),
    "MEL00": (-37.8200, 144.9834, "out"), "MEX00": (19.3029, -99.1505, "out"),
    "MIA00": (25.9580, -80.2389, "out"), "MIN01": (44.9737, -93.2581, "dome"),
    "MUN01": (48.2188, 11.6247, "out"), "NAS00": (36.1665, -86.7713, "out"),
    "NOR00": (29.9511, -90.0812, "dome"), "NYC01": (40.8135, -74.0745, "out"),
    "PAR00": (48.9245, 2.3602, "out"), "PHI00": (39.9008, -75.1675, "out"),
    "PHO00": (33.5276, -112.2626, "retract"), "PIT00": (40.4468, -80.0158, "out"),
    "RIO00": (-22.9122, -43.2302, "out"), "SEA00": (47.5952, -122.3316, "out"),
    "SFO01": (37.4030, -121.9700, "out"), "TAM00": (27.9759, -82.5033, "out"),
    "VEG00": (36.0909, -115.1833, "dome"), "WAS00": (38.9078, -76.8645, "out"),
}

def load_history():
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            h = json.load(f)
        if h.get("season") == CURRENT_SEASON:
            return h
    except Exception:
        pass
    return {"season": CURRENT_SEASON, "games": {}, "props": {}, "lines": {}}

history = load_history()
for k in ("first", "lines_first", "game_odds", "alerted", "pulls"):
    history.setdefault(k, {} if k != "alerted" else [])
# The website's HTML/JavaScript is stored at the bottom of this file
TEMPLATE = open(__file__, encoding="utf-8").read().split("#" + "__TEMPLATE__\n", 1)[1].rsplit("\n#__END__", 1)[0]

# ======================================================================
print("Step 1/7: Downloading data (the first run takes a few minutes)...")
# ======================================================================
sched_all = nfl.load_schedules(seasons).to_pandas()
games = sched_all[sched_all.game_type == "REG"].copy()
pbp_cols = ["game_id", "season", "week", "posteam", "defteam", "play_type", "epa",
            "success", "qb_dropback", "qb_epa", "id", "sack", "qb_hit"]
pbp = pd.concat([nfl.load_pbp([y]).select(pbp_cols).to_pandas() for y in seasons], ignore_index=True)

ps_all = nfl.load_player_stats(list(range(PROP_START, CURRENT_SEASON + 1))).to_pandas()
ps_all = ps_all[(ps_all.season_type == "REG") & ps_all.position.isin(POS)].copy()

snap_seasons = [CURRENT_SEASON - 1, CURRENT_SEASON]
snaps = nfl.load_snap_counts(snap_seasons).to_pandas()
ros = nfl.load_rosters(snap_seasons).to_pandas()[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
snaps = snaps.merge(ros, left_on="pfr_player_id", right_on="pfr_id")[["gsis_id", "season", "week", "offense_pct"]]
snaps = snaps.drop_duplicates(["gsis_id", "season", "week"])
roster_cur = nfl.load_rosters([CURRENT_SEASON]).to_pandas()
team_info = nfl.load_teams().to_pandas()
ps_all = ps_all.merge(snaps, left_on=["player_id", "season", "week"],
                      right_on=["gsis_id", "season", "week"], how="left").drop(columns="gsis_id")

cur_games = games[games.season == CURRENT_SEASON]
open_games = cur_games[cur_games.result.isna()]
pwk = int(open_games.week.min()) if len(open_games) else None
week_games = open_games[open_games.week == pwk] if pwk else open_games.iloc[0:0]
try:
    inj = nfl.load_injuries([CURRENT_SEASON]).to_pandas()
    inj = inj[inj.week == pwk] if pwk else inj.iloc[0:0]
except Exception:
    inj = pd.DataFrame(columns=["gsis_id", "report_status"])
OUT_IDS = set(inj.loc[inj.report_status.isin(["Out", "Doubtful"]), "gsis_id"])
Q_IDS = set(inj.loc[inj.report_status == "Questionable", "gsis_id"])
names = (ps_all.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
               .set_index("player_id").player_display_name.to_dict())

# ======================================================================
print("Step 2/7: Team form, quarterback ratings and weather...")
# ======================================================================
plays = pbp[pbp.play_type.isin(["pass", "run"]) & pbp.epa.notna()].copy()
plays["pass_epa"] = np.where(plays.play_type == "pass", plays.epa, np.nan)
plays["rush_epa"] = np.where(plays.play_type == "run", plays.epa, np.nan)
def summarize(side):
    return (plays.groupby(["game_id", "season", "week", side])
                 .agg(epa=("epa", "mean"), pass_epa=("pass_epa", "mean"),
                      rush_epa=("rush_epa", "mean"), success=("success", "mean"))
                 .reset_index().rename(columns={side: "team"}))
tg = summarize("posteam").merge(summarize("defteam"), on=["game_id", "season", "week", "team"],
                                suffixes=("_o", "_d"))
REN = {"epa_o": "off_epa", "pass_epa_o": "off_pass_epa", "rush_epa_o": "off_rush_epa",
       "success_o": "off_success", "epa_d": "def_epa", "pass_epa_d": "def_pass_epa",
       "rush_epa_d": "def_rush_epa", "success_d": "def_success"}
tg = tg.rename(columns=REN).sort_values(["team", "season", "week"])
S = list(REN.values())
form = tg.copy()
form[S] = tg.groupby("team")[S].transform(
    lambda s: s.shift(1).ewm(span=ROLLING_GAMES, min_periods=3).mean())
latest = tg.groupby("team")[S].apply(lambda d: d.ewm(span=ROLLING_GAMES).mean().iloc[-1])

# Pass protection and pass rush: share of dropbacks with a sack or QB hit
dbk = pbp[pbp.qb_dropback == 1].copy()
dbk["press"] = ((dbk.sack == 1) | (dbk.qb_hit == 1)).astype(float)
pg = dbk.groupby(["game_id", "season", "week", "posteam", "defteam"]).press.mean().reset_index()
pt = (pg.rename(columns={"posteam": "team", "press": "off_pr"})[["game_id", "season", "week", "team", "off_pr"]]
        .merge(pg.rename(columns={"defteam": "team", "press": "def_pr"})[["game_id", "team", "def_pr"]], on=["game_id", "team"])
        .sort_values(["team", "season", "week"]))
PR_BEFORE, PR_NOW = {}, {}
for c in ["off_pr", "def_pr"]:
    before = pt.groupby("team")[c].transform(lambda x: x.shift(1).ewm(span=ROLLING_GAMES, min_periods=3).mean())
    PR_BEFORE[c] = dict(zip(zip(pt.game_id, pt.team), before))
    PR_NOW[c] = pt.groupby("team")[c].apply(lambda x: x.ewm(span=ROLLING_GAMES).mean().iloc[-1]).to_dict()

# Quarterback ratings: EPA per dropback, recent games weighted more, and
# pulled toward a backup-level number until a QB has a real track record.
QB_PRIOR, QB_K = -0.02, 120
db = pbp[(pbp.qb_dropback == 1) & pbp.qb_epa.notna() & pbp.id.notna()]
qg = (db.groupby(["id", "game_id", "season", "week", "posteam"])
        .agg(n=("qb_epa", "size"), e=("qb_epa", "sum")).reset_index()
        .sort_values(["id", "season", "week"]))
def shrink(en, nn, cnt):
    eff = (nn * np.minimum(cnt, 30)).fillna(0)
    return ((en / nn).fillna(QB_PRIOR) * eff + QB_PRIOR * QB_K) / (eff + QB_K)
g_ = qg.groupby("id")
qg["rating"] = shrink(g_.e.transform(lambda s: s.shift(1).ewm(halflife=16, min_periods=1).mean()),
                      g_.n.transform(lambda s: s.shift(1).ewm(halflife=16, min_periods=1).mean()),
                      g_.cumcount())
now = qg.groupby("id").agg(en=("e", lambda s: s.ewm(halflife=16).mean().iloc[-1]),
                           nn=("n", lambda s: s.ewm(halflife=16).mean().iloc[-1]), cnt=("n", "size"))
QB_NOW = shrink(now.en, now.nn, now.cnt).to_dict()
QB_BEFORE = qg.set_index(["id", "game_id"]).rating.to_dict()
# Each team's most recent main passer (used when the listed starter is out)
cur_qg = qg[(qg.season == CURRENT_SEASON) & ~qg.id.isin(OUT_IDS)].sort_values(["week", "n"])
LAST_QB = cur_qg.groupby("posteam").id.last().to_dict()

def starter(row, side):
    """Starting QB id for a game: the actual starter for played games, the
    listed starter for upcoming games unless he's ruled out."""
    listed = row[f"{side}_qb_id"]
    team = row[f"{side}_team"]
    if pd.notna(row["result"]):
        return listed
    if isinstance(listed, str) and not (row["week"] == pwk and listed in OUT_IDS):
        return listed
    return LAST_QB.get(team, listed)

def qb_rating(pid, gid, played):
    if not isinstance(pid, str):
        return QB_PRIOR
    if played:
        return QB_BEFORE.get((pid, gid), QB_NOW.get(pid, QB_PRIOR))
    return QB_NOW.get(pid, QB_PRIOR)

for side in ["home", "away"]:
    games[f"{side}_qb_id"] = [starter(r, side) for _, r in games.iterrows()]
    games[f"{side}_qbr"] = [qb_rating(p, g, pd.notna(r)) for p, g, r in
                            zip(games[f"{side}_qb_id"], games.game_id, games.result)]
    games[f"{side}_qb_name"] = [names.get(p, n) if isinstance(p, str) else n
                                for p, n in zip(games[f"{side}_qb_id"], games[f"{side}_qb_name"])]

# Weather forecast for this week's games (free Open-Meteo service, no key)
WEATHER = {}
def get_forecast(g):
    st = STADIUMS.get(g.stadium_id)
    if not st:
        return None
    lat, lon, roof = st
    if roof == "dome":
        return {"roof": "dome"}
    q = urllib.parse.urlencode(dict(latitude=lat, longitude=lon, forecast_days=16,
        hourly="temperature_2m,precipitation_probability,wind_speed_10m,wind_gusts_10m",
        wind_speed_unit="mph", temperature_unit="fahrenheit", timezone="America/New_York"))
    with urllib.request.urlopen(f"https://api.open-meteo.com/v1/forecast?{q}", timeout=20) as r:
        h = json.load(r)["hourly"]
    stamp = f"{g.gameday}T{str(g.gametime)[:2]}:00"
    if stamp not in h["time"]:
        return None
    i = h["time"].index(stamp)
    return {"roof": roof, "temp": nz(h["temperature_2m"][i], 0), "wind": nz(h["wind_speed_10m"][i], 0),
            "gust": nz(h["wind_gusts_10m"][i], 0), "rain": nz(h["precipitation_probability"][i], 0)}
weather_fail = 0
for _, g in week_games.iterrows():
    try:
        w = get_forecast(g)
        if w:
            WEATHER[g.game_id] = w
    except Exception:
        weather_fail += 1
if weather_fail:
    print(f"   Weather forecast unavailable for {weather_fail} games right now.")

# ======================================================================
print("Step 3/7: Training the game models (spread and total points)...")
# ======================================================================
def attach(df, side):
    tc = f"{side}_team"
    m = df.merge(form[["game_id", "team"] + S], left_on=["game_id", tc],
                 right_on=["game_id", "team"], how="left").drop(columns="team")
    un = m[S[0]].isna() & m.result.isna()
    fill = latest.reindex(m.loc[un, tc]).values
    m.loc[un, S] = fill
    return m.rename(columns={c: f"{side}_{c}" for c in S})
df = attach(attach(games, "home"), "away")
df["pass_matchup"] = df.home_off_pass_epa - df.away_off_pass_epa - (df.home_def_pass_epa - df.away_def_pass_epa)
df["rush_matchup"] = df.home_off_rush_epa - df.away_off_rush_epa - (df.home_def_rush_epa - df.away_def_rush_epa)
df["success_matchup"] = df.home_off_success - df.away_off_success - (df.home_def_success - df.away_def_success)
df["rest_diff"] = (df.home_rest - df.away_rest).fillna(0)
df["neutral_site"] = (df.location == "Neutral").astype(int)
df["div_game"] = df.div_game.fillna(0)
roof_type = df.stadium_id.map(lambda s: STADIUMS.get(s, (0, 0, "out"))[2])
indoor = df.roof.isin(["dome", "closed"]) | (df.result.isna() & roof_type.isin(["dome", "retract"]))
fc_wind = df.game_id.map(lambda g: WEATHER.get(g, {}).get("wind"))
df["wind"] = df.wind.fillna(fc_wind.astype(float))
df["wind"] = np.where(indoor, 0, df.wind.fillna(games.wind.median()))
df["high_wind"] = (df.wind >= 15).astype(int)
df["indoor"] = indoor.astype(int)
df["qb_diff"] = df.home_qbr - df.away_qbr
df["qb_sum"] = df.home_qbr + df.away_qbr
df["off_sum"] = df.home_off_epa + df.away_off_epa
df["def_sum"] = df.home_def_epa + df.away_def_epa
df["succ_sum"] = df.home_off_success + df.away_off_success + df.home_def_success + df.away_def_success
df["total"] = df.home_score + df.away_score
F_SPREAD = ["pass_matchup", "rush_matchup", "success_matchup", "rest_diff", "neutral_site",
            "div_game", "high_wind", "qb_diff"]
F_TOTAL = ["off_sum", "def_sum", "succ_sum", "qb_sum", "high_wind", "indoor", "wind"]
df = df.dropna(subset=F_SPREAD)
played = df[df.result.notna()]

# Honest test: learn from 2015-2023, grade on 2024-2025
tr, te = played[played.season < CURRENT_SEASON - 2], played[played.season.isin([CURRENT_SEASON - 2, CURRENT_SEASON - 1])]
te = te[te.result != 0]
ms = Ridge(alpha=1.0).fit(tr[F_SPREAD], tr.result)
mt = Ridge(alpha=1.0).fit(tr[F_TOTAL], tr.total)
p, pt = ms.predict(te[F_SPREAD]), mt.predict(te[F_TOTAL])
TOTAL_STD = float(np.std(tr.total - mt.predict(tr[F_TOTAL])))
nopush = te.result != te.spread_line
big = nopush & (np.abs(p - te.spread_line) >= 3)
tot_np = te.total != te.total_line
game_report = dict(
    seasons=f"{CURRENT_SEASON - 2}-{str(CURRENT_SEASON - 1)[2:]}", n=int(len(te)),
    acc=nz(((p > 0) == (te.result > 0)).mean() * 100), vegas_acc=nz(((te.spread_line > 0) == (te.result > 0)).mean() * 100),
    mae=nz(np.abs(p - te.result).mean()), vegas_mae=nz(np.abs(te.spread_line - te.result).mean()),
    ats3=nz(((te.result > te.spread_line) == (p > te.spread_line))[big].mean() * 100), ats3_n=int(big.sum()),
    tmae=nz(np.abs(pt - te.total).mean()), vegas_tmae=nz(np.abs(te.total_line - te.total).mean()),
    ou=nz(((te.total > te.total_line) == (pt > te.total_line))[tot_np].mean() * 100))

# Predictions for this season. Finished games use a model trained only on
# earlier seasons (so the record is honest); upcoming games use everything.
cur = df[df.season == CURRENT_SEASON].copy()
ms_pre = Ridge(alpha=1.0).fit(played[played.season < CURRENT_SEASON][F_SPREAD], played[played.season < CURRENT_SEASON].result)
mt_pre = Ridge(alpha=1.0).fit(played[played.season < CURRENT_SEASON][F_TOTAL], played[played.season < CURRENT_SEASON].total)
ms_all = Ridge(alpha=1.0).fit(played[F_SPREAD], played.result)
mt_all = Ridge(alpha=1.0).fit(played[F_TOTAL], played.total)
fut = cur.result.isna()
cur["pred"] = np.where(fut, ms_all.predict(cur[F_SPREAD]), ms_pre.predict(cur[F_SPREAD]))
cur["tpred"] = np.where(fut, mt_all.predict(cur[F_TOTAL]), mt_pre.predict(cur[F_TOTAL]))
# If the site saved its prediction before kickoff, grade that one instead
for i, r in cur.iterrows():
    snap = history["games"].get(r.game_id)
    if snap and pd.notna(r.result):
        cur.at[i, "pred"], cur.at[i, "tpred"] = snap["pred"], snap["tpred"]
cur["prob"] = norm.cdf(cur.pred / MARGIN_STD)
for _, r in cur[fut & (cur.week == pwk)].iterrows():   # snapshot this week's picks
    history["games"][r.game_id] = {"pred": round(float(r.pred), 2), "tpred": round(float(r.tpred), 2)}
    if r.game_id not in history["first"] and pd.notna(r.spread_line):
        # the first line and pick the site saw, for closing-line value
        history["first"][r.game_id] = {"spread": float(r.spread_line), "pred": round(float(r.pred), 2),
                                       "total": nz(r.total_line), "tpred": round(float(r.tpred), 2)}

lt = latest[latest.index.isin(set(cur_games.home_team))]
ranks = pd.DataFrame({
    "off": lt.off_epa.rank(ascending=False), "def": lt.def_epa.rank(ascending=True),
    "pass_off": lt.off_pass_epa.rank(ascending=False), "rush_off": lt.off_rush_epa.rank(ascending=False),
    "pass_def": lt.def_pass_epa.rank(ascending=True), "rush_def": lt.def_rush_epa.rank(ascending=True),
    "pass_block": pd.Series({t: PR_NOW["off_pr"].get(t) for t in lt.index}).rank(ascending=True),
    "pass_rush": pd.Series({t: PR_NOW["def_pr"].get(t) for t in lt.index}).rank(ascending=False)}).astype(int)
teams = {t: {k: int(v) for k, v in ranks.loc[t].items()} for t in lt.index}

season_games = []
for _, g in cur.sort_values(["week", "gameday", "gametime"]).iterrows():
    season_games.append(dict(
        id=g.game_id, wk=int(g.week), date=g.gameday, time=g.gametime, day=g.weekday,
        away=g.away_team, home=g.home_team, aqb=txt(g.away_qb_name), hqb=txt(g.home_qb_name),
        aqbr=nz(g.away_qbr, 3), hqbr=nz(g.home_qbr, 3),
        as_=nz(g.away_score, 0), hs=nz(g.home_score, 0),
        spread=nz(g.spread_line), total=nz(g.total_line),
        hml=nz(g.home_moneyline, 0), aml=nz(g.away_moneyline, 0),
        pred=nz(g.pred), prob=nz(g.prob, 3), tpred=nz(g.tpred),
        venue=txt(g.stadium), neutral=bool(g.location == "Neutral"), wx=WEATHER.get(g.game_id),
        first=history["first"].get(g.game_id)))

# ======================================================================
print("Step 4/7: Player stats, game logs and usage trends...")
# ======================================================================
psc = ps_all[ps_all.season == CURRENT_SEASON]
def s(col): return psc.groupby("player_id")[col].sum()
g = psc.groupby("player_id")
agg = pd.DataFrame({
    "name": g.player_display_name.last(), "pos": g.position.last(), "team": g.team.last(),
    "gp": g.week.nunique(), "snap": g.offense_pct.mean(),
    "att": s("attempts"), "pyd": s("passing_yards"), "ptd": s("passing_tds"),
    "int": s("passing_interceptions"), "sacks": s("sacks_suffered"), "pepa": s("passing_epa"),
    "pair": s("passing_air_yards"), "car": s("carries"), "ryd": s("rushing_yards"), "rtd": s("rushing_tds"),
    "repa": s("rushing_epa"), "tgt": s("targets"), "rec": s("receptions"), "recyd": s("receiving_yards"),
    "rectd": s("receiving_tds"), "rair": s("receiving_air_yards"), "yac": s("receiving_yards_after_catch"),
    "recepa": s("receiving_epa"), "ppr": s("fantasy_points_ppr"),
    "tshare": g.target_share.mean(), "ashare": g.air_yards_share.mean(), "wopr": g.wopr.mean()})
cp = psc.dropna(subset=["passing_cpoe"])
agg["cpoe"] = cp.groupby("player_id").apply(lambda d: np.average(d.passing_cpoe, weights=d.attempts.clip(lower=1)))
agg = agg.reset_index()
keep = ((agg.pos == "QB") & (agg.att >= 15)) | ((agg.pos == "RB") & (agg.car + agg.tgt >= 8)) | \
       (agg.pos.isin(["WR", "TE"]) & (agg.tgt >= 4))
players = []
for _, p in agg[keep].iterrows():
    gp = p.gp
    d = dict(n=p["name"], pos=p.pos, tm=p.team, gp=int(gp),
             snap=nz(p.snap * 100 if pd.notna(p.snap) else None, 0), ppg=nz(p.ppr / gp))
    if p.pos == "QB":
        d.update(epa=nz(p.pepa / (p.att + p.sacks), 2), cpoe=nz(p.cpoe), adot=nz(p.pair / p.att),
                 ypa=nz(p.pyd / p.att), td=int(p.ptd), int_=int(p["int"]), ypg=nz(p.pyd / gp, 0), rypg=nz(p.ryd / gp, 0))
    elif p.pos == "RB":
        d.update(cpg=nz(p.car / gp), ypc=nz(p.ryd / p.car) if p.car else None,
                 repa=nz(p.repa / p.car, 2) if p.car else None, tpg=nz(p.tgt / gp), ts=nz(p.tshare * 100),
                 ypg=nz((p.ryd + p.recyd) / gp, 0), td=int(p.rtd + p.rectd))
    else:
        d.update(tpg=nz(p.tgt / gp), ts=nz(p.tshare * 100), as_=nz(p.ashare * 100), wopr=nz(p.wopr, 2),
                 adot=nz(p.rair / p.tgt), racr=nz(p.recyd / p.rair, 2) if p.rair > 0 else None,
                 yac=nz(p.yac / p.rec) if p.rec else None, epat=nz(p.recepa / p.tgt, 2),
                 ypg=nz(p.recyd / gp, 0), td=int(p.rectd), cr=nz(p.rec / p.tgt * 100, 0))
    players.append(d)

# Game logs: every game from last season and this season for relevant players
recent_ps = ps_all[ps_all.season >= CURRENT_SEASON - 1].sort_values(["season", "week"])
logs = {}
for pid, grp in recent_ps.groupby("player_id"):
    if grp.season.max() < CURRENT_SEASON:
        continue
    name = grp.player_display_name.iloc[-1]
    logs[name] = [[int(r.season), int(r.week), r.opponent_team,
                   nz(r.offense_pct * 100 if pd.notna(r.offense_pct) else None, 0),
                   int(r.attempts or 0), int(r.completions or 0), int(r.passing_yards or 0), int(r.passing_tds or 0),
                   int(r.passing_interceptions or 0), int(r.carries or 0), int(r.rushing_yards or 0),
                   int(r.rushing_tds or 0), int(r.targets or 0), int(r.receptions or 0),
                   int(r.receiving_yards or 0), int(r.receiving_tds or 0), nz(r.fantasy_points_ppr)]
                  for r in pd.concat([grp[grp.season < CURRENT_SEASON].tail(max(0, 12 - int((grp.season == CURRENT_SEASON).sum()))),
                                      grp[grp.season == CURRENT_SEASON]]).itertuples()]

# Rising usage: last 2 games compared with the 6 games before that
risers = []
for pid, grp in recent_ps[recent_ps.position.isin(["RB", "WR", "TE"])].groupby("player_id"):
    grp = grp[grp.team == grp.team.iloc[-1]].tail(8)     # only games with his current team
    if (grp.season == CURRENT_SEASON).sum() < 2 or len(grp) < 4:
        continue
    last, before = grp.tail(2), grp.iloc[:-2]
    opp_l = (last.targets + last.carries).mean(); opp_b = (before.targets + before.carries).mean()
    snap_l = last.offense_pct.mean() * 100; snap_b = before.offense_pct.mean() * 100
    tsh_l = last.target_share.mean() * 100; tsh_b = before.target_share.mean() * 100
    if opp_l < 5 or pd.isna(snap_l):
        continue
    d_opp, d_snap = opp_l - opp_b, (snap_l - snap_b) if pd.notna(snap_b) else 0
    score = d_opp + 0.15 * d_snap
    if score <= 1.5:
        continue
    risers.append(dict(n=grp.player_display_name.iloc[-1], pos=grp.position.iloc[-1], tm=grp.team.iloc[-1],
                       opp_l=nz(opp_l), opp_b=nz(opp_b), snap_l=nz(snap_l, 0), snap_b=nz(snap_b, 0),
                       ts_l=nz(tsh_l), ts_b=nz(tsh_b), ppr_l=nz(last.fantasy_points_ppr.mean()),
                       score=nz(score, 2)))
risers = sorted(risers, key=lambda r: -r["score"])[:25]

# ======================================================================
print("Step 5/7: Projecting player stat lines and fantasy points...")
# ======================================================================
QUANTS = [0.1, 0.25, 0.5, 0.75, 0.9]
PROPS = {  # stat: (positions, usage stat, model type, Odds API market, label)
    "passing_yards":   (["QB"], "attempts", "q", "player_pass_yds", "Pass yds"),
    "passing_tds":     (["QB"], "attempts", "pois", "player_pass_tds", "Pass TDs"),
    "rushing_yards":   (["QB", "RB"], "carries", "q", "player_rush_yds", "Rush yds"),
    "receptions":      (["RB", "WR", "TE"], "targets", "q", "player_receptions", "Receptions"),
    "receiving_yards": (["RB", "WR", "TE"], "targets", "q", "player_reception_yds", "Rec yds"),
    "tds":             (["QB", "RB", "WR", "TE"], "touches", "pois", "player_anytime_td", "Anytime TD"),
}
HGB = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80, random_state=0)

pp = ps_all[["player_id", "player_display_name", "position", "season", "week", "game_id", "team",
             "opponent_team", "attempts", "passing_yards", "passing_tds", "passing_interceptions", "carries",
             "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds",
             "target_share", "fantasy_points_ppr", "fantasy_points"]].copy()
pp["tds"] = pp.rushing_tds + pp.receiving_tds
pp["touches"] = pp.carries + pp.targets
backup = (pp.position == "QB") & (pp.attempts < 10)      # QB passing only counts from real starts
pp.loc[backup, ["attempts", "passing_yards", "passing_tds", "passing_interceptions"]] = np.nan

# Who plays this week: the starting QB, plus RB/WR/TE active in the team's
# last 2 games, minus anyone ruled Out or Doubtful.
curp = pp[pp.season == CURRENT_SEASON]
last_wk = curp.groupby("team").week.max()
recent = curp[curp.week >= curp.team.map(last_wk) - 1].drop_duplicates("player_id", keep="last")
up_rows = []
for _, gm in week_games.iterrows():
    row = cur[cur.game_id == gm.game_id]
    for tm, opp, side in [(gm.home_team, gm.away_team, "home"), (gm.away_team, gm.home_team, "away")]:
        base = dict(season=CURRENT_SEASON, week=int(gm.week), game_id=gm.game_id, team=tm, opponent_team=opp)
        qid = row[f"{side}_qb_id"].iloc[0] if len(row) else None
        if isinstance(qid, str):
            up_rows.append(dict(player_id=qid, player_display_name=names.get(qid, row[f"{side}_qb_name"].iloc[0]),
                                position="QB", **base))
        for _, p in recent[(recent.team == tm) & (recent.position != "QB")].iterrows():
            if p.player_id not in OUT_IDS:
                up_rows.append(dict(player_id=p.player_id, player_display_name=p.player_display_name,
                                    position=p.position, **base))
up = pd.DataFrame(up_rows, columns=["player_id", "player_display_name", "position", "season", "week",
                                    "game_id", "team", "opponent_team"])
up["upcoming"] = True
pp["upcoming"] = False
pp = pd.concat([pp, up], ignore_index=True).sort_values(["player_id", "season", "week"])

EWM_STATS = set(PROPS) | {"attempts", "carries", "targets", "touches", "target_share",
                          "passing_interceptions", "fantasy_points_ppr"}
for st in EWM_STATS:
    pp[f"p_{st}"] = pp.groupby("player_id")[st].transform(lambda s: s.shift(1).ewm(span=8, min_periods=1).mean())
pp["n_prior"] = pp.groupby("player_id").cumcount()
pp["new_season"] = (pp.season != pp.groupby("player_id").season.shift(1)).astype(int)
pp["pos_code"] = pp.position.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3})
ALLOW = list(PROPS) + ["fantasy_points_ppr"]
allow = (pp.groupby(["game_id", "opponent_team", "position", "season", "week"])[ALLOW]
           .sum(min_count=1).reset_index().sort_values(["opponent_team", "position", "season", "week"]))
for st in ALLOW:
    allow[f"d_{st}"] = allow.groupby(["opponent_team", "position"])[st].transform(
        lambda s: s.shift(1).ewm(span=10, min_periods=1).mean())
pp = pp.merge(allow[["game_id", "opponent_team", "position"] + [f"d_{s}" for s in ALLOW]],
              on=["game_id", "opponent_team", "position"], how="left")
sl = games[["game_id", "home_team", "away_team", "spread_line", "total_line"]]
lines_tv = pd.concat([
    pd.DataFrame({"game_id": sl.game_id, "team": sl.home_team, "team_spread": sl.spread_line,
                  "implied": (sl.total_line + sl.spread_line) / 2}),
    pd.DataFrame({"game_id": sl.game_id, "team": sl.away_team, "team_spread": -sl.spread_line,
                  "implied": (sl.total_line - sl.spread_line) / 2})])
pp = pp.merge(lines_tv, on=["game_id", "team"], how="left")

# Pass protection of the player's team and pass rush of the opponent (for QBs)
pp["off_pr"] = [PR_BEFORE["off_pr"].get((g, t), PR_NOW["off_pr"].get(t)) if u else PR_BEFORE["off_pr"].get((g, t))
                for g, t, u in zip(pp.game_id, pp.team, pp.upcoming)]
pp["opp_def_pr"] = [PR_BEFORE["def_pr"].get((g, t), PR_NOW["def_pr"].get(t)) if u else PR_BEFORE["def_pr"].get((g, t))
                    for g, t, u in zip(pp.game_id, pp.opponent_team, pp.upcoming)]

# Vacated work: targets and carries left behind by teammates who played the
# team's previous game but not this one (or who are ruled out this week).
VAC_TGT, VAC_CAR = 5.5, 10
done_pp = pp[~pp.upcoming]
done_pp = done_pp.assign(
    c_targets=done_pp.groupby("player_id").targets.transform(lambda x: x.ewm(span=8, min_periods=1).mean()),
    c_carries=done_pp.groupby("player_id").carries.transform(lambda x: x.ewm(span=8, min_periods=1).mean()))
tgames = done_pp[["team", "season", "week", "game_id"]].drop_duplicates().sort_values(["team", "season", "week"])
tgames["prev_game"] = tgames.groupby("team").game_id.shift(1)
players_in = done_pp.groupby("game_id").player_id.apply(set).to_dict()
by_team_game = dict(tuple(done_pp[["game_id", "team", "player_id", "c_targets", "c_carries"]].groupby(["game_id", "team"])))
vac = {}
for r in tgames.itertuples():
    prev = by_team_game.get((r.prev_game, r.team)) if isinstance(r.prev_game, str) else None
    if prev is None:
        continue
    miss = prev[~prev.player_id.isin(players_in.get(r.game_id, set()))]
    vac[(r.game_id, r.team)] = (miss.c_targets.where(miss.c_targets >= VAC_TGT, 0).sum(),
                                miss.c_carries.where(miss.c_carries >= VAC_CAR, 0).sum())
latest_c = done_pp[done_pp.season == CURRENT_SEASON].drop_duplicates("player_id", keep="last").set_index("player_id")
BOOST = {}
for _, gm in week_games.iterrows():
    for tm in (gm.home_team, gm.away_team):
        outs = latest_c[(latest_c.team == tm) & latest_c.index.isin(OUT_IDS)]
        outs = outs[outs.index.isin(recent.player_id)]
        t_out, c_out = outs[outs.c_targets >= VAC_TGT], outs[outs.c_carries >= VAC_CAR]
        vac[(gm.game_id, tm)] = (t_out.c_targets.sum(), c_out.c_carries.sum())
        BOOST[tm] = {"tgt": list(t_out.player_display_name), "car": list(c_out.player_display_name)}
pp["vac_tgt"] = [vac.get((g, t), (0, 0))[0] for g, t in zip(pp.game_id, pp.team)]
pp["vac_car"] = [vac.get((g, t), (0, 0))[1] for g, t in zip(pp.game_id, pp.team)]
pp["vac_tgt_share"] = pp.vac_tgt * pp.p_target_share

def feats(st):
    F = [f"p_{st}", f"p_{PROPS[st][1]}", f"d_{st}", "implied", "team_spread", "n_prior", "new_season", "pos_code"]
    if st in ("receptions", "receiving_yards", "tds"):
        F.append("p_target_share")
    if st == "receiving_yards":
        F += ["vac_tgt", "vac_tgt_share"]
    if st == "rushing_yards":
        F += ["vac_car"]
    if st == "tds":
        F += ["vac_tgt", "vac_car"]
    if st in ("passing_yards", "passing_tds"):
        F += ["off_pr", "opp_def_pr"]
    return F

def predict_all(train, target):
    """Fit every stat model on `train` and add predictions to `target`."""
    out = target.copy()
    for st, (poss, use, kind, _, _) in PROPS.items():
        F = feats(st)
        trn = train[train.position.isin(poss)].dropna(subset=[st, f"p_{st}", f"p_{use}", "implied"])
        mask = out.position.isin(poss) & out[f"p_{st}"].notna() & out[f"p_{use}"].notna() & out.implied.notna()
        if not mask.any():
            continue
        if kind == "pois":
            m = HistGradientBoostingRegressor(loss="poisson", **HGB).fit(trn[F], trn[st])
            out.loc[mask, f"{st}_mu"] = m.predict(out.loc[mask, F])
        else:
            qs = np.column_stack([HistGradientBoostingRegressor(loss="quantile", quantile=q, **HGB)
                                  .fit(trn[F], trn[st]).predict(out.loc[mask, F]) for q in QUANTS])
            qs = np.sort(qs, axis=1)
            for i, q in enumerate(QUANTS):
                out.loc[mask, f"{st}_q{int(q * 100)}"] = qs[:, i]
    # Fantasy points from the projected stat lines (averages, not medians)
    def mean_of(st, fallback):
        cols = [f"{st}_q{int(q * 100)}" for q in QUANTS]
        m = out[cols].mean(axis=1) if cols[0] in out else pd.Series(np.nan, index=out.index)
        return m.fillna(fallback)
    zero = pd.Series(0.0, index=out.index)
    py = mean_of("passing_yards", zero)
    ptd = out.get("passing_tds_mu", zero).fillna(0)
    ints = out.p_passing_interceptions.fillna(0).where(out.position == "QB", 0)
    ry = mean_of("rushing_yards", out.p_rushing_yards.fillna(0))
    rec = mean_of("receptions", out.p_receptions.fillna(0))
    recy = mean_of("receiving_yards", out.p_receiving_yards.fillna(0))
    td = out.get("tds_mu", zero).fillna(out.p_tds.fillna(0))
    out["f_std"] = 0.04 * py + 4 * ptd - 2 * ints + 0.1 * ry + 0.1 * recy + 6 * td
    out["f_rec"] = rec
    out["f_ppr"] = out.f_std + rec
    return out

hist = pp[~pp.upcoming & (pp.n_prior >= 3)]
upc = pp[pp.upcoming]
test_season = CURRENT_SEASON - 1
# 1) Calibration: learn from older seasons, check against last season
cal = predict_all(hist[hist.season < test_season], hist[hist.season == test_season])
calib, prop_report = {}, {}
for st, (poss, use, kind, _, label) in PROPS.items():
    c = cal[cal.position.isin(poss) & cal[st].notna()]
    if kind == "pois":
        c = c[c[f"{st}_mu"].notna()]
        calib[st] = {}
        for k in ([1] if st == "tds" else [1, 2, 3]):
            pr = np.asarray(1 - poisson.cdf(k - 1, c[f"{st}_mu"]))
            hit = (c[st] >= k).values
            table = []
            for x in np.linspace(0, 1, 21):   # real hit rate near each predicted chance
                near = np.abs(pr - x) <= 0.05
                table.append(hit[near].mean() if near.sum() >= 60 else x)
            table = np.maximum.accumulate(np.clip(table, 0.01, 0.97))
            calib[st][k] = [round(float(v), 3) for v in table]
    else:
        c = c[c[f"{st}_q50"].notna()]
        calib[st] = [0.0] + [round(float((c[st] <= c[f"{st}_q{int(q * 100)}"]).mean()), 3) for q in QUANTS] + [1.0]
        prop_report[st] = dict(label=label, mae=nz(np.abs(c[f"{st}_q50"] - c[st]).mean()),
                               base=nz(np.abs(c[f"p_{st}"] - c[st]).mean()))
# Fantasy points: blend the projection with the player's recent average and
# correct each position's bias, using how both did on last season's games.
FANT_FIX = {}
for pos in POS:
    c = cal[(cal.position == pos) & cal.f_ppr.notna() & cal.fantasy_points_ppr.notna() & cal.p_fantasy_points_ppr.notna()]
    FANT_FIX[pos] = Ridge(alpha=1.0).fit(c[["f_ppr", "p_fantasy_points_ppr"]], c.fantasy_points_ppr)
def fix_fantasy(frame):
    frame = frame.copy()
    for pos, m in FANT_FIX.items():
        ok = (frame.position == pos) & frame.f_ppr.notna()
        X = frame.loc[ok, ["f_ppr", "p_fantasy_points_ppr"]].copy()
        X["p_fantasy_points_ppr"] = X.p_fantasy_points_ppr.fillna(X.f_ppr)
        if ok.any():
            frame.loc[ok, "f_ppr"] = np.maximum(m.predict(X), 0)
    frame["f_std"] = frame.f_ppr - frame.f_rec
    return frame
cal = fix_fantasy(cal)
cf = cal[cal.f_ppr.notna() & cal.fantasy_points_ppr.notna()]
fant_report = dict(mae=nz(np.abs(cf.f_ppr - cf.fantasy_points_ppr).mean()),
                   base=nz(np.abs(cf.p_fantasy_points_ppr - cf.fantasy_points_ppr).mean()))
# Floor and ceiling: the range real PPR results landed in last season,
# relative to the projection (25th to 75th percentile), by position
cf2 = cf[cf.f_ppr >= 4]
RATIO = {pos: (float((g.fantasy_points_ppr / g.f_ppr).quantile(0.25)), float((g.fantasy_points_ppr / g.f_ppr).quantile(0.75)))
         for pos, g in cf2.groupby("position")}
def floor_ceiling(pos, pts):
    lo, hi = RATIO.get(pos, (0.55, 1.4))
    return pts * lo, pts * hi

# 2) This season's finished games, predicted by a model that never saw them
grade = fix_fantasy(predict_all(hist[hist.season < CURRENT_SEASON], hist[hist.season == CURRENT_SEASON]))
# 3) This week's projections, using everything
final = fix_fantasy(predict_all(hist, upc)) if len(upc) else upc

def prop_rows(frame):
    rows = []
    for _, r in frame.iterrows():
        for st, (poss, use, kind, _, _) in PROPS.items():
            if r.position not in poss:
                continue
            e = dict(n=r.player_display_name, pos=r.position, tm=r.team, opp=r.opponent_team,
                     gid=r.game_id, wk=int(r.week), st=st)
            if kind == "pois":
                mu = r.get(f"{st}_mu")
                if pd.isna(mu) or (st == "tds" and (mu < 0.08 or r.position == "QB" and mu < 0.2)):
                    continue
                e["mu"] = round(float(mu), 3)
            else:
                qs = [r.get(f"{st}_q{int(q * 100)}") for q in QUANTS]
                if any(pd.isna(v) for v in qs):
                    continue
                med = qs[2]
                if (st == "rushing_yards" and med < 6) or (st == "receiving_yards" and med < 8) or \
                   (st == "receptions" and med < 1):
                    continue
                e["qs"] = [round(float(v), 1) for v in qs]
            rows.append(e)
    return rows

# ======================================================================
print("Step 6/7: Prop lines, fantasy rankings and the results tracker...")
# ======================================================================
def norm_name(n):
    n = re.sub(r"[^a-z ]", "", str(n).lower().replace("-", " "))
    return " ".join(w for w in n.split() if w not in ("jr", "sr", "ii", "iii", "iv", "v"))

ODDS_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
FULL2ABBR = dict(zip(team_info.team_name, team_info.team_abbr))

def odds_get(path, **params):
    q = urllib.parse.urlencode(dict(apiKey=ODDS_API_KEY, **params))
    with urllib.request.urlopen(f"{ODDS_BASE}{path}?{q}", timeout=30) as r:
        return json.load(r)

def fetch_prop_lines(wg):
    """Every sportsbook's line for each player prop: {player|stat: [books]}"""
    want = {(g.away_team, g.home_team) for _, g in wg.iterrows()}
    mk = {v[3]: k for k, v in PROPS.items()}
    lines = {}
    try:
        for ev in odds_get("/events"):
            if (FULL2ABBR.get(ev["away_team"]), FULL2ABBR.get(ev["home_team"])) not in want:
                continue
            odds = odds_get(f"/events/{ev['id']}/odds", regions="us", oddsFormat="american", markets=",".join(mk))
            for b in odds.get("bookmakers", []):
                for m in b.get("markets", []):
                    st = mk.get(m["key"])
                    for o in m.get("outcomes", []) if st else []:
                        key = norm_name(o.get("description", "")) + "|" + st
                        rec = lines.setdefault(key, {}).setdefault(b["title"], {})
                        side = o.get("name")
                        if side in ("Over", "Yes"):
                            rec["line"], rec["over"] = o.get("point", 0.5), o.get("price")
                        elif side in ("Under", "No"):
                            rec.setdefault("line", o.get("point", 0.5))
                            rec["under"] = o.get("price")
        lines = {k: [dict(book=bk, **v) for bk, v in d.items() if "line" in v] for k, d in lines.items()}
        print(f"   Pulled lines for {len(lines)} player props from The Odds API.")
    except Exception as e:
        print(f"   Could not get prop lines from The Odds API ({e}). Continuing without them.")
    return lines

def fetch_game_odds(wg):
    """Every sportsbook's moneyline, spread and total for this week's games."""
    gid = {(g.away_team, g.home_team): g.game_id for _, g in wg.iterrows()}
    out = {}
    try:
        for ev in odds_get("/odds", regions="us", oddsFormat="american", markets="h2h,spreads,totals"):
            g = gid.get((FULL2ABBR.get(ev["away_team"]), FULL2ABBR.get(ev["home_team"])))
            if not g:
                continue
            books = []
            for b in ev.get("bookmakers", []):
                rec = {"book": b["title"]}
                for m in b.get("markets", []):
                    for o in m.get("outcomes", []):
                        home = o.get("name") == ev["home_team"]
                        if m["key"] == "h2h":
                            rec["ml_h" if home else "ml_a"] = o.get("price")
                        elif m["key"] == "spreads":
                            rec["sp_h" if home else "sp_a"] = o.get("point")
                            rec["sp_hp" if home else "sp_ap"] = o.get("price")
                        elif m["key"] == "totals":
                            rec["tot"] = o.get("point")
                            rec["ov" if o.get("name") == "Over" else "un"] = o.get("price")
                books.append(rec)
            out[g] = books
        print(f"   Pulled game lines for {len(out)} games from The Odds API.")
    except Exception as e:
        print(f"   Could not get game lines from The Odds API ({e}).")
    return out

book, game_books = {}, {}
if pwk:
    wk_key = str(pwk)
    on_github = bool(os.environ.get("GITHUB_ACTIONS"))
    forced = os.environ.get("FETCH_PROPS", "").lower() in ("1", "true", "yes")
    today = pd.Timestamp.now(tz="America/New_York")
    day, dow = today.strftime("%Y-%m-%d"), today.dayofweek
    pulls = history["pulls"].setdefault(wk_key, [])
    gpulls = history["pulls"].setdefault("g" + wk_key, [])
    prop_days = (5, 6) if PROP_PULLS_PER_WEEK == 1 else (4, 6)   # Sat/Sun, or Fri + Sun
    # If this week's lines were lost (for example history.json didn't save),
    # recover them from the live website instead of spending credits again.
    repo_ = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" in repo_ and (wk_key not in history["lines"] or wk_key not in history["game_odds"]):
        try:
            live = f"https://{repo_.split('/')[0].lower()}.github.io/{repo_.split('/')[1]}/"
            with urllib.request.urlopen(live + "?nocache=" + day, timeout=30) as r:
                page = r.read().decode("utf-8")
            old = json.loads(re.sub(r"\bNaN\b", "null", page.split("const D = ", 1)[1].split(";\nconst $", 1)[0]))
            if (old.get("prop_meta") or {}).get("week") == pwk:
                got = {}
                for e in old.get("props", []):
                    key = norm_name(e["n"]) + "|" + e["st"]
                    if e.get("books"):
                        got[key] = e["books"]
                    elif e.get("line") is not None:   # older site format
                        got[key] = [{"book": e.get("book") or "Sportsbook", "line": e["line"],
                                     "over": e.get("over"), "under": e.get("under")}]
                if got and wk_key not in history["lines"]:
                    history["lines"][wk_key] = got
                    print(f"   Recovered {len(got)} prop lines from the live site.")
                gb = {g["id"]: g["books"] for g in old.get("games", []) if g.get("books")}
                if gb and wk_key not in history["game_odds"]:
                    history["game_odds"][wk_key] = gb
        except Exception as e:
            print(f"   Couldn't check the live site for saved lines ({e}).")
    if ODDS_API_KEY:
        no_lines_yet = wk_key not in history["lines"]
        want_props = forced or (day not in pulls and (
            (len(pulls) < PROP_PULLS_PER_WEEK and (dow in prop_days or not on_github))
            or (no_lines_yet and dow in (3, 4, 5, 6))))     # Thu-Sun, if the week has none
        if want_props:
            fresh = fetch_prop_lines(week_games)
            if fresh:
                history["lines_first"].setdefault(wk_key, fresh)
                history["lines"][wk_key] = fresh
                pulls.append(day)
        if forced or (day not in gpulls and (dow in (1, 5, 6) or not on_github)):
            fresh = fetch_game_odds(week_games)
            if fresh:
                history["game_odds"][wk_key] = fresh
                gpulls.append(day)
    book = history["lines"].get(wk_key, {})
    game_books = history["game_odds"].get(wk_key, {})
    # older saved format (one book per prop) -> list of books
    book = {k: (v if isinstance(v, list) else [v]) for k, v in book.items()}
    first_book = history["lines_first"].get(wk_key, {})

for g in season_games:
    if g["id"] in game_books:
        g["books"] = game_books[g["id"]]
    if g["wk"] == pwk:   # notable players ruled out (big roles only)
        outs = {}
        for tm in (g["away"], g["home"]):
            o = latest_c[(latest_c.team == tm) & latest_c.index.isin(OUT_IDS) &
                         ((latest_c.c_targets >= VAC_TGT) | (latest_c.c_carries >= VAC_CAR) | (latest_c.position == "QB"))]
            if len(o):
                outs[tm] = [f"{n} ({p})" for n, p in zip(o.player_display_name, o.position)]
        if outs:
            g["outs"] = outs

def add_lines(rows, lines, first=None):
    for e in rows:
        key = norm_name(e["n"]) + "|" + e["st"]
        if lines.get(key):
            e["books"] = lines[key]
        if first and first.get(key) and first.get(key) != lines.get(key):
            e["fbooks"] = first[key]      # earlier pull, for closing-line value
    return rows

props = add_lines(prop_rows(final), book, first_book if pwk else None) if len(final) else []
q_names = {names.get(i) for i in Q_IDS}
game_order = {gid: i for i, gid in enumerate(week_games.sort_values(["gameday", "gametime"]).game_id)}
for e in props:
    e["q"] = e["n"] in q_names
    e["go"] = game_order.get(e["gid"], 99)
    b = BOOST.get(e["tm"], {})
    who = (b.get("tgt", []) if e["st"] in ("receptions", "receiving_yards", "tds") else []) + \
          (b.get("car", []) if e["st"] in ("rushing_yards", "tds") and e["pos"] == "RB" else [])
    who = [w for w in dict.fromkeys(who) if w != e["n"]]
    if who and e["pos"] != "QB":
        e["boost"] = who

# Save what the site shows before kickoff, so it can be graded afterwards.
if pwk:
    wk_key = str(pwk)
    upcoming_ids = set(week_games.game_id)
    kept = [e for e in history["props"].get(wk_key, []) if e["gid"] not in upcoming_ids]
    history["props"][wk_key] = kept + [{k: v for k, v in e.items() if k not in ("q", "go", "boost")} for e in props]
# Weeks from before the site started saving: fill in with honest backfilled projections
for wk, grp in grade.groupby("week"):
    if str(int(wk)) not in history["props"]:
        history["props"][str(int(wk))] = [dict(e, bf=1) for e in prop_rows(grp)]

actual = {}
for r in pp[(pp.season == CURRENT_SEASON) & ~pp.upcoming].itertuples():
    for st in PROPS:
        v = getattr(r, st)
        if pd.notna(v):
            actual[f"{int(r.week)}|{r.player_display_name}|{st}"] = float(v)
past_props = []
for wk, rows in history["props"].items():
    if pwk and int(wk) >= pwk:
        continue
    for e in rows:
        a = actual.get(f"{wk}|{e['n']}|{e['st']}")
        if a is not None:
            past_props.append(dict(e, act=a))

# Fantasy rankings for this week
fantasy = []
for _, r in final.iterrows() if len(final) else []:
    if pd.isna(r.f_ppr) or r.f_ppr < 3:
        continue
    lo, hi = floor_ceiling(r.position, r.f_ppr)
    fantasy.append(dict(n=r.player_display_name, pos=r.position, tm=r.team, opp=r.opponent_team, gid=r.game_id,
                        std=nz(r.f_std), rec=nz(r.f_rec), ppr=nz(r.f_ppr), lo=nz(lo), hi=nz(hi),
                        dpts=nz(r.get("d_fantasy_points_ppr")), q=r.player_id in Q_IDS))
# Matchup rank: PPR points each defense has allowed to the position lately (1 = easiest)
dp = {}
for f in fantasy:
    dp.setdefault(f["pos"], {})[f["opp"]] = f["dpts"]
for f in fantasy:
    vals = sorted({v for v in dp[f["pos"]].values() if v is not None}, reverse=True)
    f["mrank"] = vals.index(f["dpts"]) + 1 if f["dpts"] in vals else None
    f.pop("dpts")

# Fantasy accuracy this season
gf = grade[grade.f_ppr.notna() & grade.fantasy_points_ppr.notna()]
fant_season = dict(mae=nz(np.abs(gf.f_ppr - gf.fantasy_points_ppr).mean()),
                   base=nz(np.abs(gf.p_fantasy_points_ppr - gf.fantasy_points_ppr).mean()), n=int(len(gf)))


# ---------------------------- Edge alerts ----------------------------
NOM = [0, .1, .25, .5, .75, .9, 1]
def prob_over(e, line):
    """Model's chance of going over a line (same math as the website)."""
    st = e["st"]
    if "qs" in e:
        a, b, c, d, f = e["qs"]
        lo = a - 2 * (b - a) if st == "rushing_yards" else max(0, a - 2 * (b - a))
        hi = f + 2.5 * max(f - d, 1)
        nominal = np.interp(line, [lo, a, b, c, d, f, hi], NOM)
        return 1 - float(np.interp(nominal, NOM, calib.get(st, NOM)))
    k = int(np.floor(line)) + 1
    raw = 1 - poisson.cdf(k - 1, e["mu"])
    tbl = calib.get(st, {}).get(k)
    return float(np.interp(raw, np.linspace(0, 1, 21), tbl)) if tbl else float(raw)

def breakeven(o):
    return None if o is None else (-o / (-o + 100) if o < 0 else 100 / (o + 100))

def best_pick(e):
    best = None
    for b in e.get("books", []):
        if b.get("line") is None:
            continue
        line = 0.5 if e["st"] == "tds" else b["line"]
        p = prob_over(e, line)
        for side, odds, prob in (("Over", b.get("over"), p), ("Under", b.get("under"), 1 - p)):
            if odds is None or (e["st"] == "tds" and side == "Under"):
                continue
            edge = prob - breakeven(odds)
            if best is None or edge > best["edge"]:
                best = dict(side="Yes" if e["st"] == "tds" else side, line=line, odds=odds, book=b["book"], edge=edge, prob=prob)
    return best

def fmt_odds(o):
    return f"+{o}" if o > 0 else str(o)

alerts = []
repo = os.environ.get("GITHUB_REPOSITORY", "")
site_url = f"https://{repo.split('/')[0]}.github.io/{repo.split('/')[1]}/" if "/" in repo else ""
if pwk:
    label = {k: v[4] for k, v in PROPS.items()}
    for e in props:
        bp = best_pick(e)
        if bp and bp["edge"] >= ALERT_PROP_EDGE:
            key = f"{pwk}|{e['n']}|{e['st']}|{bp['side']}"
            if key not in history["alerted"]:
                history["alerted"].append(key)
                alerts.append(f"| {e['n']} ({e['tm']}) | {label[e['st']]} {bp['side']} {'' if e['st'] == 'tds' else bp['line']} "
                              f"| {fmt_odds(bp['odds'])} at {bp['book']} | {bp['prob'] * 100:.0f}% | +{bp['edge'] * 100:.1f}% |")
    for g in season_games:
        if g["wk"] != pwk or g["hs"] is not None or g["pred"] is None:
            continue
        for kind, mine, vegas, thr in (("spread", g["pred"], g["spread"], ALERT_SPREAD_EDGE),
                                       ("total", g["tpred"], g["total"], ALERT_TOTAL_EDGE)):
            if vegas is None or abs(mine - vegas) < thr:
                continue
            key = f"{pwk}|{g['id']}|{kind}|{'up' if mine > vegas else 'down'}"
            if key in history["alerted"]:
                continue
            history["alerted"].append(key)
            if kind == "spread":
                side = g["home"] if mine > vegas else g["away"]
                side_line = -vegas if side == g["home"] else vegas
                fav = g["home"] if mine > 0 else g["away"]
                alerts.append(f"| {g['away']} @ {g['home']} | {side} {side_line:+.1f} | | Model: {fav} by {abs(mine):.1f} | {abs(mine - vegas):.1f} pts |")
            else:
                alerts.append(f"| {g['away']} @ {g['home']} | {'Over' if mine > vegas else 'Under'} {vegas} | | Model total: {mine:.1f} | {abs(mine - vegas):.1f} pts |")
if os.path.exists("alerts.md"):
    os.remove("alerts.md")
if alerts:
    body = (f"The model found {len(alerts)} new edge(s) for Week {pwk}.\n\n"
            "| Pick | Bet | Best odds | Model chance | Edge |\n|---|---|---|---|---|\n" + "\n".join(alerts) +
            (f"\n\nSee everything at {site_url}" if site_url else "") +
            "\n\nCheck injury news before betting. Big edges often mean the model is missing something.")
    with open("alerts.md", "w", encoding="utf-8") as f:
        f.write(body)
    print(f"   {len(alerts)} new edge alert(s) written to alerts.md")

# ---------------------------- Playoff odds ----------------------------
print("   Simulating the rest of the season for playoff odds...")
TEAMS = sorted(lt.index)
TI = {t: i for i, t in enumerate(TEAMS)}
tinfo = team_info.drop_duplicates("team_abbr").set_index("team_abbr").reindex(TEAMS)
CONF, DIV = tinfo.team_conf.values, tinfo.team_division.values
# Power rating: predicted margin against an average team on a neutral field
avg = latest.loc[TEAMS, S].mean()
next_qb = {}
for _, r in cur[cur.result.isna()].sort_values(["week"]).iterrows():
    next_qb.setdefault(r.home_team, r.home_qbr); next_qb.setdefault(r.away_team, r.away_qbr)
qb_avg = np.mean([next_qb.get(t, QB_PRIOR) for t in TEAMS])
rows = []
for t in TEAMS:
    f = latest.loc[t]
    rows.append(dict(pass_matchup=(f.off_pass_epa - avg.off_pass_epa) - (f.def_pass_epa - avg.def_pass_epa),
                     rush_matchup=(f.off_rush_epa - avg.off_rush_epa) - (f.def_rush_epa - avg.def_rush_epa),
                     success_matchup=(f.off_success - avg.off_success) - (f.def_success - avg.def_success),
                     rest_diff=0, neutral_site=1, div_game=0, high_wind=0, qb_diff=next_qb.get(t, QB_PRIOR) - qb_avg))
raw = ms_all.predict(pd.DataFrame(rows)[F_SPREAD])
RATING = raw - raw.mean()
HFA = float(ms_all.intercept_)
rng = np.random.default_rng(7)
N = SIMULATIONS
wins = np.zeros((N, len(TEAMS))); pdiff = np.zeros((N, len(TEAMS)))
for _, g in cur[cur.result.notna()].iterrows():
    h, a, m = TI[g.home_team], TI[g.away_team], g.result
    wins[:, h] += 1 if m > 0 else 0.5 if m == 0 else 0
    wins[:, a] += 1 if m < 0 else 0.5 if m == 0 else 0
    pdiff[:, h] += m; pdiff[:, a] -= m
cur_w = wins[0].copy()
cur_l = np.array([((cur.result.notna()) & (((cur.home_team == t) & (cur.result < 0)) | ((cur.away_team == t) & (cur.result > 0)))).sum()
                  for t in TEAMS])
cur_t = np.array([((cur.result == 0) & ((cur.home_team == t) | (cur.away_team == t))).sum() for t in TEAMS])
cur_w = cur_w - 0.5 * cur_t
rem = cur[cur.result.isna()]
if len(rem):
    M = rem.pred.values[None, :] + rng.normal(0, MARGIN_STD, (N, len(rem)))
    hi_, ai_ = rem.home_team.map(TI).values, rem.away_team.map(TI).values
    for j in range(len(rem)):
        wins[:, hi_[j]] += M[:, j] > 0; wins[:, ai_[j]] += M[:, j] < 0
        pdiff[:, hi_[j]] += M[:, j]; pdiff[:, ai_[j]] -= M[:, j]
score = wins + pdiff * 1e-4 + rng.random(wins.shape) * 1e-7
cnt = {k: np.zeros(len(TEAMS)) for k in ("playoffs", "division", "bye", "cchamp", "sb")}
def play(a, b, home_adv):
    p = norm.cdf((RATING[a] - RATING[b] + home_adv) / MARGIN_STD)
    return np.where(rng.random(len(a)) < p, a, b)
conf_champ = {}
for c in ("AFC", "NFC"):
    ct = np.where(CONF == c)[0]
    divs = sorted(set(DIV[ct]))
    winners = np.column_stack([np.array([np.where(DIV == d)[0]])[0][np.argmax(score[:, DIV == d], axis=1)] for d in divs])
    for d_i in range(winners.shape[1]):
        np.add.at(cnt["division"], winners[:, d_i], 1)
    w_sc = np.take_along_axis(score, winners, axis=1)
    order = np.argsort(-w_sc, axis=1)
    seeds14 = np.take_along_axis(winners, order, axis=1)
    sc_c = score[:, ct].copy()
    is_w = np.zeros_like(sc_c, dtype=bool)
    for d_i in range(winners.shape[1]):
        is_w |= (ct[None, :] == winners[:, d_i][:, None])
    sc_c[is_w] = -1e9
    wc = ct[np.argsort(-sc_c, axis=1)[:, :3]]
    seeds = np.column_stack([seeds14, wc])            # seeds 1..7
    for k in range(7):
        np.add.at(cnt["playoffs"], seeds[:, k], 1)
    np.add.at(cnt["bye"], seeds[:, 0], 1)
    # Wild card round: 2v7, 3v6, 4v5 (higher seed at home)
    w27, w36, w45 = play(seeds[:, 1], seeds[:, 6], HFA), play(seeds[:, 2], seeds[:, 5], HFA), play(seeds[:, 3], seeds[:, 4], HFA)
    seed_of = lambda team_arr: np.argmax(seeds == team_arr[:, None], axis=1)
    alive = np.column_stack([w27, w36, w45])
    alive_seed = np.column_stack([seed_of(w27), seed_of(w36), seed_of(w45)])
    o = np.argsort(alive_seed, axis=1)
    alive, alive_seed = np.take_along_axis(alive, o, 1), np.take_along_axis(alive_seed, o, 1)
    # Divisional: 1 vs lowest remaining seed, other two play each other
    d1 = play(seeds[:, 0], alive[:, 2], HFA)
    d2 = play(alive[:, 0], alive[:, 1], HFA)
    s1, s2 = seed_of(d1), seed_of(d2)
    home = np.where(s1 <= s2, d1, d2); away = np.where(s1 <= s2, d2, d1)
    champ = play(home, away, HFA)
    np.add.at(cnt["cchamp"], champ, 1)
    conf_champ[c] = champ
sb = play(conf_champ["AFC"], conf_champ["NFC"], 0.0)
np.add.at(cnt["sb"], sb, 1)
playoffs = []
for t in TEAMS:
    i = TI[t]
    playoffs.append(dict(t=t, name=tinfo.loc[t, "team_name"], conf=CONF[i], div=DIV[i],
                         w=int(cur_w[i]), l=int(cur_l[i]), ties=int(cur_t[i]), pw=nz(wins[:, i].mean()),
                         rating=nz(RATING[i]), **{k: nz(v[i] / N * 100) for k, v in cnt.items()}))

# ---------------------------- Matchup trends ----------------------------
# For each defense: how players at a position did against it, compared with
# what those same players usually do (their average going into the game).
MATCH_KEYS = [("QB", "passing_yards", "attempts", 15), ("RB", "rushing_yards", "carries", 5),
              ("RB", "receiving_yards", "targets", 2), ("WR", "receiving_yards", "targets", 3),
              ("TE", "receiving_yards", "targets", 2), ("WR", "receptions", "targets", 3),
              ("RB", "tds", "touches", 6), ("WR", "tds", "targets", 3), ("TE", "tds", "targets", 2)]
done_rows = pp[~pp.upcoming]
matchups = {}
for pos_, st_, use_, mn_ in MATCH_KEYS:
    d_ = done_rows[(done_rows.position == pos_) & (done_rows[f"p_{use_}"] >= mn_) &
                   done_rows[st_].notna() & done_rows[f"p_{st_}"].notna()]
    d_ = d_.assign(diff=d_[st_] - d_[f"p_{st_}"])
    per_game = (d_.groupby(["opponent_team", "game_id", "season", "week"])["diff"].sum()
                  .reset_index().sort_values(["season", "week"]))
    recent_g = per_game.groupby("opponent_team").tail(8)          # each defense's last 8 games
    ag = recent_g.groupby("opponent_team")["diff"].agg(["size", "mean", lambda x: int((x > 0).sum())])
    ag.columns = ["n", "avg", "over"]
    ag = ag[ag.index.isin(TEAMS)]
    ag["rank"] = ag["avg"].rank(ascending=False, method="min")    # 1 = gives up the most
    for t_, r_ in ag.iterrows():
        matchups.setdefault(t_, {})[f"{pos_}|{st_}"] = [round(float(r_["avg"]), 2 if st_ == "tds" else 1),
                                                        int(r_["over"]), int(r_["n"]), int(r_["rank"])]

# ---------------------------- Trivia ----------------------------
tr_rng = np.random.default_rng(int(pd.Timestamp.now().strftime("%Y%m%d")))
ros_c = roster_cur.drop_duplicates("gsis_id").set_index("gsis_id")
season_tot = psc.groupby("player_id").agg(
    n=("player_display_name", "last"), pos=("position", "last"), tm=("team", "last"), gp=("week", "nunique"),
    pyd=("passing_yards", "sum"), ptd=("passing_tds", "sum"), ryd=("rushing_yards", "sum"), rtd=("rushing_tds", "sum"),
    rec=("receptions", "sum"), recyd=("receiving_yards", "sum"), rectd=("receiving_tds", "sum"),
    ppr=("fantasy_points_ppr", "sum"), att=("attempts", "sum"))
season_tot = season_tot.join(ros_c[["college", "draft_number", "draft_club", "entry_year", "jersey_number",
                                    "height", "weight", "years_exp", "birth_date"]], how="left")
conf_of = dict(zip(tinfo.index, tinfo.team_conf)); div_of = dict(zip(tinfo.index, tinfo.team_division))
full_of = dict(zip(tinfo.index, tinfo.team_name))
def stat_line(r):
    if r.pos == "QB":
        return f"{int(r.pyd)} passing yards and {int(r.ptd)} TD passes in {r.gp} games"
    if r.pos == "RB":
        return f"{int(r.ryd)} rushing yards, {int(r.rec)} catches and {int(r.rtd + r.rectd)} TDs in {r.gp} games"
    return f"{int(r.rec)} catches for {int(r.recyd)} yards and {int(r.rectd)} TDs in {r.gp} games"
def height_txt(h):
    try:
        h = int(h); return f"{h // 12}'{h % 12}\""
    except (TypeError, ValueError):
        return None
guess = []
pool = season_tot[season_tot.college.notna()].sort_values("ppr", ascending=False)
pool = pd.concat([pool[pool.pos == "QB"].head(20), pool[pool.pos != "QB"].head(70)])
for pid, r in pool.iterrows():
    age = None
    try:
        age = int((pd.Timestamp.now() - pd.Timestamp(r.birth_date)).days // 365.25)
    except Exception:
        pass
    yrs = int(r.years_exp) if pd.notna(r.years_exp) else None
    draft = (f"Picked No. {int(r.draft_number)} overall in the {int(r.entry_year)} draft"
             + (f" by {full_of.get(r.draft_club, r.draft_club)}" if isinstance(r.draft_club, str) and r.draft_club != r.tm else "")
             if pd.notna(r.draft_number) and pd.notna(r.entry_year) else
             f"Went undrafted in {int(r.entry_year)}" if pd.notna(r.entry_year) else None)
    college = str(r.college).split(";")[0].strip()
    clues = [f"Plays {r.pos} in the {conf_of.get(r.tm, 'NFL')}",
             (f"Rookie season" if yrs == 0 else f"Season No. {yrs + 1} in the NFL" if yrs is not None else None)
             and ((f"Rookie season" if yrs == 0 else f"Season No. {yrs + 1} in the NFL") + (f", age {age}" if age else "")),
             f"Played college football at {college}", draft,
             ". ".join(x for x in [f"Wears No. {int(r.jersey_number)}" if pd.notna(r.jersey_number) else None,
                                  f"{height_txt(r.height)}, {int(r.weight)} lbs" if pd.notna(r.weight) and height_txt(r.height) else None] if x),
             f"This season: {stat_line(r)}", f"Plays in the {div_of.get(r.tm, '')}", f"Plays for the {full_of.get(r.tm, r.tm)}"]
    clues = [c for c in clues if c]
    if len(clues) >= 6:
        guess.append(dict(n=r.n, pos=r.pos, tm=r.tm, clues=clues))

quiz = []
def add_q(q, right, wrong, explain):
    wrong = [w for w in dict.fromkeys(wrong) if w != right][:3]
    if len(wrong) < 3:
        return
    opts = wrong + [right]
    tr_rng.shuffle(opts)
    quiz.append(dict(q=q, o=[str(o) for o in opts], a=opts.index(right), x=explain))
st_min = season_tot[season_tot.gp >= 1]
for col, what, pos in [("recyd", "receiving yards", None), ("ryd", "rushing yards", None), ("pyd", "passing yards", "QB"),
                       ("rec", "catches", None), ("ptd", "touchdown passes", "QB"), ("rtd", "rushing touchdowns", None),
                       ("rectd", "receiving touchdowns", None)]:
    d = st_min if pos is None else st_min[st_min.pos == pos]
    top = d.sort_values(col, ascending=False).head(10)
    if len(top) < 4 or top[col].iloc[0] == top[col].iloc[1]:
        continue
    lead = top.iloc[0]
    add_q(f"Who leads the NFL in {what} this season?", lead.n, list(tr_rng.permutation(top.n.iloc[1:8].tolist())),
          f"{lead.n} ({lead.tm}) has {int(lead[col])} {what}.")
    a_, b_ = top.iloc[int(tr_rng.integers(1, 4))], top.iloc[int(tr_rng.integers(4, 10))] if len(top) >= 10 else top.iloc[-1]
    if a_[col] != b_[col]:
        pair = [a_, b_]; tr_rng.shuffle(pair)
        win = a_ if a_[col] > b_[col] else b_
        quiz.append(dict(q=f"Who has more {what} this season?", o=[pair[0].n, pair[1].n], a=[pair[0].n, pair[1].n].index(win.n),
                         x=f"{a_.n}: {int(a_[col])}. {b_.n}: {int(b_[col])}."))
stars = season_tot.sort_values("ppr", ascending=False).head(60)
team_list = list(TEAMS)
for _, r in stars.sample(min(8, len(stars)), random_state=int(tr_rng.integers(1e6))).iterrows():
    add_q(f"Which team does {r.n} play for?", full_of.get(r.tm, r.tm),
          [full_of[t] for t in tr_rng.permutation(team_list) if t != r.tm], f"{r.n} plays for the {full_of.get(r.tm, r.tm)}.")
stars = stars.assign(college=stars.college.map(lambda c: str(c).split(";")[0].strip() if isinstance(c, str) else c))
colleges = stars.college.dropna().unique().tolist()
for _, r in stars[stars.college.notna()].sample(min(6, stars.college.notna().sum()), random_state=int(tr_rng.integers(1e6))).iterrows():
    add_q(f"Where did {r.n} play college football?", r.college, list(tr_rng.permutation([c for c in colleges if c != r.college])),
          f"{r.n} played at {r.college}.")
firsts = stars[stars.draft_number <= 32]; lates = stars[(stars.draft_number > 64) | stars.draft_number.isna()]
for _ in range(3):
    if len(firsts) and len(lates) >= 3:
        f1 = firsts.sample(1, random_state=int(tr_rng.integers(1e6))).iloc[0]
        add_q("Which of these players was a first-round pick?", f1.n, lates.sample(3, random_state=int(tr_rng.integers(1e6))).n.tolist(),
              f"{f1.n} went No. {int(f1.draft_number)} overall in {int(f1.entry_year)}.")
seas = tg[tg.season == CURRENT_SEASON]
if len(seas):
    team_epa = seas.groupby("team").agg(off=("off_epa", "mean"), dfn=("def_epa", "mean"))
    team_epa = team_epa[team_epa.index.isin(TEAMS)]
    for col, asc, what in [("off", False, "the best offense by EPA per play"), ("dfn", True, "the best defense by EPA per play allowed")]:
        o = team_epa.sort_values(col, ascending=asc)
        add_q(f"Which team has {what} this season?", full_of[o.index[0]], [full_of[t] for t in tr_rng.permutation(o.index[3:16].tolist())],
              f"The {full_of[o.index[0]]} rank first.")
    fin = cur[cur.result.notna()]
    pts = pd.concat([fin[["home_team", "home_score", "away_score"]].set_axis(["t", "pf", "pa"], axis=1),
                     fin[["away_team", "away_score", "home_score"]].set_axis(["t", "pf", "pa"], axis=1)]).groupby("t").sum()
    if len(pts) > 8:
        o = pts.sort_values("pf", ascending=False)
        if o.pf.iloc[0] != o.pf.iloc[1]:
            add_q("Which team has scored the most points this season?", full_of[o.index[0]],
                  [full_of[t] for t in tr_rng.permutation(o.index[2:14].tolist())], f"The {full_of[o.index[0]]} have scored {int(o.pf.iloc[0])}.")
        o = pts.sort_values("pa")
        if o.pa.iloc[0] != o.pa.iloc[1]:
            add_q("Which team has allowed the fewest points this season?", full_of[o.index[0]],
                  [full_of[t] for t in tr_rng.permutation(o.index[2:14].tolist())], f"The {full_of[o.index[0]]} have allowed {int(o.pa.iloc[0])}.")
    for t in tr_rng.permutation(TEAMS)[:4]:
        i = TI[t]; w, l = int(cur_w[i]), int(cur_l[i])
        rec = f"{w}-{l}"
        add_q(f"What is the {full_of[t]}' record so far?", rec,
              [f"{w + d}-{l - d}" for d in tr_rng.permutation([-2, -1, 1, 2]) if l - d >= 0 and w + d >= 0], f"The {full_of[t]} are {rec}.")
tr_rng.shuffle(quiz)

# ======================================================================
print("Step 7/7: Writing the website...")
# ======================================================================
with open(HISTORY_FILE, "w", encoding="utf-8") as f:
    json.dump(history, f)
data = dict(season=CURRENT_SEASON, updated=pd.Timestamp.now(tz="America/New_York").strftime("%b %d, %Y at %-I:%M %p ET")
            if os.name != "nt" else pd.Timestamp.now().strftime("%b %d, %Y"),
            games=season_games, players=players, teams=teams, logs=logs, risers=risers,
            props=props, past=past_props, fantasy=fantasy, total_std=round(TOTAL_STD, 2),
            playoffs=playoffs, hfa=round(HFA, 2), sims=N, trivia=dict(guess=guess, quiz=quiz),
            colors={t: [txt(r.team_color), txt(r.team_color2)] for t, r in tinfo.iterrows()},
            assistant_url=ASSISTANT_URL, matchups=matchups,
            prop_meta=dict(week=pwk, calib=calib, report=prop_report, has_book=bool(book)),
            reports=dict(game=game_report, fantasy=fant_report, fantasy_season=fant_season))
data_json = json.dumps(data, default=lambda o: None if pd.isna(o) else (o.item() if hasattr(o, "item") else str(o)))
data_json = re.sub(r"\bNaN\b", "null", data_json)
if os.path.dirname(OUT_FILE):
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(TEMPLATE.replace("__DATA__", data_json))
print(f"Done! {len(season_games)} games, {len(players)} players, {len(props)} projections, "
      f"{len(fantasy)} fantasy players.")
print(f"Open {OUT_FILE} (double-click it) to view your site.")


r'''
#__TEMPLATE__
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Model Board, NFL picks and projections</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;700;800&family=Archivo:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#EEF1F4; --panel:#FFFFFF; --ink:#13213C; --muted:#586379; --line:#D2D8E0; --faint:#E4E8EE;
  --turf:#237A45; --turf-soft:rgba(35,122,69,.12); --brass:#C4541A; --loss:#B42318; --navy:#13213C; --pylon:#F26A21;
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px);
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --paper:#0F1726; --panel:#172136; --ink:#E7ECF4; --muted:#9AA5BA; --line:#2A3752; --faint:#1D2940;
  --turf:#5CC98A; --turf-soft:rgba(92,201,138,.14); --brass:#FF8B45; --loss:#FF7B6E; } }
:root[data-theme="dark"]{
  --paper:#0F1726; --panel:#172136; --ink:#E7ECF4; --muted:#9AA5BA; --line:#2A3752; --faint:#1D2940;
  --turf:#5CC98A; --turf-soft:rgba(92,201,138,.14); --brass:#FF8B45; --loss:#FF7B6E; }
*,*::before,*::after{box-sizing:inherit}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"Archivo",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:16px;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:32px 18px 64px}
.cond,h1,h2,h3{font-family:"Big Shoulders Display","Oswald","Arial Narrow",sans-serif}
.mast{margin:0 0 4px}
.field{position:relative;height:clamp(170px,24vw,240px);border-radius:6px 6px 0 0;overflow:hidden;background:#2B6A3E}
.field-svg{position:absolute;inset:0;width:100%;height:100%;display:block}
.paint{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:0 16px}
.paint h1{font-family:"Big Shoulders Display","Oswald","Arial Narrow",sans-serif;font-weight:800;text-transform:uppercase;font-size:clamp(46px,9.5vw,104px);line-height:.85;letter-spacing:.03em;margin:0;color:#F4F6F1;
  -webkit-text-stroke:2px #13213C;paint-order:stroke fill;text-shadow:0 3px 0 rgba(19,33,60,.55)}
.mast .updated{margin:8px 2px 0;color:var(--muted);font-size:13.5px;text-align:right}
.scoreboard{display:grid;grid-template-columns:repeat(3,1fr);background:#13213C;border-radius:0 0 6px 6px;border-top:3px solid var(--pylon)}
.scoreboard div{padding:10px 14px 12px;text-align:center;border-left:1px solid rgba(255,255,255,.12)}
.scoreboard div:first-child{border-left:0}
.scoreboard .cond{display:block;font-size:clamp(28px,4.4vw,40px);font-weight:700;line-height:1;color:#FFB547;letter-spacing:.04em}
.scoreboard small{display:block;color:#C3CCDC;font-size:13px;margin-top:4px}
nav.views{display:flex;gap:4px;margin:28px 0 0;border-bottom:2px solid var(--ink);overflow-x:auto;scrollbar-width:none}
nav.views::-webkit-scrollbar{display:none}
nav.views button{flex:0 0 auto;font:600 22px/1 "Big Shoulders Display","Arial Narrow",sans-serif;background:none;border:0;color:var(--muted);padding:10px 14px 9px;cursor:pointer;border-bottom:4px solid transparent;margin-bottom:-2px}
nav.views button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--pylon)}
button:focus-visible,select:focus-visible,summary:focus-visible{outline:2px solid var(--turf);outline-offset:2px}
.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px;margin:18px 0 8px}
.toolbar label{font-size:14px;color:var(--muted);display:flex;align-items:center;gap:8px}
select{font:inherit;font-size:15px;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 10px}
.weeks{display:flex;gap:4px;overflow-x:auto;padding:4px 0 8px;scrollbar-width:thin}
.weeks button{font:600 17px/1 "Big Shoulders Display","Arial Narrow",sans-serif;min-width:40px;padding:8px 6px;border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:6px;cursor:pointer}
.weeks button[aria-pressed="true"]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.weeks button.done{color:var(--muted)}
.weeks button.done[aria-pressed="true"]{color:var(--paper)}
/* game cards */
.game{border-bottom:1px solid var(--line)}
.game summary{background:linear-gradient(90deg,var(--ca,transparent) 0 5px,var(--ch,transparent) 5px 10px,transparent 10px);padding-left:22px!important}
.tchip{display:inline-block;font:700 26px/1 "Big Shoulders Display","Oswald","Arial Narrow",sans-serif;letter-spacing:.03em;padding:5px 7px 4px;border-radius:3px;min-width:56px;text-align:center}
.tbadge{display:inline-block;font:700 13px/1 "Big Shoulders Display","Oswald","Arial Narrow",sans-serif;letter-spacing:.04em;padding:3px 4px 2px;border-radius:3px;min-width:32px;text-align:center;margin-right:8px;vertical-align:1px;flex:0 0 auto}
.pv-head .tchip{margin-right:12px;vertical-align:10px;font-size:30px}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px;vertical-align:0;box-shadow:inset 0 0 0 1px rgba(0,0,0,.15)}
.game summary{list-style:none;cursor:pointer;display:grid;grid-template-columns:minmax(150px,1.2fr) repeat(3,minmax(110px,1fr)) 22px;gap:10px 18px;align-items:center;padding:16px 4px}
.game summary::-webkit-details-marker{display:none}
.matchup .cond{font-size:30px;font-weight:700;line-height:1}
.matchup .at{color:var(--muted);font-weight:500;margin:0 4px}
.matchup small,.cell small{display:block;font-size:12.5px;color:var(--muted)}
.cell .v{display:block;font:600 22px/1.15 "Big Shoulders Display","Arial Narrow",sans-serif}
.cell .ml{display:block;font-size:13px;line-height:1.35;color:var(--muted);margin-top:2px}
.edge3 .v{color:var(--turf)}
.res-w{color:var(--turf);font-weight:600}.res-l{color:var(--loss);font-weight:600}
.chev{width:10px;height:10px;border-right:2px solid var(--muted);border-bottom:2px solid var(--muted);transform:rotate(45deg);justify-self:center;transition:transform .15s}
details[open] .chev{transform:rotate(-135deg)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:18px;margin:0 0 20px}
.form{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:8px}
.form h3{font-size:24px;margin:0 0 6px}
.bars{display:grid;grid-template-columns:auto 1fr auto;gap:5px 10px;align-items:center;font-size:13.5px}
.bar{height:8px;background:var(--faint);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--turf);border-radius:4px}
.rk{font:600 15px "Big Shoulders Display","Arial Narrow",sans-serif;text-align:right;min-width:36px}
.pos-block{margin-top:18px}
.pos-block h4{margin:0 0 4px;font-size:14px;font-weight:600;color:var(--muted)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--faint)}
th{font-weight:600;color:var(--muted);font-size:12.5px;cursor:pointer;user-select:none}
th[aria-sort]{color:var(--ink)}
th:first-child,td:first-child{text-align:left;padding-left:0;position:sticky;left:0;background:var(--panel)}
td.tm{color:var(--muted);text-align:left}


.hot{color:var(--turf);font-weight:600}
.cold{color:var(--loss)}
.empty{color:var(--muted);font-size:14px;margin:6px 0}
/* season table */
.season-wrap{overflow-x:auto;margin-top:6px}
.season td,.season th{padding:9px 10px}
.season th:first-child,.season td:first-child{background:var(--paper)}
.season tr.wkrow td{background:var(--faint);font:600 16px "Big Shoulders Display","Arial Narrow",sans-serif;color:var(--ink);text-align:left;position:static}
.teamrec{font-size:15px;margin:4px 0 0}
.teamrec b{font-family:"Big Shoulders Display","Arial Narrow",sans-serif;font-size:20px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
.search{position:relative;margin-top:24px;max-width:460px}
.search input{width:100%;font:inherit;font-size:17px;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 14px}
.search input:focus-visible{outline:2px solid var(--turf);outline-offset:1px}
#qres{position:absolute;z-index:5;left:0;right:0;top:calc(100% + 4px);margin:0;padding:4px;list-style:none;background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.12);max-height:360px;overflow-y:auto}
#qres li{padding:8px 10px;border-radius:6px;cursor:pointer;display:flex;justify-content:space-between;gap:12px}
#qres li small{color:var(--muted)}
#qres li[aria-selected="true"]{background:var(--turf-soft)}
#qres li.none{cursor:default;color:var(--muted)}
.pcard{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:16px 18px;margin:16px 0 4px}
.pcard-top{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}
.pcard h3{font-size:30px;margin:0;line-height:1}
.pcard .who{color:var(--muted);font-size:14px;margin:4px 0 0}
.pcard button.clear{font:inherit;font-size:14px;background:none;border:1px solid var(--line);color:var(--ink);border-radius:4px;padding:5px 12px;cursor:pointer}
.pcard table th:first-child,.pcard table td:first-child{background:var(--panel)}
.pcard .empty{margin-top:10px}
h2.sub{font-size:30px;margin:34px 0 4px}
.booknote{display:block;font-size:11.5px;color:var(--muted);margin-top:3px;text-align:right}
.props td.pl small.boost{color:var(--turf);font-weight:600}
.outs{font-size:14px;margin:0 0 14px;padding:8px 12px;border-left:3px solid var(--loss);background:var(--faint);border-radius:0 6px 6px 0}
.shop{margin-top:18px}
.shop h4{font-size:14px;font-weight:600;color:var(--muted);margin:0}
.pct{position:relative;display:inline-block;min-width:64px;text-align:right}
.pct i{position:absolute;left:0;bottom:-3px;height:3px;background:var(--turf);border-radius:2px}
.po td,.po th{padding:8px 10px}
.po th:first-child,.po td:first-child{background:var(--paper)}
.po tr.divrow td{background:var(--faint);font:600 16px "Big Shoulders Display","Arial Narrow",sans-serif;text-align:left}
.chalk-wrap{max-width:720px;margin-top:18px}
.chalk{width:100%;height:auto;display:block}
.trivia-card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:20px 22px;max-width:720px;margin-top:16px}
.clues{list-style:none;padding:0;margin:10px 0 16px;counter-reset:c}
.clues li{counter-increment:c;padding:8px 0 8px 34px;position:relative;border-bottom:1px solid var(--faint)}
.clues li::before{content:counter(c);position:absolute;left:0;top:7px;width:24px;height:24px;border-radius:50%;background:var(--turf-soft);color:var(--turf);font:600 14px/24px "Archivo",system-ui,sans-serif;text-align:center}
.guess-row{display:flex;gap:8px;flex-wrap:wrap}
.guess-row input{flex:1 1 220px;font:inherit;font-size:16px;padding:9px 12px;border:1px solid var(--line);border-radius:8px;background:var(--paper);color:var(--ink)}
.msg{margin:12px 0 0;font-weight:600}
.msg.ok{color:var(--turf)} .msg.no{color:var(--loss)}
.qopts{display:grid;gap:8px;margin:14px 0}
.qopts button{font:inherit;font-size:16px;text-align:left;padding:11px 14px;border:1px solid var(--line);border-radius:8px;background:var(--paper);color:var(--ink);cursor:pointer}
.qopts button:hover:not(:disabled){border-color:var(--turf)}
.qopts button.right{border-color:var(--turf);background:var(--turf-soft)}
.qopts button.wrong{border-color:var(--loss);color:var(--loss)}
.qprog{color:var(--muted);font-size:14px}
.tq{font:600 24px/1.25 "Big Shoulders Display","Arial Narrow",sans-serif;margin:6px 0 0}
.subtabs{display:flex;width:max-content;max-width:100%;margin:18px 0 0;border:1px solid var(--line);border-radius:4px;overflow-x:auto;scrollbar-width:none;background:var(--panel)}
.subtabs::-webkit-scrollbar{display:none}
.subtabs button{flex:0 0 auto;white-space:nowrap}
.subtabs button{font:600 15px/1 "Archivo",system-ui,sans-serif;border:0;border-left:1px solid var(--line);background:transparent;color:var(--muted);padding:10px 16px;cursor:pointer}
.subtabs button:first-child{border-left:0}
.subtabs button[aria-selected="true"]{background:var(--navy);color:#fff}
.subtabs button:focus-visible{outline:2px solid var(--turf);outline-offset:2px}
.fsub h2.sub:first-child{margin-top:18px}
.lede{color:var(--muted);font-size:15px;max-width:72ch;margin:4px 0 12px}
.fant td,.fant th{padding:8px 10px}
.fant th:first-child,.fant td:first-child{background:var(--paper);text-align:right;width:36px}
.fant td.pl,.risers td.pl{text-align:left;white-space:normal;min-width:160px}
.fant td.pl small,.pv small.m{display:block;color:var(--muted);font-size:12.5px}
a.plink{color:var(--ink);text-decoration:none;border-bottom:1px solid var(--line);cursor:pointer}
a.plink:hover{border-bottom-color:var(--turf)}
.chip{display:inline-block;font-size:12px;border-radius:4px;padding:1px 8px;border:1px solid var(--line);color:var(--muted)}
.chip.easy{color:var(--turf);border-color:var(--turf)}
.chip.hard{color:var(--loss);border-color:var(--loss)}
.rng{position:relative;height:8px;width:110px;background:var(--faint);border-radius:4px;display:inline-block;vertical-align:middle;margin-left:8px}
.rng i{position:absolute;top:0;bottom:0;background:var(--turf-soft);border:1px solid var(--turf);border-radius:4px}
.rng b{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--turf)}
.keyst{color:var(--muted);font-size:13px;white-space:normal;min-width:180px;text-align:left}
.ss-inputs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;max-width:760px}
.ss-inputs input{font:inherit;font-size:16px;padding:9px 12px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink)}
.ss-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:14px}
.ss-card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px 16px}
.ss-card.best{border-color:var(--turf);box-shadow:inset 0 0 0 1px var(--turf)}
.ss-card h3{font-size:26px;margin:0}
.ss-card .big{font:700 40px/1 "Big Shoulders Display","Arial Narrow",sans-serif;margin:8px 0 2px}
.ss-card dl{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;font-size:14px;margin:10px 0 0}
.ss-card dt{color:var(--muted)} .ss-card dd{margin:0;text-align:right}
.verdict{font:600 15px "Archivo",system-ui,sans-serif;color:var(--turf)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:10px 0 14px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 14px}
.tile .cond{font-size:32px;font-weight:600;line-height:1}
.tile small{display:block;color:var(--muted);font-size:13px;margin-top:4px}
#gWeeks td,#gWeeks th,#pByStat td,#pByStat th,#pList td,#pList th,#projAcc td,#projAcc th,#riseTbl td,#riseTbl th{padding:8px 10px}
#gWeeks th:first-child,#gWeeks td:first-child,#pByStat td:first-child,#pByStat th:first-child,#pList td:first-child,#pList th:first-child,#projAcc td:first-child,#projAcc th:first-child,#riseTbl td:first-child,#riseTbl th:first-child{background:var(--paper)}
.pv-head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:12px 24px;align-items:flex-end;margin-top:24px}
.pv-head h2{font-size:clamp(40px,7vw,64px);line-height:.95;margin:0}
.pv-head .who{color:var(--muted);margin:6px 0 0}
.pv-week{display:flex;gap:26px}
.pv-week div{text-align:right}
.pv-week .cond{font-size:34px;font-weight:600;line-height:1}
.pv-week small{display:block;color:var(--muted);font-size:13px}
.pv-proj{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0 4px}
.pv-proj .tile{min-width:130px}
.pv-actions{margin:10px 0 0}
.btn{font:inherit;font-size:15px;font-weight:600;background:var(--navy);color:#fff;border:0;border-radius:4px;padding:8px 16px;cursor:pointer}
.btn.ghost{background:none;color:var(--ink);border:1px solid var(--line)}
.btn:focus-visible{outline:2px solid var(--turf);outline-offset:2px}
.metric{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 8px}
.metric button{font:inherit;font-size:14px;border:1px solid var(--line);background:none;color:var(--ink);padding:4px 12px;border-radius:4px;cursor:pointer}
.metric button[aria-pressed="true"]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.trend svg{width:100%;max-width:760px;height:auto;display:block}
.wx{color:var(--muted)}
.wx.windy{color:var(--brass);font-weight:600}
.ask-fab{position:fixed;right:max(18px,env(safe-area-inset-right,0px));bottom:calc(18px + env(safe-area-inset-bottom,0px));z-index:20;font:700 17px/1 "Big Shoulders Display","Oswald","Arial Narrow",sans-serif;letter-spacing:.04em;background:var(--navy);color:#fff;border:0;border-bottom:3px solid var(--pylon);border-radius:4px;padding:13px 18px 11px;cursor:pointer;box-shadow:0 6px 18px rgba(19,33,60,.28)}
.ask-fab.on{background:var(--pylon);border-bottom-color:var(--navy)}
.ask-fab:focus-visible{outline:2px solid var(--pylon);outline-offset:3px}
.ask-panel{position:fixed;right:max(18px,env(safe-area-inset-right,0px));bottom:calc(76px + env(safe-area-inset-bottom,0px));z-index:21;width:min(430px,calc(100vw - 24px));height:min(640px,calc(100vh - 110px));display:flex;flex-direction:column;background:var(--panel);border:1px solid var(--line);border-top:4px solid var(--navy);border-radius:6px;box-shadow:0 18px 40px rgba(19,33,60,.25)}
.ask-panel[hidden]{display:none}
.ask-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;padding:12px 14px;border-bottom:1px solid var(--line)}
.ask-head b{font:700 22px/1 "Big Shoulders Display","Oswald","Arial Narrow",sans-serif;letter-spacing:.02em}
.ask-head small{display:block;color:var(--muted);font-size:12.5px;margin-top:4px}
.ask-x{font-size:24px;line-height:1;background:none;border:0;color:var(--muted);cursor:pointer;padding:2px 6px}
.ask-log{flex:1;overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:10px}
.ask-empty p{color:var(--muted);font-size:14.5px;margin:4px 0 10px}
.ask-sugs{display:flex;flex-direction:column;gap:6px}
.ask-sugs button{font:inherit;font-size:14px;text-align:left;padding:8px 10px;border:1px solid var(--line);border-radius:4px;background:var(--paper);color:var(--ink);cursor:pointer}
.ask-sugs button:hover{border-color:var(--navy)}
.ask-msg{font-size:14.5px;line-height:1.5;max-width:92%}
.ask-msg p{margin:0 0 8px}.ask-msg p:last-child{margin-bottom:0}
.ask-msg ul{margin:4px 0 8px;padding-left:18px}.ask-msg li{margin:3px 0}
.ask-msg.user{align-self:flex-end;background:var(--navy);color:#fff;padding:8px 12px;border-radius:6px 6px 2px 6px}
.ask-msg.bot{align-self:flex-start;background:var(--paper);padding:10px 12px;border-radius:6px 6px 6px 2px;border-left:3px solid var(--pylon)}
.ask-think,.ask-note{color:var(--muted)}
.ask-status{margin:0;padding:4px 14px;font-size:12.5px;color:var(--muted)}
.ask-form{border-top:1px solid var(--line);padding:10px 12px}
.ask-form textarea{width:100%;resize:none;font:inherit;font-size:15px;padding:8px 10px;border:1px solid var(--line);border-radius:4px;background:var(--paper);color:var(--ink)}
.ask-form textarea:focus-visible{outline:2px solid var(--navy);outline-offset:1px}
.ask-actions{display:flex;justify-content:flex-end;gap:6px;margin-top:8px}
.ask-actions .btn{padding:7px 14px;font-size:14px}
@media (max-width:560px){.ask-panel{right:0;left:0;width:auto;bottom:0;height:calc(100% - 40px);border-radius:8px 8px 0 0;padding-bottom:env(safe-area-inset-bottom,0px)}
  .ask-fab.on{display:none}}
.bet-dlg{border:1px solid var(--line);border-top:4px solid var(--navy);border-radius:6px;padding:18px 20px;width:min(460px,calc(100vw - 24px));background:var(--panel);color:var(--ink)}
.bet-dlg::backdrop{background:rgba(19,33,60,.45)}
.bet-dlg h3{font-size:26px;margin:0 0 10px}
.bet-desc{font-weight:600;margin:0 0 4px}
.fld{display:flex;flex-direction:column;gap:4px;font-size:13.5px;color:var(--muted);margin:0 0 10px;flex:1}
.fld input,.fld select{font:inherit;font-size:16px;color:var(--ink);padding:8px 10px;border:1px solid var(--line);border-radius:4px;background:var(--paper)}
.fld-row{display:flex;gap:10px}
.bet-err{color:var(--loss);font-size:14px;margin:0 0 8px}
.dlg-actions{display:flex;justify-content:flex-end;gap:8px;flex-wrap:wrap;margin-top:6px}
.trk{font:600 12.5px/1 "Archivo",system-ui,sans-serif;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:4px;padding:6px 9px;cursor:pointer;white-space:nowrap}
.trk:hover{border-color:var(--navy)}
.trk-row{display:flex;flex-wrap:wrap;gap:6px;margin:16px 0 0}
.trk-row h4{width:100%}
.bet-list td,.bet-list th{padding:8px 10px;vertical-align:top}
.bet-list td.desc{text-align:left;white-space:normal;min-width:200px}
.bet-list td.desc small{display:block;color:var(--muted)}
.st{display:inline-block;font:700 12px/1 "Archivo",system-ui,sans-serif;padding:4px 7px;border-radius:3px}
.st.W{background:var(--turf-soft);color:var(--turf)} .st.L{background:rgba(180,35,24,.1);color:var(--loss)} .st.P{background:var(--faint);color:var(--muted)} .st.open{background:rgba(242,106,33,.12);color:var(--brass)}
.mini{font:inherit;font-size:12.5px;border:1px solid var(--line);background:none;color:var(--ink);border-radius:4px;padding:3px 7px;cursor:pointer;margin:2px 2px 0 0}
.slip{max-width:760px}
.slip-leg{display:grid;grid-template-columns:minmax(0,1fr) 90px 90px 30px;gap:10px;align-items:center;padding:10px 0;border-bottom:1px solid var(--line)}
.slip-leg input{font:inherit;width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:4px;background:var(--panel);color:var(--ink);text-align:right}
.warn{border-left:3px solid var(--brass);padding:6px 12px;background:var(--faint);font-size:14px;margin:10px 0}
a.tlink{color:inherit;text-decoration:none;border-bottom:1px solid var(--line);cursor:pointer}
a.tlink:hover{border-bottom-color:var(--pylon)}
.seed{font:700 16px "Big Shoulders Display","Arial Narrow",sans-serif;color:var(--muted);width:26px;display:inline-block}
.recap-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:12px}
.recap-card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px 16px}
.recap-card h3{font-size:22px;margin:0 0 8px}
.recap-card ul{margin:0;padding-left:18px}.recap-card li{margin:5px 0}
#toast{position:fixed;left:50%;bottom:calc(84px + env(safe-area-inset-bottom,0px));transform:translateX(-50%);background:var(--navy);color:#fff;padding:10px 16px;border-radius:4px;font-size:14.5px;opacity:0;pointer-events:none;transition:opacity .2s;z-index:30;max-width:calc(100vw - 32px)}
#toast.show{opacity:1}
.season-wrap th:first-child,.season-wrap td:first-child{background:var(--paper)}
.panel .season-wrap th:first-child,.panel .season-wrap td:first-child{background:var(--panel)}
.best-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.picks{list-style:none;padding:0;margin:0}
.picks li{padding:9px 0;border-bottom:1px solid var(--faint)}
.picks li:last-child{border-bottom:0}
.picks li small{display:block;color:var(--muted);font-size:13.5px;margin-top:3px}
.picks.cols{columns:2 300px;column-gap:24px}
.picks.cols li{break-inside:avoid}
.edge-tag{float:right;font:700 16px/1.2 "Big Shoulders Display","Arial Narrow",sans-serif;color:var(--turf);margin-left:10px}
.props-intro{max-width:72ch;color:var(--muted);font-size:15px;margin:18px 0 0}
.chk{cursor:pointer}
.chk input{accent-color:var(--turf);width:16px;height:16px}
.props td,.props th{padding:8px 9px;vertical-align:middle}
.props th:first-child,.props td:first-child{background:var(--paper)}
.props td.pl{white-space:normal;min-width:150px}
.props td.pl small{display:block;color:var(--muted);font-size:12.5px}
.tag{display:inline-block;font-size:11px;font-weight:600;color:var(--brass);border:1px solid var(--brass);border-radius:4px;padding:0 4px;margin-left:6px;vertical-align:1px}
.proj{font:600 19px "Big Shoulders Display","Arial Narrow",sans-serif}
.range{display:block;font-size:12px;color:var(--muted)}
.props input{font:inherit;font-size:15px;width:74px;padding:5px 6px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);text-align:right}
.props input.odds{width:62px}
.props input:focus-visible{outline:2px solid var(--turf);outline-offset:1px}
.props input.fromBook{border-color:var(--turf)}
.pick{font:600 17px "Big Shoulders Display","Arial Narrow",sans-serif}
.pick small{display:block;font:500 12px "Archivo",system-ui,sans-serif;color:var(--muted)}
.pick.good{color:var(--turf)}
.bar2{position:relative;height:6px;width:90px;background:var(--faint);border-radius:3px;margin-top:4px}
.bar2 i{position:absolute;left:0;top:0;bottom:0;background:var(--turf);border-radius:3px}
.bar2 b{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink)}
.prop-notes{max-width:72ch;font-size:14.5px;margin-top:18px}
.prop-notes table{max-width:520px;margin:8px 0 12px}
.prop-notes th:first-child,.prop-notes td:first-child{background:transparent}
.gloss{margin-top:48px;max-width:760px}
.gloss h2{font-size:32px;margin:0 0 8px}
.gloss dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 18px;font-size:14.5px;margin:0}
.gloss dt{font-weight:600}
.gloss dd{margin:0;color:var(--muted)}
.note{border-left:3px solid var(--brass);padding:4px 0 4px 14px;margin:22px 0 0;max-width:70ch;font-size:15px}
[hidden]{display:none!important}
@media (max-width:720px){
  .ss-inputs{grid-template-columns:1fr}
  .pv-week div{text-align:left}
  .game summary{grid-template-columns:1fr 1fr 22px}
  .matchup{grid-column:1 / 3}
  .game summary .chev{grid-column:3;grid-row:1}
  .form{grid-template-columns:1fr}
  .record{gap:22px}
  .record div{text-align:left}
  .game summary .res{grid-column:1 / -1}
  nav.views button{font-size:19px;white-space:nowrap;padding:10px 10px 9px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<div class="wrap">
<header class="mast">
  <div class="field"><svg class="field-svg" viewBox="0 0 1200 240" preserveAspectRatio="xMidYMid slice" aria-hidden="true" focusable="false"><rect x="0.0" y="0" width="86.2" height="240" fill="#2B6A3E"/><rect x="85.7" y="0" width="86.2" height="240" fill="#30744A"/><rect x="171.4" y="0" width="86.2" height="240" fill="#2B6A3E"/><rect x="257.1" y="0" width="86.2" height="240" fill="#30744A"/><rect x="342.9" y="0" width="86.2" height="240" fill="#2B6A3E"/><rect x="428.6" y="0" width="86.2" height="240" fill="#30744A"/><rect x="514.3" y="0" width="86.2" height="240" fill="#2B6A3E"/><rect x="600.0" y="0" width="86.2" height="240" fill="#30744A"/><rect x="685.7" y="0" width="86.2" height="240" fill="#2B6A3E"/><rect x="771.4" y="0" width="86.2" height="240" fill="#30744A"/><rect x="857.1" y="0" width="86.2" height="240" fill="#2B6A3E"/><rect x="942.9" y="0" width="86.2" height="240" fill="#30744A"/><rect x="1028.6" y="0" width="86.2" height="240" fill="#2B6A3E"/><rect x="1114.3" y="0" width="86.2" height="240" fill="#30744A"/><line x1="0" x2="1200" y1="14" y2="14" stroke="#F4F6F1" stroke-width="3" opacity=".9"/><line x1="0" x2="1200" y1="226" y2="226" stroke="#F4F6F1" stroke-width="3" opacity=".9"/><line x1="0.0" x2="0.0" y1="14" y2="226" stroke="#F4F6F1" stroke-width="2" opacity=".85"/><line x1="17.1" x2="17.1" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="17.1" x2="17.1" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="17.1" x2="17.1" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="17.1" x2="17.1" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="34.3" x2="34.3" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="34.3" x2="34.3" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="34.3" x2="34.3" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="34.3" x2="34.3" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="51.4" x2="51.4" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="51.4" x2="51.4" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="51.4" x2="51.4" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="51.4" x2="51.4" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="68.6" x2="68.6" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="68.6" x2="68.6" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="68.6" x2="68.6" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="68.6" x2="68.6" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="85.7" x2="85.7" y1="14" y2="226" stroke="#F4F6F1" stroke-width="2" opacity=".85"/><text x="85.7" y="200" text-anchor="middle" font-family="Big Shoulders Display, Oswald, Arial Narrow, sans-serif" font-weight="700" font-size="34" letter-spacing="10" fill="#F4F6F1" opacity=".8">20</text><text x="85.7" y="40" text-anchor="middle" transform="rotate(180 85.7 28)" font-family="Big Shoulders Display, Oswald, Arial Narrow, sans-serif" font-weight="700" font-size="34" letter-spacing="10" fill="#F4F6F1" opacity=".8">20</text><line x1="102.9" x2="102.9" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="102.9" x2="102.9" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="102.9" x2="102.9" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="102.9" x2="102.9" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="120.0" x2="120.0" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="120.0" x2="120.0" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="120.0" x2="120.0" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="120.0" x2="120.0" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="137.1" x2="137.1" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="137.1" x2="137.1" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="137.1" x2="137.1" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="137.1" x2="137.1" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="154.3" x2="154.3" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="154.3" x2="154.3" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="154.3" x2="154.3" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="154.3" x2="154.3" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="171.4" x2="171.4" y1="14" y2="226" stroke="#F4F6F1" stroke-width="2" opacity=".85"/><line x1="188.6" x2="188.6" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="188.6" x2="188.6" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="188.6" x2="188.6" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="188.6" x2="188.6" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="205.7" x2="205.7" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="205.7" x2="205.7" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="205.7" x2="205.7" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="205.7" x2="205.7" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="222.9" x2="222.9" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="222.9" x2="222.9" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="222.9" x2="222.9" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="222.9" x2="222.9" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="240.0" x2="240.0" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="240.0" x2="240.0" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="240.0" x2="240.0" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="240.0" x2="240.0" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="257.1" x2="257.1" y1="14" y2="226" stroke="#F4F6F1" stroke-width="2" opacity=".85"/><text x="257.1" y="200" text-anchor="middle" font-family="Big Shoulders Display, Oswald, Arial Narrow, sans-serif" font-weight="700" font-size="34" letter-spacing="10" fill="#F4F6F1" opacity=".8">30</text><text x="257.1" y="40" text-anchor="middle" transform="rotate(180 257.1 28)" font-family="Big Shoulders Display, Oswald, Arial Narrow, sans-serif" font-weight="700" font-size="34" letter-spacing="10" fill="#F4F6F1" opacity=".8">30</text><line x1="274.3" x2="274.3" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="274.3" x2="274.3" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="274.3" x2="274.3" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="274.3" x2="274.3" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="291.4" x2="291.4" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="291.4" x2="291.4" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="291.4" x2="291.4" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="291.4" x2="291.4" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="308.6" x2="308.6" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="308.6" x2="308.6" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="308.6" x2="308.6" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="308.6" x2="308.6" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="325.7" x2="325.7" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="325.7" x2="325.7" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="325.7" x2="325.7" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="325.7" x2="325.7" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="342.9" x2="342.9" y1="14" y2="226" stroke="#F4F6F1" stroke-width="2" opacity=".85"/><line x1="360.0" x2="360.0" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="360.0" x2="360.0" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="360.0" x2="360.0" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="360.0" x2="360.0" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="377.1" x2="377.1" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="377.1" x2="377.1" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="377.1" x2="377.1" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="377.1" x2="377.1" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="394.3" x2="394.3" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="394.3" x2="394.3" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="394.3" x2="394.3" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="394.3" x2="394.3" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="411.4" x2="411.4" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="411.4" x2="411.4" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="411.4" x2="411.4" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="411.4" x2="411.4" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="428.6" x2="428.6" y1="14" y2="226" stroke="#F4F6F1" stroke-width="2" opacity=".85"/><text x="428.6" y="200" text-anchor="middle" font-family="Big Shoulders Display, Oswald, Arial Narrow, sans-serif" font-weight="700" font-size="34" letter-spacing="10" fill="#F4F6F1" opacity=".8">40</text><text x="428.6" y="40" text-anchor="middle" transform="rotate(180 428.6 28)" font-family="Big Shoulders Display, Oswald, Arial Narrow, sans-serif" font-weight="700" font-size="34" letter-spacing="10" fill="#F4F6F1" opacity=".8">40</text><line x1="445.7" x2="445.7" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="445.7" x2="445.7" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="445.7" x2="445.7" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="445.7" x2="445.7" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="462.9" x2="462.9" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="462.9" x2="462.9" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="462.9" x2="462.9" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="462.9" x2="462.9" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="480.0" x2="480.0" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="480.0" x2="480.0" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="480.0" x2="480.0" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="480.0" x2="480.0" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="497.1" x2="497.1" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="497.1" x2="497.1" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="497.1" x2="497.1" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="497.1" x2="497.1" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="514.3" x2="514.3" y1="14" y2="226" stroke="#F4F6F1" stroke-width="2" opacity=".85"/><line x1="531.4" x2="531.4" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="531.4" x2="531.4" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="531.4" x2="531.4" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="531.4" x2="531.4" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="548.6" x2="548.6" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="548.6" x2="548.6" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="548.6" x2="548.6" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="548.6" x2="548.6" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="565.7" x2="565.7" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="565.7" x2="565.7" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="565.7" x2="565.7" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="565.7" x2="565.7" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="582.9" x2="582.9" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="582.9" x2="582.9" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="582.9" x2="582.9" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="582.9" x2="582.9" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="600.0" x2="600.0" y1="14" y2="226" stroke="#F4F6F1" stroke-width="3" opacity=".85"/><text x="600.0" y="200" text-anchor="middle" font-family="Big Shoulders Display, Oswald, Arial Narrow, sans-serif" font-weight="700" font-size="34" letter-spacing="10" fill="#F4F6F1" opacity=".8">50</text><text x="600.0" y="40" text-anchor="middle" transform="rotate(180 600.0 28)" font-family="Big Shoulders Display, Oswald, Arial Narrow, sans-serif" font-weight="700" font-size="34" letter-spacing="10" fill="#F4F6F1" opacity=".8">50</text><line x1="617.1" x2="617.1" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="617.1" x2="617.1" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="617.1" x2="617.1" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="617.1" x2="617.1" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="634.3" x2="634.3" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="634.3" x2="634.3" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="634.3" x2="634.3" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="634.3" x2="634.3" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="651.4" x2="651.4" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="651.4" x2="651.4" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="651.4" x2="651.4" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="651.4" x2="651.4" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="668.6" x2="668.6" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="668.6" x2="668.6" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="668.6" x2="668.6" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="668.6" x2="668.6" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="685.7" x2="685.7" y1="14" y2="226" stroke="#F4F6F1" stroke-width="2" opacity=".85"/><line x1="702.9" x2="702.9" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="702.9" x2="702.9" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="702.9" x2="702.9" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="702.9" x2="702.9" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="720.0" x2="720.0" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="720.0" x2="720.0" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="720.0" x2="720.0" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="720.0" x2="720.0" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="737.1" x2="737.1" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="737.1" x2="737.1" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="737.1" x2="737.1" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="737.1" x2="737.1" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="754.3" x2="754.3" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="754.3" x2="754.3" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="754.3" x2="754.3" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="754.3" x2="754.3" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="771.4" x2="771.4" y1="14" y2="226" stroke="#F4F6F1" stroke-width="2" opacity=".85"/><text x="771.4" y="200" text-anchor="middle" font-family="Big Shoulders Display, Oswald, Arial Narrow, sans-serif" font-weight="700" font-size="34" letter-spacing="10" fill="#F4F6F1" opacity=".8">40</text><text x="771.4" y="40" text-anchor="middle" transform="rotate(180 771.4 28)" font-family="Big Shoulders Display, Oswald, Arial Narrow, sans-serif" font-weight="700" font-size="34" letter-spacing="10" fill="#F4F6F1" opacity=".8">40</text><line x1="788.6" x2="788.6" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="788.6" x2="788.6" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="788.6" x2="788.6" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="788.6" x2="788.6" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="805.7" x2="805.7" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="805.7" x2="805.7" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="805.7" x2="805.7" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="805.7" x2="805.7" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="822.9" x2="822.9" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="822.9" x2="822.9" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="822.9" x2="822.9" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="822.9" x2="822.9" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="840.0" x2="840.0" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="840.0" x2="840.0" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="840.0" x2="840.0" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="840.0" x2="840.0" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="857.1" x2="857.1" y1="14" y2="226" stroke="#F4F6F1" stroke-width="2" opacity=".85"/><line x1="874.3" x2="874.3" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="874.3" x2="874.3" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="874.3" x2="874.3" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="874.3" x2="874.3" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="891.4" x2="891.4" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="891.4" x2="891.4" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="891.4" x2="891.4" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="891.4" x2="891.4" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="908.6" x2="908.6" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="908.6" x2="908.6" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="908.6" x2="908.6" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="908.6" x2="908.6" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="925.7" x2="925.7" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="925.7" x2="925.7" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="925.7" x2="925.7" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="925.7" x2="925.7" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="942.9" x2="942.9" y1="14" y2="226" stroke="#F4F6F1" stroke-width="2" opacity=".85"/><text x="942.9" y="200" text-anchor="middle" font-family="Big Shoulders Display, Oswald, Arial Narrow, sans-serif" font-weight="700" font-size="34" letter-spacing="10" fill="#F4F6F1" opacity=".8">30</text><text x="942.9" y="40" text-anchor="middle" transform="rotate(180 942.9 28)" font-family="Big Shoulders Display, Oswald, Arial Narrow, sans-serif" font-weight="700" font-size="34" letter-spacing="10" fill="#F4F6F1" opacity=".8">30</text><line x1="960.0" x2="960.0" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="960.0" x2="960.0" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="960.0" x2="960.0" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="960.0" x2="960.0" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="977.1" x2="977.1" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="977.1" x2="977.1" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="977.1" x2="977.1" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="977.1" x2="977.1" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="994.3" x2="994.3" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="994.3" x2="994.3" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="994.3" x2="994.3" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="994.3" x2="994.3" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1011.4" x2="1011.4" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1011.4" x2="1011.4" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1011.4" x2="1011.4" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1011.4" x2="1011.4" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1028.6" x2="1028.6" y1="14" y2="226" stroke="#F4F6F1" stroke-width="2" opacity=".85"/><line x1="1045.7" x2="1045.7" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1045.7" x2="1045.7" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1045.7" x2="1045.7" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1045.7" x2="1045.7" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1062.9" x2="1062.9" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1062.9" x2="1062.9" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1062.9" x2="1062.9" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1062.9" x2="1062.9" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1080.0" x2="1080.0" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1080.0" x2="1080.0" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1080.0" x2="1080.0" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1080.0" x2="1080.0" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1097.1" x2="1097.1" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1097.1" x2="1097.1" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1097.1" x2="1097.1" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1097.1" x2="1097.1" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1114.3" x2="1114.3" y1="14" y2="226" stroke="#F4F6F1" stroke-width="2" opacity=".85"/><text x="1114.3" y="200" text-anchor="middle" font-family="Big Shoulders Display, Oswald, Arial Narrow, sans-serif" font-weight="700" font-size="34" letter-spacing="10" fill="#F4F6F1" opacity=".8">20</text><text x="1114.3" y="40" text-anchor="middle" transform="rotate(180 1114.3 28)" font-family="Big Shoulders Display, Oswald, Arial Narrow, sans-serif" font-weight="700" font-size="34" letter-spacing="10" fill="#F4F6F1" opacity=".8">20</text><line x1="1131.4" x2="1131.4" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1131.4" x2="1131.4" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1131.4" x2="1131.4" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1131.4" x2="1131.4" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1148.6" x2="1148.6" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1148.6" x2="1148.6" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1148.6" x2="1148.6" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1148.6" x2="1148.6" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1165.7" x2="1165.7" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1165.7" x2="1165.7" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1165.7" x2="1165.7" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1165.7" x2="1165.7" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1182.9" x2="1182.9" y1="14" y2="22" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1182.9" x2="1182.9" y1="92" y2="100" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1182.9" x2="1182.9" y1="142" y2="150" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1182.9" x2="1182.9" y1="218" y2="226" stroke="#F4F6F1" stroke-width="1.5" opacity=".7"/><line x1="1200.0" x2="1200.0" y1="14" y2="226" stroke="#F4F6F1" stroke-width="2" opacity=".85"/></svg>
    <div class="paint"><h1 id="title">Model Board</h1></div>
  </div>
  <div class="scoreboard" id="record"></div>
  <p class="updated" id="updated"></p>
</header>

<div class="search">
  <label for="q" class="sr">Search players and teams</label>
  <input id="q" type="search" autocomplete="off" placeholder="Search a player or team" role="combobox" aria-expanded="false" aria-controls="qres" aria-autocomplete="list">
  <ul id="qres" role="listbox" hidden></ul>
</div>

<nav class="views" role="tablist">
  <button role="tab" aria-selected="true" data-view="week">Games</button>
  <button role="tab" aria-selected="false" data-view="props">Betting</button>
  <button role="tab" aria-selected="false" data-view="fantasy">Fantasy</button>
  <button role="tab" aria-selected="false" data-view="league">League</button>
  <button role="tab" aria-selected="false" data-view="results">Model record</button>
  <button role="tab" aria-selected="false" data-view="trivia">Trivia</button>
</nav>

<section id="view-week">
  <nav class="subtabs" role="tablist" aria-label="Games views">
    <button role="tab" aria-selected="true" data-gsub="wk">This week</button>
    <button role="tab" aria-selected="false" data-gsub="sched">Full schedule</button>
  </nav>
  <div id="gsub-wk">
  <div class="weeks" id="weeks" role="group" aria-label="Choose week" style="margin-top:12px"></div>
  <p class="empty">Tap a game to see both teams' form and every offensive skill player's advanced stats.</p>
  <div id="games"></div>
  </div>
  <div id="gsub-sched" hidden>
  <div class="toolbar">
    <label>Team <select id="teamSel"><option value="">All teams</option></select></label>
    <label>Show <select id="showSel"><option value="all">All games</option><option value="done">Finished</option><option value="up">Upcoming</option></select></label>
  </div>
  <p class="teamrec" id="teamrec"></p>
  <div class="season-wrap"><table class="season" id="seasonTbl"></table></div>
  </div>
</section>

<section id="view-props" hidden>
  <nav class="subtabs" role="tablist" aria-label="Betting views">
    <button role="tab" aria-selected="true" data-psb="best">Best bets</button>
    <button role="tab" aria-selected="false" data-psb="all">All player props</button>
  </nav>
  <div id="psb-best"></div>
  <div id="psb-all" hidden>
  <p class="props-intro" id="propsIntro"></p>
  <div id="playerCard"></div>
  <div class="toolbar">
    <label>Game <select id="pGame"><option value="">All games</option></select></label>
    <label>Stat <select id="pStat"><option value="">All stats</option></select></label>
    <label>Position <select id="pPos"><option value="">All</option><option>QB</option><option>RB</option><option>WR</option><option>TE</option></select></label>
    <label class="chk"><input type="checkbox" id="pEdge"> Only plays with 5%+ edge</label>
  </div>
  <div class="season-wrap"><table class="props" id="propsTbl"></table></div>
  <div class="prop-notes" id="propNotes"></div>
  </div>
</section>

<section id="view-fantasy" hidden>
  <nav class="subtabs" role="tablist" aria-label="Fantasy tools">
    <button role="tab" aria-selected="true" data-sub="rank">Rankings</button>
    <button role="tab" aria-selected="false" data-sub="ss">Start or sit</button>
    <button role="tab" aria-selected="false" data-sub="rise">Rising usage</button>
  </nav>
  <div class="toolbar" id="fToolbar">
    <label>Scoring <select id="fScore"><option value="1">PPR</option><option value="0.5">Half PPR</option><option value="0">Standard</option></select></label>
    <label>Position <select id="fPos"><option value="">All</option><option>QB</option><option>RB</option><option>WR</option><option>TE</option><option value="FLEX">FLEX (RB/WR/TE)</option></select></label>
  </div>
  <div class="fsub" id="fsub-rank">
  <h2 class="sub" id="fTitle">Rankings</h2>
  <p class="lede">Projected fantasy points with a likely range (the middle half of outcomes). Matchup rank is how generous the opponent has been to that position lately, 1 being the easiest.</p>
  <div class="season-wrap"><table class="fant" id="fantTbl"></table></div>
  </div>

  <div class="fsub" id="fsub-ss" hidden>
  <h2 class="sub">Start or sit</h2>
  <p class="lede">Type up to three players to compare them side by side.</p>
  <datalist id="fantNames"></datalist>
  <div class="ss-inputs">
    <input list="fantNames" class="ssIn" placeholder="Player 1" aria-label="Player 1">
    <input list="fantNames" class="ssIn" placeholder="Player 2" aria-label="Player 2">
    <input list="fantNames" class="ssIn" placeholder="Player 3 (optional)" aria-label="Player 3">
  </div>
  <div id="ssOut"></div>
  </div>

  <div class="fsub" id="fsub-rise" hidden>
  <h2 class="sub">Rising usage</h2>
  <p class="lede">Running backs and receivers whose role grew over the last two games compared with their previous games for the same team. Bigger roles often show up in fantasy points a week or two later, which makes these good waiver-wire targets.</p>
  <div class="season-wrap"><table id="riseTbl"></table></div>
  </div>
</section>

<section id="view-results" hidden>
  <h2 class="sub">Game picks</h2>
  <div class="tiles" id="gTiles"></div>
  <div class="season-wrap"><table id="gWeeks"></table></div>
  <p class="lede" id="gTest"></p>
  <h2 class="sub">Closing line value</h2>
  <p class="lede">Did the line move toward the model's pick after the site first made it? Consistently getting a better number than the closing line is the strongest sign an edge is real, even before the win-loss record has enough games to mean much.</p>
  <div class="tiles" id="clvTiles"></div>

  <h2 class="sub">Prop picks</h2>
  <div class="toolbar">
    <label>Lines <select id="rSrc"><option value="book">Sportsbook lines the site pulled</option><option value="mine">Lines I entered</option></select></label>
    <label>Minimum edge <select id="rEdge"><option value="0">Any edge</option><option value="0.03">3%</option><option value="0.05" selected>5%</option><option value="0.08">8%</option></select></label>
  </div>
  <div class="tiles" id="pTiles"></div>
  <div class="season-wrap"><table id="pByStat"></table></div>
  <div class="season-wrap"><table id="pList"></table></div>

  <h2 class="sub">Projection accuracy</h2>
  <p class="lede" id="projLede"></p>
  <div class="season-wrap"><table id="projAcc"></table></div>
</section>


<section id="view-trivia" hidden>
  <div class="chalk-wrap"><svg class="chalk" viewBox="0 0 720 150" role="img" aria-label="Chalkboard play diagram">
<rect width="720" height="150" rx="6" fill="#23372D"/>
<g fill="none" stroke="#EDEFE6" stroke-width="2.5" stroke-linecap="round" opacity=".92">
<line x1="30" y1="96" x2="690" y2="96" stroke-dasharray="6 8" opacity=".5"/>
<circle cx="300" cy="112" r="9"/><circle cx="330" cy="112" r="9"/><circle cx="360" cy="112" r="9"/><circle cx="390" cy="112" r="9"/><circle cx="420" cy="112" r="9"/>
<circle cx="360" cy="136" r="9"/>
<circle cx="150" cy="112" r="9"/><circle cx="560" cy="112" r="9"/><circle cx="470" cy="120" r="9"/>
<path d="M150 103 C150 70 170 40 230 30"/><path d="M224 24 L232 30 L223 36"/>
<path d="M560 103 L560 60 L500 44"/><path d="M506 38 L498 44 L506 50"/>
<path d="M470 111 C480 80 520 70 600 70"/><path d="M593 64 L601 70 L593 76"/>

</g>
<g stroke="#EDEFE6" stroke-width="2.5" stroke-linecap="round" opacity=".85">
<path d="M290 70 l14 14 M304 70 l-14 14"/><path d="M350 66 l14 14 M364 66 l-14 14"/><path d="M410 70 l14 14 M424 70 l-14 14"/>
<path d="M150 48 l14 14 M164 48 l-14 14"/><path d="M548 30 l14 14 M562 30 l-14 14"/><path d="M620 52 l14 14 M634 52 l-14 14"/><path d="M252 34 l14 14 M266 34 l-14 14"/>
</g>
</svg></div>
  <nav class="subtabs" role="tablist" aria-label="Trivia games">
    <button role="tab" aria-selected="true" data-tsub="guess">Guess the player</button>
    <button role="tab" aria-selected="false" data-tsub="quiz">Quiz</button>
  </nav>
  <div id="tGuess"></div>
  <div id="tQuiz" hidden></div>
  <datalist id="allNames"></datalist>
</section>


<section id="view-league" hidden>
  <nav class="subtabs" role="tablist" aria-label="League views">
    <button role="tab" aria-selected="true" data-lsub="stand">Standings</button>
    <button role="tab" aria-selected="false" data-lsub="odds">Playoff odds</button>
    <button role="tab" aria-selected="false" data-lsub="power">Power rankings</button>
    <button role="tab" aria-selected="false" data-lsub="leaders">Stat leaders</button>
    <button role="tab" aria-selected="false" data-lsub="recap">Weekly recap</button>
  </nav>
  <div id="lsub-stand" class="lsub"></div>
  <div id="lsub-po" class="lsub" hidden><p class="lede" id="poLede" style="margin-top:14px"></p><div id="poOut"></div></div>
  <div id="lsub-leaders" class="lsub" hidden>
    <div class="toolbar" style="margin-top:14px">
      <label>Position <select id="ldPos"><option>QB</option><option>RB</option><option selected>WR</option><option>TE</option></select></label>
      <label>Stat <select id="ldStat"></select></label>
      <label>Minimum games <select id="ldMin"><option>1</option><option selected>2</option><option>4</option><option>6</option><option>8</option></select></label>
    </div>
    <div class="season-wrap"><table id="ldTbl" class="po"></table></div>
  </div>
  <div id="lsub-recap" class="lsub" hidden></div>
</section>

<section id="view-team" hidden><div id="teamView"></div></section>


<section id="view-player" hidden><div id="playerView"></div></section>


<section class="gloss">
  <h2>What the numbers mean</h2>
  <dl>
    <dt>Spread</dt><dd>Favorite shown with a minus. "BUF -7" means Buffalo is expected to win by 7. The model's spread is its own estimate.</dd>
    <dt>Moneyline</dt><dd>Vegas odds to win outright. The model's "fair" line is what the odds would be if its win probability were exactly right.</dd>
    <dt>Edge</dt><dd>Points between the model's spread and Vegas. Big edges usually mean the model is missing news, like an injured quarterback.</dd>
    <dt>EPA</dt><dd>Expected points added. How much a play raised or lowered the offense's expected points. Positive is good.</dd>
    <dt>EPA/db</dt><dd>QB passing EPA per dropback (attempts plus sacks). About +0.15 or higher is very good.</dd>
    <dt>CPOE</dt><dd>Completion % over expected, given how hard each throw was.</dd>
    <dt>aDOT</dt><dd>Average depth of target, in yards downfield.</dd>
    <dt>Tgt%</dt><dd>Share of the team's targets the player received.</dd>
    <dt>Air%</dt><dd>Share of the team's air yards (downfield passing yards on targets).</dd>
    <dt>WOPR</dt><dd>Weighted opportunity rating, blending target share and air yards share. Above 0.5 is elite usage.</dd>
    <dt>RACR</dt><dd>Receiving yards per air yard. Above 1 means the player adds yards after the catch.</dd>
    <dt>Prop edge</dt><dd>Model's chance of a result minus the chance the odds need to break even. At -110 you need 52.4%.</dd>
    <dt>Form ranks</dt><dd>Team EPA per play over roughly the last 10 games, ranked 1 (best) to 32.</dd>
    <dt>Total</dt><dd>Combined points scored by both teams. The model's total is compared with the Vegas over/under.</dd>
    <dt>QB rating</dt><dd>The model rates each starting quarterback by his EPA per dropback, weighting recent games more. When a backup starts, the spread adjusts.</dd>
  </dl>
  <p class="note" id="glossNote"></p>
</section>
</div>

<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const fmt = (v, d=1) => v == null ? "–" : (+v).toFixed(d);
const sgn = v => v == null ? "–" : (v > 0 ? "+" : "") + Math.round(v);
function spreadTxt(g, v){ // v = expected home margin
  if (v == null) return "–";
  if (Math.abs(v) < 0.25) return "Pick'em";
  const t = v > 0 ? g.home : g.away, n = Math.abs(v);
  return t + " -" + (Math.round(n*2)/2).toString();
}
function fairML(p){ if (p == null) return null; return p >= .5 ? -100*p/(1-p) : 100*(1-p)/p; }
function mlTxt(v){ return v == null ? "–" : (v > 0 ? "+" : "") + Math.round(v); }
function timeTxt(g){ const [h,m] = g.time.split(":"); let hh=+h, ap=hh>=12?"pm":"am"; hh=hh%12||12;
  const d = new Date(g.date+"T12:00:00"); return d.toLocaleDateString("en-US",{weekday:"short",month:"short",day:"numeric"})+", "+hh+":"+m+ap+" ET"; }
const done = g => g.hs != null;

// header
$("#updated").textContent = `${D.season} NFL season. Updated ${D.updated}.`;
function gradeGames(list){
  const r = {su:[0,0], ats:[0,0,0], ats3:[0,0], ou:[0,0,0], ou3:[0,0]};
  list.filter(done).forEach(g=>{
    const m = g.hs-g.as_, t = g.hs+g.as_;
    if (m!==0 && g.pred!=null) r.su[(g.pred>0)===(m>0)?0:1]++;
    if (g.spread!=null && g.pred!=null){
      if (m===g.spread) r.ats[2]++;
      else { const ok=(g.pred>g.spread)===(m>g.spread); r.ats[ok?0:1]++; if(Math.abs(g.pred-g.spread)>=3) r.ats3[ok?0:1]++; }
    }
    if (g.total!=null && g.tpred!=null){
      if (t===g.total) r.ou[2]++;
      else { const ok=(g.tpred>g.total)===(t>g.total); r.ou[ok?0:1]++; if(Math.abs(g.tpred-g.total)>=3) r.ou3[ok?0:1]++; }
    }
  });
  return r;
}
const wl = a => a[0]+"-"+a[1]+(a[2]?"-"+a[2]:"");
const pct = a => (a[0]+a[1]) ? Math.round(a[0]/(a[0]+a[1])*1000)/10+"%" : "–";
const SEASON_REC = gradeGames(D.games);
$("#record").innerHTML =
  `<div><span class="cond">${wl(SEASON_REC.su)}</span><small>Picking winners</small></div>
   <div><span class="cond">${wl(SEASON_REC.ats)}</span><small>Vs the spread</small></div>
   <div><span class="cond">${wl(SEASON_REC.ou)}</span><small>Totals</small></div>`;
const GR = D.reports.game;
$("#glossNote").textContent = `Tested on ${GR.seasons} games it never saw, the game model picked ${GR.acc}% of winners (Vegas favorites: ${GR.vegas_acc}%) and went ${GR.ats3}% against the spread when it disagreed with Vegas by 3+ points. You need about 52.4% to break even. Use it as a starting point for research, not as a betting system.`;

// views
function showView(v){
  document.querySelectorAll("nav.views button").forEach(o => o.setAttribute("aria-selected", o.dataset.view===v));
  ["week","props","fantasy","results","player","trivia","league","team"].forEach(x => $("#view-"+x).hidden = x !== v);
  if (v==="league" && typeof renderLeague==="function") renderLeague();
}
document.querySelectorAll("nav.views button").forEach(b => b.onclick = () => showView(b.dataset.view));

// ---------- week view ----------
const weeks = [...new Set(D.games.map(g=>g.wk))];
const curWk = (D.games.find(g=>!done(g)) || D.games[D.games.length-1]).wk;
let selWk = curWk;
function renderWeeks(){
  $("#weeks").innerHTML = weeks.map(w=>{
    const all = D.games.filter(g=>g.wk===w).every(done);
    return `<button class="${all?'done':''}" aria-pressed="${w===selWk}" data-w="${w}" aria-label="Week ${w}">${w}</button>`}).join("");
  $("#weeks").querySelectorAll("button").forEach(b=>b.onclick=()=>{selWk=+b.dataset.w;renderWeeks();renderGames();});
}
function resultCell(g){
  if (!done(g)) return `<div class="cell res"><small>Kickoff</small><span class="ml">${timeTxt(g)}</span></div>`;
  const margin = g.hs - g.as_;
  let ats = "", pick = "";
  if (g.spread != null && margin !== g.spread){
    const homeCovered = margin > g.spread;
    ats = (homeCovered ? g.home : g.away) + " covered";
    const modelHome = g.pred > g.spread;
    pick = modelHome === homeCovered ? '<span class="res-w">Model right</span>' : '<span class="res-l">Model wrong</span>';
  } else if (g.spread != null) ats = "Push";
  let tot = "";
  if (g.total!=null && g.tpred!=null){ const t=g.hs+g.as_; if (t===g.total) tot=" Total pushed."; else {
    const ok=(g.tpred>g.total)===(t>g.total); tot = ` ${t>g.total?"Over":"Under"} hit, <span class="${ok?'res-w':'res-l'}">model ${ok?"right":"wrong"}</span>.`; } }
  return `<div class="cell res"><small>Final</small><span class="v">${g.away} ${g.as_}, ${g.home} ${g.hs}</span><span class="ml">${ats}${pick? ". "+pick:""}.${tot}</span></div>`;
}
function formPanel(t){
  const f = D.teams[t]; if(!f) return "";
  const row = (lbl, r) => `<span>${lbl}</span><span class="bar"><i style="width:${Math.round((33-r)/32*100)}%;background:${tcol(t)}"></i></span><span class="rk">${r}${r===1?"st":r===2?"nd":r===3?"rd":"th"}</span>`;
  return `<div><h3>${teamChip(t)}</h3><div class="bars">${row("Offense",f.off)}${row("Pass offense",f.pass_off)}${row("Run offense",f.rush_off)}${row("Defense",f.def)}${row("Pass defense",f.pass_def)}${row("Run defense",f.rush_def)}${f.pass_block?row("Pass protection",f.pass_block):""}${f.pass_rush?row("Pass rush",f.pass_rush):""}</div></div>`;
}
const COLS = {
  QB: [["G","gp",0],["Snap%","snap",0],["EPA/db","epa",2,.15,-.05],["CPOE","cpoe",1,3,-3],["aDOT","adot",1],["Y/A","ypa",1,8,6],["Pass Y/G","ypg",0],["TD","td",0],["INT","int_",0],["Rush Y/G","rypg",0],["PPR/G","ppg",1]],
  RB: [["G","gp",0],["Snap%","snap",0,65,35],["Car/G","cpg",1],["YPC","ypc",1,5,3.5],["EPA/car","repa",2,.05,-.15],["Tgt/G","tpg",1],["Tgt%","ts",1,12],["Scrim Y/G","ypg",0],["TD","td",0],["PPR/G","ppg",1]],
  WR: [["G","gp",0],["Snap%","snap",0,80,50],["Tgt/G","tpg",1],["Tgt%","ts",1,24,10],["Air%","as_",1,30],["WOPR","wopr",2,.55,.25],["aDOT","adot",1],["RACR","racr",2,1],["YAC/rec","yac",1,5.5],["EPA/tgt","epat",2,.3,-.1],["Catch%","cr",0],["Rec Y/G","ypg",0],["TD","td",0],["PPR/G","ppg",1]],
};
COLS.TE = COLS.WR;
const sortState = {};
function playerTable(key, pos, g){
  const list = D.players.filter(p=>p.pos===pos && (p.tm===g.away || p.tm===g.home));
  if (!list.length) return "";
  const cols = COLS[pos];
  const st = sortState[key+pos] || {k:"ppg", dir:-1};
  list.sort((a,b)=>((a[st.k]??-1e9)-(b[st.k]??-1e9))*st.dir);
  const head = `<tr><th>Player</th>${cols.map(c=>`<th data-k="${c[1]}" ${st.k===c[1]?`aria-sort="${st.dir<0?'descending':'ascending'}"`:""}>${c[0]}</th>`).join("")}</tr>`;
  const rows = list.map(p=>`<tr class="${p.tm===g.away?'team-a':'team-b'}"><td>${teamBadge(p.tm)}${plink(p.n)}</td>${cols.map(c=>{
      const v=p[c[1]]; let cls="";
      if(v!=null && c[3]!=null && v>=c[3]) cls="hot"; else if(v!=null && c[4]!=null && v<=c[4]) cls="cold";
      return `<td class="${cls}">${fmt(v,c[2])}</td>`;}).join("")}</tr>`).join("");
  const label = {QB:"Quarterbacks",RB:"Running backs",WR:"Wide receivers",TE:"Tight ends"}[pos];
  return `<div class="pos-block"><h4>${label}</h4><div class="scroll"><table data-key="${key}" data-pos="${pos}">${head}${rows}</table></div></div>`;
}
function outsHTML(g){
  if (!g.outs) return "";
  return `<p class="outs">${Object.entries(g.outs).map(([tm,l])=>`<b>${tm} out:</b> ${l.join(", ")}`).join(". ")}. The model gives their work to teammates.</p>`;
}
function shopHTML(g){
  const B = g.books; if (!B || !B.length) return "";
  const sideHome = g.pred > g.spread, overPick = g.tpred > g.total;
  const bestBy = (arr, key, better) => arr.filter(b=>b[key]!=null).reduce((a,b)=>!a||better(b,a)?b:a, null);
  const bs = sideHome ? bestBy(B,"sp_h",(b,a)=>b.sp_h>a.sp_h||(b.sp_h===a.sp_h&&b.sp_hp>a.sp_hp)) : bestBy(B,"sp_a",(b,a)=>b.sp_a>a.sp_a||(b.sp_a===a.sp_a&&b.sp_ap>a.sp_ap));
  const bt = overPick ? bestBy(B,"tot",(b,a)=>b.tot<a.tot||(b.tot===a.tot&&b.ov>a.ov)) : bestBy(B,"tot",(b,a)=>b.tot>a.tot||(b.tot===a.tot&&b.un>a.un));
  const winHome = g.prob >= .5, bm = bestBy(B, winHome?"ml_h":"ml_a", (b,a)=>winHome? b.ml_h>a.ml_h : b.ml_a>a.ml_a);
  const sp = x => x==null?"–":(x>0?"+":"")+x;
  let s = `<div class="shop"><h4>Line shopping, ${B.length} sportsbooks</h4><p class="lede" style="margin:2px 0 8px">Best available price for each side the model likes.</p><div class="tiles">`;
  if (bs) s += `<div class="tile"><span class="cond">${sideHome?g.home:g.away} ${sp(sideHome?bs.sp_h:bs.sp_a)}</span><small>${fmtOdds(sideHome?bs.sp_hp:bs.sp_ap)} at ${bs.book}</small></div>`;
  if (bm) s += `<div class="tile"><span class="cond">${winHome?g.home:g.away} ${fmtOdds(winHome?bm.ml_h:bm.ml_a)}</span><small>Moneyline at ${bm.book}</small></div>`;
  if (bt) s += `<div class="tile"><span class="cond">${overPick?"Over":"Under"} ${bt.tot}</span><small>${fmtOdds(overPick?bt.ov:bt.un)} at ${bt.book}</small></div>`;
  s += `</div><div class="scroll"><table><tr><th>Book</th><th>${g.away} spread</th><th>${g.home} spread</th><th>${g.away} ML</th><th>${g.home} ML</th><th>Total</th><th>Over</th><th>Under</th></tr>` +
    B.map(b=>`<tr><td>${b.book}</td><td>${sp(b.sp_a)} (${fmtOdds(b.sp_ap)})</td><td>${sp(b.sp_h)} (${fmtOdds(b.sp_hp)})</td><td>${fmtOdds(b.ml_a)}</td><td>${fmtOdds(b.ml_h)}</td><td>${b.tot??"–"}</td><td>${fmtOdds(b.ov)}</td><td>${fmtOdds(b.un)}</td></tr>`).join("") + `</table></div></div>`;
  return s;
}
function panelHTML(g){
  return `<div class="panel">${outsHTML(g)}<div class="form">${formPanel(g.away)}${formPanel(g.home)}</div>${shopHTML(g)}
    ${["QB","RB","WR","TE"].map(p=>playerTable(g.id,p,g)).join("") || '<p class="empty">No player stats yet this season.</p>'}
    <p class="empty" style="margin-top:12px">Player stats are ${D.season} season to date. Green numbers are strong, red are weak. Tap a column header to sort.</p></div>`;
}
const COL = D.colors || {};
const TNAME = Object.fromEntries((D.playoffs||[]).map(x=>[x.t, x.name]));
const tcol = t => (COL[t]&&COL[t][0]) || "#5B6475";
function lum(hex){ const h=hex.replace("#",""); if(h.length<6) return .5; const c=[0,2,4].map(i=>parseInt(h.substr(i,2),16)/255).map(v=>v<=.03928?v/12.92:Math.pow((v+.055)/1.055,2.4)); return .2126*c[0]+.7152*c[1]+.0722*c[2]; }
function teamBadge(t){ if(!t) return ""; const bg=tcol(t); return `<span class="tbadge" style="background:${bg};color:${lum(bg)>.4?"#13213C":"#FFFFFF"}" title="${TNAME[t]||t}">${t}</span>`; }
function teamChip(t){ const bg=tcol(t); return `<span class="tchip" style="background:${bg};color:${lum(bg)>.4?"#13213C":"#FFFFFF"}">${t}</span>`; }
function wxTxt(g){
  const w = g.wx;
  if (!w) return g.neutral && g.venue ? `<small class="wx">At ${g.venue}</small>` : "";
  const at = g.neutral && g.venue ? `At ${g.venue}. ` : "";
  if (w.roof==="dome") return `<small class="wx">${at}Indoors</small>`;
  if (w.temp==null) return at ? `<small class="wx">${at}</small>` : "";
  const windy = w.wind>=15;
  return `<small class="wx ${windy?'windy':''}">${at}${w.roof==="retract"?"Retractable roof. ":""}${w.temp}°F, wind ${w.wind} mph${w.gust>=25?` (gusts ${w.gust})`:""}, ${w.rain}% rain${windy?". Windy: tough for passing and kicking":""}</small>`;
}
function renderGames(){
  const gs = D.games.filter(g=>g.wk===selWk);
  $("#games").innerHTML = gs.map(g=>{
    const edge = (g.pred!=null && g.spread!=null) ? Math.abs(g.pred-g.spread) : null;
    const mw = g.prob>=.5 ? g.home : g.away, mp = Math.round((g.prob>=.5?g.prob:1-g.prob)*100);
    const fh = fairML(g.prob), fa = fairML(1-g.prob);
    return `<details class="game" data-id="${g.id}" style="--ca:${tcol(g.away)};--ch:${tcol(g.home)}"><summary>
      <div class="matchup"><span>${teamChip(g.away)}<span class="at">at</span>${teamChip(g.home)}</span>${g.aqb&&g.hqb?`<small>${g.aqb} vs ${g.hqb}</small>`:""}${wxTxt(g)}</div>
      <div class="cell"><small>Vegas</small><span class="v">${spreadTxt(g,g.spread)}</span><span class="ml">${g.aml!=null?`${g.away} ${mlTxt(g.aml)}, ${g.home} ${mlTxt(g.hml)}`:"Line not posted"}${g.total!=null?`. O/U ${g.total}`:""}</span></div>
      <div class="cell ${edge!=null&&edge>=3?'edge3':''}"><small>Model${edge!=null?`, edge ${edge.toFixed(1)}`:""}</small><span class="v">${spreadTxt(g,g.pred)}</span><span class="ml">${mw} ${mp}%. Fair ${g.away} ${mlTxt(fa)}, ${g.home} ${mlTxt(fh)}${g.tpred!=null?`. Total ${g.tpred.toFixed(1)}${g.total!=null?` (${g.tpred>g.total?"over":"under"} ${g.total})`:""}`:""}</span></div>
      ${resultCell(g)}
      <span class="chev" aria-hidden="true"></span></summary><div class="slot"></div></details>`;
  }).join("");
  $("#games").querySelectorAll("details").forEach(d=>d.addEventListener("toggle",()=>{
    if(d.open) fillPanel(d);
  }));
}
function fillPanel(d){
  const g = D.games.find(x=>x.id===d.dataset.id);
  const slot = d.querySelector(".slot"); slot.innerHTML = panelHTML(g);
  slot.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{
    const t = th.closest("table"), key=t.dataset.key+t.dataset.pos, k=th.dataset.k;
    const cur = sortState[key] || {k:"ppg",dir:-1};
    sortState[key] = {k, dir: cur.k===k ? -cur.dir : -1};
    fillPanel(d);
  });
}
renderWeeks(); renderGames();

// ---------- season view ----------
const teams = Object.keys(D.teams).sort();
$("#teamSel").innerHTML += teams.map(t=>`<option>${t}</option>`).join("");
function renderSeason(){
  const tm = $("#teamSel").value, show = $("#showSel").value;
  let gs = D.games.filter(g=>(!tm || g.home===tm || g.away===tm) && (show==="all" || (show==="done")===done(g)));
  let html = `<thead><tr><th>Matchup</th><th>Vegas spread</th><th>Moneyline</th><th>Model spread</th><th>Model win %</th><th>Fair ML</th><th>Edge</th><th>O/U</th><th>Model total</th><th>Final (away–home)</th><th>ATS</th><th>Total</th></tr></thead><tbody>`;
  let lastWk = null;
  for (const g of gs){
    if (g.wk!==lastWk && !tm){ html += `<tr class="wkrow"><td colspan="12">Week ${g.wk}</td></tr>`; lastWk=g.wk; }
    const edge = (g.pred!=null&&g.spread!=null)?Math.abs(g.pred-g.spread):null;
    const mw = g.prob>=.5?g.home:g.away, mp=Math.round((g.prob>=.5?g.prob:1-g.prob)*100);
    let fin="–", ats="–", tres="–";
    if (done(g)){
      fin = `${g.as_}–${g.hs}`;
      if (g.spread!=null){ const m=g.hs-g.as_; if(m===g.spread) ats="Push"; else {
        const hc=m>g.spread, right=(g.pred>g.spread)===hc;
        ats = `${hc?g.home:g.away} <span class="${right?'res-w':'res-l'}">${right?'✓':'✗'}</span>`; } }
      if (g.total!=null && g.tpred!=null){ const t=g.hs+g.as_; if(t===g.total) tres="Push"; else {
        const ok=(g.tpred>g.total)===(t>g.total); tres = `${t>g.total?"Over":"Under"} <span class="${ok?'res-w':'res-l'}">${ok?'✓':'✗'}</span>`; } }
    }
    html += `<tr><td>${tm?`Wk ${g.wk}: `:""}${g.away} @ ${g.home}</td><td>${spreadTxt(g,g.spread)}</td><td>${g.aml!=null?`${mlTxt(g.aml)} / ${mlTxt(g.hml)}`:"–"}</td><td>${spreadTxt(g,g.pred)}</td><td>${mw} ${mp}%</td><td>${mlTxt(fairML(1-g.prob))} / ${mlTxt(fairML(g.prob))}</td><td class="${edge>=3?'hot':''}">${edge==null?"–":edge.toFixed(1)}</td><td>${g.total??"–"}</td><td>${g.tpred!=null?g.tpred.toFixed(1):"–"}</td><td>${fin}</td><td>${ats}</td><td>${tres}</td></tr>`;
  }
  $("#seasonTbl").innerHTML = html + "</tbody>";
  // team record line
  if (tm){
    let w=0,l=0,aw=0,al=0,ap=0;
    D.games.filter(g=>done(g)&&(g.home===tm||g.away===tm)).forEach(g=>{
      const home=g.home===tm, m=(g.hs-g.as_)*(home?1:-1); if(m>0)w++; else if(m<0)l++;
      if(g.spread!=null){ const s=g.spread*(home?1:-1); if(m>s)aw++; else if(m<s)al++; else ap++; }
    });
    $("#teamrec").innerHTML = `${tm}: <b>${w}-${l}</b> straight up, <b>${aw}-${al}${ap?"-"+ap:""}</b> against the spread. Moneylines shown away / home.`;
  } else $("#teamrec").textContent = "Moneylines shown away / home. Vegas lines appear once books post them, usually a week or two ahead.";
}
$("#teamSel").onchange = renderSeason; $("#showSel").onchange = renderSeason;
renderSeason();

// ---------- player props ----------
const PM = D.prop_meta || {}, PR = D.props || [];
const STATL = {passing_yards:"Pass yds",passing_tds:"Pass TDs",rushing_yards:"Rush yds",receptions:"Receptions",receiving_yards:"Rec yds",tds:"Anytime TD"};
const NOM = [0,.1,.25,.5,.75,.9,1];
function interp(x, xs, ys){ if (x<=xs[0]) return ys[0]; for(let i=1;i<xs.length;i++){ if(x<=xs[i]){ const t=(x-xs[i-1])/((xs[i]-xs[i-1])||1); return ys[i-1]+t*(ys[i]-ys[i-1]); } } return ys[ys.length-1]; }
function pois(k, mu){ let p=Math.exp(-mu), c=0; for(let i=0;i<=k;i++){ if(i>0) p*=mu/i; c+=p; } return c; } // P(X<=k)
function probOver(r, line){
  if (r.qs){
    const [a,b,c,d,e]=r.qs;
    const lo = r.st==="rushing_yards" ? a-2*(b-a) : Math.max(0, a-2*(b-a));
    const hi = e + 2.5*Math.max(e-d, 1);
    const nominal = interp(line, [lo,a,b,c,d,e,hi], NOM);
    const cal = (PM.calib && PM.calib[r.st]) || NOM;
    return 1 - interp(nominal, NOM, cal);
  }
  const k = Math.floor(line) + 1;             // need at least k
  const raw = 1 - pois(k-1, r.mu);
  const tbl = PM.calib && PM.calib[r.st] && PM.calib[r.st][k];
  if (!tbl) return raw;
  return interp(raw, tbl.map((_,i)=>i/20), tbl);
}
function fmtOdds(o){ return o==null||isNaN(o) ? "" : (o>0?"+":"")+o; }
function breakeven(o){ o=+o; if(!o) return null; return o<0 ? -o/(-o+100) : 100/(o+100); }
function store(){ try { return JSON.parse(localStorage.getItem("nflprops:"+D.season+":"+PM.week)||"{}"); } catch(e){ return {}; } }
function save(obj){ try { localStorage.setItem("nflprops:"+D.season+":"+PM.week, JSON.stringify(obj)); } catch(e){} }
let saved = store();
const keyOf = r => r.n+"|"+r.st;
function vals(r){
  const u = saved[keyOf(r)] || {};
  const line = r.st==="tds" ? 0.5 : (u.line!=null && u.line!=="" ? +u.line : (r.line!=null ? r.line : null));
  const over = u.over!=null && u.over!=="" ? +u.over : (r.over!=null ? r.over : (r.st==="tds" ? null : -110));
  const under = u.under!=null && u.under!=="" ? +u.under : (r.under!=null ? r.under : -110);
  return {line, over, under, fromBook: r.line!=null && u.line==null};
}
function evalRow(r){
  const v = vals(r);
  if (v.line==null || (r.st==="tds" && v.over==null)) return {v, p:null};
  const p = probOver(r, v.line);
  const beO = breakeven(v.over), beU = r.st==="tds" ? null : breakeven(v.under);
  const eO = beO!=null ? p-beO : -1, eU = beU!=null ? (1-p)-beU : -1;
  const side = eO>=eU ? "over" : "under";
  return {v, p, side, edge: Math.max(eO,eU), sideP: side==="over"?p:1-p};
}
function projCell(r){
  if (r.st==="tds") { const p=probOver(r,.5); return `<span class="proj">${Math.round(p*100)}%</span><span class="range">chance to score</span>`; }
  if (r.st==="passing_tds") return `<span class="proj">${r.mu.toFixed(1)}</span><span class="range">avg TDs</span>`;
  const d = r.st==="receptions"?1:0;
  return `<span class="proj">${r.qs[2].toFixed(d)}</span><span class="range">${r.qs[1].toFixed(d)}–${r.qs[3].toFixed(d)} likely</span>`;
}
function applyBest(r){
  // Pick the sportsbook and side with the biggest edge for the model
  if (!r.books || !r.books.length) return;
  let best = null;
  for (const b of r.books){
    if (b.line==null) continue;
    const line = r.st==="tds" ? .5 : b.line, p = probOver(r, line);
    const opts = [["over", b.over, p]]; if (r.st!=="tds") opts.push(["under", b.under, 1-p]);
    for (const [side, odds, prob] of opts){ const be = breakeven(odds); if (be==null) continue;
      const e = prob - be; if (!best || e > best.e) best = {b, e}; }
  }
  if (!best) return;
  r.line = best.b.line; r.over = best.b.over; r.under = best.b.under ?? null; r.book = best.b.book; r.nbooks = r.books.length;
}
PR.forEach(applyBest); (D.past||[]).forEach(applyBest);
function initProps(){
  if (!PR.length){ $("#propsIntro").textContent = "Projections appear once next week's games are on the schedule."; return; }
  const gm = {}; D.games.forEach(g=>gm[g.id]=g);
  $("#propsIntro").innerHTML = `Week ${PM.week} projections for every starting QB and active RB, WR and TE. ` +
    (PM.has_book ? "Lines outlined in green came from the sportsbook; you can change any of them." :
     "Type in the line and odds from your sportsbook, and the model shows its chance of going over or under and whether that beats the odds. Your entries are saved in this browser.");
  const gids = [...new Set(PR.map(r=>r.gid))].sort((a,b)=>PR.find(r=>r.gid===a).go-PR.find(r=>r.gid===b).go);
  $("#pGame").innerHTML += gids.map(id=>`<option value="${id}">${gm[id]?gm[id].away+" @ "+gm[id].home:id}</option>`).join("");
  $("#pStat").innerHTML += Object.entries(STATL).map(([k,l])=>`<option value="${k}">${l}</option>`).join("");
  ["#pGame","#pStat","#pPos","#pEdge"].forEach(id=>$(id).onchange=renderProps);
  const rep = PM.report || {};
  const rows = Object.values(rep).filter(x=>x.mae!=null).map(x=>`<tr><td>${x.label}</td><td>${x.mae}</td><td>${x.base}</td></tr>`).join("");
  $("#propNotes").innerHTML = `<p><b>How it works.</b> For each player the model looks at his recent usage and production, what the opponent's defense has allowed to his position, and the Vegas spread and total (which predict game script and how much his team will score). It predicts a full range, not just one number, so it can estimate the chance of beating any line. Probabilities are adjusted using how the model actually did on ${D.season-1} games it never saw.</p>
  <p><b>Tested on ${D.season-1}:</b> average miss of the middle projection, compared with just using the player's recent average.</p>
  <table><thead><tr><th>Stat</th><th>Model</th><th>Recent avg</th></tr></thead><tbody>${rows}</tbody></table>
  <p class="note">"Edge" is the model's chance minus the chance the odds require to break even. Sportsbooks are good at setting props, and a real edge of a few percent is rare. Before betting, check the latest injury news. Players ruled Out are removed, but game-time decisions (tagged Q) and changes in role are not.</p>`;
  renderProps();
}
let pSel = null;
function renderProps(){
  renderCard();
  const g=$("#pGame").value, st=$("#pStat").value, pos=$("#pPos").value, onlyEdge=$("#pEdge").checked;
  let list = PR.filter(r=>(!pSel||r.n===pSel)&&(pSel||((!g||r.gid===g)&&(!pos||r.pos===pos)))&&(!st||r.st===st)).map(r=>({r, e:evalRow(r)}));
  if (onlyEdge) list = list.filter(x=>x.e.p!=null && x.e.edge>=.05);
  const order = Object.keys(STATL);
  list.sort((a,b)=>{
    const ea=a.e.p!=null, eb=b.e.p!=null;
    if (ea!==eb) return ea?-1:1;
    if (ea) return b.e.edge-a.e.edge;
    return a.r.go-b.r.go || a.r.tm.localeCompare(b.r.tm) || order.indexOf(a.r.st)-order.indexOf(b.r.st) || (b.r.qs?b.r.qs[2]:b.r.mu)-(a.r.qs?a.r.qs[2]:a.r.mu);
  });
  const shown = list.slice(0, 400);
  let html = `<thead><tr><th>Player</th><th>Prop</th><th>Model</th><th>Line</th><th>Over odds</th><th>Under odds</th><th>Model over %</th><th>Model pick</th></tr></thead><tbody>`;
  html += shown.map(({r,e})=>{
    const i = PR.indexOf(r), v=e.v, td=r.st==="tds";
    const bookNote = r.book && v.fromBook ? `<small class="booknote">${r.book}${r.nbooks>1?`, best of ${r.nbooks}`:""}</small>` : "";
    const lineCell = td ? `<td>Yes${r.book?`<small class="booknote">${r.book}${r.nbooks>1?`, best of ${r.nbooks}`:""}</small>`:""}</td>` : `<td><input inputmode="decimal" data-i="${i}" data-f="line" class="${v.fromBook?'fromBook':''}" value="${v.line??""}" placeholder="line" aria-label="${r.n} ${STATL[r.st]} line">${bookNote}</td>`;
    const ov = `<td><input class="odds" inputmode="numeric" data-i="${i}" data-f="over" value="${fmtOdds(v.over)}" placeholder="${td?'+odds':''}" aria-label="${r.n} over odds"></td>`;
    const un = td ? `<td>–</td>` : `<td><input class="odds" inputmode="numeric" data-i="${i}" data-f="under" value="${fmtOdds(v.under)}" aria-label="${r.n} under odds"></td>`;
    let pc="<td>–</td>", pk=`<td class="pick"><small>${td?"Enter the Yes odds":"Enter a line"}</small></td>`;
    if (e.p!=null){
      pc = `<td>${Math.round(e.p*100)}%<div class="bar2"><i style="width:${Math.round(e.p*100)}%"></i><b style="left:${Math.round((breakeven(v.over)||.5)*100)}%"></b></div></td>`;
      const lbl = td ? (e.side==="over"?"Yes":"No value") : (e.side==="over"?"Over":"Under");
      pk = e.edge>0 ? `<td class="pick ${e.edge>=.05?'good':''}">${lbl} ${!td||e.side==="over"?Math.round(e.sideP*100)+"%":""}<small>${(e.edge*100).toFixed(1)}% edge</small></td>`
                    : `<td class="pick"><small>No edge</small></td>`;
    }
    return `<tr><td class="pl">${teamBadge(r.tm)}${plink(r.n)}${r.q?'<span class="tag" title="Questionable on injury report">Q</span>':''}<small>${r.pos}, vs ${r.opp}</small>${r.boost?`<small class="boost">More work with ${r.boost.join(", ")} out</small>`:""}</td><td>${STATL[r.st]}</td><td>${projCell(r)}</td>${lineCell}${ov}${un}${pc}${pk}</tr>`;
  }).join("");
  $("#propsTbl").innerHTML = html + "</tbody>";
  $("#propsTbl").querySelectorAll("input").forEach(inp=>inp.addEventListener("change",()=>{
    const r = PR[+inp.dataset.i], k=keyOf(r);
    saved[k] = saved[k] || {}; saved[k][inp.dataset.f] = inp.value.trim()===""?null:inp.value.trim();
    save(saved); renderProps();
  }));
}
initProps();

// ---------- search ----------
const NAMES = {ARI:"Arizona Cardinals",ATL:"Atlanta Falcons",BAL:"Baltimore Ravens",BUF:"Buffalo Bills",CAR:"Carolina Panthers",CHI:"Chicago Bears",CIN:"Cincinnati Bengals",CLE:"Cleveland Browns",DAL:"Dallas Cowboys",DEN:"Denver Broncos",DET:"Detroit Lions",GB:"Green Bay Packers",HOU:"Houston Texans",IND:"Indianapolis Colts",JAX:"Jacksonville Jaguars",KC:"Kansas City Chiefs",LA:"Los Angeles Rams",LAC:"Los Angeles Chargers",LV:"Las Vegas Raiders",MIA:"Miami Dolphins",MIN:"Minnesota Vikings",NE:"New England Patriots",NO:"New Orleans Saints",NYG:"New York Giants",NYJ:"New York Jets",PHI:"Philadelphia Eagles",PIT:"Pittsburgh Steelers",SEA:"Seattle Seahawks",SF:"San Francisco 49ers",TB:"Tampa Bay Buccaneers",TEN:"Tennessee Titans",WAS:"Washington Commanders"};
const people = new Map();
D.players.forEach(p=>people.set(p.n,{type:"p",n:p.n,pos:p.pos,tm:p.tm}));
PR.forEach(r=>{ if(!people.has(r.n)) people.set(r.n,{type:"p",n:r.n,pos:r.pos,tm:r.tm}); });
(D.fantasy||[]).forEach(r=>{ if(!people.has(r.n)) people.set(r.n,{type:"p",n:r.n,pos:r.pos,tm:r.tm}); });
const INDEX = [...people.values(), ...Object.keys(D.teams).map(t=>({type:"t",n:NAMES[t]||t,tm:t}))];
const norm = x => x.toLowerCase().normalize("NFD").replace(/[^a-z0-9 ]/g,"");
let hits = [], hi = -1;
function searchFor(q){
  const w = norm(q).split(" ").filter(Boolean); if(!w.length) return [];
  const scored = [];
  for (const it of INDEX){
    const hay = norm(it.n+" "+it.tm+(it.pos?" "+it.pos:""));
    if (!w.every(x=>hay.includes(x))) continue;
    let sc = 0; const nm = norm(it.n);
    if (nm.startsWith(w[0]) || nm.split(" ").some(p=>p.startsWith(w[0]))) sc -= 2;
    if (it.type==="t" && norm(it.tm)===w.join(" ")) sc -= 5;
    scored.push([sc, it]);
  }
  return scored.sort((a,b)=>a[0]-b[0] || a[1].n.localeCompare(b[1].n)).slice(0,10).map(x=>x[1]);
}
function drawHits(){
  const ul = $("#qres");
  if (!$("#q").value.trim()){ ul.hidden = true; $("#q").setAttribute("aria-expanded","false"); return; }
  ul.innerHTML = hits.length ? hits.map((h,i)=>`<li role="option" id="opt${i}" aria-selected="${i===hi}" data-i="${i}"><span>${h.type==="t"?teamBadge(h.tm):teamBadge(h.tm)}${h.n}</span><small>${h.type==="t"?"Team page":h.pos}</small></li>`).join("")
                             : `<li class="none">No players or teams match</li>`;
  ul.hidden = false; $("#q").setAttribute("aria-expanded","true");
  $("#q").setAttribute("aria-activedescendant", hi>=0 ? "opt"+hi : "");
  ul.querySelectorAll("li[data-i]").forEach(li=>li.onmousedown=e=>{ e.preventDefault(); choose(hits[+li.dataset.i]); });
}
function choose(h){
  $("#qres").hidden = true; $("#q").setAttribute("aria-expanded","false"); $("#q").value = h.n;
  if (h.type==="t"){
    openTeam(h.tm);
  } else {
    openPlayer(h.n);
  }
  window.scrollTo({top: $("nav.views").getBoundingClientRect().top + window.scrollY - 12});
}
$("#q").addEventListener("input", ()=>{ hits = searchFor($("#q").value); hi = hits.length?0:-1; drawHits(); });
$("#q").addEventListener("keydown", e=>{
  if (e.key==="ArrowDown"){ e.preventDefault(); if(hits.length){ hi=(hi+1)%hits.length; drawHits(); } }
  else if (e.key==="ArrowUp"){ e.preventDefault(); if(hits.length){ hi=(hi-1+hits.length)%hits.length; drawHits(); } }
  else if (e.key==="Enter"){ if(hi>=0 && hits[hi]){ e.preventDefault(); choose(hits[hi]); } }
  else if (e.key==="Escape"){ $("#qres").hidden = true; $("#q").setAttribute("aria-expanded","false"); }
});
$("#q").addEventListener("blur", ()=>setTimeout(()=>{ $("#qres").hidden = true; $("#q").setAttribute("aria-expanded","false"); }, 100));
$("#q").addEventListener("focus", ()=>{ if($("#q").value.trim()){ hits=searchFor($("#q").value); drawHits(); } });

function renderCard(){
  const box = $("#playerCard");
  if (!pSel){ box.innerHTML = ""; ["#pGame","#pPos"].forEach(id=>$(id).disabled=false); return; }
  ["#pGame","#pPos"].forEach(id=>$(id).disabled=true);
  const p = D.players.find(x=>x.n===pSel), info = people.get(pSel) || {};
  let stats = "";
  if (p){
    const cols = COLS[p.pos];
    stats = `<div class="scroll"><table><tr><th>${D.season} so far</th>${cols.map(c=>`<th>${c[0]}</th>`).join("")}</tr>
      <tr><td></td>${cols.map(c=>{ const v=p[c[1]]; let cls=""; if(v!=null&&c[3]!=null&&v>=c[3])cls="hot"; else if(v!=null&&c[4]!=null&&v<=c[4])cls="cold"; return `<td class="${cls}">${fmt(v,c[2])}</td>`; }).join("")}</tr></table></div>`;
  } else stats = `<p class="empty">Not enough ${D.season} snaps yet for season stats.</p>`;
  const has = PR.some(r=>r.n===pSel);
  const g = has ? PR.find(r=>r.n===pSel) : null;
  box.innerHTML = `<div class="pcard"><div class="pcard-top"><div><h3>${teamBadge(info.tm)}${pSel}</h3><p class="who">${info.pos||""}, ${NAMES[info.tm]||info.tm||""}${g?`. Week ${PM.week} vs ${g.opp}`:""}</p></div>
    <button class="clear" type="button">Show all players</button></div>${stats}
    ${has?"":`<p class="empty">No projection this week. He may be ruled out, on bye, or not in a big enough role.</p>`}</div>`;
  box.querySelector(".clear").onclick = ()=>{ pSel=null; $("#q").value=""; renderProps(); };
}

// ---------- shared helpers ----------
const ord = n => n + (n%100>=11&&n%100<=13 ? "th" : ({1:"st",2:"nd",3:"rd"}[n%10]||"th"));
const FS = D.fantasy || [];
const propsBy = {}; PR.forEach(r => { (propsBy[r.n] = propsBy[r.n] || {})[r.st] = r; });
const fantBy = {}; FS.forEach(f => fantBy[f.n] = f);
function plink(n){ return `<a class="plink" data-p="${n.replace(/"/g,"&quot;")}">${n}</a>`; }
function wireLinks(root){ root.querySelectorAll("a.plink").forEach(a => a.onclick = () => openPlayer(a.dataset.p)); }
function matchupChip(pos, r){
  if (r==null) return "";
  const n = new Set(FS.filter(f=>f.pos===pos).map(f=>f.opp)).size || 32;
  const easy = r <= Math.ceil(n/4), hard = r > n - Math.ceil(n/4);
  const label = r <= n/2 ? `${ord(r)} easiest matchup` : `${ord(n-r+1)} toughest matchup`;
  return `<span class="chip ${easy?'easy':hard?'hard':''}">${label}</span>`;
}
function keyLine(n){
  const p = propsBy[n] || {}, out = [];
  if (p.passing_yards) out.push(`${Math.round(p.passing_yards.qs[2])} pass yds`);
  if (p.passing_tds) out.push(`${p.passing_tds.mu.toFixed(1)} pass TD`);
  if (p.rushing_yards) out.push(`${Math.round(p.rushing_yards.qs[2])} rush yds`);
  if (p.receptions) out.push(`${p.receptions.qs[2].toFixed(1)} rec`);
  if (p.receiving_yards) out.push(`${Math.round(p.receiving_yards.qs[2])} rec yds`);
  if (p.tds) out.push(`${Math.round(probOver(p.tds,.5)*100)}% TD`);
  return out.join(", ");
}
const fk = () => +$("#fScore").value;
const fpts = f => f.std + fk()*f.rec;
const frange = f => { const s = f.ppr>0 ? fpts(f)/f.ppr : 1; return [f.lo*s, f.hi*s]; };

// ---------- fantasy tab ----------
function renderFantasy(){
  if (!FS.length){ $("#fantTbl").innerHTML = `<tbody><tr><td>Rankings appear once next week's games are on the schedule.</td></tr></tbody>`; return; }
  const pos = $("#fPos").value, lbl = {"1":"PPR","0.5":"Half PPR","0":"Standard"}[$("#fScore").value];
  $("#fTitle").textContent = `Week ${PM.week} rankings, ${lbl}`;
  let list = FS.filter(f => !pos || (pos==="FLEX" ? f.pos!=="QB" : f.pos===pos)).sort((a,b)=>fpts(b)-fpts(a)).slice(0, pos ? 80 : 150);
  const maxHi = Math.max(...list.map(f=>frange(f)[1]), 1);
  $("#fantTbl").innerHTML = `<thead><tr><th>#</th><th>Player</th><th>Opponent</th><th>Projected</th><th>Likely range</th><th>Projected stat line</th></tr></thead><tbody>` +
    list.map((f,i)=>{ const [lo,hi] = frange(f), p = fpts(f);
      return `<tr><td>${i+1}</td><td class="pl">${teamBadge(f.tm)}${plink(f.n)}${f.q?'<span class="tag">Q</span>':''}<small>${f.pos}</small></td>
        <td style="text-align:left">vs ${f.opp}<br>${matchupChip(f.pos,f.mrank)}</td>
        <td><span class="proj">${p.toFixed(1)}</span></td>
        <td style="white-space:nowrap">${lo.toFixed(0)}–${hi.toFixed(0)}<span class="rng"><i style="left:${lo/maxHi*100}%;width:${(hi-lo)/maxHi*100}%"></i><b style="left:${p/maxHi*100}%"></b></span></td>
        <td class="keyst">${keyLine(f.n)}</td></tr>`; }).join("") + "</tbody>";
  wireLinks($("#fantTbl"));
  renderStartSit();
}
function lastPPR(n, k){ const lg = D.logs[n] || []; return lg.slice(-k).map(x=>x[16]); }
function renderStartSit(){
  const picks = [...document.querySelectorAll(".ssIn")].map(i=>i.value.trim()).filter(Boolean);
  const found = picks.map(n=>FS.find(f=>f.n.toLowerCase()===n.toLowerCase())).filter(Boolean);
  const missing = picks.filter(n=>!FS.find(f=>f.n.toLowerCase()===n.toLowerCase()));
  if (!found.length){ $("#ssOut").innerHTML = missing.length ? `<p class="empty">No projection this week for ${missing.join(", ")}. They may be on bye or ruled out.</p>` : ""; return; }
  const sorted = [...found].sort((a,b)=>fpts(b)-fpts(a));
  const best = sorted[0], gap = sorted.length>1 ? fpts(best)-fpts(sorted[1]) : 99;
  const maxHi = Math.max(...found.map(f=>frange(f)[1]), 1);
  const verdict = found.length<2 ? "" : gap < 1.0 ? `Close call. The projections are within a point, so go with the better matchup or your gut. Slight lean: ${best.n}.` : `Start ${best.n}. Projected ${gap.toFixed(1)} points ahead.`;
  $("#ssOut").innerHTML = (verdict?`<p class="verdict">${verdict}</p>`:"") + `<div class="ss-grid">` + found.map(f=>{
    const [lo,hi] = frange(f), p = fpts(f), rec = lastPPR(f.n,3);
    return `<div class="ss-card ${f===best&&found.length>1?'best':''}"><h3>${teamBadge(f.tm)}${plink(f.n)}</h3><p class="who" style="margin:2px 0 0;color:var(--muted);font-size:14px">${f.pos}, vs ${f.opp}${f.q?' <span class="tag">Q</span>':''}</p>
      <div class="big">${p.toFixed(1)}</div><span class="rng" style="margin:0;width:100%"><i style="left:${lo/maxHi*100}%;width:${(hi-lo)/maxHi*100}%"></i><b style="left:${p/maxHi*100}%"></b></span>
      <dl><dt>Likely range</dt><dd>${lo.toFixed(0)}–${hi.toFixed(0)} pts</dd><dt>Matchup</dt><dd>${matchupChip(f.pos,f.mrank)||"–"}</dd>
      <dt>Last 3 games (PPR)</dt><dd>${rec.length?rec.map(v=>v==null?"–":v.toFixed(1)).join(", "):"–"}</dd></dl>
      <p class="keyst" style="margin:10px 0 0">${keyLine(f.n)}</p></div>`; }).join("") + `</div>` +
    (missing.length?`<p class="empty">No projection this week for ${missing.join(", ")}.</p>`:"");
  wireLinks($("#ssOut"));
}
function renderRisers(){
  const R = D.risers || [];
  if (!R.length){ $("#riseTbl").innerHTML = `<tbody><tr><td>No big usage jumps yet. Check back after a couple of games.</td></tr></tbody>`; return; }
  const arrow = (a,b,d=1) => `${a==null?"–":(+a).toFixed(d)} → <b>${b==null?"–":(+b).toFixed(d)}</b>`;
  $("#riseTbl").innerHTML = `<thead><tr><th>Player</th><th>Touches + targets per game</th><th>Snap %</th><th>Target share %</th><th>PPR last 2</th></tr></thead><tbody>` +
    R.map(r=>`<tr><td class="pl" style="text-align:left">${teamBadge(r.tm)}${plink(r.n)}<small style="display:block;color:var(--muted);font-size:12.5px">${r.pos}</small></td>
      <td>${arrow(r.opp_b,r.opp_l)}</td><td>${arrow(r.snap_b,r.snap_l,0)}</td><td>${arrow(r.ts_b,r.ts_l)}</td><td>${r.ppr_l==null?"–":r.ppr_l.toFixed(1)}</td></tr>`).join("") + "</tbody>";
  wireLinks($("#riseTbl"));
}
$("#fantNames").innerHTML = FS.map(f=>`<option value="${f.n.replace(/"/g,"&quot;")}">${f.pos}, ${f.tm}</option>`).join("");
document.querySelectorAll(".ssIn").forEach(i=>i.addEventListener("change", renderStartSit));
$("#fScore").onchange = renderFantasy; $("#fPos").onchange = renderFantasy;
function showSub(k){
  document.querySelectorAll(".subtabs button").forEach(b=>b.setAttribute("aria-selected", b.dataset.sub===k));
  ["rank","ss","rise"].forEach(x=>$("#fsub-"+x).hidden = x!==k);
  $("#fPos").closest("label").hidden = k!=="rank";
  $("#fToolbar").hidden = k==="rise";
}
document.querySelectorAll(".subtabs button").forEach(b=>b.onclick=()=>showSub(b.dataset.sub));
renderFantasy(); renderRisers();

// ---------- results tab ----------
function renderResults(){
  const r = SEASON_REC;
  const tile = (v, lbl) => `<div class="tile"><span class="cond">${v}</span><small>${lbl}</small></div>`;
  $("#gTiles").innerHTML = tile(wl(r.su), `Winners, ${pct(r.su)}`) + tile(wl(r.ats), `Vs the spread, ${pct(r.ats)}`) +
    tile(wl(r.ats3), `Spread, 3+ pt edges, ${pct(r.ats3)}`) + tile(wl(r.ou), `Totals, ${pct(r.ou)}`) + tile(wl(r.ou3), `Totals, 3+ pt edges, ${pct(r.ou3)}`);
  const wks = [...new Set(D.games.filter(done).map(g=>g.wk))];
  $("#gWeeks").innerHTML = wks.length ? `<thead><tr><th>Week</th><th>Winners</th><th>Spread</th><th>Spread, 3+ edges</th><th>Totals</th><th>Totals, 3+ edges</th></tr></thead><tbody>` +
    wks.map(w=>{ const x = gradeGames(D.games.filter(g=>g.wk===w)); return `<tr><td>Week ${w}</td><td>${wl(x.su)}</td><td>${wl(x.ats)}</td><td>${wl(x.ats3)}</td><td>${wl(x.ou)}</td><td>${wl(x.ou3)}</td></tr>`; }).join("") + "</tbody>"
    : `<tbody><tr><td>No games finished yet this season.</td></tr></tbody>`;
  $("#gTest").textContent = `For comparison, on ${GR.seasons} games the model never saw: ${GR.acc}% of winners, ${GR.ats3}% against the spread on 3+ point edges (${GR.ats3_n} games), and ${GR.ou}% on totals. Its average miss was ${(+GR.mae).toFixed(1)} points on the margin (Vegas: ${(+GR.vegas_mae).toFixed(1)}) and ${(+GR.tmae).toFixed(1)} on the total (Vegas: ${(+GR.vegas_tmae).toFixed(1)}). You need 52.4% against the spread or on totals to break even at standard odds.`;
  renderPropResults(); renderProjAcc();
}
function pickFor(r, line, over, under){
  const td = r.st==="tds";
  if (line==null || isNaN(line) || (td && (over==null||isNaN(over)))) return null;
  const p = probOver(r, line), beO = breakeven(over), beU = td ? null : breakeven(under);
  const eO = beO!=null ? p-beO : -1, eU = beU!=null ? (1-p)-beU : -1;
  const side = eO>=eU ? "over" : "under", edge = Math.max(eO,eU);
  if (edge <= 0) return null;
  const odds = side==="over" ? over : under;
  let res;
  if (r.act === line) res = "P";
  else res = ((r.act > line) === (side==="over")) ? "W" : "L";
  const units = res==="W" ? (odds>0 ? odds/100 : 100/-odds) : res==="L" ? -1 : 0;
  return {side, edge, res, units, line, odds, p: side==="over"?p:1-p};
}
function allSaved(){
  const out = {};
  try { for (let i=0;i<localStorage.length;i++){ const k = localStorage.key(i), m = k && k.match(/^nflprops:(\d+):(\d+)$/);
    if (m && +m[1]===D.season) out[m[2]] = JSON.parse(localStorage.getItem(k)||"{}"); } } catch(e){}
  return out;
}
function renderPropResults(){
  const src = $("#rSrc").value, thr = +$("#rEdge").value, saved = allSaved();
  const picks = [];
  for (const r of D.past || []){
    let line, over, under;
    if (src==="book"){ if (r.line==null) continue; line=r.line; over=r.over; under=r.under??-110; if (r.st==="tds") line=.5; }
    else {
      const u = (saved[String(r.wk)]||{})[r.n+"|"+r.st]; if (!u) continue;
      line = r.st==="tds" ? .5 : (u.line!=null ? +u.line : r.line); over = u.over!=null ? +u.over : (r.over ?? (r.st==="tds"?null:-110));
      under = u.under!=null ? +u.under : (r.under ?? -110);
    }
    const pk = pickFor(r, line, over, under);
    if (pk && pk.edge >= thr) picks.push({r, pk});
  }
  const tally = arr => { const w=arr.filter(x=>x.pk.res==="W").length, l=arr.filter(x=>x.pk.res==="L").length, p=arr.filter(x=>x.pk.res==="P").length;
    return {w,l,p,units:arr.reduce((s,x)=>s+x.pk.units,0), risked:w+l}; };
  const t = tally(picks);
  const tile = (v, lbl) => `<div class="tile"><span class="cond">${v}</span><small>${lbl}</small></div>`;
  if (!picks.length){
    $("#pTiles").innerHTML = ""; $("#pByStat").innerHTML = ""; 
    $("#pList").innerHTML = `<tbody><tr><td>${src==="book" ? "No graded picks yet. Once the site pulls sportsbook lines (with an Odds API key), every pick with an edge gets graded here after the games." : "No graded picks yet. Lines you type in on the Betting tab are graded here after the games, as long as you use the same browser."}</td></tr></tbody>`;
    return;
  }
  $("#pTiles").innerHTML = tile(`${t.w}-${t.l}${t.p?"-"+t.p:""}`, `Record, ${pct([t.w,t.l])}`) +
    tile(`${t.units>=0?"+":""}${t.units.toFixed(1)}u`, `Profit, betting 1 unit per pick`) +
    tile(t.risked?`${(t.units/t.risked*100).toFixed(1)}%`:"–", "Return on money risked");
  const by = {}; picks.forEach(x=>(by[x.r.st]=by[x.r.st]||[]).push(x));
  $("#pByStat").innerHTML = `<thead><tr><th>Prop</th><th>Record</th><th>Win %</th><th>Units</th></tr></thead><tbody>` +
    Object.entries(by).map(([st,a])=>{ const x=tally(a); return `<tr><td>${STATL[st]}</td><td>${x.w}-${x.l}${x.p?"-"+x.p:""}</td><td>${pct([x.w,x.l])}</td><td class="${x.units>=0?'hot':'cold'}">${x.units>=0?"+":""}${x.units.toFixed(1)}</td></tr>`; }).join("") + "</tbody>";
  const recent = picks.sort((a,b)=>b.r.wk-a.r.wk || b.pk.edge-a.pk.edge).slice(0,40);
  $("#pList").innerHTML = `<thead><tr><th>Week</th><th>Player</th><th>Prop</th><th>Pick</th><th>Odds</th><th>Edge</th><th>Result</th></tr></thead><tbody>` +
    recent.map(({r,pk})=>`<tr><td>${r.wk}</td><td style="text-align:left">${teamBadge(r.tm)}${plink(r.n)}</td><td>${STATL[r.st]}</td>
      <td>${r.st==="tds"?"Yes":(pk.side==="over"?"Over ":"Under ")+pk.line}</td><td>${fmtOdds(pk.odds)}</td><td>${(pk.edge*100).toFixed(1)}%</td>
      <td><span class="${pk.res==="W"?'res-w':pk.res==="L"?'res-l':''}">${pk.res==="W"?"Won":pk.res==="L"?"Lost":"Push"}</span> (${r.act})</td></tr>`).join("") + "</tbody>";
  wireLinks($("#pList"));
}
function renderProjAcc(){
  const P = D.past || [];
  const bf = P.some(r=>r.bf);
  const FSR = D.reports.fantasy_season, FR = D.reports.fantasy;
  $("#projLede").textContent = `How this season's projections compare with what actually happened. A well-calibrated model lands over its middle projection about half the time and inside its likely range about half the time.` +
    (bf ? " Weeks before the site started saving its projections are filled in by re-running the model as of before those games." : "") +
    (FSR && FSR.n ? ` Fantasy points: average miss of ${FSR.mae} PPR points this season (recent averages: ${FSR.base}).` : "");
  if (!P.length){ $("#projAcc").innerHTML = `<tbody><tr><td>No finished games to grade yet.</td></tr></tbody>`; return; }
  const rows = Object.keys(STATL).map(st=>{
    const a = P.filter(r=>r.st===st); if (!a.length) return "";
    if (st==="tds" || st==="passing_tds"){
      const k = st==="tds" ? 1 : 2;
      const pred = a.reduce((s,r)=>s+probOver(r,k-.5),0)/a.length, act = a.filter(r=>r.act>=k).length/a.length;
      return `<tr><td>${STATL[st]}</td><td>${a.length}</td><td colspan="3" style="text-align:left">Predicted ${st==="tds"?"scoring":"2+ TD"} rate ${(pred*100).toFixed(0)}%, actual ${(act*100).toFixed(0)}%</td></tr>`;
    }
    const mae = a.reduce((s,r)=>s+Math.abs(r.qs[2]-r.act),0)/a.length;
    const over = a.filter(r=>r.act>r.qs[2]).length/a.length, inr = a.filter(r=>r.act>=r.qs[1]&&r.act<=r.qs[3]).length/a.length;
    return `<tr><td>${STATL[st]}</td><td>${a.length}</td><td>${mae.toFixed(1)}</td><td>${(over*100).toFixed(0)}%</td><td>${(inr*100).toFixed(0)}%</td></tr>`;
  }).join("");
  $("#projAcc").innerHTML = `<thead><tr><th>Stat</th><th>Graded</th><th>Avg miss</th><th>Went over projection</th><th>Inside likely range</th></tr></thead><tbody>${rows}</tbody>`;
}
$("#rSrc").onchange = renderPropResults; $("#rEdge").onchange = renderPropResults;
renderResults();

// ---------- player pages ----------
const TREND = {
  QB: [["Fantasy (PPR)",16],["Pass yds",6],["Rush yds",10],["Snap %",3]],
  RB: [["Fantasy (PPR)",16],["Carries",9],["Targets",12],["Snap %",3]],
  WR: [["Fantasy (PPR)",16],["Targets",12],["Rec yds",14],["Snap %",3]],
};
TREND.TE = TREND.WR;
let trendIdx = 0;
function openPlayer(n){
  trendIdx = 0; renderPlayer(n); showView("player");
  window.scrollTo({top: $("nav.views").getBoundingClientRect().top + window.scrollY - 12});
}
function trendSVG(lg, idx){
  const pts = lg.slice(-12), W = 760, H = 220, pad = 28, bw = (W - pad*2) / Math.max(pts.length,1);
  const vals = pts.map(x=>x[idx]==null?0:+x[idx]), max = Math.max(...vals, 1);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Trend over the last ${pts.length} games">`;
  s += `<line x1="${pad}" x2="${W-pad}" y1="${H-40}" y2="${H-40}" stroke="var(--line)"/>`;
  pts.forEach((x,i)=>{
    const v = vals[i], h = (H-70) * v / max, X = pad + i*bw + bw*.15, Y = H-40-h;
    const cur = x[0]===D.season;
    s += `<rect x="${X}" y="${Y}" width="${bw*.7}" height="${Math.max(h,1)}" rx="3" fill="${cur?'var(--turf)':'var(--line)'}"/>`;
    s += `<text x="${X+bw*.35}" y="${Y-6}" text-anchor="middle" font-size="13" fill="var(--ink)" font-family="Big Shoulders Display, Arial Narrow, sans-serif" font-weight="600">${x[idx]==null?"–":(+x[idx]).toFixed(idx===16?1:0)}</text>`;
    s += `<text x="${X+bw*.35}" y="${H-22}" text-anchor="middle" font-size="12" fill="var(--muted)">W${x[1]}</text>`;
    s += `<text x="${X+bw*.35}" y="${H-7}" text-anchor="middle" font-size="11" fill="var(--muted)">${x[2]}</text>`;
    if (i>0 && pts[i-1][0]!==x[0]) s += `<line x1="${X-bw*.15}" x2="${X-bw*.15}" y1="10" y2="${H-40}" stroke="var(--muted)" stroke-dasharray="3 3"/><text x="${X-bw*.15+4}" y="20" font-size="12" fill="var(--muted)">${x[0]} season</text>`;
  });
  return s + "</svg>";
}
function renderPlayer(n){
  const info = people.get(n) || {}, pos = info.pos || (D.players.find(p=>p.n===n)||{}).pos || "WR";
  const f = fantBy[n], pr = PR.filter(r=>r.n===n), st = D.players.find(p=>p.n===n), lg = D.logs[n] || [];
  const tm = info.tm || (f&&f.tm) || "";
  const g = pr[0] || f;
  let html = `<div class="pv-head"><div><h2>${tm?teamChip(tm):""}${n}</h2><p class="who">${pos}, ${NAMES[tm]||tm}</p></div>`;
  if (f) html += `<div class="pv-week"><div><span class="cond">${f.ppr.toFixed(1)}</span><small>Week ${PM.week} PPR points vs ${f.opp}</small></div><div><span class="cond">${f.lo.toFixed(0)}–${f.hi.toFixed(0)}</span><small>Likely range</small></div></div>`;
  html += `</div>`;
  if (f) html += `<p style="margin:8px 0 0">${matchupChip(f.pos,f.mrank)}${f.q?' <span class="tag">Questionable</span>':''}</p>`;
  if (pr.length){
    html += `<div class="pv-proj">` + pr.map(r=>{
      if (r.st==="tds") return `<div class="tile"><span class="cond">${Math.round(probOver(r,.5)*100)}%</span><small>Chance to score a TD</small></div>`;
      if (r.st==="passing_tds") return `<div class="tile"><span class="cond">${r.mu.toFixed(1)}</span><small>Pass TDs (average)</small></div>`;
      const d = r.st==="receptions"?1:0;
      return `<div class="tile"><span class="cond">${r.qs[2].toFixed(d)}</span><small>${STATL[r.st]}, likely ${r.qs[1].toFixed(d)}–${r.qs[3].toFixed(d)}</small></div>`;
    }).join("") + `</div><div class="pv-actions"><button class="btn" type="button" id="pvProps">Enter prop lines for ${n}</button></div>`;
  } else html += `<p class="empty">No projection this week. He may be on bye, ruled out, or not in a big enough role.</p>`;
  if (st){
    const cols = COLS[st.pos];
    html += `<h2 class="sub">${D.season} advanced stats</h2><div class="scroll"><table><tr>${cols.map(c=>`<th>${c[0]}</th>`).join("")}</tr><tr>${cols.map(c=>{ const v=st[c[1]]; let cls=""; if(v!=null&&c[3]!=null&&v>=c[3])cls="hot"; else if(v!=null&&c[4]!=null&&v<=c[4])cls="cold"; return `<td class="${cls}">${fmt(v,c[2])}</td>`; }).join("")}</tr></table></div>`;
  }
  if (lg.length){
    const T = TREND[pos] || TREND.WR;
    html += `<h2 class="sub">Trend, last ${Math.min(lg.length,12)} games</h2><div class="metric" role="group" aria-label="Choose stat">${T.map((t,i)=>`<button type="button" data-i="${i}" aria-pressed="${i===trendIdx}">${t[0]}</button>`).join("")}</div><div class="trend">${trendSVG(lg, T[trendIdx][1])}</div>`;
    const C = pos==="QB" ? [["Cmp/Att",x=>x[5]+"/"+x[4]],["Pass yds",x=>x[6]],["TD",x=>x[7]],["INT",x=>x[8]],["Rush yds",x=>x[10]],["Rush TD",x=>x[11]]]
            : pos==="RB" ? [["Car",x=>x[9]],["Rush yds",x=>x[10]],["Tgt",x=>x[12]],["Rec",x=>x[13]],["Rec yds",x=>x[14]],["TD",x=>x[11]+x[15]]]
            : [["Tgt",x=>x[12]],["Rec",x=>x[13]],["Rec yds",x=>x[14]],["TD",x=>x[15]+x[11]],["Car",x=>x[9]],["Rush yds",x=>x[10]]];
    html += `<h2 class="sub">Game log</h2><div class="scroll"><table><tr><th>Game</th><th>Opp</th><th>Snap %</th>${C.map(c=>`<th>${c[0]}</th>`).join("")}<th>PPR</th></tr>` +
      [...lg].reverse().map(x=>`<tr><td>${x[0]} W${x[1]}</td><td>${x[2]}</td><td>${x[3]??"–"}</td>${C.map(c=>`<td>${c[1](x)}</td>`).join("")}<td>${x[16]==null?"–":(+x[16]).toFixed(1)}</td></tr>`).join("") + `</table></div>`;
  }
  html += `<p style="margin-top:24px"><button class="btn ghost" type="button" id="pvBack">Back to games</button></p>`;
  const box = $("#playerView"); box.innerHTML = html;
  box.querySelectorAll(".metric button").forEach(b=>b.onclick=()=>{ trendIdx=+b.dataset.i; renderPlayer(n); });
  const pb = $("#pvProps"); if (pb) pb.onclick = ()=>{ pSel = n; showView("props"); renderProps(); };
  $("#pvBack").onclick = ()=>{ $("#q").value=""; showView("week"); };
}

// ---------- closing line value ----------
function renderCLV(){
  const tile = (v, lbl) => `<div class="tile"><span class="cond">${v}</span><small>${lbl}</small></div>`;
  const sp = [], tt = [];
  D.games.forEach(g=>{
    const f = g.first; if (!f || g.spread==null) return;
    if (!done(g) && g.wk!==PM.week) return;
    const home = f.pred > f.spread;
    sp.push({v: home ? g.spread - f.spread : f.spread - g.spread, big: Math.abs(f.pred-f.spread)>=3});
    if (f.total!=null && g.total!=null && f.tpred!=null){ const over = f.tpred > f.total; tt.push({v: over ? g.total - f.total : f.total - g.total, big: Math.abs(f.tpred-f.total)>=3}); }
  });
  const summ = a => { if (!a.length) return null; const avg = a.reduce((s,x)=>s+x.v,0)/a.length;
    return {n:a.length, avg, beat:a.filter(x=>x.v>0).length, same:a.filter(x=>x.v===0).length}; };
  const S1 = summ(sp), S2 = summ(tt), S3 = summ(sp.filter(x=>x.big));
  let html = "";
  if (S1) html += tile(`${S1.avg>=0?"+":""}${S1.avg.toFixed(2)} pts`, `Spread picks: average move toward the pick (${S1.n} games)`) +
                  tile(`${S1.beat} of ${S1.n}`, `Spread picks that beat the closing line (${S1.same} unchanged)`);
  if (S3) html += tile(`${S3.avg>=0?"+":""}${S3.avg.toFixed(2)} pts`, `Spread picks with 3+ pt edges (${S3.n})`);
  if (S2) html += tile(`${S2.avg>=0?"+":""}${S2.avg.toFixed(2)} pts`, `Totals picks: average move toward the pick (${S2.n})`);
  // props: only when two pulls a week are saved
  const pc = [];
  (D.past||[]).forEach(r=>{
    if (!r.fbooks || !r.books) return;
    const first = {books:r.fbooks, st:r.st, qs:r.qs, mu:r.mu}; applyBest(first);
    if (first.line==null) return;
    const fin = r.books.find(b=>b.book===first.book); if (!fin || fin.line==null) return;
    const p = probOver(first, first.line), eO = p - (breakeven(first.over)??1), eU = (1-p) - (breakeven(first.under)??1);
    const over = eO >= eU;
    pc.push(r.st==="tds" ? (breakeven(fin.over) - breakeven(first.over)) : (over ? fin.line - first.line : first.line - fin.line));
  });
  if (pc.length) html += tile(`${pc.filter(v=>v>0).length} of ${pc.length}`, "Prop picks that beat the closing line at the same book");
  $("#clvTiles").innerHTML = html || `<p class="empty">Closing line value starts showing once the site has recorded a week of opening lines.</p>`;
}
renderCLV();

// ---------- playoff odds ----------
const POD = D.playoffs || [];
let poSub = "odds";
function pctCell(v){ return `<td><span class="pct">${v==null?"–":(v>=99.95?">99.9":v<0.05&&v>0?"<0.1":v.toFixed(1))+"%"}<i style="width:${Math.min(100,v||0)}%"></i></span></td>`; }
function renderPlayoffs(){
  if (!POD.length){ $("#poOut").innerHTML = ""; return; }
  $("#poLede").textContent = poSub==="odds"
    ? `Based on ${D.sims.toLocaleString()} simulations of the rest of the season, using the model's prediction for every remaining game. Updated with every build. Tiebreakers are simplified (point differential), so close races can differ slightly from the real rules.`
    : `Each team's rating is how many points the model expects it to beat an average team by on a neutral field, based on recent play and its current quarterback.`;
  const rec = t => `${t.w}-${t.l}${t.ties?"-"+t.ties:""}`;
  if (poSub==="power"){
    const L = [...POD].sort((a,b)=>b.rating-a.rating);
    $("#poOut").innerHTML = `<div class="season-wrap"><table class="po"><thead><tr><th>#</th><th style="text-align:left">Team</th><th>Rating</th><th>Record</th><th>Projected wins</th><th>Win Super Bowl</th></tr></thead><tbody>` +
      L.map((t,i)=>`<tr><td>${i+1}</td><td style="text-align:left"><span class="swatch" style="background:${tcol(t.t)}"></span>${tlink(t.t)}</td><td class="${t.rating>=3?'hot':t.rating<=-3?'cold':''}">${t.rating>0?"+":""}${t.rating.toFixed(1)}</td><td>${rec(t)}</td><td>${t.pw.toFixed(1)}</td>${pctCell(t.sb)}</tr>`).join("") + `</tbody></table></div>`;
    return;
  }
  let html = "";
  for (const c of ["AFC","NFC"]){
    html += `<h2 class="sub">${c}</h2><div class="season-wrap"><table class="po"><thead><tr><th style="text-align:left">Team</th><th>Record</th><th>Projected wins</th><th>Make playoffs</th><th>Win division</th><th>No. 1 seed</th><th>Win ${c}</th><th>Win Super Bowl</th></tr></thead><tbody>`;
    const divs = [...new Set(POD.filter(t=>t.conf===c).map(t=>t.div))].sort();
    for (const d of divs){
      html += `<tr class="divrow"><td colspan="8">${d}</td></tr>`;
      POD.filter(t=>t.div===d).sort((a,b)=>b.playoffs-a.playoffs).forEach(t=>{
        html += `<tr><td style="text-align:left"><span class="swatch" style="background:${tcol(t.t)}"></span>${tlink(t.t)}</td><td>${rec(t)}</td><td>${t.pw.toFixed(1)}</td>${pctCell(t.playoffs)}${pctCell(t.division)}${pctCell(t.bye)}${pctCell(t.cchamp)}${pctCell(t.sb)}</tr>`;
      });
    }
    html += `</tbody></table></div>`;
  }
  $("#poOut").innerHTML = html;
}
renderPlayoffs();

// ---------- trivia ----------
const TV = D.trivia || {guess:[], quiz:[]};
const tget = k => { try { return JSON.parse(localStorage.getItem("nfltrivia:"+k)||"null"); } catch(e){ return null; } };
const tset = (k,v) => { try { localStorage.setItem("nfltrivia:"+k, JSON.stringify(v)); } catch(e){} };
document.querySelectorAll("[data-tsub]").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("[data-tsub]").forEach(o=>o.setAttribute("aria-selected", o===b));
  $("#tGuess").hidden = b.dataset.tsub!=="guess"; $("#tQuiz").hidden = b.dataset.tsub!=="quiz";
});
$("#allNames").innerHTML = [...new Set([...TV.guess.map(g=>g.n), ...D.players.map(p=>p.n), ...Object.keys(D.logs||{})])].sort()
  .map(n=>`<option value="${n.replace(/"/g,"&quot;")}"></option>`).join("");
// Guess the player
const today = new Date().toISOString().slice(0,10);
const dayIdx = [...today].reduce((h,c)=>(h*31+c.charCodeAt(0))>>>0, 7) % Math.max(TV.guess.length,1);
let gp = {i: dayIdx, shown: 1, tries: [], over: false, daily: true};
const gnorm = s => s.toLowerCase().normalize("NFD").replace(/[^a-z]/g,"");
function renderGuess(){
  const box = $("#tGuess");
  if (!TV.guess.length){ box.innerHTML = `<p class="empty">Trivia appears once the season has a few games.</p>`; return; }
  const P = TV.guess[gp.i], st = tget("guess") || {played:0, solved:0, clues:0};
  const solved = gp.tries.length && gnorm(gp.tries[gp.tries.length-1])===gnorm(P.n);
  let msg = "";
  if (gp.over) msg = solved ? `<p class="msg ok">Got it! ${P.n}, in ${gp.shown} clue${gp.shown>1?"s":""}. Score: ${P.clues.length + 1 - gp.shown} of ${P.clues.length}.</p>`
                            : `<p class="msg no">It was ${P.n}.</p>`;
  else if (gp.tries.length) msg = `<p class="msg no">Not ${gp.tries[gp.tries.length-1]}. Here's another clue.</p>`;
  box.innerHTML = `<div class="trivia-card"><p class="qprog">${gp.daily?`Today's player`:"Bonus round"}. Clues revealed: ${gp.shown} of ${P.clues.length}. Fewer clues, higher score.</p>
    <ol class="clues">${P.clues.slice(0, gp.over ? P.clues.length : gp.shown).map(c=>`<li>${c}</li>`).join("")}</ol>
    ${gp.over ? "" : `<div class="guess-row"><input id="gIn" list="allNames" placeholder="Type a player's name" aria-label="Your guess"><button class="btn" type="button" id="gGo">Guess</button><button class="btn ghost" type="button" id="gHint">${gp.shown<P.clues.length?"Next clue":"Give up"}</button></div>`}
    ${msg}
    ${gp.over ? `<p style="margin-top:14px"><button class="btn" type="button" id="gNew">Play another</button> <a class="plink" data-p="${P.n.replace(/"/g,"&quot;")}" style="margin-left:10px">See ${P.n}'s page</a></p>` : ""}
    <p class="qprog" style="margin-top:14px">Your record: ${st.solved} solved of ${st.played} played${st.solved?`, ${(st.clues/st.solved).toFixed(1)} clues on average`:""}.</p></div>`;
  wireLinks(box);
  const finish = ok => { gp.over = true; const s = tget("guess") || {played:0, solved:0, clues:0}; s.played++; if (ok){ s.solved++; s.clues += gp.shown; } tset("guess", s); renderGuess(); };
  const go = () => { const v = $("#gIn").value.trim(); if (!v) return; gp.tries.push(v);
    if (gnorm(v)===gnorm(P.n)) finish(true); else if (gp.shown >= P.clues.length) finish(false); else { gp.shown++; renderGuess(); $("#gIn") && $("#gIn").focus(); } };
  if ($("#gGo")){ $("#gGo").onclick = go; $("#gIn").addEventListener("keydown", e=>{ if (e.key==="Enter") go(); });
    $("#gHint").onclick = () => { if (gp.shown < P.clues.length){ gp.shown++; renderGuess(); } else finish(false); }; }
  if ($("#gNew")) $("#gNew").onclick = () => { let j; do { j = Math.floor(Math.random()*TV.guess.length); } while (TV.guess.length>1 && j===gp.i);
    gp = {i:j, shown:1, tries:[], over:false, daily:false}; renderGuess(); };
}
renderGuess();
// Quiz
let qz = null;
function newQuiz(){ const idx = TV.quiz.map((_,i)=>i).sort(()=>Math.random()-.5).slice(0,10); qz = {idx, at:0, score:0, picked:null}; renderQuiz(); }
function renderQuiz(){
  const box = $("#tQuiz");
  if (!TV.quiz.length){ box.innerHTML = `<p class="empty">Quiz questions appear once the season has a few games.</p>`; return; }
  const best = tget("quizbest") || 0;
  if (qz.at >= qz.idx.length){
    if (qz.score > best) tset("quizbest", qz.score);
    box.innerHTML = `<div class="trivia-card"><p class="tq">You got ${qz.score} of ${qz.idx.length}.</p><p class="qprog">${qz.score>best?"New personal best!":`Your best: ${Math.max(best,qz.score)} of ${qz.idx.length}.`} Questions change every day as the season goes on.</p>
      <p style="margin-top:14px"><button class="btn" type="button" id="qNew">New quiz</button></p></div>`;
    $("#qNew").onclick = newQuiz; return;
  }
  const Q = TV.quiz[qz.idx[qz.at]], answered = qz.picked!==null;
  box.innerHTML = `<div class="trivia-card"><p class="qprog">Question ${qz.at+1} of ${qz.idx.length}. Score: ${qz.score}</p><p class="tq">${Q.q}</p>
    <div class="qopts">${Q.o.map((o,i)=>`<button type="button" data-i="${i}" ${answered?"disabled":""} class="${answered&&i===Q.a?"right":answered&&i===qz.picked?"wrong":""}">${o}</button>`).join("")}</div>
    ${answered?`<p class="msg ${qz.picked===Q.a?"ok":"no"}">${qz.picked===Q.a?"Correct.":"Not quite."} ${Q.x}</p><p style="margin-top:12px"><button class="btn" type="button" id="qNext">${qz.at+1<qz.idx.length?"Next question":"See score"}</button></p>`:""}</div>`;
  box.querySelectorAll(".qopts button").forEach(b=>b.onclick=()=>{ if (qz.picked!==null) return; qz.picked=+b.dataset.i; if (qz.picked===Q.a) qz.score++; renderQuiz(); });
  if ($("#qNext")) $("#qNext").onclick = ()=>{ qz.at++; qz.picked=null; renderQuiz(); };
}
newQuiz();

// ================= Shared: teams =================
const GAME = {}; D.games.forEach(g => GAME[g.id] = g);
function ncdf(x){ const t=1/(1+.2316419*Math.abs(x)), d=.3989423*Math.exp(-x*x/2); const p=d*t*(.3193815+t*(-.3565638+t*(1.781478+t*(-1.821256+t*1.330274)))); return x>0?1-p:p; }
const TEAMINFO = Object.fromEntries((D.playoffs||[]).map(t=>[t.t,t]));
function tlink(t){ return `<a class="tlink" data-t="${t}">${TNAME[t]||t}</a>`; }
document.addEventListener("click", e => { const a = e.target.closest && e.target.closest("a.tlink"); if (a){ e.preventDefault(); openTeam(a.dataset.t); } });
// ================= League =================
let lsub = "stand";
document.querySelectorAll("[data-lsub]").forEach(b=>b.onclick=()=>{ lsub=b.dataset.lsub; renderLeague(); });
function standings(){
  const T = {}; (D.playoffs||[]).forEach(t=>T[t.t] = {t:t.t, name:t.name, conf:t.conf, div:t.div, w:0,l:0,ti:0,pf:0,pa:0,dw:0,dl:0,cw:0,cl:0,res:[]});
  D.games.filter(done).sort((a,b)=>a.wk-b.wk).forEach(g=>{
    const H = T[g.home], A = T[g.away]; if (!H||!A) return;
    H.pf+=g.hs; H.pa+=g.as_; A.pf+=g.as_; A.pa+=g.hs;
    const r = g.hs>g.as_ ? 1 : g.hs<g.as_ ? -1 : 0;
    [[H,r],[A,-r]].forEach(([X,v])=>{ if (v>0) X.w++; else if (v<0) X.l++; else X.ti++; X.res.push(v);
      const O = X===H?A:H; if (O.div===X.div){ if(v>0)X.dw++; else if(v<0)X.dl++; } if (O.conf===X.conf){ if(v>0)X.cw++; else if(v<0)X.cl++; } });
  });
  Object.values(T).forEach(x=>{ x.pct = (x.w + x.ti/2)/Math.max(1,x.w+x.l+x.ti); x.diff = x.pf-x.pa;
    let s = 0, v = x.res[x.res.length-1]; for (let i=x.res.length-1;i>=0 && x.res[i]===v;i--) s++; x.streak = x.res.length ? (v>0?"W":v<0?"L":"T")+s : "–"; });
  return T;
}
const byRank = (a,b) => b.pct-a.pct || (b.dw-b.dl)-(a.dw-a.dl) || b.diff-a.diff;
function renderLeague(){
  document.querySelectorAll("[data-lsub]").forEach(b=>b.setAttribute("aria-selected", b.dataset.lsub===lsub));
  ["stand","leaders","recap"].forEach(k=>$("#lsub-"+k).hidden = k!==lsub);
  $("#lsub-po").hidden = !(lsub==="odds" || lsub==="power");
  if (lsub==="stand") renderStandings(); else if (lsub==="leaders") renderLeaders(); else if (lsub==="recap") renderRecap();
  else { poSub = lsub; renderPlayoffs(); }
}
function renderStandings(){
  const T = standings(), rec = x => `${x.w}-${x.l}${x.ti?"-"+x.ti:""}`;
  let h = `<p class="lede" style="margin-top:14px">Current standings, with the playoff picture if the season ended today. Tiebreakers are simplified (win percentage, then division record, then point differential), so some close races may be ordered differently than the NFL's official rules.</p>`;
  for (const c of ["AFC","NFC"]){
    const teams = Object.values(T).filter(x=>x.conf===c), divs = [...new Set(teams.map(x=>x.div))].sort();
    const leaders = divs.map(d=>teams.filter(x=>x.div===d).sort(byRank)[0]).sort(byRank);
    const rest = teams.filter(x=>!leaders.includes(x)).sort(byRank);
    const seeds = [...leaders, ...rest.slice(0,3)];
    h += `<h2 class="sub">${c} playoff picture</h2><div class="season-wrap"><table class="po"><tbody>` +
      seeds.map((x,i)=>`<tr><td style="text-align:left"><span class="seed">${i+1}</span><span class="swatch" style="background:${tcol(x.t)}"></span>${tlink(x.t)}</td><td>${rec(x)}</td><td style="text-align:left;color:var(--muted)">${i<4?(i===0?"Division leader, first-round bye":"Division leader"):"Wild card"}</td></tr>`).join("") +
      rest.slice(3,5).map(x=>`<tr><td style="text-align:left"><span class="seed"></span><span class="swatch" style="background:${tcol(x.t)}"></span>${tlink(x.t)}</td><td>${rec(x)}</td><td style="text-align:left;color:var(--muted)">In the hunt</td></tr>`).join("") + `</tbody></table></div>`;
    h += `<div class="season-wrap"><table class="po"><thead><tr><th style="text-align:left">Team</th><th>W-L</th><th>Pct</th><th>PF</th><th>PA</th><th>Diff</th><th>Div</th><th>Conf</th><th>Streak</th></tr></thead><tbody>`;
    for (const d of divs){
      h += `<tr class="divrow"><td colspan="9">${d}</td></tr>` + teams.filter(x=>x.div===d).sort(byRank).map(x=>`<tr><td style="text-align:left"><span class="swatch" style="background:${tcol(x.t)}"></span>${tlink(x.t)}</td><td>${rec(x)}</td><td>${x.pct.toFixed(3).replace(/^0/,"")}</td><td>${x.pf}</td><td>${x.pa}</td><td class="${x.diff>0?'hot':x.diff<0?'cold':''}">${x.diff>0?"+":""}${x.diff}</td><td>${x.dw}-${x.dl}</td><td>${x.cw}-${x.cl}</td><td>${x.streak}</td></tr>`).join("");
    }
    h += `</tbody></table></div>`;
  }
  $("#lsub-stand").innerHTML = h;
}
function ldOptions(){ const cols = COLS[$("#ldPos").value].filter(c=>c[1]!=="gp"); $("#ldStat").innerHTML = cols.map(c=>`<option value="${c[1]}" ${c[1]==="ppg"?"selected":""}>${c[0]}</option>`).join(""); }
ldOptions(); $("#ldPos").onchange = () => { ldOptions(); renderLeaders(); }; $("#ldStat").onchange = renderLeaders; $("#ldMin").onchange = renderLeaders;
function renderLeaders(){
  const pos = $("#ldPos").value, k = $("#ldStat").value, mn = +$("#ldMin").value, cols = COLS[pos];
  const low = ["int_"].includes(k);   // fewer is better
  const L = D.players.filter(p=>p.pos===pos && p.gp>=mn && p[k]!=null).sort((a,b)=>low ? a[k]-b[k] : b[k]-a[k]).slice(0,40);
  $("#ldTbl").innerHTML = `<thead><tr><th>#</th><th style="text-align:left">Player</th>${cols.map(c=>`<th ${c[1]===k?'aria-sort="descending"':""}>${c[0]}</th>`).join("")}</tr></thead><tbody>` +
    L.map((p,i)=>`<tr><td>${i+1}</td><td style="text-align:left">${teamBadge(p.tm)}${plink(p.n)}</td>${cols.map(c=>{ const v=p[c[1]]; return `<td class="${c[1]===k?'hot':''}">${fmt(v,c[2])}</td>`; }).join("")}</tr>`).join("") + `</tbody>`;
  wireLinks($("#ldTbl"));
}
let recapWk = null;
function renderRecap(){
  const wks = [...new Set(D.games.filter(done).map(g=>g.wk))].sort((a,b)=>a-b);
  if (!wks.length){ $("#lsub-recap").innerHTML = `<p class="empty">The recap appears after the first week of games.</p>`; return; }
  if (!recapWk || !wks.includes(recapWk)){ const full = wks.filter(w=>D.games.filter(g=>g.wk===w).every(done)); recapWk = full.length ? full[full.length-1] : wks[wks.length-1]; }
  const G = D.games.filter(g=>g.wk===recapWk && done(g)), R = gradeGames(G);
  const judged = G.filter(g=>g.spread!=null && g.pred!=null && (g.hs-g.as_)!==g.spread).map(g=>{ const m = g.hs-g.as_, ok = (g.pred>g.spread)===(m>g.spread); return {g, ok, edge:Math.abs(g.pred-g.spread)}; });
  const best = judged.filter(x=>x.ok).sort((a,b)=>b.edge-a.edge).slice(0,2), worst = judged.filter(x=>!x.ok).sort((a,b)=>b.edge-a.edge).slice(0,2);
  const pickTxt = x => `${x.g.away} at ${x.g.home}: the model took ${x.g.pred>x.g.spread?x.g.home:x.g.away} against a ${spreadTxt(x.g,x.g.spread)} line (${x.edge.toFixed(1)}-point edge). Final: ${x.g.away} ${x.g.as_}, ${x.g.home} ${x.g.hs}.`;
  const upsets = G.filter(g=>g.spread!=null && g.spread!==0 && ((g.spread>0) !== (g.hs>g.as_)) && g.hs!==g.as_).sort((a,b)=>Math.abs(b.spread)-Math.abs(a.spread)).slice(0,4);
  const perf = []; Object.entries(D.logs||{}).forEach(([n,lg])=>{ const x = lg.find(r=>r[0]===D.season && r[1]===recapWk); if (x && x[16]!=null) perf.push({n, x, tm:(people.get(n)||{}).tm}); });
  perf.sort((a,b)=>b.x[16]-a.x[16]);
  const line = x => { const p=[]; if (x[4]) p.push(`${x[5]}/${x[4]}, ${x[6]} pass yds, ${x[7]} TD`); if (x[9]) p.push(`${x[9]} car, ${x[10]} yds${x[11]?`, ${x[11]} TD`:""}`); if (x[12]) p.push(`${x[13]} rec, ${x[14]} yds${x[15]?`, ${x[15]} TD`:""}`); return p.join("; "); };
  const surpr = (D.past||[]).filter(r=>r.wk===recapWk && r.qs && ["rushing_yards","receiving_yards","passing_yards"].includes(r.st)).map(r=>({r, d:r.act-r.qs[2]}));
  const up = surpr.sort((a,b)=>b.d-a.d).slice(0,4), down = [...surpr].sort((a,b)=>a.d-b.d).slice(0,3);
  let h = `<div class="toolbar" style="margin-top:14px"><label>Week <select id="rcWk">${wks.map(w=>`<option ${w===recapWk?"selected":""}>${w}</option>`).join("")}</select></label></div>
    <div class="recap-grid">
    <div class="recap-card"><h3>How the model did</h3><ul><li>Winners: ${wl(R.su)}</li><li>Against the spread: ${wl(R.ats)} (3+ point edges: ${wl(R.ats3)})</li><li>Totals: ${wl(R.ou)}</li></ul></div>
    <div class="recap-card"><h3>Best calls</h3>${best.length?`<ul>${best.map(x=>`<li>${pickTxt(x)}</li>`).join("")}</ul>`:`<p class="empty">No winning picks this week.</p>`}</div>
    <div class="recap-card"><h3>Worst calls</h3>${worst.length?`<ul>${worst.map(x=>`<li>${pickTxt(x)}</li>`).join("")}</ul>`:`<p class="empty">No losing picks this week.</p>`}</div>
    <div class="recap-card"><h3>Upsets</h3>${upsets.length?`<ul>${upsets.map(g=>{ const dog = g.spread>0?g.away:g.home; return `<li>${dog} won as a ${Math.abs(g.spread)}-point underdog, ${g.away} ${g.as_}, ${g.home} ${g.hs}.</li>`; }).join("")}</ul>`:`<p class="empty">Every favorite won.</p>`}</div>
    <div class="recap-card"><h3>Top fantasy performances</h3><ul>${perf.slice(0,6).map(p=>`<li>${teamBadge(p.tm)}${plink(p.n)}: ${p.x[16].toFixed(1)} PPR. ${line(p.x)}.</li>`).join("")}</ul></div>
    <div class="recap-card"><h3>Biggest surprises</h3><ul>${up.map(x=>`<li>${plink(x.r.n)}: ${Math.round(x.r.act)} ${STATL[x.r.st].toLowerCase()}, projected ${Math.round(x.r.qs[2])}.</li>`).join("")}${down.map(x=>`<li>${plink(x.r.n)}: ${Math.round(x.r.act)} ${STATL[x.r.st].toLowerCase()}, projected ${Math.round(x.r.qs[2])}.</li>`).join("")}</ul></div>
    </div>`;
  $("#lsub-recap").innerHTML = h; wireLinks($("#lsub-recap"));
  $("#rcWk").onchange = e => { recapWk = +e.target.value; renderRecap(); };
}
// ================= Team pages =================
function openTeam(t){ if (!TNAME[t]) return; renderTeam(t); showView("team"); window.scrollTo({top: $("nav.views").getBoundingClientRect().top + window.scrollY - 12}); }
function renderTeam(t){
  const T = standings()[t], P = TEAMINFO[t] || {}, games = D.games.filter(g=>g.home===t||g.away===t).sort((a,b)=>a.wk-b.wk);
  const tile = (v, lbl) => `<div class="tile"><span class="cond">${v}</span><small>${lbl}</small></div>`;
  let ats = [0,0,0]; games.filter(done).forEach(g=>{ if (g.spread==null) return; const m = (g.hs-g.as_)*(g.home===t?1:-1), s = g.spread*(g.home===t?1:-1); if (m>s) ats[0]++; else if (m<s) ats[1]++; else ats[2]++; });
  let h = `<div class="pv-head"><div><h2>${teamChip(t)}${TNAME[t]}</h2><p class="who">${P.div||""}. ${T?`${T.w}-${T.l}${T.ti?"-"+T.ti:""}, ${wl(ats)} against the spread`:""}</p></div></div>
    <div class="tiles" style="margin-top:14px">${tile(P.rating!=null?(P.rating>0?"+":"")+P.rating.toFixed(1):"–","Power rating (pts vs average team)")}${tile(P.pw!=null?P.pw.toFixed(1):"–","Projected wins")}${tile(P.playoffs!=null?P.playoffs.toFixed(0)+"%":"–","Chance to make playoffs")}${tile(P.sb!=null?P.sb.toFixed(1)+"%":"–","Chance to win Super Bowl")}</div>`;
  h += `<div class="panel" style="margin-top:14px">${formPanel(t)}</div>`;
  h += `<h2 class="sub">Schedule</h2><div class="season-wrap"><table class="po"><thead><tr><th>Wk</th><th style="text-align:left">Opponent</th><th>Vegas</th><th>Model</th><th style="text-align:left">Result</th></tr></thead><tbody>` +
    games.map(g=>{ const home = g.home===t, opp = home?g.away:g.home;
      let res = done(g) ? (()=>{ const m=(g.hs-g.as_)*(home?1:-1); return `${m>0?"W":m<0?"L":"T"} ${home?g.hs:g.as_}-${home?g.as_:g.hs}`; })() : `<span class="qprog">${timeTxt(g)}</span>`;
      return `<tr><td>${g.wk}</td><td style="text-align:left">${home?"vs":"at"} <span class="swatch" style="background:${tcol(opp)}"></span>${tlink(opp)}</td><td>${spreadTxt(g,g.spread)}</td><td>${spreadTxt(g,g.pred)}</td><td style="text-align:left">${res}</td></tr>`; }).join("") + `</tbody></table></div>`;
  const key = D.players.filter(p=>p.tm===t).sort((a,b)=>(b.ppg||0)-(a.ppg||0)).slice(0,8);
  if (key.length) h += `<h2 class="sub">Key players</h2><div class="season-wrap"><table class="po"><thead><tr><th style="text-align:left">Player</th><th>Pos</th><th>Games</th><th>Snap %</th><th>PPR per game</th></tr></thead><tbody>` +
    key.map(p=>`<tr><td style="text-align:left">${plink(p.n)}</td><td>${p.pos}</td><td>${p.gp}</td><td>${p.snap??"–"}</td><td>${fmt(p.ppg,1)}</td></tr>`).join("") + `</tbody></table></div>`;
  const nx = games.find(g=>!done(g)); if (nx && nx.outs && nx.outs[t]) h += `<p class="outs" style="margin-top:14px"><b>Ruled out for week ${nx.wk}:</b> ${nx.outs[t].join(", ")}.</p>`;
  h += `<p style="margin-top:24px"><button class="btn ghost" type="button" id="tvBack">Back to games</button></p>`;
  $("#teamView").innerHTML = h; wireLinks($("#teamView"));
  $("#tvBack").onclick = () => { $("#q").value=""; showView("week"); };
}

// ================= Sub-tabs for Games and Betting =================
document.querySelectorAll("[data-gsub]").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("[data-gsub]").forEach(o=>o.setAttribute("aria-selected", o===b));
  $("#gsub-wk").hidden = b.dataset.gsub!=="wk"; $("#gsub-sched").hidden = b.dataset.gsub!=="sched";
});
document.querySelectorAll("[data-psb]").forEach(b=>b.onclick=()=>showPsb(b.dataset.psb));
function showPsb(k){
  document.querySelectorAll("[data-psb]").forEach(o=>o.setAttribute("aria-selected", o.dataset.psb===k));
  $("#psb-best").hidden = k!=="best"; $("#psb-all").hidden = k!=="all";
}

// ================= Best bets =================
const MU = D.matchups || {};
const MU_LABEL = {"QB|passing_yards":["passing yards","QBs"], "RB|rushing_yards":["rushing yards","RBs"], "RB|receiving_yards":["receiving yards","RBs"],
  "WR|receiving_yards":["receiving yards","WRs"], "TE|receiving_yards":["receiving yards","TEs"], "WR|receptions":["catches","WRs"],
  "RB|tds":["touchdowns","RBs"], "WR|tds":["touchdowns","WRs"], "TE|tds":["touchdowns","TEs"]};
const nDef = Object.keys(MU).length || 32;
function muFor(def, key){ const m = MU[def] && MU[def][key]; return m ? {avg:m[0], over:m[1], n:m[2], rank:m[3]} : null; }
function muSentence(def, key, m){
  const [what, who] = MU_LABEL[key], dec = key.endsWith("tds") ? 2 : 0, amt = Math.abs(m.avg).toFixed(dec);
  const dir = m.avg >= 0 ? "more" : "fewer";
  const tail = m.avg >= 0 ? `more than usual in ${m.over} of ${m.n}` : `less than usual in ${m.n - m.over} of ${m.n}`;
  return `${who} average ${amt} ${dir} ${what} per game than usual against the ${TNAME[def]||def} (${tail} games${m.n<5?", small sample":""}).`;
}
function streakFor(n, st){
  const lg = (D.logs[n]||[]).slice(-6); if (lg.length < 4) return null;
  const idx = {receiving_yards:14, rushing_yards:10, receptions:13, passing_yards:6}[st];
  if (st==="tds"){ let s = 0; for (let i=lg.length-1;i>=0 && (lg[i][11]+lg[i][15])>0;i--) s++; return s>=3 ? {text:`a touchdown in ${s} straight games`, mark:.5, s} : null; }
  if (idx==null) return null;
  const marks = {receiving_yards:[125,100,80,70,60,50], rushing_yards:[125,100,80,70,60,50], receptions:[9,8,7,6,5,4], passing_yards:[325,300,275,250,225]}[st];
  const need = lg.length>=6 ? 5 : lg.length-1;
  for (const mk of marks){ const hits = lg.filter(x=>x[idx]>=mk).length; if (hits>=need) return {text:`${mk}+ ${STATL[st].toLowerCase()} in ${hits} of his last ${lg.length}`, mark:mk, hits, of:lg.length}; }
  return null;
}
function renderBest(){
  const box = $("#psb-best");
  const wkGames = D.games.filter(g=>g.wk===PM.week && !done(g));
  const tile = (v, lbl) => `<div class="tile"><span class="cond">${v}</span><small>${lbl}</small></div>`;
  let h = `<p class="lede" style="margin-top:14px">The strongest bets and trends for week ${PM.week}. Edges compare the model with the sportsbook line. Trends show what's been happening lately; sportsbooks know about matchups too, so treat a trend as a tiebreaker rather than a reason to bet on its own.</p>`;
  // --- 1. Top picks
  const gp = [];
  wkGames.forEach(g=>{
    if (g.spread!=null && g.pred!=null){ const e = Math.abs(g.pred-g.spread); if (e>=2){ const side = g.pred>g.spread ? g.home : g.away, line = side===g.home ? -g.spread : g.spread;
      gp.push({e, txt:`${teamBadge(side)}<b>${side} ${line>0?"+":""}${line===0?"pick'em":line}</b>`, sub:`${g.away} at ${g.home}. Model: ${spreadTxt(g,g.pred)}, Vegas: ${spreadTxt(g,g.spread)}.`, pts:`${e.toFixed(1)} pts`}); } }
    if (g.total!=null && g.tpred!=null){ const e = Math.abs(g.tpred-g.total); if (e>=3) gp.push({e:e*.8, txt:`<b>${g.tpred>g.total?"Over":"Under"} ${g.total}</b>`, sub:`${g.away} at ${g.home}. Model total: ${g.tpred.toFixed(1)}.`, pts:`${e.toFixed(1)} pts`}); }
  });
  gp.sort((a,b)=>b.e-a.e);
  const pp = PR.map(r=>({r, e:evalRow(r)})).filter(x=>x.e.p!=null && x.e.edge>=.04).sort((a,b)=>b.e.edge-a.e.edge).slice(0,10);
  h += `<h2 class="sub">Top picks</h2><div class="best-grid"><div class="recap-card"><h3>Games</h3>${gp.length ? `<ul class="picks">${gp.slice(0,6).map(x=>`<li>${x.txt}<span class="edge-tag">${x.pts}</span><small>${x.sub}</small></li>`).join("")}</ul>` : `<p class="empty">No big spread or total edges this week.</p>`}</div>
    <div class="recap-card"><h3>Player props</h3>${pp.length ? `<ul class="picks">${pp.map(({r,e})=>{ const v=e.v, td=r.st==="tds";
      return `<li>${teamBadge(r.tm)}<b>${plink(r.n)} ${td?"anytime TD":(e.side==="over"?"over ":"under ")+v.line+" "+STATL[r.st].toLowerCase()}</b><span class="edge-tag">+${(e.edge*100).toFixed(1)}%</span><small>${fmtOdds(e.side==="under"?v.under:v.over)}${r.book&&v.fromBook?` at ${r.book}`:""}. Model gives it ${Math.round(e.sideP*100)}%.</small></li>`; }).join("")}</ul>`
      : `<p class="empty">${PR.some(r=>evalRow(r).p!=null) ? "No props with a 4%+ edge right now." : "Prop picks appear once the site has sportsbook lines, or when you type lines into All player props."}</p>`}</div></div>`;
  // --- 2. Where they agree
  const agree = [];
  PR.forEach(r=>{
    const e = evalRow(r); if (e.p==null || e.edge<.03) return;
    const key = r.pos+"|"+r.st, m = muFor(r.opp, key), s = streakFor(r.n, r.st), why = [];
    const overSide = r.st==="tds" || e.side==="over";
    if (m && m.n>=4){ if (overSide && m.rank<=8 && m.over/m.n>=.6) why.push(`soft matchup (${m.rank===1?"most":ord(m.rank)+" most"} allowed)`); if (!overSide && m.rank>nDef-8 && (m.n-m.over)/m.n>=.6) why.push(`tough matchup (${nDef-m.rank+1===1?"fewest":ord(nDef-m.rank+1)+" fewest"} allowed)`); }
    if (s && overSide && (r.st==="tds" || s.mark>=e.v.line)) why.push(s.text);
    if (why.length) agree.push({r, e, why});
  });
  agree.sort((a,b)=>b.why.length-a.why.length || b.e.edge-a.e.edge);
  h += `<h2 class="sub">Where the model and trends agree</h2><div class="recap-card">${agree.length ? `<ul class="picks">${agree.slice(0,10).map(({r,e,why})=>`<li>${teamBadge(r.tm)}<b>${plink(r.n)} ${r.st==="tds"?"anytime TD":(e.side==="over"?"over ":"under ")+e.v.line+" "+STATL[r.st].toLowerCase()}</b><span class="edge-tag">+${(e.edge*100).toFixed(1)}%</span><small>Model edge, plus ${why.join(" and ")}.</small></li>`).join("")}</ul>`
    : `<p class="empty">${PR.some(r=>evalRow(r).p!=null) ? "Nothing lines up strongly this week." : "This list fills in once props have sportsbook lines."}</p>`}</div>`;
  // --- 3. Matchup trends
  const soft = [], tough = [];
  wkGames.forEach(g=>[[g.home,g.away],[g.away,g.home]].forEach(([def, off])=>{
    Object.keys(MU_LABEL).forEach(key=>{
      const m = muFor(def, key); if (!m || m.n<3) return;
      const [pos, st] = key.split("|");
      const players = PR.filter(r=>r.tm===off && r.pos===pos && r.st===st).sort((a,b)=>(b.qs?b.qs[2]:b.mu)-(a.qs?a.qs[2]:a.mu)).slice(0,2);
      if (!players.length) return;
      const item = {def, off, key, m, players};
      if (m.rank<=4) soft.push(item); else if (m.rank>nDef-4) tough.push(item);
    });
  }));
  const muItem = x => `<li>${teamBadge(x.def)}<b>${MU_LABEL[x.key][1]} vs the ${TNAME[x.def]}</b><small>${muSentence(x.def, x.key, x.m)} This week: ${x.players.map(r=>`${plink(r.n)} (${r.qs?`projected ${x.key.endsWith("receptions")?r.qs[2].toFixed(1):Math.round(r.qs[2])}`:`${Math.round(probOver(r,.5)*100)}% TD`})`).join(", ")}.</small></li>`;
  soft.sort((a,b)=>a.m.rank-b.m.rank || b.m.over/b.m.n-a.m.over/a.m.n); tough.sort((a,b)=>b.m.rank-a.m.rank);
  h += `<h2 class="sub">Matchup trends</h2><p class="lede">How players at each position have done against this week's opponents over each defense's last 8 games, compared with what those same players usually do.</p>
    <div class="best-grid"><div class="recap-card"><h3>Softest matchups</h3>${soft.length?`<ul class="picks">${soft.slice(0,8).map(muItem).join("")}</ul>`:`<p class="empty">No standout soft matchups.</p>`}</div>
    <div class="recap-card"><h3>Toughest matchups</h3>${tough.length?`<ul class="picks">${tough.slice(0,6).map(muItem).join("")}</ul>`:`<p class="empty">No standout tough matchups.</p>`}</div></div>`;
  // --- 4. Hot streaks
  const seen = new Set(), hot = [];
  PR.forEach(r=>{ const k = r.n+"|"+r.st; if (seen.has(k)) return; seen.add(k); const s = streakFor(r.n, r.st); if (s) hot.push({r, s}); });
  const score = x => x.r.st==="tds" ? x.s.s*20 : x.s.mark * (x.r.st==="receptions"?15:x.r.st==="passing_yards"?.35:1) * (x.s.hits/x.s.of);
  hot.sort((a,b)=>score(b)-score(a));
  h += `<h2 class="sub">Hot streaks</h2><div class="recap-card">${hot.length?`<ul class="picks cols">${hot.slice(0,14).map(({r,s})=>`<li>${teamBadge(r.tm)}<b>${plink(r.n)}</b><small>${s.text[0].toUpperCase()+s.text.slice(1)}. Faces ${TNAME[r.opp]||r.opp} this week.</small></li>`).join("")}</ul>`:`<p class="empty">No long streaks among this week's players.</p>`}</div>`;
  box.innerHTML = h; wireLinks(box);
}
renderBest();

// ---------- Ask the model (AI assistant, uses the viewer's Claude account) ----------
(function(){
  const ASK = {turns: [], busy: false, ctl: null};
  const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
  function md(s){
    const lines = esc(s).split("\n"); let out = "", list = false;
    for (let l of lines){
      l = l.replace(/\*\*(.+?)\*\*/g,"<b>$1</b>");
      const m = l.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)/);
      if (m){ if (!list){ out += "<ul>"; list = true; } out += `<li>${m[1]}</li>`; continue; }
      if (list){ out += "</ul>"; list = false; }
      if (/^#{1,4}\s/.test(l)) out += `<p><b>${l.replace(/^#+\s/,"")}</b></p>`;
      else if (l.trim()) out += `<p>${l}</p>`;
    }
    return out + (list ? "</ul>" : "");
  }
  const r1 = x => x==null ? null : Math.round(x*10)/10;
  const findPlayer = q => { const h = searchFor(String(q||"")).find(x=>x.type==="p"); return h ? h.n : null; };
  const findTeam = q => { const s = String(q||"").toLowerCase(); if (!s) return null;
    return Object.keys(D.teams).find(t => t.toLowerCase()===s || (TNAME[t]||"").toLowerCase().includes(s)) || null; };
  const matchupText = (pos, r) => { if (r==null) return null; const n = new Set(FS.filter(f=>f.pos===pos).map(f=>f.opp)).size || 32;
    return r <= n/2 ? `${ord(r)} easiest of ${n}` : `${ord(n-r+1)} toughest of ${n}`; };
  function propOut(r){
    const e = evalRow(r), v = e.v;
    const o = {player:r.n, team:r.tm, opp:r.opp, prop:STATL[r.st],
      model: r.qs ? {middle:r.qs[2], likely_range:[r.qs[1], r.qs[3]]} : r.st==="tds" ? {chance_to_score_pct:Math.round(probOver(r,.5)*100)} : {average:r1(r.mu)}};
    if (r.q) o.injury = "Questionable";
    if (r.boost) o.more_work_because_out = r.boost;
    if (e.p!=null){ o.line = v.line; o.over_odds = v.over; if (r.st!=="tds") o.under_odds = v.under; o.book = v.fromBook ? r.book : "entered by you";
      o.model_over_pct = Math.round(e.p*100); o.best_side = r.st==="tds" ? "Yes" : (e.side==="over"?"Over":"Under"); o.edge_pct = r1(e.edge*100); }
    return o;
  }
  function gameOut(g){
    const edge = (g.pred!=null && g.spread!=null) ? r1(Math.abs(g.pred-g.spread)) : null;
    const o = {game:`${g.away} at ${g.home}`, week:g.wk, kickoff: done(g) ? null : timeTxt(g), qbs:`${g.aqb||"?"} vs ${g.hqb||"?"}`,
      vegas_spread: spreadTxt(g,g.spread), vegas_total: g.total, vegas_moneyline: g.aml!=null ? `${g.away} ${fmtOdds(g.aml)}, ${g.home} ${fmtOdds(g.hml)}` : null,
      model_spread: spreadTxt(g,g.pred), model_winner: g.prob>=.5 ? g.home : g.away, model_win_pct: Math.round(Math.max(g.prob,1-g.prob)*100),
      model_total: r1(g.tpred), spread_edge_pts: edge, spread_side_model_likes: edge!=null ? (g.pred>g.spread ? g.home : g.away) : null,
      total_side_model_likes: (g.total!=null && g.tpred!=null) ? (g.tpred>g.total ? "Over" : "Under") : null,
      total_edge_pts: (g.total!=null && g.tpred!=null) ? r1(Math.abs(g.tpred-g.total)) : null};
    if (g.wx) o.weather = g.wx.roof==="dome" ? "indoors" : (g.wx.temp!=null ? `${g.wx.temp}F, wind ${g.wx.wind} mph, ${g.wx.rain}% rain${g.wx.roof==="retract"?", retractable roof":""}` : null);
    if (g.outs) o.ruled_out = g.outs;
    if (g.books && g.books.length) o.sportsbooks = g.books.length;
    if (done(g)) o.final = `${g.away} ${g.as_}, ${g.home} ${g.hs}`;
    return o;
  }
  const TOOLS = [
    {name:"get_best_bets", description:"This week's biggest edges: game spreads and totals where the model disagrees with Vegas most, and player props where the model's chance beats the odds (only props that have a line, from a sportsbook or typed in by the user). Returns edges, lines and model numbers.",
     inputSchema:{type:"object", properties:{min_prop_edge_pct:{type:"number", description:"Minimum prop edge in percent, default 4"}}},
     execute(i){ status("Looking for the biggest edges");
       const thr = (Number(i.min_prop_edge_pct)||4)/100;
       const games = D.games.filter(g=>g.wk===PM.week && !done(g)).map(gameOut)
         .filter(g=>(g.spread_edge_pts||0)>=2 || (g.total_edge_pts||0)>=3).sort((a,b)=>Math.max(b.spread_edge_pts||0,b.total_edge_pts||0)-Math.max(a.spread_edge_pts||0,a.total_edge_pts||0)).slice(0,8);
       const props = PR.map(r=>({r,e:evalRow(r)})).filter(x=>x.e.p!=null && x.e.edge>=thr).sort((a,b)=>b.e.edge-a.e.edge).slice(0,12).map(x=>propOut(x.r));
       return {week:PM.week, games, props, props_with_lines: PR.filter(r=>evalRow(r).p!=null).length,
         note: props.length ? undefined : "No player props have lines yet. Props need sportsbook lines (Odds API key) or lines typed in on the Betting tab."}; }},
    {name:"get_games", description:"Games for a week (default: the current week): Vegas spread, total and moneyline, the model's spread, win chance and total, weather, notable injuries, and final scores for finished games. Optional team filter.",
     inputSchema:{type:"object", properties:{week:{type:"number"}, team:{type:"string", description:"Team name or abbreviation"}}},
     execute(i){ status("Checking the games"); const wk = Number(i.week)||PM.week, tm = findTeam(i.team);
       return D.games.filter(g=>tm ? (g.home===tm||g.away===tm) : g.wk===wk).filter(g=>tm ? true : true).slice(0, tm ? 18 : 17).map(gameOut); }},
    {name:"get_player", description:"One player's info: team, position, this season's advanced stats, recent game log (last 5 games), this week's projections and prop lines, and fantasy projection with range and matchup. Use for any question about a specific player.",
     inputSchema:{type:"object", properties:{name:{type:"string"}}, required:["name"]},
     execute(i){ const n = findPlayer(i.name); if (!n) throw new Error(`No player found matching "${i.name}"`); status(`Looking up ${n}`);
       const st = D.players.find(p=>p.n===n), f = fantBy[n], lg = (D.logs[n]||[]).slice(-5);
       return {player:n, season_stats: st || "not enough snaps this season",
         last_games: lg.map(x=>({season:x[0], week:x[1], opp:x[2], snap_pct:x[3], pass:`${x[5]}/${x[4]}, ${x[6]} yds, ${x[7]} TD, ${x[8]} INT`, rush:`${x[9]} car, ${x[10]} yds, ${x[11]} TD`, rec:`${x[13]}/${x[12]} tgt, ${x[14]} yds, ${x[15]} TD`, ppr:x[16]})),
         this_week_props: PR.filter(r=>r.n===n).map(propOut),
         fantasy: f ? {opp:f.opp, ppr:f.ppr, half_ppr:r1(f.std+.5*f.rec), standard:f.std, likely_range_ppr:[f.lo, f.hi], matchup:matchupText(f.pos,f.mrank), questionable:!!f.q} : "no fantasy projection this week (bye, out, or small role)"}; }},
    {name:"compare_start_sit", description:"Side-by-side fantasy comparison of 2 to 4 players for this week: projected points in the chosen scoring, likely range, matchup, last 3 games and key projected stats.",
     inputSchema:{type:"object", properties:{players:{type:"array", items:{type:"string"}}, scoring:{type:"string", enum:["ppr","half","standard"]}}, required:["players"]},
     execute(i){ status("Comparing players"); const k = {ppr:1, half:.5, standard:0}[String(i.scoring||"ppr")] ?? 1;
       return (Array.isArray(i.players)?i.players:[]).slice(0,4).map(q=>{ const n = findPlayer(q), f = n && fantBy[n];
         if (!f) return {query:q, player:n, note:"no projection this week (bye, ruled out, or small role)"};
         const pts = f.std + k*f.rec, s = f.ppr>0 ? pts/f.ppr : 1;
         return {player:n, pos:f.pos, team:f.tm, opp:f.opp, projected:r1(pts), likely_range:[r1(f.lo*s), r1(f.hi*s)], matchup:matchupText(f.pos,f.mrank), questionable:!!f.q,
           last_3_ppr:(D.logs[n]||[]).slice(-3).map(x=>x[16]), key_stats:keyLine(n)}; }); }},
    {name:"get_fantasy_rankings", description:"This week's fantasy rankings by projected points. Position QB, RB, WR, TE or FLEX; scoring ppr, half or standard.",
     inputSchema:{type:"object", properties:{position:{type:"string"}, scoring:{type:"string", enum:["ppr","half","standard"]}, limit:{type:"number"}}},
     execute(i){ status("Pulling fantasy rankings"); const pos = String(i.position||"").toUpperCase(), k = {ppr:1, half:.5, standard:0}[String(i.scoring||"ppr")] ?? 1;
       return FS.filter(f=>!pos || (pos==="FLEX" ? f.pos!=="QB" : f.pos===pos)).map(f=>({player:f.n, pos:f.pos, team:f.tm, opp:f.opp, pts:r1(f.std+k*f.rec), matchup:matchupText(f.pos,f.mrank)}))
         .sort((a,b)=>b.pts-a.pts).slice(0, Math.min(Number(i.limit)||15, 40)); }},
    {name:"get_rising_usage", description:"Running backs and receivers whose role grew over their last two games (waiver-wire targets).",
     execute(){ status("Checking usage trends"); return (D.risers||[]).slice(0,15).map(r=>({player:r.n, pos:r.pos, team:r.tm, touches_targets_per_game:`${r.opp_b} to ${r.opp_l}`, snap_pct:`${r.snap_b} to ${r.snap_l}`, ppr_last_2:r.ppr_l})); }},
    {name:"get_playoff_odds", description:"Playoff odds from 10,000 season simulations, plus power ratings (points better than an average team). Optional team or conference filter.",
     inputSchema:{type:"object", properties:{team:{type:"string"}, conference:{type:"string", enum:["AFC","NFC"]}}},
     execute(i){ status("Checking playoff odds"); const tm = findTeam(i.team), c = String(i.conference||"").toUpperCase();
       return POD.filter(t=>tm ? t.t===tm : c ? t.conf===c : true).sort((a,b)=>b.sb-a.sb).slice(0, tm?1:16)
         .map(t=>({team:t.name, record:`${t.w}-${t.l}${t.ties?"-"+t.ties:""}`, projected_wins:t.pw, playoffs_pct:t.playoffs, division_pct:t.division, top_seed_pct:t.bye, conference_pct:t.cchamp, super_bowl_pct:t.sb, power_rating:t.rating})); }},
    {name:"get_model_record", description:"How the model has done: this season's record picking winners, against the spread and on totals (overall and on 3+ point edges), and its tested accuracy on past seasons.",
     execute(){ status("Checking the model's record"); const r = SEASON_REC;
       return {season:D.season, winners:wl(r.su), spread:wl(r.ats), spread_3plus_edges:wl(r.ats3), totals:wl(r.ou), totals_3plus_edges:wl(r.ou3),
         past_seasons_test:{seasons:GR.seasons, winners_pct:GR.acc, vegas_favorites_pct:GR.vegas_acc, spread_3plus_edge_pct:GR.ats3, totals_pct:GR.ou},
         break_even_note:"At standard -110 odds you need 52.4% to break even."}; }},
  ];
  const RULES = `You are the assistant built into "Model Board", a personal NFL prediction site. Today is ${new Date().toDateString()}; it's Week ${PM.week} of the ${D.season} season, and the site's data was updated ${D.updated}.
Use the tools to get this site's model numbers: projections, edges, lines, fantasy rankings, playoff odds and records. Never invent stats, lines or odds; if a tool doesn't have something, say so.
For general NFL questions (rules, history, strategy, how stats work), answer from your own knowledge, but say plainly that you can't see news newer than your training data or injury updates after ${D.updated}.
Best bets: rank by edge, give the line, odds and book when known, and say why the model likes it in one short clause. Remind the user once, briefly, that edges are estimates, the model isn't guaranteed, and big edges often mean the model is missing news like an injury. Never pressure anyone to bet or to bet more.
Start/sit: use compare_start_sit, pick one player, and say how close the call is.
Style: short and direct, plain language, short bullet lists for multiple picks, no tables, no headings, no emoji.`;
  // ---- UI ----
  const fab = document.createElement("button"); fab.type = "button"; fab.className = "ask-fab"; fab.hidden = true;
  fab.setAttribute("aria-expanded","false"); fab.setAttribute("aria-controls","askPanel"); fab.textContent = "Ask the model";
  const panel = document.createElement("section"); panel.id = "askPanel"; panel.className = "ask-panel"; panel.hidden = true;
  panel.setAttribute("role","dialog"); panel.setAttribute("aria-label","Ask the model");
  panel.innerHTML = `<div class="ask-head"><div><b>Ask the model</b><small class="ask-sub">Answers use this week's model data.</small></div><button type="button" class="ask-x" aria-label="Close">×</button></div>
    <div class="ask-log" aria-live="polite"><div class="ask-empty"><p>Ask about this week's best bets, start/sit decisions, any player or team, or anything NFL.</p>
    <div class="ask-sugs">${["What are the best bets this week?","Who should I start at flex this week?","Which teams are the best bets to make the playoffs?","How has the model done this season?"].map(s=>`<button type="button">${s}</button>`).join("")}</div></div></div>
    <p class="ask-status" hidden></p>
    <form class="ask-form"><textarea rows="2" placeholder="Should I start Zay Flowers or Chris Olave?" aria-label="Your question"></textarea>
    <div class="ask-actions"><button type="button" class="btn ghost ask-new">New chat</button><button type="button" class="btn ghost ask-stop" hidden>Stop</button><button type="submit" class="btn ask-send">Send</button></div></form>`;
  document.body.append(fab, panel);
  const log = panel.querySelector(".ask-log"), ta = panel.querySelector("textarea"), stat = panel.querySelector(".ask-status");
  const sendB = panel.querySelector(".ask-send"), stopB = panel.querySelector(".ask-stop");
  function status(t){ stat.hidden = !t; stat.textContent = t ? t + "…" : ""; }
  function open(v){ panel.hidden = !v; fab.setAttribute("aria-expanded", String(v)); fab.classList.toggle("on", v); if (v) ta.focus(); }
  fab.onclick = () => open(panel.hidden);
  panel.querySelector(".ask-x").onclick = () => { open(false); fab.focus(); };
  document.addEventListener("keydown", e => { if (e.key==="Escape" && !panel.hidden){ open(false); fab.focus(); } });
  panel.querySelector(".ask-new").onclick = () => { if (ASK.busy) return; ASK.turns = []; log.innerHTML = ""; status(""); ta.focus(); };
  stopB.onclick = () => ASK.ctl && ASK.ctl.abort();
  ta.addEventListener("keydown", e => { if (e.key==="Enter" && !e.shiftKey){ e.preventDefault(); panel.querySelector("form").requestSubmit(); } });
  log.querySelectorAll(".ask-sugs button").forEach(b => b.onclick = () => ask(b.textContent));
  panel.querySelector("form").addEventListener("submit", e => { e.preventDefault(); const q = ta.value.trim(); if (!q){ ta.focus(); return; } ask(q); });
  function bubble(role, html){ const d = document.createElement("div"); d.className = "ask-msg " + role; d.innerHTML = html; log.append(d); log.scrollTop = log.scrollHeight; return d; }
  const COPY = {rate_limited:"You've hit a usage limit for now. Try again in a little while.", session_expired:"Sign in to Claude again, then retry.",
    refused:"Claude didn't answer that one. Try asking it a different way.", empty_completion:"No answer came back. Try a shorter or simpler question.",
    prompt_too_large:"That conversation got too long. Start a new chat.", upstream_error:"The answer was interrupted. Try again."};
  let SAMPLE = null, API = (D.assistant_url || "").trim();
  const API_TOOLS = TOOLS.map(t => ({name:t.name, description:t.description, input_schema:t.inputSchema || {type:"object", properties:{}}}));
  async function callApi(messages, round, signal){
    let r;
    try {
      r = await fetch(API, {method:"POST", headers:{"Content-Type":"application/json"}, signal,
        body: JSON.stringify({system: RULES, messages, tools: API_TOOLS, round})});
    } catch(e){ if (e.name==="AbortError") throw {code:"cancelled"}; throw {code:"upstream_error", message:"Couldn't reach the assistant. Check your connection and try again."}; }
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw {code: r.status===429 ? "rate_limited" : "upstream_error", message: d.error};
    return d;
  }
  async function askApi(q, out, signal){
    const hist = ASK.turns.slice(0, -1).slice(-8);
    while (hist.length && hist[0].role !== "user") hist.shift();
    const msgs = [...hist, {role:"user", content:q}];
    for (let round = 0; round < 6; round++){
      const d = await callApi(msgs, round, signal);
      const blocks = Array.isArray(d.content) ? d.content : [];
      const said = blocks.filter(b=>b.type==="text").map(b=>b.text).join("\n\n").trim();
      if (said) out.innerHTML = md(said);
      if (d.stop_reason !== "tool_use") return said;
      msgs.push({role:"assistant", content:blocks});
      const results = [];
      for (const b of blocks.filter(b=>b.type==="tool_use")){
        const tool = TOOLS.find(t=>t.name===b.name);
        try { const res = tool ? await tool.execute(b.input || {}, {signal}) : (()=>{ throw new Error("Unknown tool"); })();
          results.push({type:"tool_result", tool_use_id:b.id, content: JSON.stringify(res).slice(0, 30000)}); }
        catch(e){ results.push({type:"tool_result", tool_use_id:b.id, content: "Error: " + (e.message || e), is_error:true}); }
      }
      msgs.push({role:"user", content:results});
    }
    return "I looked at a lot of data for that one and ran out of steps. Try asking something narrower.";
  }
  async function ask(q){
    if (ASK.busy || !(SAMPLE || API)) return;
    const em = log.querySelector(".ask-empty"); if (em) em.remove();
    ta.value = ""; bubble("user", `<p>${esc(q)}</p>`);
    ASK.turns.push({role:"user", content:q});
    const out = bubble("bot", `<p class="ask-think">Thinking…</p>`);
    ASK.busy = true; sendB.disabled = true; stopB.hidden = false; ASK.ctl = new AbortController();
    try {
      let text, truncated = false;
      if (SAMPLE){
        ({text, truncated} = await SAMPLE([{role:"user", content:RULES}, {role:"assistant", content:"Got it."}, ...ASK.turns.slice(-10)], {
          signal: ASK.ctl.signal, tools: TOOLS, modelTier: "default",
          onText: ({text}) => { status(""); out.innerHTML = md(text); log.scrollTop = log.scrollHeight; }}));
      } else {
        text = await askApi(q, out, ASK.ctl.signal);
      }
      out.innerHTML = md(text) + (truncated ? `<p class="ask-note">The answer was cut short. Ask for less at once.</p>` : "");
      ASK.turns.push({role:"assistant", content:text});
    } catch(e){
      ASK.turns.pop();
      const hide = ["not_granted","sampling_disabled","not_declared","capability_disabled","capability_removed","tools_unavailable"].includes(e.code);
      if (e.code==="cancelled"){ out.innerHTML = e.text ? md(e.text) + `<p class="ask-note">Stopped.</p>` : `<p class="ask-note">Stopped.</p>`; }
      else if (hide){ out.innerHTML = `<p class="ask-note">The assistant isn't available here. Allowing it to use your Claude account turns it on.</p>`; if (e.code!=="not_granted"){ fab.hidden = true; } }
      else if (e.code==="refused"){ out.innerHTML = `<p class="ask-note">${COPY.refused}</p>`; }
      else { out.innerHTML = (e.text ? md(e.text) : "") + `<p class="ask-note">${esc(e.message || COPY[e.code] || COPY.upstream_error)}</p>`; }
    } finally {
      ASK.busy = false; sendB.disabled = false; stopB.hidden = true; status(""); log.scrollTop = log.scrollHeight;
    }
  }
  if (API){ fab.hidden = false; panel.querySelector(".ask-sub").textContent = "Powered by Claude. Answers use this week's model data."; }
  if (window.claude && typeof window.claude.use === "function"){
    window.claude.use("sample").then(s => { if (!s) return; SAMPLE = s; fab.hidden = false;
      panel.querySelector(".ask-sub").textContent = "Answers use your Claude account and this week's model data."; }).catch(()=>{});
  }
})();
</script>
</body>
</html>

#__END__
'''
