"""
NFL Model Board builder
=======================
Downloads the latest NFL data, runs the game model, projects every skill
player's stat line for the upcoming week, and writes a website file:
nfl_model_board.html

How to run (from Command Prompt, in the folder with this file):
    py nfl_site.py
Then double-click nfl_model_board.html to open it in your browser.
Re-run it each week to refresh lines, results, projections, and stats.

Optional: to fill in sportsbook prop lines automatically, get a free key at
https://the-odds-api.com and paste it into ODDS_API_KEY below.
"""
import os
import numpy as np, pandas as pd, nflreadpy as nfl, json
from scipy.stats import norm
from sklearn.linear_model import Ridge

FIRST_SEASON, CURRENT_SEASON = 2015, 2026
# Optional: paste a free key from https://the-odds-api.com to pull real prop
# lines automatically. On GitHub, add it as a secret named ODDS_API_KEY instead.
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")   # or paste your key between the quotes
ROLLING_GAMES, MARGIN_STD = 10, 13.5
seasons = list(range(FIRST_SEASON, CURRENT_SEASON + 1))

# ======================= PLAYER PROJECTIONS =======================
# Predicts each skill player's stat line for the upcoming week as a range
# (10th to 90th percentile), then compares it with sportsbook prop lines.
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from scipy.stats import poisson
import re, urllib.request, urllib.parse

PROP_START = 2018
QUANTS = [0.1, 0.25, 0.5, 0.75, 0.9]
PROPS = {  # stat: (positions, usage stat, model type, Odds API market, label)
    "passing_yards":   (["QB"], "attempts", "q", "player_pass_yds", "Pass yds"),
    "passing_tds":     (["QB"], "attempts", "pois", "player_pass_tds", "Pass TDs"),
    "rushing_yards":   (["QB", "RB"], "carries", "q", "player_rush_yds", "Rush yds"),
    "receptions":      (["RB", "WR", "TE"], "targets", "q", "player_receptions", "Receptions"),
    "receiving_yards": (["RB", "WR", "TE"], "targets", "q", "player_reception_yds", "Rec yds"),
    "tds":             (["RB", "WR", "TE"], "touches", "pois", "player_anytime_td", "Anytime TD"),
}
HGB = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80, random_state=0)

def prop_features(sched, week_games, injuries):
    yrs = list(range(PROP_START, CURRENT_SEASON + 1))
    ps = nfl.load_player_stats(yrs).to_pandas()
    ps = ps[(ps.season_type == "REG") & ps.position.isin(["QB", "RB", "WR", "TE"])]
    ps = ps[["player_id", "player_display_name", "position", "season", "week", "game_id", "team",
             "opponent_team", "attempts", "passing_yards", "passing_tds", "carries", "rushing_yards",
             "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds", "target_share"]].copy()
    ps["tds"] = ps.rushing_tds + ps.receiving_tds
    ps["touches"] = ps.carries + ps.targets
    # A QB's passing numbers only count from games he really played (10+ attempts)
    backup = (ps.position == "QB") & (ps.attempts < 10)
    ps.loc[backup, ["attempts", "passing_yards", "passing_tds"]] = np.nan

    # Who plays this week: players active in their team's last 2 games,
    # minus anyone listed Out/Doubtful. QBs: only the listed starter.
    out = set(injuries.loc[injuries.report_status.isin(["Out", "Doubtful"]), "gsis_id"])
    cur = ps[ps.season == CURRENT_SEASON]
    last_wk = cur.groupby("team").week.max()
    recent = cur[cur.week >= cur.team.map(last_wk) - 1].drop_duplicates("player_id", keep="last")
    # Starting QB: the schedule's listed starter unless he's Out/Doubtful,
    # otherwise whoever threw the most passes for that team most recently.
    qb_ids = (ps[ps.position == "QB"].sort_values(["season", "week"])
                .drop_duplicates("player_display_name", keep="last")
                .set_index("player_display_name").player_id)
    qb_pool = cur[(cur.position == "QB") & cur.attempts.notna() & ~cur.player_id.isin(out)]
    def starting_qb(tm, listed_name):
        pid = qb_ids.get(listed_name) if isinstance(listed_name, str) else None
        if pid is not None and pid not in out:
            return pid, listed_name
        grp = qb_pool[qb_pool.team == tm].sort_values(["week", "attempts"])
        if len(grp):
            return grp.player_id.iloc[-1], grp.player_display_name.iloc[-1]
        return None, None
    rows = []
    for _, g in week_games.iterrows():
        for tm, opp, qbn in [(g.home_team, g.away_team, g.home_qb_name), (g.away_team, g.home_team, g.away_qb_name)]:
            base = dict(season=CURRENT_SEASON, week=int(g.week), game_id=g.game_id, team=tm, opponent_team=opp)
            pid, name = starting_qb(tm, qbn)
            if pid is not None:
                rows.append(dict(player_id=pid, player_display_name=name, position="QB", **base))
            for _, p in recent[(recent.team == tm) & (recent.position != "QB")].iterrows():
                if p.player_id not in out:
                    rows.append(dict(player_id=p.player_id, player_display_name=p.player_display_name,
                                     position=p.position, **base))
    up = pd.DataFrame(rows)
    up["upcoming"] = True
    ps["upcoming"] = False
    ps = pd.concat([ps, up], ignore_index=True).sort_values(["player_id", "season", "week"])

    # Each player's recent averages BEFORE each game (recent games count more)
    for st in set(PROPS) | {"attempts", "carries", "targets", "touches", "target_share"}:
        ps[f"p_{st}"] = ps.groupby("player_id")[st].transform(
            lambda s: s.shift(1).ewm(span=8, min_periods=1).mean())
    ps["n_prior"] = ps.groupby("player_id").cumcount()
    ps["new_season"] = (ps.season != ps.groupby("player_id").season.shift(1)).astype(int)
    ps["pos_code"] = ps.position.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3})

    # What each defense has recently allowed to each position
    allow = (ps.groupby(["game_id", "opponent_team", "position", "season", "week"])[list(PROPS)]
               .sum(min_count=1).reset_index()
               .sort_values(["opponent_team", "position", "season", "week"]))
    for st in PROPS:
        allow[f"d_{st}"] = allow.groupby(["opponent_team", "position"])[st].transform(
            lambda s: s.shift(1).ewm(span=10, min_periods=1).mean())
    ps = ps.merge(allow[["game_id", "opponent_team", "position"] + [f"d_{s}" for s in PROPS]],
                  on=["game_id", "opponent_team", "position"], how="left")

    # Vegas implied team points and spread (projects game script and volume)
    h = pd.DataFrame({"game_id": sched.game_id, "team": sched.home_team, "team_spread": sched.spread_line,
                      "implied": (sched.total_line + sched.spread_line) / 2})
    a = pd.DataFrame({"game_id": sched.game_id, "team": sched.away_team, "team_spread": -sched.spread_line,
                      "implied": (sched.total_line - sched.spread_line) / 2})
    return ps.merge(pd.concat([h, a]), on=["game_id", "team"], how="left")

def feature_list(st):
    poss, use = PROPS[st][0], PROPS[st][1]
    F = [f"p_{st}", f"p_{use}", f"d_{st}", "implied", "team_spread", "n_prior", "new_season", "pos_code"]
    if st in ("receptions", "receiving_yards", "tds"):
        F.append("p_target_share")
    return F

def fit_predict(st, tr, te):
    """Returns quantile predictions (dict) or Poisson means for rows of te."""
    F = feature_list(st)
    if PROPS[st][2] == "pois":
        m = HistGradientBoostingRegressor(loss="poisson", **HGB).fit(tr[F], tr[st])
        return m.predict(te[F])
    return {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **HGB)
               .fit(tr[F], tr[st]).predict(te[F]) for q in QUANTS}

def build_projections(sched, week_games, injuries):
    ps = prop_features(sched, week_games, injuries)
    hist = ps[~ps.upcoming & (ps.n_prior >= 3)]
    up = ps[ps.upcoming].copy()
    test_season = CURRENT_SEASON - 1
    calib, report = {}, {}
    for st, (poss, use, kind, _, label) in PROPS.items():
        d = hist[hist.position.isin(poss)].dropna(subset=[st, f"p_{st}", f"p_{use}", "implied"])
        # 1) Honest test: train on older seasons, grade on last season
        tr, te = d[d.season < test_season], d[d.season == test_season]
        pred = fit_predict(st, tr, te)
        if kind == "pois":
            ks = [1] if st == "tds" else [1, 2, 3]
            calib[st] = {}
            for k in ks:   # calibrate P(at least k) against what really happened
                p = 1 - poisson.cdf(k - 1, pred)
                iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(p, te[st] >= k)
                xs = np.linspace(0, 1, 21)
                calib[st][k] = [round(float(v), 3) for v in iso.predict(xs)]
            report[st] = dict(label=label, n=int(len(te)))
        else:
            cover = [float((te[st] <= pred[q]).mean()) for q in QUANTS]
            calib[st] = [0.0] + [round(c, 3) for c in cover] + [1.0]
            report[st] = dict(label=label, n=int(len(te)),
                              mae=round(float(np.mean(np.abs(pred[0.5] - te[st]))), 1),
                              base=round(float(np.mean(np.abs(te[f"p_{st}"] - te[st]))), 1))
        # 2) Final model: train on everything, predict this week
        mask = up.position.isin(poss) & up[f"p_{st}"].notna() & up[f"p_{use}"].notna()
        if mask.any():
            out = fit_predict(st, d, up[mask])
            if kind == "pois":
                up.loc[mask, f"{st}_mean"] = out
            else:
                qs = np.sort(np.column_stack([out[q] for q in QUANTS]), axis=1)  # keep quantiles in order
                for i, q in enumerate(QUANTS):
                    up.loc[mask, f"{st}_q{int(q*100)}"] = qs[:, i]
    return up, calib, report

# ---------- Optional: pull real prop lines from The Odds API ----------
def norm_name(n):
    n = re.sub(r"[^a-z ]", "", str(n).lower().replace("-", " "))
    return " ".join(w for w in n.split() if w not in ("jr", "sr", "ii", "iii", "iv", "v"))

def fetch_prop_lines(api_key, week_games):
    """Returns {(player name, stat): {line, over, under, book}} for this week's games."""
    if not api_key:
        return {}
    base = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
    teams = nfl.load_teams().to_pandas()
    full = dict(zip(teams.team_name, teams.team_abbr))
    want = {(g.away_team, g.home_team) for _, g in week_games.iterrows()}
    mk = {v[3]: k for k, v in PROPS.items()}
    book_rank = ["draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "hardrockbet"]
    lines = {}
    try:
        with urllib.request.urlopen(f"{base}/events?apiKey={api_key}", timeout=30) as r:
            events = json.load(r)
        for ev in events:
            if (full.get(ev["away_team"]), full.get(ev["home_team"])) not in want:
                continue
            q = urllib.parse.urlencode(dict(apiKey=api_key, regions="us", oddsFormat="american",
                                            markets=",".join(mk)))
            with urllib.request.urlopen(f"{base}/events/{ev['id']}/odds?{q}", timeout=30) as r:
                odds = json.load(r)
            books = sorted(odds.get("bookmakers", []),
                           key=lambda b: book_rank.index(b["key"]) if b["key"] in book_rank else 99)
            for b in books:
                for m in b.get("markets", []):
                    st = mk.get(m["key"])
                    if not st:
                        continue
                    for o in m.get("outcomes", []):
                        key = (norm_name(o.get("description", "")), st)
                        rec = lines.setdefault(key, {"book": b["title"]})
                        if rec["book"] != b["title"]:
                            continue  # keep one book per prop so line and odds match
                        side = o.get("name", "")
                        if side in ("Over", "Yes"):
                            rec["line"] = o.get("point", 0.5)
                            rec["over"] = o.get("price")
                        elif side in ("Under", "No"):
                            rec["under"] = o.get("price")
        print(f"   Pulled {len(lines)} prop lines from The Odds API.")
    except Exception as e:
        print(f"   Could not get prop lines from The Odds API ({e}). Continuing without them.")
    return lines

TEMPLATE = open(__file__, encoding="utf-8").read().split("#" + "__TEMPLATE__\n", 1)[1].rsplit("\n#__END__", 1)[0]

PROP_CACHE = "prop_lines_cache.json"
def get_prop_lines(api_key, week_games, week):
    """Saves credits: pulls lines once per week and reuses them on other runs.
    On GitHub it pulls on Saturday/Sunday (or when you tick 'fetch props' on a
    manual run). On your own computer it pulls once per week the first time you run it."""
    cache = {}
    try:
        with open(PROP_CACHE, encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        pass
    have = cache.get("season") == CURRENT_SEASON and cache.get("week") == week
    on_github = bool(os.environ.get("GITHUB_ACTIONS"))
    forced = os.environ.get("FETCH_PROPS", "").lower() in ("1", "true", "yes")
    weekend = pd.Timestamp.now(tz="UTC").dayofweek in (5, 6)
    should_fetch = api_key and (forced or (not have and (weekend or not on_github)))
    if should_fetch:
        lines = fetch_prop_lines(api_key, week_games)
        if lines:
            cache = dict(season=CURRENT_SEASON, week=week,
                         lines=[[k[0], k[1], v] for k, v in lines.items()])
            with open(PROP_CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            return lines
    if cache.get("season") == CURRENT_SEASON and cache.get("week") == week:
        print(f"   Using saved prop lines for week {week}.")
        return {(n, st): v for n, st, v in cache["lines"]}
    return {}

print("Step 1/6: Downloading schedules and play-by-play (a few minutes)...")
games = nfl.load_schedules(seasons).to_pandas()
games = games[games["game_type"] == "REG"].copy()
cols = ["game_id","season","week","posteam","defteam","play_type","epa","success"]
pbp = pd.concat([nfl.load_pbp([y]).select(cols).to_pandas() for y in seasons], ignore_index=True)

print("Step 2/6: Building team form...")
plays = pbp[pbp.play_type.isin(["pass","run"]) & pbp.epa.notna()].copy()
plays["pass_epa"] = np.where(plays.play_type=="pass", plays.epa, np.nan)
plays["rush_epa"] = np.where(plays.play_type=="run", plays.epa, np.nan)
def summarize(side):
    return (plays.groupby(["game_id","season","week",side])
            .agg(epa=("epa","mean"), pass_epa=("pass_epa","mean"),
                 rush_epa=("rush_epa","mean"), success=("success","mean"))
            .reset_index().rename(columns={side:"team"}))
off = summarize("posteam"); dfn = summarize("defteam")
tg = off.merge(dfn, on=["game_id","season","week","team"], suffixes=("_o","_d"))
tg = tg.rename(columns={"epa_o":"off_epa","pass_epa_o":"off_pass_epa","rush_epa_o":"off_rush_epa","success_o":"off_success",
                        "epa_d":"def_epa","pass_epa_d":"def_pass_epa","rush_epa_d":"def_rush_epa","success_d":"def_success"})
S = ["off_epa","off_pass_epa","off_rush_epa","off_success","def_epa","def_pass_epa","def_rush_epa","def_success"]
tg = tg.sort_values(["team","season","week"])
form = tg.copy()
form[S] = tg.groupby("team")[S].transform(lambda s: s.shift(1).ewm(span=ROLLING_GAMES, min_periods=3).mean())
latest = tg.groupby("team")[S].apply(lambda d: d.ewm(span=ROLLING_GAMES).mean().iloc[-1]).reset_index()

def attach(df, side):
    tc = f"{side}_team"
    m = df.merge(form[["game_id","team"]+S], left_on=["game_id",tc], right_on=["game_id","team"], how="left")
    un = m[S[0]].isna() & m["result"].isna()
    fill = m.loc[un,[tc]].merge(latest, left_on=tc, right_on="team", how="left")
    m.loc[un, S] = fill[S].values
    return m.drop(columns="team").rename(columns={c:f"{side}_{c}" for c in S})
df = attach(attach(games, "home"), "away")

print("Step 3/6: Training the game model...")
df["pass_matchup"] = df.home_off_pass_epa - df.away_off_pass_epa - (df.home_def_pass_epa - df.away_def_pass_epa)
df["rush_matchup"] = df.home_off_rush_epa - df.away_off_rush_epa - (df.home_def_rush_epa - df.away_def_rush_epa)
df["success_matchup"] = df.home_off_success - df.away_off_success - (df.home_def_success - df.away_def_success)
df["rest_diff"] = (df.home_rest - df.away_rest).fillna(0)
df["neutral_site"] = (df.location == "Neutral").astype(int)
df["div_game"] = df.div_game.fillna(0)
indoor = df.roof.isin(["dome","closed"])
df["wind"] = np.where(indoor, 0, df.wind.fillna(df.wind.median()))
df["high_wind"] = (df.wind >= 15).astype(int)
F = ["pass_matchup","rush_matchup","success_matchup","rest_diff","neutral_site","div_game","high_wind"]
df = df.dropna(subset=F)
played = df[df.result.notna()]
model = Ridge(alpha=1.0).fit(played[played.season < CURRENT_SEASON][F], played[played.season < CURRENT_SEASON].result)
cur = df[df.season == CURRENT_SEASON].copy()
cur["pred"] = model.predict(cur[F])   # 2026 games graded out-of-sample
# Future games: refit on everything played so far
model_all = Ridge(alpha=1.0).fit(played[F], played.result)
fut = cur.result.isna()
cur.loc[fut, "pred"] = model_all.predict(cur.loc[fut, F])
cur["prob"] = norm.cdf(cur.pred / MARGIN_STD)

# team form ranks for the matchup panel (latest form, 1 = best)
lt = latest.set_index("team")
lt = lt[lt.index.isin(set(games[games.season == CURRENT_SEASON].home_team))]
ranks = pd.DataFrame({
    "off": lt.off_epa.rank(ascending=False), "def": lt.def_epa.rank(ascending=True),
    "pass_off": lt.off_pass_epa.rank(ascending=False), "rush_off": lt.off_rush_epa.rank(ascending=False),
    "pass_def": lt.def_pass_epa.rank(ascending=True), "rush_def": lt.def_rush_epa.rank(ascending=True)}).astype(int)
teams = {t: dict(offv=round(lt.off_epa[t],3), defv=round(lt.def_epa[t],3), **{k:int(v) for k,v in ranks.loc[t].items()}) for t in lt.index}

def nz(v, d=1):
    return None if pd.isna(v) else round(float(v), d)
season_games = []
for _, g in cur.sort_values(["week","gameday","gametime"]).iterrows():
    season_games.append(dict(
        id=g.game_id, wk=int(g.week), date=g.gameday, time=g.gametime, day=g.weekday,
        away=g.away_team, home=g.home_team, aqb=g.away_qb_name if isinstance(g.away_qb_name,str) else None, hqb=g.home_qb_name if isinstance(g.home_qb_name,str) else None,
        as_=nz(g.away_score,0), hs=nz(g.home_score,0),
        spread=nz(g.spread_line), total=nz(g.total_line),
        hml=nz(g.home_moneyline,0), aml=nz(g.away_moneyline,0),
        pred=nz(g.pred), prob=nz(g.prob,3), roof=g.roof if isinstance(g.roof,str) else None))

print("Step 4/6: Building player advanced stats...")
ps = nfl.load_player_stats([CURRENT_SEASON]).to_pandas()
ps = ps[(ps.season_type == "REG") & ps.position.isin(["QB","RB","WR","TE"])]
sc = nfl.load_snap_counts([CURRENT_SEASON]).to_pandas()
ro = nfl.load_rosters([CURRENT_SEASON]).to_pandas()[["gsis_id","pfr_id"]].dropna().drop_duplicates("gsis_id")
sc = sc.merge(ro, left_on="pfr_player_id", right_on="pfr_id")[["gsis_id","week","offense_pct"]]
ps = ps.merge(sc, left_on=["player_id","week"], right_on=["gsis_id","week"], how="left")

def s(col): return ps.groupby(["player_id"])[col].sum()
g = ps.groupby("player_id")
agg = pd.DataFrame({
    "name": g.player_display_name.last(), "pos": g.position.last(), "team": g.team.last(),
    "gp": g.week.nunique(), "snap": g.offense_pct.mean(),
    "att": s("attempts"), "cmp": s("completions"), "pyd": s("passing_yards"), "ptd": s("passing_tds"),
    "int": s("passing_interceptions"), "sacks": s("sacks_suffered"), "pepa": s("passing_epa"),
    "pair": s("passing_air_yards"), "car": s("carries"), "ryd": s("rushing_yards"), "rtd": s("rushing_tds"),
    "repa": s("rushing_epa"), "rfd": s("rushing_first_downs"), "tgt": s("targets"), "rec": s("receptions"), "recyd": s("receiving_yards"),
    "rectd": s("receiving_tds"), "rair": s("receiving_air_yards"), "yac": s("receiving_yards_after_catch"),
    "recepa": s("receiving_epa"), "ppr": s("fantasy_points_ppr"),
    "tshare": g.target_share.mean(), "ashare": g.air_yards_share.mean(), "wopr": g.wopr.mean(),
})
# attempt-weighted CPOE
cp = ps.dropna(subset=["passing_cpoe"])
agg["cpoe"] = cp.groupby("player_id").apply(lambda d: np.average(d.passing_cpoe, weights=d.attempts.clip(lower=1)))
agg = agg.reset_index()
keep = ((agg.pos=="QB") & (agg.att >= 15)) | ((agg.pos=="RB") & (agg.car + agg.tgt >= 8)) | (agg.pos.isin(["WR","TE"]) & (agg.tgt >= 4))
agg = agg[keep]

def sd(a, b): return a / b.where(b > 0)
players = []
for _, p in agg.iterrows():
    gp = p.gp
    d = dict(n=p["name"], pos=p.pos, tm=p.team, gp=int(gp), snap=nz(p.snap*100 if pd.notna(p.snap) else np.nan, 0), ppg=nz(p.ppr/gp))
    if p.pos == "QB":
        db = p.att + p.sacks
        d.update(epa=nz(p.pepa/db,2), cpoe=nz(p.cpoe), adot=nz(p.pair/p.att), ypa=nz(p.pyd/p.att),
                 td=int(p.ptd), int_=int(p["int"]), ypg=nz(p.pyd/gp,0), rypg=nz(p.ryd/gp,0))
    elif p.pos == "RB":
        d.update(cpg=nz(p.car/gp), ypc=nz(p.ryd/p.car) if p.car else None, repa=nz(p.repa/p.car,2) if p.car else None,
                 tpg=nz(p.tgt/gp), ts=nz(p.tshare*100), ypg=nz((p.ryd+p.recyd)/gp,0), td=int(p.rtd+p.rectd))
    else:
        d.update(tpg=nz(p.tgt/gp), ts=nz(p.tshare*100), as_=nz(p.ashare*100), wopr=nz(p.wopr,2),
                 adot=nz(p.rair/p.tgt), racr=nz(p.recyd/p.rair,2) if p.rair>0 else None,
                 yac=nz(p.yac/p.rec) if p.rec else None, epat=nz(p.recepa/p.tgt,2),
                 ypg=nz(p.recyd/gp,0), td=int(p.rectd), cr=nz(p.rec/p.tgt*100,0))
    players.append(d)

print("Step 5/6: Projecting player stat lines for the upcoming week...")
open_games = cur[cur.result.isna()]
props, prop_meta = [], {}
if len(open_games):
    pwk = int(open_games.week.min())
    week_games = open_games[open_games.week == pwk]
    try:
        inj = nfl.load_injuries([CURRENT_SEASON]).to_pandas()
        inj = inj[inj.week == pwk]
    except Exception:
        inj = pd.DataFrame(columns=["gsis_id", "report_status"])
    quest = set(inj.loc[inj.report_status == "Questionable", "gsis_id"])
    up, calib, report = build_projections(games[games.season >= PROP_START], week_games, inj)
    book = get_prop_lines(ODDS_API_KEY, week_games, pwk)
    game_order = {gid: i for i, gid in enumerate(week_games.sort_values(["gameday", "gametime"]).game_id)}
    for _, r in up.iterrows():
        for st, (poss, use, kind, _, label) in PROPS.items():
            if r.position not in poss:
                continue
            e = dict(n=r.player_display_name, pos=r.position, tm=r.team, opp=r.opponent_team,
                     gid=r.game_id, go=game_order.get(r.game_id, 99), st=st, q=r.player_id in quest)
            if kind == "pois":
                mu = r.get(f"{st}_mean")
                if pd.isna(mu) or (st == "tds" and mu < 0.08):
                    continue
                e["mu"] = round(float(mu), 3)
            else:
                qs = [r.get(f"{st}_q{int(q*100)}") for q in QUANTS]
                if any(pd.isna(v) for v in qs):
                    continue
                med = qs[2]
                if (st == "rushing_yards" and med < 6) or (st == "receiving_yards" and med < 8) \
                        or (st == "receptions" and med < 1):
                    continue
                e["qs"] = [round(float(v), 1) for v in qs]
            bl = book.get((norm_name(r.player_display_name), st))
            if bl and "line" in bl:
                e.update(line=bl["line"], over=bl.get("over"), under=bl.get("under"), book=bl["book"])
            props.append(e)
    prop_meta = dict(week=pwk, calib=calib, report=report, has_book=bool(book))

# 2026 model record so far
done = cur[cur.result.notna() & cur.spread_line.notna()]
done = done[done.result != done.spread_line]
side_home = done.pred > done.spread_line
ats = int(((done.result > done.spread_line) == side_home).sum())
su = int(((done.pred > 0) == (done.result > 0)).sum())
data = dict(season=CURRENT_SEASON, games=season_games, players=players, teams=teams, props=props, prop_meta=prop_meta,
            record=dict(ats_w=ats, ats_l=int(len(done)-ats), su_w=su, su_l=int(len(done)-su)),
            updated=pd.Timestamp.now().strftime("%b %d, %Y"))
data_json = json.dumps(data, default=lambda o: None if pd.isna(o) else o)
print("Step 6/6: Writing the website...")
OUT_FILE = os.environ.get("OUTPUT_FILE", "nfl_model_board.html")
if os.path.dirname(OUT_FILE):
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(TEMPLATE.replace("__DATA__", data_json))
print(f"Done! {len(season_games)} games, {len(players)} players, {len(props)} player projections.")
print("Double-click nfl_model_board.html to open your site.")


'''
#__TEMPLATE__
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>NFL Model Board</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#F2F4EF; --panel:#FAFBF8; --ink:#17231B; --muted:#5E6B62; --line:#C6CFC8; --faint:#E3E8E3;
  --turf:#2E6A47; --turf-soft:rgba(46,106,71,.12); --brass:#9C7019; --loss:#A4412F;
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px);
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --paper:#121A15; --panel:#18221C; --ink:#E6ECE7; --muted:#9AA79E; --line:#34423A; --faint:#1F2A23;
  --turf:#63B887; --turf-soft:rgba(99,184,135,.14); --brass:#E0AE4E; --loss:#E08A74; } }
:root[data-theme="dark"]{
  --paper:#121A15; --panel:#18221C; --ink:#E6ECE7; --muted:#9AA79E; --line:#34423A; --faint:#1F2A23;
  --turf:#63B887; --turf-soft:rgba(99,184,135,.14); --brass:#E0AE4E; --loss:#E08A74; }
*,*::before,*::after{box-sizing:inherit}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"Barlow",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:16px;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:32px 18px 64px}
.cond,h1,h2,h3{font-family:"Barlow Condensed","Arial Narrow",sans-serif}
header{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:12px 24px}
h1{font-size:clamp(44px,8vw,80px);line-height:.9;margin:0;font-weight:700}
.updated{color:var(--muted);font-size:14px;margin:6px 0 0}
.record{display:flex;gap:28px}
.record div{text-align:right}
.record .cond{font-size:34px;font-weight:600;line-height:1}
.record small{display:block;color:var(--muted);font-size:13px}
nav.views{display:flex;gap:4px;margin:28px 0 0;border-bottom:2px solid var(--ink)}
nav.views button{font:600 22px/1 "Barlow Condensed",sans-serif;background:none;border:0;color:var(--muted);padding:10px 14px 9px;cursor:pointer;border-bottom:4px solid transparent;margin-bottom:-2px}
nav.views button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--turf)}
button:focus-visible,select:focus-visible,summary:focus-visible{outline:2px solid var(--turf);outline-offset:2px}
.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px;margin:18px 0 8px}
.toolbar label{font-size:14px;color:var(--muted);display:flex;align-items:center;gap:8px}
select{font:inherit;font-size:15px;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 10px}
.weeks{display:flex;gap:4px;overflow-x:auto;padding:4px 0 8px;scrollbar-width:thin}
.weeks button{font:600 17px/1 "Barlow Condensed",sans-serif;min-width:40px;padding:8px 6px;border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:6px;cursor:pointer}
.weeks button[aria-pressed="true"]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.weeks button.done{color:var(--muted)}
.weeks button.done[aria-pressed="true"]{color:var(--paper)}
/* game cards */
.game{border-bottom:1px solid var(--line)}
.game summary{list-style:none;cursor:pointer;display:grid;grid-template-columns:minmax(150px,1.2fr) repeat(3,minmax(110px,1fr)) 22px;gap:10px 18px;align-items:center;padding:16px 4px}
.game summary::-webkit-details-marker{display:none}
.matchup .cond{font-size:30px;font-weight:700;line-height:1}
.matchup .at{color:var(--muted);font-weight:500;margin:0 4px}
.matchup small,.cell small{display:block;font-size:12.5px;color:var(--muted)}
.cell .v{display:block;font:600 22px/1.15 "Barlow Condensed",sans-serif}
.cell .ml{display:block;font-size:13px;line-height:1.35;color:var(--muted);margin-top:2px}
.edge3 .v{color:var(--turf)}
.res-w{color:var(--turf);font-weight:600}.res-l{color:var(--loss);font-weight:600}
.chev{width:10px;height:10px;border-right:2px solid var(--muted);border-bottom:2px solid var(--muted);transform:rotate(45deg);justify-self:center;transition:transform .15s}
details[open] .chev{transform:rotate(-135deg)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;margin:0 0 20px}
.form{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:8px}
.form h3{font-size:24px;margin:0 0 6px}
.bars{display:grid;grid-template-columns:auto 1fr auto;gap:5px 10px;align-items:center;font-size:13.5px}
.bar{height:8px;background:var(--faint);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--turf);border-radius:4px}
.rk{font:600 15px "Barlow Condensed",sans-serif;text-align:right;min-width:36px}
.pos-block{margin-top:18px}
.pos-block h4{margin:0 0 4px;font-size:14px;font-weight:600;color:var(--muted)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--faint)}
th{font-weight:600;color:var(--muted);font-size:12.5px;cursor:pointer;user-select:none}
th[aria-sort]{color:var(--ink)}
th:first-child,td:first-child{text-align:left;padding-left:0;position:sticky;left:0;background:var(--panel)}
td.tm{color:var(--muted);text-align:left}
tr.team-a td:first-child{box-shadow:inset 3px 0 0 var(--turf);padding-left:8px}
tr.team-b td:first-child{box-shadow:inset 3px 0 0 var(--brass);padding-left:8px}
.hot{color:var(--turf);font-weight:600}
.cold{color:var(--loss)}
.empty{color:var(--muted);font-size:14px;margin:6px 0}
/* season table */
.season-wrap{overflow-x:auto;margin-top:6px}
.season td,.season th{padding:9px 10px}
.season th:first-child,.season td:first-child{background:var(--paper)}
.season tr.wkrow td{background:var(--faint);font:600 16px "Barlow Condensed",sans-serif;color:var(--ink);text-align:left;position:static}
.teamrec{font-size:15px;margin:4px 0 0}
.teamrec b{font-family:"Barlow Condensed",sans-serif;font-size:20px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
.search{position:relative;margin-top:24px;max-width:460px}
.search input{width:100%;font:inherit;font-size:17px;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 14px}
.search input:focus-visible{outline:2px solid var(--turf);outline-offset:1px}
#qres{position:absolute;z-index:5;left:0;right:0;top:calc(100% + 4px);margin:0;padding:4px;list-style:none;background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.12);max-height:360px;overflow-y:auto}
#qres li{padding:8px 10px;border-radius:6px;cursor:pointer;display:flex;justify-content:space-between;gap:12px}
#qres li small{color:var(--muted)}
#qres li[aria-selected="true"]{background:var(--turf-soft)}
#qres li.none{cursor:default;color:var(--muted)}
.pcard{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:16px 0 4px}
.pcard-top{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}
.pcard h3{font-size:30px;margin:0;line-height:1}
.pcard .who{color:var(--muted);font-size:14px;margin:4px 0 0}
.pcard button.clear{font:inherit;font-size:14px;background:none;border:1px solid var(--line);color:var(--ink);border-radius:999px;padding:5px 12px;cursor:pointer}
.pcard table th:first-child,.pcard table td:first-child{background:var(--panel)}
.pcard .empty{margin-top:10px}
.props-intro{max-width:72ch;color:var(--muted);font-size:15px;margin:18px 0 0}
.chk{cursor:pointer}
.chk input{accent-color:var(--turf);width:16px;height:16px}
.props td,.props th{padding:8px 9px;vertical-align:middle}
.props th:first-child,.props td:first-child{background:var(--paper)}
.props td.pl{white-space:normal;min-width:150px}
.props td.pl small{display:block;color:var(--muted);font-size:12.5px}
.tag{display:inline-block;font-size:11px;font-weight:600;color:var(--brass);border:1px solid var(--brass);border-radius:4px;padding:0 4px;margin-left:6px;vertical-align:1px}
.proj{font:600 19px "Barlow Condensed",sans-serif}
.range{display:block;font-size:12px;color:var(--muted)}
.props input{font:inherit;font-size:15px;width:74px;padding:5px 6px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);text-align:right}
.props input.odds{width:62px}
.props input:focus-visible{outline:2px solid var(--turf);outline-offset:1px}
.props input.fromBook{border-color:var(--turf)}
.pick{font:600 17px "Barlow Condensed",sans-serif}
.pick small{display:block;font:500 12px "Barlow",sans-serif;color:var(--muted)}
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
<header>
  <div>
    <h1 id="title">NFL Model Board</h1>
    <p class="updated" id="updated"></p>
  </div>
  <div class="record" id="record"></div>
</header>

<div class="search">
  <label for="q" class="sr">Search players and teams</label>
  <input id="q" type="search" autocomplete="off" placeholder="Search a player or team" role="combobox" aria-expanded="false" aria-controls="qres" aria-autocomplete="list">
  <ul id="qres" role="listbox" hidden></ul>
</div>

<nav class="views" role="tablist">
  <button role="tab" aria-selected="true" data-view="week">Games &amp; players</button>
  <button role="tab" aria-selected="false" data-view="props">Player props</button>
  <button role="tab" aria-selected="false" data-view="season">Season lines</button>
</nav>

<section id="view-week">
  <div class="weeks" id="weeks" role="group" aria-label="Choose week"></div>
  <p class="empty">Tap a game to see both teams' form and every offensive skill player's advanced stats.</p>
  <div id="games"></div>
</section>

<section id="view-props" hidden>
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
</section>

<section id="view-season" hidden>
  <div class="toolbar">
    <label>Team <select id="teamSel"><option value="">All teams</option></select></label>
    <label>Show <select id="showSel"><option value="all">All games</option><option value="done">Finished</option><option value="up">Upcoming</option></select></label>
  </div>
  <p class="teamrec" id="teamrec"></p>
  <div class="season-wrap"><table class="season" id="seasonTbl"></table></div>
</section>

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
  </dl>
  <p class="note">The model's 2024–25 test hit 52.2% against the spread on big edges, just under the 52.4% needed to break even. Use it as a starting point for research, not as a betting system.</p>
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
$("#updated").textContent = D.season + " season. Data through " + D.updated + ".";
$("#record").innerHTML =
  `<div><span class="cond">${D.record.su_w}-${D.record.su_l}</span><small>Model picking winners, ${D.season}</small></div>
   <div><span class="cond">${D.record.ats_w}-${D.record.ats_l}</span><small>Model vs the spread, ${D.season}</small></div>`;

// views
function showView(v){
  document.querySelectorAll("nav.views button").forEach(o => o.setAttribute("aria-selected", o.dataset.view===v));
  ["week","season","props"].forEach(x => $("#view-"+x).hidden = x !== v);
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
  return `<div class="cell res"><small>Final</small><span class="v">${g.away} ${g.as_}, ${g.home} ${g.hs}</span><span class="ml">${ats}${pick? ". "+pick:""}</span></div>`;
}
function formPanel(t){
  const f = D.teams[t]; if(!f) return "";
  const row = (lbl, r) => `<span>${lbl}</span><span class="bar"><i style="width:${Math.round((33-r)/32*100)}%"></i></span><span class="rk">${r}${r===1?"st":r===2?"nd":r===3?"rd":"th"}</span>`;
  return `<div><h3>${t}</h3><div class="bars">${row("Offense",f.off)}${row("Pass offense",f.pass_off)}${row("Run offense",f.rush_off)}${row("Defense",f.def)}${row("Pass defense",f.pass_def)}${row("Run defense",f.rush_def)}</div></div>`;
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
  const head = `<tr><th>Player</th><th>Team</th>${cols.map(c=>`<th data-k="${c[1]}" ${st.k===c[1]?`aria-sort="${st.dir<0?'descending':'ascending'}"`:""}>${c[0]}</th>`).join("")}</tr>`;
  const rows = list.map(p=>`<tr class="${p.tm===g.away?'team-a':'team-b'}"><td>${p.n}</td><td class="tm">${p.tm}</td>${cols.map(c=>{
      const v=p[c[1]]; let cls="";
      if(v!=null && c[3]!=null && v>=c[3]) cls="hot"; else if(v!=null && c[4]!=null && v<=c[4]) cls="cold";
      return `<td class="${cls}">${fmt(v,c[2])}</td>`;}).join("")}</tr>`).join("");
  const label = {QB:"Quarterbacks",RB:"Running backs",WR:"Wide receivers",TE:"Tight ends"}[pos];
  return `<div class="pos-block"><h4>${label}</h4><div class="scroll"><table data-key="${key}" data-pos="${pos}">${head}${rows}</table></div></div>`;
}
function panelHTML(g){
  return `<div class="panel"><div class="form">${formPanel(g.away)}${formPanel(g.home)}</div>
    ${["QB","RB","WR","TE"].map(p=>playerTable(g.id,p,g)).join("") || '<p class="empty">No player stats yet this season.</p>'}
    <p class="empty" style="margin-top:12px">Player stats are ${D.season} season to date. Green left edge is ${g.away}, gold is ${g.home}. Green numbers are strong, red are weak. Tap a column header to sort.</p></div>`;
}
function renderGames(){
  const gs = D.games.filter(g=>g.wk===selWk);
  $("#games").innerHTML = gs.map(g=>{
    const edge = (g.pred!=null && g.spread!=null) ? Math.abs(g.pred-g.spread) : null;
    const mw = g.prob>=.5 ? g.home : g.away, mp = Math.round((g.prob>=.5?g.prob:1-g.prob)*100);
    const fh = fairML(g.prob), fa = fairML(1-g.prob);
    return `<details class="game" data-id="${g.id}"><summary>
      <div class="matchup"><span class="cond">${g.away}<span class="at">@</span>${g.home}</span>${g.aqb&&g.hqb?`<small>${g.aqb} vs ${g.hqb}</small>`:""}</div>
      <div class="cell"><small>Vegas</small><span class="v">${spreadTxt(g,g.spread)}</span><span class="ml">${g.aml!=null?`${g.away} ${mlTxt(g.aml)}, ${g.home} ${mlTxt(g.hml)}`:"Line not posted"}${g.total!=null?`. O/U ${g.total}`:""}</span></div>
      <div class="cell ${edge!=null&&edge>=3?'edge3':''}"><small>Model${edge!=null?`, edge ${edge.toFixed(1)}`:""}</small><span class="v">${spreadTxt(g,g.pred)}</span><span class="ml">${mw} ${mp}%. Fair ${g.away} ${mlTxt(fa)}, ${g.home} ${mlTxt(fh)}</span></div>
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
  let html = `<thead><tr><th>Matchup</th><th>Vegas spread</th><th>Moneyline</th><th>Model spread</th><th>Model win %</th><th>Fair ML</th><th>Edge</th><th>Final (away–home)</th><th>ATS</th></tr></thead><tbody>`;
  let lastWk = null;
  for (const g of gs){
    if (g.wk!==lastWk && !tm){ html += `<tr class="wkrow"><td colspan="9">Week ${g.wk}</td></tr>`; lastWk=g.wk; }
    const edge = (g.pred!=null&&g.spread!=null)?Math.abs(g.pred-g.spread):null;
    const mw = g.prob>=.5?g.home:g.away, mp=Math.round((g.prob>=.5?g.prob:1-g.prob)*100);
    let fin="–", ats="–";
    if (done(g)){
      fin = `${g.as_}–${g.hs}`;
      if (g.spread!=null){ const m=g.hs-g.as_; if(m===g.spread) ats="Push"; else {
        const hc=m>g.spread, right=(g.pred>g.spread)===hc;
        ats = `${hc?g.home:g.away} <span class="${right?'res-w':'res-l'}">${right?'✓':'✗'}</span>`; } }
    }
    html += `<tr><td>${tm?`Wk ${g.wk}: `:""}${g.away} @ ${g.home}</td><td>${spreadTxt(g,g.spread)}</td><td>${g.aml!=null?`${mlTxt(g.aml)} / ${mlTxt(g.hml)}`:"–"}</td><td>${spreadTxt(g,g.pred)}</td><td>${mw} ${mp}%</td><td>${mlTxt(fairML(1-g.prob))} / ${mlTxt(fairML(g.prob))}</td><td class="${edge>=3?'hot':''}">${edge==null?"–":edge.toFixed(1)}</td><td>${fin}</td><td>${ats}</td></tr>`;
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
    const lineCell = td ? `<td>Yes</td>` : `<td><input inputmode="decimal" data-i="${i}" data-f="line" class="${v.fromBook?'fromBook':''}" value="${v.line??""}" placeholder="line" aria-label="${r.n} ${STATL[r.st]} line"></td>`;
    const ov = `<td><input class="odds" inputmode="numeric" data-i="${i}" data-f="over" value="${fmtOdds(v.over)}" placeholder="${td?'+odds':''}" aria-label="${r.n} over odds"></td>`;
    const un = td ? `<td>–</td>` : `<td><input class="odds" inputmode="numeric" data-i="${i}" data-f="under" value="${fmtOdds(v.under)}" aria-label="${r.n} under odds"></td>`;
    let pc="<td>–</td>", pk=`<td class="pick"><small>${td?"Enter the Yes odds":"Enter a line"}</small></td>`;
    if (e.p!=null){
      pc = `<td>${Math.round(e.p*100)}%<div class="bar2"><i style="width:${Math.round(e.p*100)}%"></i><b style="left:${Math.round((breakeven(v.over)||.5)*100)}%"></b></div></td>`;
      const lbl = td ? (e.side==="over"?"Yes":"No value") : (e.side==="over"?"Over":"Under");
      pk = e.edge>0 ? `<td class="pick ${e.edge>=.05?'good':''}">${lbl} ${!td||e.side==="over"?Math.round(e.sideP*100)+"%":""}<small>${(e.edge*100).toFixed(1)}% edge</small></td>`
                    : `<td class="pick"><small>No edge</small></td>`;
    }
    return `<tr><td class="pl">${r.n}${r.q?'<span class="tag" title="Questionable on injury report">Q</span>':''}<small>${r.pos}, ${r.tm} vs ${r.opp}</small></td><td>${STATL[r.st]}</td><td>${projCell(r)}</td>${lineCell}${ov}${un}${pc}${pk}</tr>`;
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
  ul.innerHTML = hits.length ? hits.map((h,i)=>`<li role="option" id="opt${i}" aria-selected="${i===hi}" data-i="${i}"><span>${h.n}</span><small>${h.type==="t"?"Team, see schedule and lines":h.pos+", "+h.tm}</small></li>`).join("")
                             : `<li class="none">No players or teams match</li>`;
  ul.hidden = false; $("#q").setAttribute("aria-expanded","true");
  $("#q").setAttribute("aria-activedescendant", hi>=0 ? "opt"+hi : "");
  ul.querySelectorAll("li[data-i]").forEach(li=>li.onmousedown=e=>{ e.preventDefault(); choose(hits[+li.dataset.i]); });
}
function choose(h){
  $("#qres").hidden = true; $("#q").setAttribute("aria-expanded","false"); $("#q").value = h.n;
  if (h.type==="t"){
    showView("season"); $("#teamSel").value = h.tm; $("#showSel").value = "all"; renderSeason();
  } else {
    pSel = h.n; showView("props"); renderProps();
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
  box.innerHTML = `<div class="pcard"><div class="pcard-top"><div><h3>${pSel}</h3><p class="who">${info.pos||""}, ${NAMES[info.tm]||info.tm||""}${g?`. Week ${PM.week} vs ${g.opp}`:""}</p></div>
    <button class="clear" type="button">Show all players</button></div>${stats}
    ${has?"":`<p class="empty">No projection this week. He may be ruled out, on bye, or not in a big enough role.</p>`}</div>`;
  box.querySelector(".clear").onclick = ()=>{ pSel=null; $("#q").value=""; renderProps(); };
}
</script>
</body>
</html>

#__END__
'''
