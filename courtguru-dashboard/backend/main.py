import sys
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Add CourtGuru src/ to Python path so we can import your modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # courtguru-dashboard/
PROJECT_DIR = os.path.dirname(BASE_DIR)  # CourtGuru/
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from loader import load_all
from model import build_features, train_model, filter_bad_odds
from odds_fetcher import get_tennis_odds, parse_odds

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global state (loaded once on startup) ---
model = None
features = None
df = None
elo = {}
elo_surface = {}
form_hist = {}
surface_record = {}
h2h = {}
last_match = {}
name_map = {}

def build_player_dicts(df):
    K = 32
    elo = {}
    elo_surface = {}
    form = {}
    surface_record = {}
    h2h = {}
    last_match = {}

    for _, row in df.iterrows():
        p1, p2, winner, surface = row["Player_1"], row["Player_2"], row["Winner"], row["Surface"]
        match_date = row["Date"]

        r1 = elo.get(p1, 1500)
        r2 = elo.get(p2, 1500)
        rs1 = elo_surface.get((p1, surface), 1500)
        rs2 = elo_surface.get((p2, surface), 1500)

        exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        exp_s1 = 1 / (1 + 10 ** ((rs2 - rs1) / 400))
        actual = 1.0 if winner == p1 else 0.0

        elo[p1] = r1 + K * (actual - exp1)
        elo[p2] = r2 + K * ((1 - actual) - (1 - exp1))
        elo_surface[(p1, surface)] = rs1 + K * (actual - exp_s1)
        elo_surface[(p2, surface)] = rs2 + K * ((1 - actual) - (1 - exp_s1))

        won_1 = 1 if winner == p1 else 0
        form.setdefault(p1, []).append(won_1)
        form.setdefault(p2, []).append(1 - won_1)

        surface_record.setdefault((p1, surface), [0, 0])
        surface_record[(p1, surface)][1] += 1
        surface_record[(p1, surface)][0] += won_1
        surface_record.setdefault((p2, surface), [0, 0])
        surface_record[(p2, surface)][1] += 1
        surface_record[(p2, surface)][0] += (1 - won_1)

        key = tuple(sorted([p1, p2]))
        h2h.setdefault(key, {})
        h2h[key][p1] = h2h[key].get(p1, 0) + won_1
        h2h[key][p2] = h2h[key].get(p2, 0) + (1 - won_1)

        last_match[p1] = match_date
        last_match[p2] = match_date

    return elo, elo_surface, form, surface_record, h2h, last_match

def build_name_map():
    atp = pd.read_csv(os.path.join(PROJECT_DIR, "data", "atp", "atp_tennis.csv"), low_memory=False)
    wta = pd.read_csv(os.path.join(PROJECT_DIR, "data", "wta", "wta.csv"), low_memory=False)
    players = set()
    for d in [atp, wta]:
        players.update(d["Player_1"].dropna().unique())
        players.update(d["Player_2"].dropna().unique())
    return {p.lower().strip(): p for p in players}

def match_name(full_name, name_map):
    parts = full_name.strip().split()
    if len(parts) < 2:
        return None
    for i in range(1, len(parts)):
        first = " ".join(parts[:i])
        last = " ".join(parts[i:])
        attempts = [f"{last} {first[0]}.", f"{last} {first[0]}", f"{first} {last[0]}."]
        for attempt in attempts:
            key = attempt.lower().strip()
            if key in name_map:
                return name_map[key]
        for map_key, original in name_map.items():
            map_last = map_key.split()[0] if map_key else ""
            if last.lower() == map_last and len(map_key.split()) > 1:
                map_initial = map_key.split()[-1][0]
                if first[0].lower() == map_initial.lower():
                    return original
    return None

def get_player_features(p1, p2, surface):
    elo_1 = elo.get(p1, 1500)
    elo_2 = elo.get(p2, 1500)
    elo_s1 = elo_surface.get((p1, surface), 1500)
    elo_s2 = elo_surface.get((p2, surface), 1500)

    h1 = form_hist.get(p1, [])
    h2 = form_hist.get(p2, [])
    f5_1 = np.mean(h1[-5:]) if len(h1) >= 5 else 0.5
    f10_1 = np.mean(h1[-10:]) if len(h1) >= 10 else 0.5
    f20_1 = np.mean(h1[-20:]) if len(h1) >= 20 else 0.5
    f5_2 = np.mean(h2[-5:]) if len(h2) >= 5 else 0.5
    f10_2 = np.mean(h2[-10:]) if len(h2) >= 10 else 0.5
    f20_2 = np.mean(h2[-20:]) if len(h2) >= 20 else 0.5

    sr1 = surface_record.get((p1, surface), [0, 0])
    sr2 = surface_record.get((p2, surface), [0, 0])
    swr1 = sr1[0] / sr1[1] if sr1[1] >= 10 else 0.5
    swr2 = sr2[0] / sr2[1] if sr2[1] >= 10 else 0.5

    key = tuple(sorted([p1, p2]))
    h2h_record = h2h.get(key, {})
    p1_h2h_wins = h2h_record.get(p1, 0)
    p2_h2h_wins = h2h_record.get(p2, 0)
    h2h_total = p1_h2h_wins + p2_h2h_wins
    h2h_wr = p1_h2h_wins / h2h_total if h2h_total >= 2 else 0.5

    today = pd.Timestamp.now()
    d1 = (today - last_match[p1]).days if p1 in last_match else 14
    d2 = (today - last_match[p2]).days if p2 in last_match else 14

    recent_1 = df[(df["Player_1"] == p1) | (df["Player_2"] == p1)].tail(1)
    if len(recent_1) > 0:
        row = recent_1.iloc[0]
        rank_1 = row["Rank_1"] if row["Player_1"] == p1 else row["Rank_2"]
        pts_1 = row["Pts_1"] if row["Player_1"] == p1 else row["Pts_2"]
    else:
        rank_1, pts_1 = 200, 100

    recent_2 = df[(df["Player_1"] == p2) | (df["Player_2"] == p2)].tail(1)
    if len(recent_2) > 0:
        row = recent_2.iloc[0]
        rank_2 = row["Rank_1"] if row["Player_1"] == p2 else row["Rank_2"]
        pts_2 = row["Pts_1"] if row["Player_1"] == p2 else row["Pts_2"]
    else:
        rank_2, pts_2 = 200, 100

    return {
        "elo_diff": elo_1 - elo_2,
        "elo_surf_diff": elo_s1 - elo_s2,
        "rank_ratio": np.log(rank_2 / rank_1) if rank_1 > 0 and rank_2 > 0 else 0,
        "pts_ratio": pts_2 / (pts_1 + pts_2 + 1) if pts_1 and pts_2 else 0.5,
        "form_diff_5": f5_1 - f5_2,
        "form_diff_10": f10_1 - f10_2,
        "form_diff_20": f20_1 - f20_2,
        "surface_wr_diff": swr1 - swr2,
        "h2h_wr_1": h2h_wr,
        "fatigue_diff": d1 - d2,
        "round_num": 2,
        "is_clay": 1 if surface == "Clay" else 0,
        "is_grass": 1 if surface == "Grass" else 0,
    }

def detect_surface(event_name):
    clay = ["roland garros", "french open", "madrid", "rome", "barcelona", "monte carlo"]
    grass = ["wimbledon", "queens", "halle", "eastbourne"]
    name = event_name.lower()
    for t in clay:
        if t in name:
            return "Clay"
    for t in grass:
        if t in name:
            return "Grass"
    return "Hard"

# --- Startup: load data + train model ---
@app.on_event("startup")
def startup():
    global model, features, df, elo, elo_surface, form_hist, surface_record, h2h, last_match, name_map

    print("\n[CourtGuru] Loading data and training model...")
    raw = load_all()
    raw = filter_bad_odds(raw)
    df = raw.sort_values("Date").reset_index(drop=True)

    df_feat = build_features(df.copy())
    model, features, _ = train_model(df_feat)

    elo, elo_surface, form_hist, surface_record, h2h, last_match = build_player_dicts(df)
    name_map = build_name_map()

    print("[CourtGuru] Ready! Model loaded.\n")

# --- API Endpoints ---
@app.get("/bets")
def get_bets():
    try:
        matches = get_tennis_odds()
        bets = parse_odds(matches)
    except Exception as e:
        return {"bets": [], "error": f"Could not fetch odds: {str(e)}", "last_updated": datetime.now().isoformat()}

    results = []
    for bet in bets:
        p1 = match_name(bet["player_1"], name_map)
        p2 = match_name(bet["player_2"], name_map)
        if not p1 or not p2:
            continue

        surface = detect_surface(bet["event"])
        odd_1, odd_2 = bet["odd_1"], bet["odd_2"]

        feat = get_player_features(p1, p2, surface)
        feat["book_logit"] = np.log((1 / odd_1) / (1 / odd_2))

        X = pd.DataFrame([feat])[features]
        our_prob = float(model.predict_proba(X)[0][1])

        book_total = (1 / odd_1) + (1 / odd_2)
        book_prob = (1 / odd_1) / book_total

        # Check player 1
        ev = (our_prob * (odd_1 - 1)) - (1 - our_prob)
        edge = our_prob - book_prob
        if ev > 0.05 and edge > 0.03:
            results.append({
                "match": f"{bet['player_1']} vs {bet['player_2']}",
                "tournament": bet["event"],
                "surface": surface,
                "bet_type": "Match Winner",
                "pick": bet["player_1"],
                "model_prob": round(our_prob, 3),
                "implied_prob": round(book_prob, 3),
                "edge": round(edge, 3),
                "odds": f"+{int((odd_1 - 1) * 100)}" if odd_1 >= 2 else f"-{int(100 / (odd_1 - 1))}",
                "kelly_fraction": round(edge / (odd_1 - 1), 3) if odd_1 > 1 else 0,
                "ev": round(ev, 3),
                "bookmaker": bet["bookmaker"],
                "confidence": "high" if edge > 0.10 else "medium" if edge > 0.05 else "low",
            })

        # Check player 2
        our_prob_2 = 1 - our_prob
        book_prob_2 = (1 / odd_2) / book_total
        ev_2 = (our_prob_2 * (odd_2 - 1)) - (1 - our_prob_2)
        edge_2 = our_prob_2 - book_prob_2
        if ev_2 > 0.05 and edge_2 > 0.03:
            results.append({
                "match": f"{bet['player_1']} vs {bet['player_2']}",
                "tournament": bet["event"],
                "surface": surface,
                "bet_type": "Match Winner",
                "pick": bet["player_2"],
                "model_prob": round(our_prob_2, 3),
                "implied_prob": round(book_prob_2, 3),
                "edge": round(edge_2, 3),
                "odds": f"+{int((odd_2 - 1) * 100)}" if odd_2 >= 2 else f"-{int(100 / (odd_2 - 1))}",
                "kelly_fraction": round(edge_2 / (odd_2 - 1), 3) if odd_2 > 1 else 0,
                "ev": round(ev_2, 3),
                "bookmaker": bet["bookmaker"],
                "confidence": "high" if edge_2 > 0.10 else "medium" if edge_2 > 0.05 else "low",
            })

    results.sort(key=lambda x: -x["edge"])

    return {
        "bets": results,
        "last_updated": datetime.now().isoformat(),
        "total_bets": len(results),
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}