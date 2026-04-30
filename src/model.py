import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from loader import load_all

# ============================================================
# STEP 1: Elo ratings
# ============================================================
def build_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    K = 32
    elo = {}
    elo_surface = {}

    elo_1, elo_2, elo_s1, elo_s2 = [], [], [], []

    for _, row in df.iterrows():
        p1, p2, winner, surface = row["Player_1"], row["Player_2"], row["Winner"], row["Surface"]

        r1 = elo.get(p1, 1500)
        r2 = elo.get(p2, 1500)
        rs1 = elo_surface.get((p1, surface), 1500)
        rs2 = elo_surface.get((p2, surface), 1500)

        elo_1.append(r1)
        elo_2.append(r2)
        elo_s1.append(rs1)
        elo_s2.append(rs2)

        exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        exp_s1 = 1 / (1 + 10 ** ((rs2 - rs1) / 400))
        actual = 1.0 if winner == p1 else 0.0

        elo[p1] = r1 + K * (actual - exp1)
        elo[p2] = r2 + K * ((1 - actual) - (1 - exp1))
        elo_surface[(p1, surface)] = rs1 + K * (actual - exp_s1)
        elo_surface[(p2, surface)] = rs2 + K * ((1 - actual) - (1 - exp_s1))

    df["elo_1"], df["elo_2"] = elo_1, elo_2
    df["elo_surf_1"], df["elo_surf_2"] = elo_s1, elo_s2
    print("Elo ratings built")
    return df

# ============================================================
# STEP 2: Player form
# ============================================================
def build_form(df: pd.DataFrame) -> pd.DataFrame:
    form = {}
    f5_1, f10_1, f20_1, f5_2, f10_2, f20_2 = [], [], [], [], [], []

    for _, row in df.iterrows():
        p1, p2, winner = row["Player_1"], row["Player_2"], row["Winner"]
        h1 = form.get(p1, [])
        h2 = form.get(p2, [])

        f5_1.append(np.mean(h1[-5:]) if len(h1) >= 5 else np.nan)
        f10_1.append(np.mean(h1[-10:]) if len(h1) >= 10 else np.nan)
        f20_1.append(np.mean(h1[-20:]) if len(h1) >= 20 else np.nan)
        f5_2.append(np.mean(h2[-5:]) if len(h2) >= 5 else np.nan)
        f10_2.append(np.mean(h2[-10:]) if len(h2) >= 10 else np.nan)
        f20_2.append(np.mean(h2[-20:]) if len(h2) >= 20 else np.nan)

        result_1 = 1 if winner == p1 else 0
        form.setdefault(p1, []).append(result_1)
        form.setdefault(p2, []).append(1 - result_1)

    df["form_5_1"], df["form_10_1"], df["form_20_1"] = f5_1, f10_1, f20_1
    df["form_5_2"], df["form_10_2"], df["form_20_2"] = f5_2, f10_2, f20_2
    print("Form stats built")
    return df

# ============================================================
# STEP 3: Surface win rates
# ============================================================
def build_surface_stats(df: pd.DataFrame) -> pd.DataFrame:
    surface_record = {}
    sw1, sw2 = [], []

    for _, row in df.iterrows():
        p1, p2, winner, surface = row["Player_1"], row["Player_2"], row["Winner"], row["Surface"]

        r1 = surface_record.get((p1, surface), [0, 0])
        r2 = surface_record.get((p2, surface), [0, 0])

        sw1.append(r1[0] / r1[1] if r1[1] >= 10 else np.nan)
        sw2.append(r2[0] / r2[1] if r2[1] >= 10 else np.nan)

        won_1 = 1 if winner == p1 else 0
        surface_record.setdefault((p1, surface), [0, 0])
        surface_record[(p1, surface)][1] += 1
        surface_record[(p1, surface)][0] += won_1
        surface_record.setdefault((p2, surface), [0, 0])
        surface_record[(p2, surface)][1] += 1
        surface_record[(p2, surface)][0] += (1 - won_1)

    df["surface_wr_1"], df["surface_wr_2"] = sw1, sw2
    print("Surface stats built")
    return df

# ============================================================
# STEP 4: Head-to-head record
# ============================================================
def build_h2h(df: pd.DataFrame) -> pd.DataFrame:
    h2h = {}
    h2h_1, h2h_total_list = [], []

    for _, row in df.iterrows():
        p1, p2, winner = row["Player_1"], row["Player_2"], row["Winner"]

        key = tuple(sorted([p1, p2]))
        record = h2h.get(key, {p1: 0, p2: 0, "total": 0})

        p1_wins = record.get(p1, 0)
        total = record["total"]
        h2h_1.append(p1_wins / total if total >= 2 else np.nan)
        h2h_total_list.append(total)

        if winner == p1:
            record[p1] = record.get(p1, 0) + 1
        else:
            record[p2] = record.get(p2, 0) + 1
        record["total"] += 1
        h2h[key] = record

    df["h2h_wr_1"] = h2h_1
    df["h2h_total"] = h2h_total_list
    print("Head-to-head stats built")
    return df

# ============================================================
# STEP 5: Fatigue — days since last match
# ============================================================
def build_fatigue(df: pd.DataFrame) -> pd.DataFrame:
    last_match = {}
    days_1, days_2 = [], []

    for _, row in df.iterrows():
        p1, p2 = row["Player_1"], row["Player_2"]
        match_date = row["Date"]

        if p1 in last_match:
            d1 = (match_date - last_match[p1]).days
            days_1.append(d1)
        else:
            days_1.append(np.nan)

        if p2 in last_match:
            d2 = (match_date - last_match[p2]).days
            days_2.append(d2)
        else:
            days_2.append(np.nan)

        last_match[p1] = match_date
        last_match[p2] = match_date

    df["days_since_1"] = days_1
    df["days_since_2"] = days_2
    print("Fatigue stats built")
    return df

# ============================================================
# STEP 6: Round encoding
# ============================================================
def build_round_features(df: pd.DataFrame) -> pd.DataFrame:
    round_map = {
        "1st Round": 1, "2nd Round": 2, "3rd Round": 3,
        "4th Round": 4, "Quarterfinals": 5, "Semifinals": 6,
        "The Final": 7, "Final": 7,
        "Round Robin": 3,
        "Qualifying": 0, "1st Qualifying": 0, "2nd Qualifying": 0
    }
    df["round_num"] = df["Round"].map(round_map).fillna(2)
    print("Round features built")
    return df

# ============================================================
# STEP 7: Combine everything
# ============================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    print("Building features (this takes a couple minutes)...")
    df = build_elo_ratings(df)
    df = build_form(df)
    df = build_surface_stats(df)
    df = build_h2h(df)
    df = build_fatigue(df)
    df = build_round_features(df)

    # Derived features
    df["elo_diff"] = df["elo_1"] - df["elo_2"]
    df["elo_surf_diff"] = df["elo_surf_1"] - df["elo_surf_2"]
    df["rank_ratio"] = np.log(df["Rank_2"] / df["Rank_1"])
    df["pts_ratio"] = df["Pts_2"] / (df["Pts_1"] + df["Pts_2"] + 1)
    df["form_diff_5"] = df["form_5_1"] - df["form_5_2"]
    df["form_diff_10"] = df["form_10_1"] - df["form_10_2"]
    df["form_diff_20"] = df["form_20_1"] - df["form_20_2"]
    df["surface_wr_diff"] = df["surface_wr_1"] - df["surface_wr_2"]
    df["fatigue_diff"] = df["days_since_1"] - df["days_since_2"]

    # Bookmaker
    df["book_prob_1"] = 1 / df["Odd_1"]
    df["book_prob_2"] = 1 / df["Odd_2"]
    total = df["book_prob_1"] + df["book_prob_2"]
    df["book_prob_1"] = df["book_prob_1"] / total
    df["book_prob_2"] = df["book_prob_2"] / total
    df["book_logit"] = np.log(df["book_prob_1"] / df["book_prob_2"])

    # Surface dummies
    df["is_clay"] = (df["Surface"] == "Clay").astype(int)
    df["is_grass"] = (df["Surface"] == "Grass").astype(int)

    # Target
    df["p1_won"] = (df["Winner"] == df["Player_1"]).astype(int)

    return df

def train_model(df: pd.DataFrame):
    features = [
        "book_logit",
        "elo_diff", "elo_surf_diff",
        "rank_ratio", "pts_ratio",
        "form_diff_5", "form_diff_10", "form_diff_20",
        "surface_wr_diff",
        "h2h_wr_1",
        "fatigue_diff",
        "round_num",
        "is_clay", "is_grass"
    ]

    clean = df.dropna(subset=["p1_won", "book_logit"]).copy()
    print(f"\nTraining on {len(clean)} matches")

    X = clean[features]
    y = clean["p1_won"]

    split = int(len(clean) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.1%}")
    print(f"Test accuracy:  {test_acc:.1%}")

    print("\nFeature importance:")
    for name, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {name:>20}: {imp:.3f} {bar}")

    return model, features, clean

def add_probabilities(df: pd.DataFrame):
    df = build_features(df)
    model, features, clean = train_model(df)
    clean["our_prob"] = model.predict_proba(clean[features])[:, 1]
    return clean, model

def filter_bad_odds(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["Odd_1"] >= 1.10) & (df["Odd_1"] <= 8.0)]
    df = df[(df["Odd_2"] >= 1.10) & (df["Odd_2"] <= 8.0)]
    print(f"Rows after filtering bad odds: {len(df)}")
    return df

def backtest(df: pd.DataFrame) -> None:
    df = df.copy()
    df["we_pick_1"] = df["our_prob"] > 0.5
    df["correct"] = df["we_pick_1"] == df["p1_won"]
    accuracy = df["correct"].mean()

    df["book_pick_1"] = df["book_prob_1"] > 0.5
    df["book_correct"] = df["book_pick_1"] == df["p1_won"]
    book_accuracy = df["book_correct"].mean()

    print(f"\nOur model accuracy:  {accuracy:.1%}")
    print(f"Bookmaker accuracy:  {book_accuracy:.1%}")
    print(f"Edge over bookmaker: {(accuracy - book_accuracy)*100:+.1f} percentage points")

if __name__ == "__main__":
    df = load_all()
    df = filter_bad_odds(df)
    df, model = add_probabilities(df)
    backtest(df)

