import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATP_PATH = os.path.join(BASE_DIR, "data", "atp", "atp_tennis.csv")
WTA_PATH = os.path.join(BASE_DIR, "data", "wta", "wta.csv")

def load_atp() -> pd.DataFrame:
    df = pd.read_csv(ATP_PATH, low_memory=False)
    df["tour"] = "ATP"
    return df

def load_wta() -> pd.DataFrame:
    df = pd.read_csv(WTA_PATH, low_memory=False)
    df["tour"] = "WTA"
    return df

def load_all() -> pd.DataFrame:
    atp = load_atp()
    wta = load_wta()

    combined = pd.concat([atp, wta], ignore_index=True)
    print(f"Total rows loaded: {len(combined)}")

    num_cols = ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2"]
    for col in num_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        combined[col] = combined[col].where(combined[col] > 0, other=pd.NA)

    combined = combined.dropna(subset=["Rank_1", "Rank_2", "Winner", "Odd_1", "Odd_2"])

    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.dropna(subset=["Date"])

    print(f"Clean rows with odds and rankings: {len(combined)}")

    return combined

if __name__ == "__main__":
    df = load_all()
    print(df[["Player_1", "Player_2", "Winner", "Rank_1", "Rank_2", "Odd_1", "Odd_2", "Surface", "tour"]].head())