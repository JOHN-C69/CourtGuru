import pandas as pd
from loader import load_all
from model import add_probabilities, filter_bad_odds, backtest

def calculate_ev(df: pd.DataFrame) -> pd.DataFrame:
    df["ev_bet_1"] = (df["our_prob"] * (df["Odd_1"] - 1)) - (1 - df["our_prob"])
    df["ev_bet_2"] = ((1 - df["our_prob"]) * (df["Odd_2"] - 1)) - df["our_prob"]
    return df

def find_plus_ev(df: pd.DataFrame, min_ev: float = 0.05) -> pd.DataFrame:
    ev1 = df[
        (df["ev_bet_1"] > min_ev) &
        (df["our_prob"] - df["book_prob_1"] > 0.05) &
        (df["our_prob"] > 0.30)
    ][["Player_1", "Player_2", "our_prob", "book_prob_1", "Odd_1", "ev_bet_1", "Surface", "tour"]].copy()
    ev1 = ev1.rename(columns={"Player_1": "Bet_On", "Player_2": "Against", "Odd_1": "Odds", "ev_bet_1": "EV", "book_prob_1": "Book_Prob"})
    ev1["Our_Prob"] = ev1["our_prob"]

    ev2 = df[
        (df["ev_bet_2"] > min_ev) &
        ((1 - df["our_prob"]) - df["book_prob_2"] > 0.05) &
        ((1 - df["our_prob"]) > 0.30)
    ][["Player_1", "Player_2", "our_prob", "book_prob_2", "Odd_2", "ev_bet_2", "Surface", "tour"]].copy()
    ev2 = ev2.rename(columns={"Player_2": "Bet_On", "Player_1": "Against", "Odd_2": "Odds", "ev_bet_2": "EV", "book_prob_2": "Book_Prob"})
    ev2["Our_Prob"] = 1 - ev2["our_prob"]

    plus_ev = pd.concat([ev1, ev2], ignore_index=True)
    plus_ev = plus_ev[["Bet_On", "Against", "Our_Prob", "Book_Prob", "Odds", "EV", "Surface", "tour"]]
    plus_ev = plus_ev.sort_values("EV", ascending=False)

    return plus_ev

if __name__ == "__main__":
    df = load_all()
    df = filter_bad_odds(df)
    df, model = add_probabilities(df)
    backtest(df)
    df = calculate_ev(df)
    plus_ev = find_plus_ev(df)
    print(f"\n+EV bets found: {len(plus_ev)} out of {len(df)} matches ({len(plus_ev)/len(df)*100:.1f}%)")
    print(plus_ev.head(20))