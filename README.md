# CourtGuru 🎾

A machine learning-powered tennis value bet finder that identifies +EV (positive expected value) opportunities by comparing model predictions against live sportsbook odds.

## What It Does

CourtGuru trains an XGBoost model on 100,000+ historical ATP and WTA matches, builds per-player Elo ratings, form metrics, and surface-specific stats, then compares its win probabilities against live odds from DraftKings, FanDuel, and BetMGM to surface bets where the sportsbook is undervaluing a player.

```
============================================================
  CourtGuru +EV Bet Finder
============================================================

Found 6 +EV bets:

  +EV Casper Ruud vs Alexander Blockx
     Our prob: 85.5%  |  Book: 70.7%  |  Edge: 14.8%
     Odds: 1.35  |  EV: +0.154  |  BetMGM (ATP Madrid Open)

  +EV Alexander Zverev vs Flavio Cobolli
     Our prob: 81.9%  |  Book: 71.0%  |  Edge: 10.9%
     Odds: 1.35  |  EV: +0.106  |  FanDuel (ATP Madrid Open)
```

## Features

- **Elo Rating System** — Chess-style ratings computed per-player across all historical matches, with separate surface-specific Elo for clay, grass, and hard court specialists
- **Player Form Tracking** — Rolling win rates over last 5, 10, and 20 matches to capture momentum and slumps
- **Head-to-Head Records** — Historical matchup win rates between specific player pairs
- **Surface Win Rates** — Career win percentages per surface to identify clay/grass specialists
- **Fatigue Detection** — Days since last match to factor in tournament scheduling and rest
- **Round Weighting** — Tournament round context (early rounds see more upsets)
- **Live Odds Integration** — Real-time odds from DraftKings, FanDuel, and BetMGM via The Odds API
- **XGBoost Classifier** — Gradient-boosted model trained on 100k+ matches with temporal train/test split

## Tech Stack

- **Python** — Core language
- **pandas** — Data loading, cleaning, and feature engineering
- **XGBoost** — Gradient-boosted decision tree classifier
- **scikit-learn** — Model evaluation and train/test splitting
- **The Odds API** — Live sportsbook odds
- **Kaggle API** — Automated dataset updates

## Project Structure

```
CourtGuru/
├── data/
│   ├── atp/              # ATP match data (2000–2026)
│   └── wta/              # WTA match data (2007–2026)
├── src/
│   ├── loader.py         # Data loading and cleaning pipeline
│   ├── model.py          # Feature engineering and XGBoost training
│   ├── odds_fetcher.py   # Live odds from The Odds API
│   └── ev_calculator.py  # Expected value calculations
├── main.py               # Entry point — runs the full pipeline
├── update.py             # Dataset refresh via Kaggle API
├── .env                  # API keys (not committed)
├── .gitignore
└── README.md
```

## How It Works

### 1. Feature Engineering

For each historical match, the system computes features *at the time of the match* (no data leakage):

| Feature | Description |
|---|---|
| `elo_diff` | Overall Elo rating difference between players |
| `elo_surf_diff` | Surface-specific Elo difference |
| `rank_ratio` | Log ratio of ATP/WTA rankings |
| `pts_ratio` | Ranking points comparison |
| `form_diff_5/10/20` | Rolling win rate difference (last 5, 10, 20 matches) |
| `h2h_wr_1` | Head-to-head win rate between the two players |
| `surface_wr_diff` | Career surface win rate difference |
| `fatigue_diff` | Difference in days since each player's last match |
| `round_num` | Tournament round (1st round through Final) |
| `book_logit` | Log-odds from sportsbook lines |

### 2. Model Training

- XGBoost classifier with 500 estimators, max depth 4
- Temporal split: trains on older matches, tests on recent ones
- Achieves ~68% accuracy vs ~67% bookmaker baseline

### 3. Value Detection

A bet is flagged as +EV when:
- `EV = (our_probability × payout) - (1 - our_probability) > 0.05`
- Our probability exceeds the bookmaker's implied probability by at least 3%

## Setup

### Prerequisites
- Python 3.10+
- The Odds API key ([the-odds-api.com](https://the-odds-api.com))
- Kaggle API token ([kaggle.com](https://kaggle.com) → Settings → API)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/CourtGuru.git
cd CourtGuru

pip install pandas xgboost scikit-learn requests python-dotenv kaggle
```

### Configuration

Create a `.env` file in the project root:

```
ODDS_API_KEY=your_odds_api_key_here
```

Set up Kaggle credentials:

```bash
mkdir ~/.kaggle
# Save your Kaggle access token to ~/.kaggle/access_token
```

### Download Data

```bash
python update.py
```

### Run

```bash
python src/main.py
```

## Model Performance

```
Training on 20,042 matches
Train accuracy: 72.5%
Test accuracy:  63.8%

Feature importance:
            book_logit: 0.307 ███████████████
              elo_diff: 0.118 █████
         elo_surf_diff: 0.061 ███
       surface_wr_diff: 0.054 ██
             round_num: 0.053 ██
              h2h_wr_1: 0.047 ██
          fatigue_diff: 0.049 ██
```

## Data Sources

- [ATP Tennis 2000–2026 (Kaggle, daily updates)](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull)
- [WTA Tennis 2007–2025 (Kaggle, daily updates)](https://www.kaggle.com/datasets/dissfya/wta-tennis-2007-2023-daily-update)
- [The Odds API](https://the-odds-api.com) — Live odds from DraftKings, FanDuel, BetMGM

## Disclaimer

This project is for educational and research purposes only. It demonstrates machine learning, feature engineering, and API integration in a real-world domain. Past model performance does not guarantee future results.
