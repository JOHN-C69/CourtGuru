import os
import requests
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
API_KEY = os.getenv("ODDS_API_KEY")

BASE_URL = "https://api.the-odds-api.com/v4/sports"

def get_tennis_odds(tour: str = "upcoming") -> list:
    """
    Fetch live tennis odds from major US sportsbooks.
    tour: 'atp' or 'wta' or 'upcoming' (both)
    """
    # The Odds API sport keys for tennis
    sport_keys = []
    if tour in ("atp", "upcoming"):
        sport_keys.append("tennis_atp_french_open")  # changes by tournament
    if tour in ("wta", "upcoming"):
        sport_keys.append("tennis_wta_french_open")

    # First, get all available tennis markets
    sports_url = f"{BASE_URL}?apiKey={API_KEY}"
    resp = requests.get(sports_url)
    resp.raise_for_status()

    # Find all active tennis sports
    all_sports = resp.json()
    tennis_sports = [s for s in all_sports if "tennis" in s["key"] and s["active"]]

    if not tennis_sports:
        print("No active tennis events found")
        return []

    print(f"Active tennis events:")
    for s in tennis_sports:
        print(f"  {s['title']} ({s['key']})")

    # Fetch odds for each active tennis event
    all_matches = []
    for sport in tennis_sports:
        url = f"{BASE_URL}/{sport['key']}/odds"
        params = {
            "apiKey": API_KEY,
            "regions": "us",
            "markets": "h2h",
            "bookmakers": "draftkings,fanduel,betmgm"
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        matches = resp.json()

        for match in matches:
            match["sport_title"] = sport["title"]

        all_matches.extend(matches)
        print(f"  Found {len(matches)} matches for {sport['title']}")

    print(f"\nTotal matches with odds: {len(all_matches)}")
    return all_matches

def parse_odds(matches: list) -> list:
    """Convert raw API data into clean bet opportunities."""
    bets = []

    for match in matches:
        p1 = match["home_team"]
        p2 = match["away_team"]
        event = match.get("sport_title", "")
        commence = match.get("commence_time", "")

        for bookmaker in match.get("bookmakers", []):
            book_name = bookmaker["title"]

            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue

                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}

                if p1 in outcomes and p2 in outcomes:
                    bets.append({
                        "player_1": p1,
                        "player_2": p2,
                        "odd_1": outcomes[p1],
                        "odd_2": outcomes[p2],
                        "bookmaker": book_name,
                        "event": event,
                        "start_time": commence
                    })

    return bets

if __name__ == "__main__":
    print("Fetching live tennis odds...\n")
    matches = get_tennis_odds()
    bets = parse_odds(matches)

    if bets:
        print(f"\nParsed {len(bets)} betting lines:\n")
        for b in bets[:10]:
            print(f"  {b['player_1']} vs {b['player_2']}")
            print(f"    {b['bookmaker']}: {b['odd_1']} / {b['odd_2']}")
            print(f"    Event: {b['event']}")
            print()
    else:
        print("No bets found — there may not be any active tennis matches right now")