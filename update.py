import kaggle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def update():
    print("=" * 50)
    print("  CourtGuru Data Updater")
    print("=" * 50)

    kaggle.api.authenticate()

    print("\nDownloading latest ATP data...")
    kaggle.api.dataset_download_files(
        "dissfya/atp-tennis-2000-2023daily-pull",
        path=os.path.join(BASE_DIR, "data", "atp"),
        unzip=True
    )
    print("ATP data updated!")

    print("\nDownloading latest WTA data...")
    kaggle.api.dataset_download_files(
        "dissfya/wta-tennis-2007-2023-daily-update",
        path=os.path.join(BASE_DIR, "data", "wta"),
        unzip=True
    )
    print("WTA data updated!")

    print("\nDone! Run 'python src/main.py' to find +EV bets.")

if __name__ == "__main__":
    update()