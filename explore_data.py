from pathlib import Path
import os
import tempfile
import pandas as pd

# Direct matplotlib cache to system temp directory to prevent permission warnings
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib_cache"))

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
    print("DATASET SHAPE:", df.shape)
    print("\nFIRST FIVE ROWS:")
    print(df.head())
    print("\nDATA TYPES AND NON-NULL COUNTS:")
    print(df.info())
    print("\nMISSING VALUES:")
    print(df.isnull().sum())
    print("\nNUMERICAL SUMMARY:")
    print(df.describe())
    counts = df["Machine failure"].value_counts().sort_index()
    print("\nMACHINE FAILURE COUNTS:\n", counts)

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Machine failure")
    plt.xticks([0, 1], ["No Failure", "Failure"])
    plt.title("Machine Failure Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(REPORTS / "failure_distribution.png", dpi=160)
    plt.close()
    print(f"\nChart saved in: {REPORTS / 'failure_distribution.png'}")


if __name__ == "__main__":
    main()
