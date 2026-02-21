import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RAW_PATH = Path("data/raw/nvda_daily.csv")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(RAW_PATH)
    df["Date"] = pd.to_datetime(df["Date"]) # convert Date strings into real dates
    df = df.set_index("Date").sort_index() # set Date as the index and sort by time
 
    df["MA20"] = df["Close"].rolling(20).mean() # create a moving average column
    
    plt.figure()
    plt.plot(df.index, df["Close"], label="Close") # plot the Close price line
    plt.plot(df.index, df["MA20"], label="MA20") # plot the moving average line
    plt.legend()
    plt.title("NVDA Close + 20-day Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Price")
    # save graph as PNG file
    out = FIG_DIR / "nvda_ma20.png"
    plt.savefig(out, bbox_inches="tight") # trim empty whitespace
    print("Saved figure:", out)

if __name__ == "__main__":
    main()