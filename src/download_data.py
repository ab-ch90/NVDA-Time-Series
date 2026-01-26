import yfinance as yf
import pandas as pd
from pathlib import Path

# create folder to store raw data
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def main():
    ticker = "NVDA" # choose stock ticker
    df = yf.download(ticker, start="2018-01-01", progress=False) # download historical stock prices
    df = df.sort_index() # sort data by date
    out_path = DATA_DIR / "nvda_daily.csv" # create output file parth
    df.to_csv(out_path) # save data to csv
    print(f"Saved raw data to {out_path}") # conformation message

# run program
if __name__ == "__main__":
    main()