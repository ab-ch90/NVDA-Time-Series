import yfinance as yf
import pandas as pd
from pathlib import Path

# create folder to store raw data
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def main():
    ticker = "NVDA" # choose stock ticker

    df = yf.download(ticker, start="2018-01-01", progress=False, auto_adjust=False, group_by="column") # download historical stock prices
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index() # sort data by date

    out_path = DATA_DIR / "nvda_daily.csv" # create output file parth
    df.to_csv(out_path, index_label="Date") # save data to csv
    
    print(f"Saved raw data to {out_path}") # confirmation message

# run program
if __name__ == "__main__":
    main()

# pulls historical NVIDIA strock data using an API, saves in a structured CSV to reuse later