import pandas as pd
from pathlib import Path
import joblib

RAW_PATH = Path("data/raw/nvda_daily.csv") # historical NVIDIA prices
MODEL_PATH = Path("data/processed/model.joblib") # trained model

def forecast_next_days(last_values, model, lags=10, days=5): # given last few days of prices, predict the next days future prices one at a time
    # convert recent prices into a list
    history = list(last_values)
    preds = []
    for _ in range(days): # run prediciton loop once per future day
        x = [history[-i] for i in range(1, lags +1)] # takes most recent lags prices, uses them as input features
        x_df = pd.DataFrame([x], columns=[f"lag_{i}" for i in range(1, lags+1)]) # convert input into a DataFrame, creates single-row table 
        yhat = model.predict(x_df)[0] # make prediction for next day's price
        # save prediction/update history
        preds.append(yhat) 
        history.append(yhat)
    return preds 

def main(days=5):
    # load the saved model/lag values
    pack = joblib.load(MODEL_PATH)
    model = pack["model"]
    lags = pack["lags"]

    df = pd.read_csv(RAW_PATH, index_col="Date", parse_dates=True).sort_index() # load and prepare stock data; sets Date as index, converts Date as index, sorts chronologically

    # extract recent prices from most recent lag days
    closes = df["Close"].dropna()
    last_values = closes.iloc[-lags:].values

    preds = forecast_next_days(last_values, model, lags=lags, days=days) # generates future predicted prices
    print(f"Next {days} predicted closes:")
    for i, p in enumerate(preds, 1):
        print(f"Day {i}: {p:.2f}")

if __name__ == "__main__":
    main()