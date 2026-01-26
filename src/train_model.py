import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

from utils import make_lagged_features

# file paths
RAW_PATH = Path("data/raw/nvda_daily.csv")
MODEL_DIR = Path("data/processed")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main(lags=10): # past 10 days
    df = pd.read_csv(RAW_PATH) # reads csv, loads into DataFrame
    df["Date"] = pd.to_datetime(df["Date"]) # convert Data column into real dates
    df = df.set_index("Date").sort_index() # set Date as index and sort by time

    series = df["Close"].dropna() # select closing price (most commonly used)
    data = make_lagged_features(series, lags=lags) # create lagged features

    # split data into training and testing by time; first 80% to training, last 20% to testing
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]

    # seperate features and target to prevent data leakage/ensure the model learned from past data; X = inputs (past prices), y = output (today's price)
    X_train, y_train = train.drop(columns=["y"]), train["y"]
    X_test, y_test = test.drop(columns=["y"]), test["y"]

    # train model w/ linear regression, patterns between past prices and today's prices
    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test) # use trained model to predict prices for unseen future data
    mae = mean_absolute_error(y_test, pred) # measure accuracy, average absolute difference between predicted price and actual price

    joblib.dump({"model": model, "lags": lags}, MODEL_DIR / "model.joblib") # save trained model
    (MODEL_DIR / "metrics.txt").write_text(f"MAE: {mae}\n") # save evaluation results

    print("Saved model to data/processed/model.joblib")
    print("Saved metrics to data/processed/metric.txt")
    print("MAE:", mae)

    if __name__ == "__main__":
        main()