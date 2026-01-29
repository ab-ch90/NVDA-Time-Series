import pandas as pd

def make_lagged_features(series: pd.Series, lags: int = 10) -> pd.DataFrame: #converts time-series into dataset
    data = pd.DataFrame({"y": series}) # create target column y
    for i in range(1, lags + 1): # loop that creates past-day feature
        data[f"lag_{i}"] = series.shift(i) # moves the data down by i rows
    return data.dropna() # first few rows don't have enough data

# converts a time series into a supervised dataset by using past prices as input features