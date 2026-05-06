import yfinance as yf
import numpy as np
import requests

def start_price(ticker: str) -> float:
    df = yf.download(ticker, period="5d", progress=False)
    if df is None or df.empty:
        raise ValueError("No data returned from yfinance")
    return float(df["Close"].iloc[-1])

def historical_vol(ticker: str, window: int = 252) -> float:
    df = yf.download(ticker, period="2y", progress=False)
    if df is None or df.empty:
        raise ValueError("No data returned from yfinance")
    close = df["Close"]
    # each value becomes the log change of previous price
    log_returns = (close / close.shift(1)).apply(np.log).dropna() # remove the first row
    return log_returns.std() * np.sqrt(252) # HV = daily volatility annualised

def _fetch_rate(ticker: str) -> float:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker}"
    r = requests.get(url) # response object
    lines = [line for line in r.text.strip().split("\n") if not line.endswith(",")] # string CSV of date and percentage
    return float(lines[-1].split(",")[1]) / 100 # latest value as a decimal

def risk_free_rate(t: float) -> float:
    TENORS = {
    1/12: "DGS1MO",
    3/12: "DGS3MO",
    6/12: "DGS6MO",
    1:    "DGS1",
    2:    "DGS2",
    5:    "DGS5",
    7:    "DGS7",
    10:   "DGS10",
    20:   "DGS20",
    30:   "DGS30",
    }
    tenors = sorted(TENORS.keys())
    if t <= tenors[0]:  
        return _fetch_rate(TENORS[tenors[0]])
    if t >= tenors[-1]: 
        return _fetch_rate(TENORS[tenors[-1]])
    lower = max(t0 for t0 in tenors if t0 <= t)
    upper = min(t0 for t0 in tenors if t0 >= t)
    if lower == upper: 
        return _fetch_rate(TENORS[lower])
    weight = (t - lower) / (upper - lower)
    weighted_rate = _fetch_rate(TENORS[lower]) + weight * (_fetch_rate(TENORS[upper]) - _fetch_rate(TENORS[lower]))
    return weighted_rate