import numpy as np
import yfinance as yf
from arch import arch_model

def simulate_gbm(
    p0: float, # current share price
    r: float, # risk free rate
    sigma: float, # volatility
    t: float, # years to expiry
    simulations: int
) -> np.ndarray:
    steps = max(1, int(t*365)) # incase 0 < t < 1
    dt = t / steps

    # random numbers initialisation
    Z = np.random.normal(size=(simulations, steps))

    # transform each value to: drift + volatility adjusted movement
    log_returns = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z

    # prepend log(p0) and add steps to p0
    log_paths = np.hstack([
        np.full((simulations, 1), np.log(p0)),
        np.log(p0) + np.cumsum(log_returns, axis=1)
    ])

    # reverse logged valued
    return np.exp(log_paths)

def simulate_garch(
    p0: float,
    r: float,
    t: float,
    simulations: int,
    ticker: str,
) -> np.ndarray:
    
    df = yf.download(ticker, period="2y", progress=False)
    if df is None or df.empty:
        raise ValueError("No data returned from yfinance")
    prices = df["Close"]
    returns = (prices / prices.shift(1)).apply(np.log).dropna()  # log returns

    # arch expects percentage returns, hence *100
    definition = arch_model(returns * 100, vol="GARCH", p=1, q=1)
    model = definition.fit(disp="off")

    omega = model.params["omega"]    # baseline variance
    alpha = model.params["alpha[1]"] # shock sensitivity
    beta  = model.params["beta[1]"]  # variance persistence/memory/decay

    # starting variance, division to undo the *100 scaling
    sigma0 = (model.conditional_volatility[-1] ** 2) / 10000

    steps = max(1, int(t * 365))
    dt = t / steps

    paths = np.zeros((simulations, steps + 1))
    paths[:, 0] = p0

    sigma2 = np.full(simulations, sigma0)  # each sim starts with same variance

    for step in range(steps):
        sigma = np.sqrt(sigma2)
        Z = np.random.normal(size=simulations) # random vector

        # sigma is a vector exclusive to every path
        log_return = (r - 0.5 * sigma2) * dt + sigma * np.sqrt(dt) * Z
        paths[:, step + 1] = paths[:, step] * np.exp(log_return)

        # update variance for next step using this step's shock
        epsilon = sigma * np.sqrt(dt) * Z # realised shock
        sigma2 = omega + alpha * epsilon**2 + beta * sigma2

    return paths