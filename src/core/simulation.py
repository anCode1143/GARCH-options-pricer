import numpy as np

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