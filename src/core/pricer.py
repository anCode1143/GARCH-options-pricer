from typing import Tuple
import simulation as sim
import payoffs
import fetch_market_data as fetch
import numpy as np


# build the option price and provide stats
def price_option(config: dict) -> Tuple[float, float]:
    ticker = config["ticker"]
    simulations = config["paths"]
    t = config["simulation"]["years_to_expiry"]
    r = fetch.risk_free_rate(t)
    p0 = fetch.start_price(ticker)

    model = config["simulation"].get("model", "GBM")
    if model == "GBM":
        sigma = fetch.historical_vol(ticker)
        path = sim.simulate_gbm(p0, r, sigma, t, simulations)
    elif model == "GARCH":
        path = sim.simulate_garch(p0, r, t, simulations, ticker)
    else:
        raise ValueError("simulation.model must be 'GBM' or 'GARCH'")

    strike = config["strike_price"]
    option_type = config["order_type"]
    # if config["option"] == "european":
    payoff = payoffs.european_payoff(path, strike, option_type)

    option_price = float(
        np.exp(-r * t) * payoff.mean()
    )  # discount mean by coumpounded r
    std_error = np.exp(-r * t) * payoff.std(ddof=1) / np.sqrt(simulations)

    return option_price, std_error
