import numpy as np

def european_payoff(
    paths: np.ndarray,
    k: float, # strike price
    option_type: str  # 'call' or 'put'
) -> np.ndarray:
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")

    PT = paths[:, -1]

    if option_type == 'call':
        payoffs = np.maximum(0, PT - k)
    else:
        payoffs = np.maximum(0, k - PT)

    return payoffs