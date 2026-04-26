# Monte Carlo Variance Reduction for Option Pricing

A Python implementation of Monte Carlo option pricing with variance reduction techniques and GARCH volatility modelling.

---

## What This Does

Prices European options using Monte Carlo simulation and benchmarks multiple variance reduction techniques against a plain MC baseline. Also fits a GARCH model of past asset data and compares pricing accuracy under GBM vs. GARCH dynamics.
I have included a report of my GARCH vs MC performance across asset classes.

---

## Project Structure

```
src/
├── core/
│   ├── simulation.py       # GBM and GARCH path generation
│   ├── payoffs.py          # Option payoff functions
│   └── pricer.py           # MC engine (orchestrates simulation → payoff → price)
├── variance_reduction/     # One file per technique (antithetic, control, etc.)
├── analysis/
│   └── excel_export.py     # All Excel output logic
├── outputs/                # Generated Excel files
└── main.py                 # Entry point
```

---

## Installation

```bash
git clone https://github.com/ancode1143/monte-carlo-option-pricing.git
cd monte-carlo-option-pricing
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

Output Excel files are saved to `/outputs/`.

---

## Results

Each run outputs:
- Estimated option price
- Standard error
- Variance reduction ratio vs. plain MC
- Convergence plot (price estimate vs. number of simulations)

---
 Techniques based on Glasserman (2003), *Monte Carlo Methods in Financial Engineering*.
