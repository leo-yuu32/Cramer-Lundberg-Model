# Cramér-Lundberg Ruin Simulator

Interactive Streamlit app for simulating the classical Cramér-Lundberg insurance surplus process and estimating ruin probabilities via Monte Carlo. Suitable for undergraduate research (e.g. Imperial College London). View simulator here: https://cramer-lundberg-simulation.streamlit.app/

## Surplus process

$$U(t) = u + ct - \sum_{i=1}^{N(t)} X_i$$

- \(u\): initial surplus  
- \(c\): premium rate  
- \(N(t)\): Poisson process with rate \(\lambda\)  
- \(X_i\): i.i.d. claim sizes  

Ruin occurs when \(U(t) < 0\) for some \(t\).

## Setup

```bash
pip install -r requirements.txt
```

## Run the app

From the **project root** (parent of `cramer_lundberg`):

```bash
streamlit run cramer_lundberg/app.py
```

Or with module syntax:

```bash
python -m streamlit run cramer_lundberg/app.py
```

## Features

- **Model parameters**: initial surplus \(u\), premium (via safety loading \(\theta\) or rate \(c\)), claim rate \(\lambda\), claim distribution (Exponential, Gamma, Lognormal, Pareto).
- **Simulation**: number of paths \(N\), time horizon \(T\), optional seed.
- **Outputs**: ruin probability with Wilson/Wald CIs, mean ruin time, adjustment coefficient \(R\), Lundberg bound.
- **Tabs**: sample paths, ruin probability vs \(u\), convergence, ruin times & deficit, claim distribution, sensitivity heatmap over \((\theta, u)\).

## Module overview

| Module         | Role                                      |
|----------------|-------------------------------------------|
| `simulation.py`| Core engine: `CramerLundbergModel`, paths |
| `estimation.py`| Ruin prob estimate, CIs, bootstrap       |
| `lundberg.py`  | Adjustment coefficient, bounds, exact \(\psi(u)\) (exponential) |
| `fisher.py`    | Fisher information, CRLB                 |
| `plots.py`     | All Plotly figures                        |
| `utils.py`     | Distribution wrappers, MGFs               |
| `app.py`       | Streamlit UI                              |

## Mathematical notes

- **Net profit condition**: \(c > \lambda\mu\) (otherwise ruin probability \(\to 1\)).
- **Safety loading**: \(c = (1+\theta)\lambda\mu\), so \(\theta = c/(\lambda\mu) - 1\).
- **Exponential claims**: adjustment coefficient \(R = \theta/(\mu(1+\theta))\); exact ruin probability \(\psi(u) = \frac{1}{1+\theta}e^{-Ru}\).
- **Lundberg bound**: \(\psi(u) \leq e^{-Ru}\) when \(R\) exists.
- **Wilson CI** is used when \(\hat{\psi}\) is near 0 or 1.
