import os
import math
import pandas as pd
import pymc as pm
import xarray as xr
import arviz as az
import pytensor.tensor as pt
from datetime import timedelta
import numpy as np
from dotenv import load_dotenv
import logging

# Logging setup
load_dotenv()
logging_level = os.environ.get("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)

def calculate_tangency_portfolio(sotck_names, prior_weights, risk_aversion, observed_data
) -> pd.DataFrame:
    """Main Function to calculate the positive tangency portfolio

    Args:
        portfolio_spec:
            This element contains 'prior_weights', 'portfolio_size', 'scale'
        trading_date_ts:
        k_stock_market_caps_df:
        k_stock_prices_df:
        treasury_bill_rate_df:

    Returns:
        A pandas DataFrame
    """
    p = len(sotck_names)
    scale = 10

    with pm.Model() as model:
        # Cholesky decomposition for the covariance matrix
        # eta is considered 2
        # p is portfolio size
        chol, corr, sigmas = pm.LKJCholeskyCov(
            "packed_L",
            n=p,
            eta=2,
            sd_dist=pm.HalfCauchy.dist(beta=5, shape=p),
            shape=(p * (p + 1) // 2),
        )

        # Omega is the result of cholesky and diagonal of covariance values
        Omega = pm.Deterministic("Omega", chol.dot(chol.T))  # Precision matrix

        # Positive weights prior
        # kappa is considered scale
        # A larger value of κ indicates a diminished reliance on the prior.
        log_nu = pm.MvNormal(
            "log_nu", mu=pt.log(prior_weights), cov=scale * pt.eye(len(prior_weights))
        )

        # Make nu from the exp(log_nu)
        nu = pm.Deterministic("nu", pt.exp(log_nu))

        # Likelihood for X
        # Convert from natural parameters to mean and covariance
        Sigma = pm.Deterministic(
            "Sigma", pt.nlinalg.matrix_inverse(Omega) + pt.eye(p) * 1e-32
        )

        # pt.dot(Sigma, nu) gives the main values for mu
        mu = pm.Deterministic("mu", pt.reshape(pt.dot(Sigma, nu), (p,)))
        
        # what is the lag for the new incoming observed data?
        observed_data = (
            observed_data
        )
        likelihood = pm.MvNormal("obs", mu=mu, cov=Sigma, observed=observed_data)

        # Sample
        trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.9)

    # Posterior mean of nu
    # This is used to construct the portfolio weights
    posterior_mean_nu = np.exp(
        trace.posterior["log_nu"].mean(dim=("chain", "draw")).values
    )

    # We used the 'posterior_mean_nu' to make the weights for portfolio
    # Formula: (1/gamma)*nu - There is no short selling constraint
    portfolio_comp_df = pd.DataFrame(
        {"Weight": 1 / risk_aversion * posterior_mean_nu},
        index=sotck_names,
    )

    # Rename the index to 'Stock'
    portfolio_comp_df.index.name = "Stock"

    return portfolio_comp_df
