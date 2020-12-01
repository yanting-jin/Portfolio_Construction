import pandas as pd

def drawdown(returns_series:pd.Series):
    """
    Takes a time series of asset returns
    Compute and returns a DataFrame that contrains:
    the wealth index
    the previous peaks
    percent drawdowns
    """
    wealth_index = 1000*(1+returns_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks"  : previous_peaks,
        "Drawdowns":drawdowns,
    })

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Declines by MarketCap
    """
    me_m=pd.read_csv('/Users/Jinyanting/Documents/GitHub/PortfolioConstruction/Week1/data/Portfolios_Formed_on_ME_monthly_EW.csv',
                     header=0, index_col=0, parse_dates=True, na_values=-99.99)

    returns = me_m[['Lo 10','Hi 10']]
    returns.columns=['SmallCap','LargeCap']
    returns = returns/100
    returns.index = pd.to_datetime(returns.index, format = "%Y%m").to_period('M')
    return returns


def get_hfi_returns():
    """
    Load and format the EDHEC Hege Fund Index Returns
    """
    hfi = pd.read_csv('/Users/Jinyanting/Documents/GitHub/PortfolioConstruction/Week1/data/edhec-hedgefundindices.csv',
                     header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def semideviation(r):
    return r[r<0].std(ddof=0)

def skewness(r):
    """
    Alternative to scipy.state.skew()
    Compute the skewness of the supplied DataFrame
    Returns a float or a Series
    """
    demeaned_r  = (r-r.mean())**3
    sigma_r = r.std(ddof = 0)
    
    return demeaned_r.mean()/sigma_r**3
    
    
def kurtosis(r):
    """
    Alternative to scipy.state.kurtosis()
    Compute the skewness of the supplied DataFrame
    Returns a float or a Series
    """
    demeaned_r  = (r-r.mean())**4
    sigma_r = r.std(ddof = 0)
    
    return demeaned_r.mean()/sigma_r**4

import scipy.stats
def is_norm(r, level=0.01):
    statistics, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


import numpy as np
def var_historic(r, level=5):
    """
    Var Historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level) # call this var_historic function on every column
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expect r to be DataFrame or Series")

def cvar_historic(r, level=5):
    """
    Compute the Conditional VaR of Series or DataFrame 
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level) # a mask that selects the values that is smaller than the VaR
        return -r[is_beyond].mean()
    
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level) # call this cvar_historic function on every column
    
    else:
        raise TypeError("Expect r to be DataFrame or Series")
        
from scipy.stats import norm       
def var_gaussian(r, level=5, modified = False):
    """
    Returns the Parametric Gaussian VaR of a Series of DataFrame
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z scorebased on observaed skewness and kurtosis
        s = skewness(r) 
        k = kurtosis(r)
        z = (z + 
            (z**2-1)*s/6 + 
            (z**3-3*z)*(k-3)/24 -
            (2*z**3-5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))