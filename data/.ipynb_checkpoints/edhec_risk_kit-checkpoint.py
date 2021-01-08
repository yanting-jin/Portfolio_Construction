import numpy as np
import pandas as pd
from scipy.stats import norm   

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


def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns

    """
    ind = pd.read_csv("/Users/Jinyanting/Documents/GitHub/PortfolioConstruction/Week1/data/ind30_m_vw_rets.csv",header = 0, index_col=0, parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_total_market_index_returns():
    """

    """
    ind_returns = get_ind_returns()
    ind_size = get_ind_size()
    ind_nfirms = get_ind_nfirms()
    ind_mktcap = ind_nfirms*ind_nfirms
    total_mktcap = ind_mktcap.sum(axis = 'columns')
    
    ind_capweight = ind_mktcap.divide(total_mktcap, axis = 'rows')
    total_market_return = (ind_capweight*ind_returns).sum(axis = 'columns')
     #total_market_index=erk.drawdown(total_market_return).Wealth 
    return total_market_return


def get_ind_size():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns

    """
    import pandas as pd
    ind = pd.read_csv("/Users/Jinyanting/Documents/GitHub/PortfolioConstruction/Week1/data/ind30_m_size.csv",header = 0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns

    """
    import pandas as pd
    ind = pd.read_csv("/Users/Jinyanting/Documents/GitHub/PortfolioConstruction/Week1/data/ind30_m_nfirms.csv",header = 0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

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

def annualize_vol(r, period_per_year):
    return r.std()*(period_per_year**0.5)

def annualize_rets(r, period_per_year):
    compounded_return = (1+r).prod()
    time = period_per_year/r.shape[0]
    return compounded_return**(time)-1

def sharpe_ratio(r, riskfree_rate,period_per_year):
    rf_per_period = (1+riskfree_rate)**(1/period_per_year)-1
    excess_ret = r-rf_per_period
    ann_ex_ret = annualize_rets(excess_ret,period_per_year)
    ann_vol = annualize_vol(r, period_per_year)
    return ann_ex_ret/ann_vol


def portfolio_return(weights, returns):
    """
    Weights --> Returns
    """
    return weights.T@returns

def portfolio_vol(weights, covmat):
    """
    Weights --> Vol
    """
    return (weights.T@covmat@weights)**0.5


def plot_ef2(er, cov, n_points):
    """
    Plot the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError('plot_ef2 can only plot 2-asset frontiers')
        
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets = [ portfolio_return(w, er) for w in weights]
    vols = [ portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns":rets,
                       "Volatility": vols})
    return ef.plot.line(x='Volatility', y = 'Returns',style='.-')


from scipy.optimize import minimize
def minimize_vol(target_return, er, cov):
    """
    Given a target return, 
    Output the weights for this portfoli that has the minimun volatility
    """
    n = er.shape[0] # number of stocks to optimize
    init_guess = np.repeat(1/n,n) # initiate equal weights
    bounds  = ((0.0,1.0),)*n
    
    return_is_target = {
        'type':'eq',
        'args': (er, ),
        'fun':lambda weights, er: target_return - portfolio_return(er,weights)
    }
    
    weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights:np.sum(weights)-1
    }
        
    results = minimize(portfolio_vol, init_guess,
                        args = (cov,), method="SLSQP",
                       options = {'disp':False},
                       constraints = (return_is_target,weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x

def msr(riskfree_rate, er, cov):
    """
    maximum sharpe ratio portfolio
    Given a target return, covariance of the stocks, and the riskfree rate 
    Output the weights for this portfoli that has the minimun volatility
    """
    n = er.shape[0] # number of stocks to optimize
    init_guess = np.repeat(1/n,n) # initiate equal weights
    bounds  = ((0.0,1.0),)*n       
    weights_sum_to_1 = {
        'type':'eq',
        'fun': lambda weights:np.sum(weights)-1
    }  
    
    def neg_sharpe_ratio(weights,riskfree_rate, er, cov):
        r = portfolio_return(weights,er)
        vol = portfolio_vol(weights,cov)
        return -(r-riskfree_rate)/vol
    results = minimize(neg_sharpe_ratio, init_guess,
                        args = (riskfree_rate,er, cov,), method="SLSQP",
                       options = {'disp':False},
                       constraints = (weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x

def optimum_weights(n_points,er, cov):
    """
    List of wegights to run the optimizer that returns the optimum weights
    """
    target_rets = np.linspace(er.min(),er.max(),n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rets]
    return weights

def gmv(cov):
    """""
    Return the weight for the global minimum Vol Portoflio
    given the covariance matrix
    """""
    n = cov.shape[0] # numner of assets
    return msr(0, np.repeat(1,n), cov)
    

def plot_ef(n_points,er, cov, riskfree_rate=0,show_cml=False, show_ew=False,show_gmv=False, style='.-' ):
    weights = optimum_weights(n_points,er,cov)
    rets = [ portfolio_return(w, er) for w in weights]
    vols = [ portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns":rets,
        "Volatility": vols
    })
    ax = ef.plot.line(x='Volatility', y = 'Returns',style='.-')
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n,n)
        r_ew = portfolio_return(w_ew,er)
        vol_ew = portfolio_vol(w_ew,cov)
        #Add Equal Weight portfolio
        ax.plot([vol_ew],[r_ew],color = "goldenrod", marker='o',markersize = 10)
    
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv,er)
        vol_gmv = portfolio_vol(w_gmv,cov)
        #Add Equal Weight portfolio
        ax.plot([vol_gmv],[r_gmv],color = "midnightblue", marker='o',markersize = 10)      
        
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate,er,cov)
        r_msr=portfolio_return(w_msr,er)
        vol_msr=portfolio_vol(w_msr,cov)
        # Add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x,cml_y, color='green', marker ='o',linestyle='dashed',markersize=12, linewidth=2)      
    return ax
        
def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03,drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the riksy asset
    Returns a dictionary containing: asset value history, risk budget history, riksy weight history
    """
    # CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = start
    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=['R'])
        
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12  

    ## use DataFrame for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_weight_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak,account_value)
            floor_value = peak*(1-drawdown)
        
        cushion = (account_value - floor_value)/account_value # in percentage
        risky_weight = m*cushion
        risky_weight =np.minimum(risky_weight,1) #compare risky_weight within 1 and return the smaller one, thus constrain risky_weight <1
        risky_weight =np.maximum(risky_weight,0)
        safe_weight = 1- risky_weight
        risky_alloc = account_value*risky_weight
        safe_alloc = account_value*safe_weight

        # Update the account value for this time step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])

        # save the values so as to track the history
        cushion_history.iloc[step] = cushion
        risky_weight_history.iloc[step] = risky_weight
        account_history.iloc[step] = account_value
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_weight_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r
    }
    return backtest_result

def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, period_per_year=12)
    ann_vol = r.aggregate(annualize_vol, period_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, period_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdowns.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })

    