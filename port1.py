# portfolio_app.py (updated: shared sidebar used in all tabs)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
st.title("📊 Portfolio Analysis")

# ============================ SIDEBAR ============================
st.sidebar.header("⚙️ Portfolio Settings")
tickers = [t.strip().upper() for t in st.sidebar.text_input("Enter ticker symbols", "AAPL, MSFT, GOOGL").split(",")]
start = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# ============================ UTILS ============================
def get_price_data(tickers, start, end):
    tickers = [t.strip().upper() for t in tickers]
    data = yf.download(tickers, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(axis=1, how='all')
    return data

def compute_returns(price_data):
    return price_data.pct_change(fill_method=None).dropna()

def compute_portfolio_returns(price_data, weights):
    returns = compute_returns(price_data)
    weighted_returns = returns.dot(weights)
    cumulative_returns = (1 + weighted_returns).cumprod()
    return weighted_returns, cumulative_returns

def portfolio_stats(weighted_returns):
    mean_return = weighted_returns.mean()
    volatility = weighted_returns.std()
    sharpe_ratio = mean_return / volatility
    return mean_return, volatility, sharpe_ratio


def calculate_risk_contributions(weights, returns):
    """
    Calculate risk contributions for each asset in the portfolio.
    
    Parameters:
    -----------
    weights : np.array or list
        Portfolio weights
    returns : pd.DataFrame
        Historical returns for each asset
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing weights, risk contributions, 
        marginal contributions, and other risk metrics
    """
    # Convert to numpy array if not already
    weights_array = np.array(weights)
    
    # Calculate covariance matrix
    cov_matrix = returns.cov().values
    
    # Calculate portfolio volatility
    port_vol = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
    
    # Calculate marginal contribution to risk (MCTR)
    mctr = np.dot(cov_matrix, weights_array) / port_vol
    
    # Calculate component contribution to risk (CCTR)
    cctr = weights_array * mctr
    
    # Percentage contribution to risk (PCTR)
    pctr = cctr / port_vol
    
    # Create a DataFrame with all risk metrics
    risk_df = pd.DataFrame({
        'Asset': returns.columns,
        'Weight': weights_array,
        'Volatility': np.sqrt(np.diag(cov_matrix)),
        'Marginal Contribution': mctr,
        'Risk Contribution': cctr,
        'Risk Contribution (%)': pctr,
        'Risk/Weight Ratio': pctr / weights_array
    })
    
    return risk_df.sort_values('Risk Contribution (%)', ascending=False)


# ============================ TABS ============================
tabs = st.tabs(["Asset Analysis", "Portfolio Comparison", "Mean Risk", "Risk Building"])


# -------------------- 1. Asset Analysis --------------------
with tabs[0]:
    # show_model_inputs = False
    # st.header("📈 Asset Analysis")
    if not tickers:
        st.warning("Please enter at least one ticker in the sidebar.")
        st.stop()
    prices = get_price_data(tickers, start, end)
    if not prices.empty:
        st.subheader("ℹ️ Asset Information")
        asset_info = []
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                asset_info.append({"Ticker": ticker, "Sector": info.get("sector", "N/A"), "Industry": info.get("industry", "N/A"), "Market Cap": f"${info.get('marketCap', 0) / 1e9:.2f} B" if info.get('marketCap') else "N/A", "Country": info.get("country", "N/A")})
            except:
                asset_info.append({"Ticker": ticker, "Sector": "N/A", "Industry": "N/A", "Market Cap": "N/A", "Country": "N/A"})
        df_info = pd.DataFrame(asset_info).set_index("Ticker")
        st.dataframe(df_info)

        st.subheader("📈 Price Chart")
        st.line_chart(prices)

        st.subheader("📈 Daily Returns")
        returns = compute_returns(prices)
        st.line_chart(returns)

        returns = compute_returns(prices)

        st.subheader("📊 Basic Statistics")
        stats = pd.DataFrame({"Mean Return": returns.mean(), "Volatility": returns.std(), "Sharpe Ratio": returns.mean() / returns.std()})
        stats.index.name = "Ticker"
        stats_reset = stats.reset_index()
        stats_melt = stats_reset.melt(id_vars="Ticker", var_name="Metric", value_name="Value")
        fig = px.bar(stats_melt, x="Ticker", y="Value", color="Metric", barmode="group", title="Mean Return, Volatility, and Sharpe Ratio by Asset")
        st.plotly_chart(fig, use_container_width=True)
        # if st.toggle("📋 Show raw statistics table"):
        #     st.dataframe(stats.style.format("{:.4f}"))

        st.subheader("📊 Correlation Heatmap")
        corr_matrix = returns.corr().round(2)
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu", title="Correlation Matrix of Returns")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏢 Sector Allocation")
        equal_weights = [1 / len(tickers)] * len(tickers)
        sectors = {}
        for i, ticker in enumerate(tickers):
            try:
                sector = yf.Ticker(ticker).info.get("sector", "Unknown")
                sectors[sector] = sectors.get(sector, 0) + equal_weights[i]
            except:
                sectors["Unknown"] = sectors.get("Unknown", 0) + equal_weights[i]
        total = sum(sectors.values())
        sector_weights = {k: v / total for k, v in sectors.items()}
        sector_df = pd.DataFrame(list(sector_weights.items()), columns=["Sector", "Weight"])
        fig = px.pie(sector_df, values="Weight", names="Sector", title="Sector Allocation", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

        
    else:
        st.warning("No data available for selection.")

# -------------------- 2. Portfolio Comparison --------------------
with tabs[1]:
    # st.header("🔄 Portfolio Comparison")
    
    if not tickers:
        st.warning("Please enter at least one ticker in the sidebar.")
        st.stop()

    prices = get_price_data(tickers, start, end)
    if prices.empty:
        st.warning("No price data returned for selected tickers.")
        st.stop()

    returns = compute_returns(prices)

    # Fix shape mismatch by filtering valid tickers
    valid_tickers = list(returns.columns)
    if set(tickers) != set(valid_tickers):
        st.warning(f"Some tickers dropped due to missing data: {set(tickers) - set(valid_tickers)}")
    tickers = valid_tickers

    # Options for portfolio optimization
    st.subheader("⚙️ Portfolio Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        allow_short = st.checkbox("Allow Short Selling", value=False, key="portfolio_comparison_short_selling")
        min_weight = -1.0 if allow_short else 0.0
    
    with col2:
        rebalance_frequency = st.selectbox(
            "Rebalance Frequency",
            ["None", "Monthly", "Quarterly", "Annually"],
            index=0,
            help="How often to rebalance the portfolio back to target weights"
        )

    # Model selection
    model_options = [
        "Equal Weighted",
        "Inverse Volatility",
        "Random",
        "Minimum Variance",
        "Target Return Portfolio",
        "Maximum Diversification",
        "Maximum Sharpe Ratio",
        "Minimum CVaR",
        "Risk Parity (Variance)",
        "Risk Budgeting (CVaR)",
        "Risk Parity (Covariance Shrinkage)",
        "Hierarchical Risk Parity (HRP)",
        "Bayesian Mean-Variance"
    ]
    
    selected_models = st.multiselect(
        "📍 Select Models to Compare", 
        model_options, 
        default=["Equal Weighted", "Minimum Variance", "Maximum Sharpe Ratio", "Hierarchical Risk Parity (HRP)"]
    )
    
    # -------------------- Dynamic Inputs for Models --------------------
    with st.expander("Advanced Model Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if "Target Return Portfolio" in selected_models:
                target_return = st.number_input(
                    "🎯 Target Annual Return (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=0.5,
                    help="Set the minimum annual return you'd like this portfolio to target."
                ) / 100
            else:
                target_return = 0.05
                
            if "Bayesian Mean-Variance" in selected_models:
                shrinkage_level = st.slider(
                    "Shrinkage Level (τ)",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.05,
                    step=0.01,
                    help="Lower = rely more on historical data. Higher = rely more on prior belief."
                )
            else:
                shrinkage_level = 0.05
        
        with col2:
            if "Minimum CVaR" in selected_models or "Risk Budgeting (CVaR)" in selected_models:
                alpha = st.slider(
                    "CVaR Confidence Level (α)",
                    min_value=0.85,
                    max_value=0.99,
                    step=0.01,
                    value=0.95,
                    help="Tail threshold for CVaR calculations."
                )
            else:
                alpha = 0.95
                
            if "Risk Parity (Covariance Shrinkage)" in selected_models:
                shrinkage_lambda = st.slider(
                    "Covariance Shrinkage Level (λ)",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    value=0.1,
                    help="0 = pure sample covariance, 1 = pure diagonal target"
                )
            else:
                shrinkage_lambda = 0.1

    # Dictionary to store portfolio weights
    weight_dict = {}
    
    # ------------------------ Model Implementations ------------------------
    # Show a spinner while calculating weights
    with st.spinner("Calculating optimal portfolios..."):
        vol = returns.std()
        
        if "Equal Weighted" in selected_models:
            w_eq = np.ones(len(tickers)) / len(tickers)
            weight_dict["Equal Weighted"] = w_eq
        
        if "Inverse Volatility" in selected_models:
            inv_vol = 1 / vol
            inv_vol = inv_vol / inv_vol.sum()
            weight_dict["Inverse Volatility"] = inv_vol.values
        
        if "Random" in selected_models:
            # Create reproducible random weights by setting seed
            np.random.seed(42)
            w = np.random.random(len(tickers))
            if not allow_short:
                w = np.abs(w)
            w /= np.sum(np.abs(w))
            weight_dict["Random"] = w
        
        if "Minimum Variance" in selected_models:
            # Get the covariance matrix
            cov = returns.cov().values
            
            # Add a small regularization term to ensure positive definiteness
            n_assets = cov.shape[0]
            cov_reg = cov + np.eye(n_assets) * 1e-8
            
            # Define the objective function for portfolio variance
            def portfolio_variance(w):
                return np.dot(w.T, np.dot(cov_reg, w))
            
            # Define constraints: weights sum to 1
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            
            # Define bounds based on whether short-selling is allowed
            bounds = tuple((min_weight, 1) for _ in range(n_assets))
            
            # Initial guess: equal weights
            init_guess = np.ones(n_assets) / n_assets
            
            # Try multiple starting points to avoid local minima
            best_result = None
            best_variance = float('inf')
            
            for attempt in range(3):
                if attempt == 0:
                    start_guess = init_guess
                else:
                    # Random starting point that sums to 1
                    rand_weights = np.random.random(n_assets)
                    start_guess = rand_weights / np.sum(rand_weights)
                
                # Run optimization with relaxed tolerance
                try:
                    result = minimize(
                        portfolio_variance, 
                        start_guess, 
                        method='SLSQP', 
                        bounds=bounds, 
                        constraints=cons,
                        options={'maxiter': 1000, 'ftol': 1e-9}
                    )
                    
                    # If optimization succeeded and found a lower variance
                    if result.success and result.fun < best_variance:
                        best_result = result
                        best_variance = result.fun
                except Exception as e:
                    pass
            
            # Use the best result if available
            if best_result is not None and best_result.success:
                # Ensure weights sum to 1 (fix any small numerical issues)
                weights = best_result.x
                weights = weights / np.sum(np.abs(weights))
                weight_dict["Minimum Variance"] = weights
        
        if "Target Return Portfolio" in selected_models:
            mean_ret = returns.mean() * 252  # annualized
            cov = returns.cov() * 252
            
            def portfolio_var(w):
                return np.dot(w.T, np.dot(cov, w))
            
            cons = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'ineq', 'fun': lambda w: np.dot(w, mean_ret) - target_return}
            )
            bounds = tuple((min_weight, 1) for _ in range(len(tickers)))
            init_guess = np.ones(len(tickers)) / len(tickers)
            
            try:
                result = minimize(portfolio_var, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
                if result.success:
                    weights = result.x / np.sum(np.abs(result.x))
                    weight_dict["Target Return Portfolio"] = weights
            except:
                st.warning(f"Target Return of {target_return*100:.1f}% could not be achieved. Try a lower target.")
        
        if "Maximum Diversification" in selected_models:
            cov = returns.cov()
            vol_vec = vol.values
            
            def diversification_ratio(w):
                portfolio_vol = np.sqrt(np.dot(w.T, np.dot(cov.values, w)))
                weighted_avg_vol = np.dot(np.abs(w), vol_vec)
                return -weighted_avg_vol / portfolio_vol  # Negative for minimization
            
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = tuple((min_weight, 1) for _ in range(len(tickers)))
            init_guess = np.ones(len(tickers)) / len(tickers)
            
            result = minimize(diversification_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
            if result.success:
                weights = result.x / np.sum(np.abs(result.x))
                weight_dict["Maximum Diversification"] = weights
        
        if "Maximum Sharpe Ratio" in selected_models:
            mean_ret = returns.mean()
            cov = returns.cov()
            
            def negative_sharpe(w):
                port_ret = np.dot(w, mean_ret)
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov.values, w)))
                return -port_ret / port_vol if port_vol > 0 else 0
            
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = tuple((min_weight, 1) for _ in range(len(tickers)))
            init_guess = np.ones(len(tickers)) / len(tickers)
            
            result = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
            if result.success:
                weights = result.x / np.sum(np.abs(result.x))
                weight_dict["Maximum Sharpe Ratio"] = weights
        
        if "Minimum CVaR" in selected_models:
            returns_array = returns.values
            
            def portfolio_cvar(weights):
                port_ret = np.dot(returns_array, weights)
                var_thresh = np.percentile(port_ret, (1 - alpha) * 100)
                tail_returns = port_ret[port_ret <= var_thresh]
                cvar = -tail_returns.mean() if len(tail_returns) > 0 else -var_thresh
                return cvar
            
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = tuple((min_weight, 1) for _ in range(len(tickers)))
            init_guess = np.ones(len(tickers)) / len(tickers)
            
            result = minimize(portfolio_cvar, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
            if result.success:
                weights = result.x / np.sum(np.abs(result.x))
                weight_dict["Minimum CVaR"] = weights
        
        if "Risk Parity (Variance)" in selected_models:
            cov = returns.cov().values
            
            def risk_parity_objective(weights):
                # Avoid division by zero by setting a minimum weight
                weights = np.maximum(weights, 1e-8)
                weights = weights / np.sum(np.abs(weights))  # Normalize weights
                
                portfolio_variance = np.dot(weights.T, np.dot(cov, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # Calculate risk contributions
                marginal_risk_contribution = np.dot(cov, weights)
                risk_contribution = weights * marginal_risk_contribution / portfolio_volatility
                
                # Target: equal risk contribution
                target_risk = portfolio_volatility / len(weights)
                
                # Return sum of squared deviations
                return np.sum((risk_contribution - target_risk) ** 2)
            
            # Multi-start optimization to avoid local minima
            best_weights = None
            best_score = float('inf')
            
            # Define constraints and bounds
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            bounds = tuple((max(0.001, min_weight), 1.0) for _ in range(len(tickers)))
            
            # Try multiple starting points (3 attempts)
            for attempt in range(3):
                if attempt == 0:
                    init_guess = np.ones(len(tickers)) / len(tickers)  # Equal weight
                else:
                    init_guess = np.random.random(len(tickers))
                    init_guess = init_guess / np.sum(init_guess)
                
                # Run optimization with relaxed tolerance
                result = minimize(
                    risk_parity_objective,
                    init_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 500, 'ftol': 1e-8}
                )
                
                if result.success and (best_weights is None or result.fun < best_score):
                    best_score = result.fun
                    best_weights = result.x
                    
                    # If we're very close to perfect solution, break early
                    if result.fun < 1e-6:
                        break
            
            # Use the best result or fall back to equal weights
            if best_weights is not None:
                # Ensure weights sum to 1
                best_weights = best_weights / np.sum(np.abs(best_weights))
                weight_dict["Risk Parity (Variance)"] = best_weights
        
        if "Risk Budgeting (CVaR)" in selected_models:
            # Get returns data
            returns_array = returns.values
            
            # Define CVaR calculation function
            def compute_cvar(weights):
                port_ret = np.dot(returns_array, weights)
                var_thresh = np.percentile(port_ret, (1 - alpha) * 100)
                # Filter returns below VaR and calculate mean
                tail_returns = port_ret[port_ret <= var_thresh]
                if len(tail_returns) > 0:
                    cvar = -tail_returns.mean()
                else:
                    cvar = -var_thresh  # Fallback if no returns are in the tail
                return cvar
            
            # Calculate marginal CVaR for each asset
            def marginal_cvar(weights):
                epsilon = 1e-5  # Small perturbation for numerical differentiation
                base_cvar = compute_cvar(weights)
                mcvar = np.zeros(len(weights))
                
                for i in range(len(weights)):
                    # Create perturbed weights
                    w_up = weights.copy()
                    
                    # Ensure we don't make any weight negative
                    if weights[i] > epsilon:
                        w_up[i] += epsilon
                    else:
                        w_up[i] = epsilon
                        
                    # Renormalize weights to sum to 1
                    w_up = w_up / np.sum(np.abs(w_up))
                    
                    # Calculate marginal change in CVaR
                    cvar_up = compute_cvar(w_up)
                    mcvar[i] = (cvar_up - base_cvar) / epsilon
                
                return mcvar
            
            # Define risk budget objective function
            def cvar_risk_budget_obj(weights):
                # Ensure weights are valid and sum to 1
                weights = np.maximum(weights, 1e-8)
                weights = weights / np.sum(np.abs(weights))
                
                # Calculate marginal CVaR
                mc = marginal_cvar(weights)
                
                # Calculate risk contribution
                rc = weights * mc
                
                # Calculate total risk
                total_risk = np.sum(rc)
                
                # Target equal risk contribution
                target_risk = total_risk / len(weights)
                
                # Minimize sum of squared deviations from target
                risk_diffs = rc - target_risk
                return np.sum(risk_diffs**2)
            
            # Optimization constraints
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            
            # Bound weights based on short-selling preference
            bounds = tuple((max(0.01, min_weight), 1) for _ in range(len(tickers)))
            
            # Initial guess - equal weights
            init_guess = np.ones(len(tickers)) / len(tickers)
            
            # Use SLSQP optimizer with multiple starting points
            best_result = None
            best_obj = float('inf')
            
            # Try multiple starting points for better convergence
            for _ in range(3):
                if _ > 0:  # Skip first iteration as it uses equal weights
                    # Randomize starting weights for subsequent attempts
                    rand_weights = np.random.random(len(tickers))
                    init_guess = rand_weights / np.sum(rand_weights)
                    
                # Run optimization
                attempt = minimize(
                    cvar_risk_budget_obj, 
                    init_guess, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=cons,
                    options={'maxiter': 500}
                )
                
                # Keep best result
                if attempt.success and attempt.fun < best_obj:
                    best_result = attempt
                    best_obj = attempt.fun
            
            # Use the best result if successful, otherwise use equal weights
            if best_result is not None and best_result.success:
                # Normalize weights to ensure they sum to 1
                result_weights = best_result.x
                result_weights = result_weights / np.sum(np.abs(result_weights))
                weight_dict["Risk Budgeting (CVaR)"] = result_weights
        
        if "Risk Parity (Covariance Shrinkage)" in selected_models:
            # Original covariance matrix
            S = returns.cov().values
            
            # Target matrix (diagonal with average variance)
            avg_var = np.mean(np.diag(S))
            T = np.eye(len(tickers)) * avg_var
            
            # Shrinkage
            shrunk_cov = shrinkage_lambda * T + (1 - shrinkage_lambda) * S
            
            def risk_parity_shrinkage(weights):
                weights = np.maximum(weights, 1e-8)  # Avoid numerical issues
                weights = weights / np.sum(np.abs(weights))
                
                port_vol = np.sqrt(np.dot(weights.T, np.dot(shrunk_cov, weights)))
                mrc = np.dot(shrunk_cov, weights) / port_vol
                rc = weights * mrc
                rc = rc / np.sum(rc)  # Normalize to percentage
                
                target = np.ones(len(weights)) / len(weights)  # Equal target
                return np.sum((rc - target) ** 2)
            
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = tuple((max(0.001, min_weight), 1) for _ in range(len(tickers)))
            init_guess = np.ones(len(tickers)) / len(tickers)
            
            # Try multiple starting points
            best_result = None
            best_score = float('inf')
            
            for attempt in range(3):
                if attempt > 0:
                    init_guess = np.random.random(len(tickers))
                    init_guess = init_guess / np.sum(init_guess)
                
                result = minimize(
                    risk_parity_shrinkage, 
                    init_guess, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=cons,
                    options={'maxiter': 500}
                )
                
                if result.success and result.fun < best_score:
                    best_score = result.fun
                    best_result = result
            
            if best_result is not None and best_result.success:
                weights = best_result.x / np.sum(np.abs(best_result.x))
                weight_dict["Risk Parity (Covariance Shrinkage)"] = weights
        
        if "Hierarchical Risk Parity (HRP)" in selected_models:
            # Get correlation and covariance matrices
            cov = returns.cov().values
            corr = returns.corr().values
            
            # Step 1: Compute distance matrix from correlation
            dist = np.sqrt(0.5 * (1 - corr))
            dist = np.clip(dist, 0, 1)  # Ensure valid distance
            
            # Step 2: Hierarchical clustering
            linkage_matrix = linkage(squareform(dist), method='single')
            
            # Step 3: Quasi-diagonalization (sort order)
            def get_quasi_diag(linkage_matrix):
                leaf_order = dendrogram(linkage_matrix, no_plot=True)['leaves']
                return leaf_order
            
            sorted_idx = get_quasi_diag(linkage_matrix)
            
            # Step 4: Recursive bisection
            def hrp_allocation(cov, sort_order):
                weights = pd.Series(1.0, index=sort_order)
                cluster_items = [sort_order]
                
                while cluster_items:
                    new_clusters = []
                    for cluster in cluster_items:
                        if len(cluster) <= 1:
                            continue
                        split = len(cluster) // 2
                        left = cluster[:split]
                        right = cluster[split:]
                        
                        # Calculate inverse variance (for weighting the clusters)
                        var_left = np.diag(cov)[left].sum()
                        var_right = np.diag(cov)[right].sum()
                        
                        # Weight inversely proportional to variance
                        if var_left + var_right > 0:
                            alpha = 1 - var_left / (var_left + var_right)
                        else:
                            alpha = 0.5  # Equal weight if variances are zero
                        
                        weights[left] *= alpha
                        weights[right] *= (1 - alpha)
                        
                        new_clusters += [left, right]
                    cluster_items = new_clusters
                
                # Normalize weights
                return weights / weights.sum()
            
            hrp_weights = hrp_allocation(cov, sorted_idx)
            
            # Reorder to match the original ticker order
            w_final = np.zeros(len(tickers))
            for i, idx in enumerate(sorted_idx):
                w_final[idx] = hrp_weights.iloc[i]
                
            # Make sure weights sum to 1
            w_final = w_final / np.sum(np.abs(w_final))
            weight_dict["Hierarchical Risk Parity (HRP)"] = w_final
        
        if "Bayesian Mean-Variance" in selected_models:
            tau = shrinkage_level
            # Get sample statistics
            sample_mean = returns.mean()
            sample_cov = returns.cov()
            
            # Prior belief (shrinkage target)
            mu_0 = pd.Series(sample_mean.mean(), index=sample_mean.index)
            
            # Create prior covariance matrix
            prior_cov = tau * np.eye(len(tickers)) * sample_cov.values.mean()
            
            # Calculate posterior distribution (combine prior with data)
            try:
                post_cov = np.linalg.inv(
                    np.linalg.inv(sample_cov.values) + np.linalg.inv(prior_cov)
                )
                post_mean = post_cov @ (
                    np.linalg.inv(sample_cov.values) @ sample_mean.values +
                    np.linalg.inv(prior_cov) @ mu_0.values
                )
                
                # Optimize for maximum Sharpe ratio using posterior
                def neg_sharpe(w):
                    port_ret = np.dot(w, post_mean)
                    port_vol = np.sqrt(np.dot(w.T, np.dot(sample_cov.values, w)))
                    return -port_ret / port_vol if port_vol > 0 else 0
                
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
                bounds = tuple((min_weight, 1) for _ in range(len(tickers)))
                init_guess = np.ones(len(tickers)) / len(tickers)
                
                result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
                if result.success:
                    weights = result.x / np.sum(np.abs(result.x))
                    weight_dict["Bayesian Mean-Variance"] = weights
            except:
                st.warning("Bayesian optimization failed, possibly due to matrix inversion issues.")
    
    # ---------------- Portfolio Weight Visualization ----------------
    if weight_dict:
        st.subheader("📊 Portfolio Composition")
        weight_df = pd.DataFrame(weight_dict, index=tickers)
        weight_df = weight_df.T  # portfolios as rows
        
        fig = px.bar(
            weight_df,
            barmode="relative",
            orientation="v",
            title="Portfolio Allocation by Model",
            labels={"value": "Weight", "index": "Portfolios", "variable": "Assets"},
        )
        fig.update_layout(
            height=500,
            yaxis_tickformat=".0%", 
            xaxis_title="Portfolios", 
            yaxis_title="Weight"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Allow users to view raw weights
        if st.checkbox("Show Portfolio Weights Table"):
            st.dataframe(
                weight_df.style.format("{:.4f}").background_gradient(cmap="YlGnBu", axis=1),
                use_container_width=True
            )
        
        # ---------------- Train-Test Split ----------------
        st.subheader("📆 Backtest Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            split_slider = st.slider(
                "Train-Test Split (%)",
                min_value=50,
                max_value=95,
                value=80,
                step=5,
                help="Percentage of data to use for training (rest will be for testing)"
            )
        
        with col2:
            transaction_cost = st.number_input(
                "Transaction Cost (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.1,
                step=0.05,
                help="Cost of trading as a percentage of transaction value"
            ) / 100
        
        # Calculate split point
        split_index = int(len(returns) * (split_slider / 100))
        train_returns = returns.iloc[:split_index]
        test_returns = returns.iloc[split_index:]
        split_date = returns.index[split_index]
        
        st.caption(f"Training period: {returns.index[0].date()} to {split_date.date()} ({len(train_returns)} days) | Testing period: {split_date.date()} to {returns.index[-1].date()} ({len(test_returns)} days)")
        
        # ---------------- Portfolio Performance Calculation ----------------
        initial_amount = st.number_input(
            "💰 Initial Investment ($)",
            min_value=1000,
            value=10000,
            step=1000,
            help="Starting amount for portfolio backtest"
        )
        
        # Determine rebalancing frequency
        def get_rebalance_dates(returns_index, frequency):
            if frequency == "None":
                return [returns_index[0]]  # Only initial investment
            elif frequency == "Monthly":
                return pd.date_range(
                    start=returns_index[0],
                    end=returns_index[-1],
                    freq='MS'  # Month Start
                )
            elif frequency == "Quarterly":
                return pd.date_range(
                    start=returns_index[0],
                    end=returns_index[-1],
                    freq='QS'  # Quarter Start
                )
            elif frequency == "Annually":
                return pd.date_range(
                    start=returns_index[0],
                    end=returns_index[-1],
                    freq='AS'  # Annual Start
                )
        
        rebalance_dates = get_rebalance_dates(returns.index, rebalance_frequency)
        
        # Calculate portfolio returns
        portfolio_returns = {}
        portfolio_values = {}
        portfolio_weights_over_time = {}
        
       # Function to calculate portfolio performance with rebalancing
        def calculate_portfolio_performance(weights, returns_data, rebalance_dates, transaction_cost):
            """
            Calculate portfolio performance with rebalancing and transaction costs.
            
            Parameters:
            -----------
            weights : numpy.ndarray
                Target weights for the portfolio
            returns_data : pandas.DataFrame
                Historical returns data
            rebalance_dates : list
                Dates on which to rebalance the portfolio
            transaction_cost : float
                Transaction cost as a percentage of transaction value
                
            Returns:
            --------
            portfolio_vals : pandas.Series
                Portfolio values over time
            portfolio_rets : pandas.Series
                Portfolio returns over time
            weights_df : pandas.DataFrame
                Portfolio weights over time
            """
            # Create a copy of the returns data to avoid modifying the original
            returns_data = returns_data.copy()
            
            # Initialize arrays to store results
            dates = returns_data.index
            n_days = len(dates)
            n_assets = len(weights)
            
            # Arrays to store results
            portfolio_values = np.zeros(n_days)
            portfolio_returns = np.zeros(n_days - 1)  # One less return than values
            weight_history = np.zeros((n_days, n_assets))
            
            # Initialize portfolio
            portfolio_values[0] = initial_amount
            weight_history[0, :] = weights
            
            # Create a mask for rebalancing days
            rebalance_mask = np.zeros(n_days, dtype=bool)
            for date in rebalance_dates:
                if date in dates:
                    idx = dates.get_loc(date)
                    rebalance_mask[idx] = True
            
            # Start with initial weights
            current_weights = weights.copy()
            
            # Calculate performance day by day
            for i in range(1, n_days):
                # Get returns for the day
                day_returns = returns_data.iloc[i-1].values
                
                # Update weights based on returns
                new_weights = current_weights * (1 + day_returns)
                new_weights = new_weights / np.sum(new_weights)  # Normalize
                
                # Calculate portfolio return
                portfolio_return = np.sum(current_weights * day_returns)
                portfolio_returns[i-1] = portfolio_return
                
                # Update portfolio value
                portfolio_values[i] = portfolio_values[i-1] * (1 + portfolio_return)
                
                # Check if rebalancing is needed
                if rebalance_mask[i]:
                    # Calculate transaction costs
                    trades = np.abs(new_weights - weights)
                    trade_value = np.sum(trades) * portfolio_values[i]
                    cost = trade_value * transaction_cost
                    
                    # Apply costs
                    portfolio_values[i] -= cost
                    
                    # Reset to target weights
                    current_weights = weights.copy()
                else:
                    # Keep the new weights
                    current_weights = new_weights
                
                # Store current weights
                weight_history[i, :] = current_weights
            
            # Convert to pandas objects with proper indices
            portfolio_vals = pd.Series(portfolio_values, index=dates)
            portfolio_rets = pd.Series(portfolio_returns, index=dates[1:])
            weights_df = pd.DataFrame(weight_history, index=dates, columns=returns_data.columns)
            
            return portfolio_vals, portfolio_rets, weights_df
        
        # Calculate performance for each model
        for model_name in selected_models:
            if model_name in weight_dict:
                weights = weight_dict[model_name]
                
                # Calculate full period performance with rebalancing
                values, rets, weights_hist = calculate_portfolio_performance(
                    weights, 
                    returns, 
                    rebalance_dates,
                    transaction_cost
                )
                
                portfolio_values[model_name] = values
                portfolio_returns[model_name] = rets
                portfolio_weights_over_time[model_name] = weights_hist
        
        # ---------------- Performance Visualization ----------------
        if portfolio_values:
            # Create DataFrames
            values_df = pd.DataFrame(portfolio_values)
            returns_df = pd.DataFrame(portfolio_returns)
            
            # Add train/test split annotation
            train_mask = values_df.index < split_date
            test_mask = values_df.index >= split_date
            
            # Calculate train and test only DataFrames
            train_values = values_df[train_mask]
            test_values = values_df[test_mask]
            
            # Normalize test values to start at same point
            test_normalized = pd.DataFrame()
            for col in test_values.columns:
                if len(train_values) > 0:
                    # Get the last value from train period
                    last_train_value = train_values[col].iloc[-1]
                    # Normalize test values to start at the last train value
                    first_test_value = test_values[col].iloc[0]
                    test_normalized[col] = test_values[col] * (last_train_value / first_test_value)
                else:
                    test_normalized[col] = test_values[col]
            
            # 1. Full period performance chart
            st.subheader("📈 Portfolio Performance")
            
            # Instead of using add_vline and add_vrect with timestamps, use shapes:

            # Create figure with data
            fig = px.line(
                values_df,
                labels={"value": "Portfolio Value ($)", "index": "Date", "variable": "Model"},
                title=f"Growth of ${initial_amount:,} Investment"
            )

            # Add vertical line for train/test split using shapes
            fig.add_shape(
                type="line",
                x0=split_date,
                y0=0,
                x1=split_date,
                y1=1,
                yref="paper",
                line=dict(color="magenta", width=2, dash="dash"),
            )

            # Add annotation for the split line
            fig.add_annotation(
                x=split_date,
                y=1,
                yref="paper",
                text="Train/Test Split",
                showarrow=False,
                font=dict(color="magenta"),
                bgcolor="rgba(255,255,255,0.7)",
                borderpad=4
            )

            # Add rectangles for train and test regions using shapes
            fig.add_shape(
                type="rect",
                x0=values_df.index[0],
                y0=0,
                x1=split_date,
                y1=1,
                yref="paper",
                # fillcolor="rgba(0,100,255,0.1)",
                line_width=0,
            )

            fig.add_shape(
                type="rect",
                x0=split_date,
                y0=0,
                x1=values_df.index[-1],
                y1=1,
                yref="paper",
                # fillcolor="rgba(255,100,0,0.1)",
                line_width=0,
            )

            # Add annotations for regions
            fig.add_annotation(
                x=values_df.index[int(len(values_df.index) * 0.25)],  # 25% into train set
                y=0.95,
                yref="paper",
                text="Train Set",
                showarrow=False,
                font=dict(color="lightgreen"),
            )

            fig.add_annotation(
                x=values_df.index[int(len(values_df.index) * 0.75)],  # 25% into test set
                y=0.95,
                yref="paper",
                text="Test Set",
                showarrow=False,
                font=dict(color="orange"),
            )

            fig.update_layout(
                height=500,
                # Improve the legend positioning and appearance
                legend=dict(
                    orientation="v",            # Vertical orientation
                    yanchor="top",             # Anchor point at the top
                    y=1.0,                     # Position at the top
                    xanchor="left",            # Anchor to the left
                    x=1.05,                    # Position outside the plot area
                    borderwidth=1,              # Border width
                    itemsizing="constant",      # Keep item size consistent
                    itemclick="toggle",         # Allow clicking to toggle visibility
                    font=dict(size=12),         # Adjust font size
                    tracegroupgap=5             # Add gap between legend groups
                ),
                # Add margin to accommodate the legend
                margin=dict(r=150)  # Increase right margin to make room for legend
            )


            # Update layout
            # fig.update_layout(height=500, legend=dict(orientation="v", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. Create additional visualization tabs
            viz_tabs = st.tabs([
                "Return Analysis", 
                "Train Period Only", 
                "Test Period Only", 
                "Weight Evolution", 
                "Drawdown Analysis"
            ])
            
            # Tab 1: Return Analysis
            with viz_tabs[0]:
                # Monthly returns heatmap - now full width
                st.subheader("📅 Monthly Returns")
                
                # Select a model for monthly returns
                selected_model = st.selectbox(
                    "Select model for monthly returns", 
                    selected_models, 
                    index=0,
                    key="monthly_returns_model_selector"
                )
                
                if selected_model in portfolio_returns:
                    # Resample to monthly returns
                    monthly_returns = portfolio_returns[selected_model].resample('M').apply(
                        lambda x: (1 + x).prod() - 1
                    )
                    
                    # Create a DataFrame with year and month
                    monthly_df = pd.DataFrame({
                        'Return': monthly_returns,
                        'Year': monthly_returns.index.year,
                        'Month': monthly_returns.index.month
                    })
                    
                    # Pivot for heatmap
                    heatmap_data = monthly_df.pivot_table(
                        index='Year', 
                        columns='Month', 
                        values='Return'
                    )
                    
                    # Create heatmap
                    fig = px.imshow(
                        heatmap_data,
                        text_auto='.2%',
                        color_continuous_scale="RdYlGn",
                        labels=dict(x="Month", y="Year", color="Return"),
                        title=f"Monthly Returns: {selected_model}"
                    )
                    
                    # Update layout
                    fig.update_layout(height=400)
                    fig.update_coloraxes(colorbar_tickformat='.0%')
                    
                    # Display
                    st.plotly_chart(fig, use_container_width=True)
                
                # Return distribution - now below monthly returns
                st.subheader("📊 Return Distribution")
                
                # Controls for return distribution
                dist_freq = st.selectbox(
                    "Return frequency", 
                    ["Daily", "Weekly", "Monthly"],
                    index=0,
                    key="return_frequency_selector"
                )
                
                freq_map = {
                    "Daily": "B",
                    "Weekly": "W",
                    "Monthly": "M"
                }
                
                # Resample returns to selected frequency
                model_returns = {}
                for model in selected_models:
                    if model in portfolio_returns:
                        if dist_freq == "Daily":
                            model_returns[model] = portfolio_returns[model]
                        else:
                            model_returns[model] = portfolio_returns[model].resample(
                                freq_map[dist_freq]
                            ).apply(lambda x: (1 + x).prod() - 1)
                
                # Convert to DataFrame
                returns_dist_df = pd.DataFrame(model_returns)
                
                # Histogram of returns
                fig = px.histogram(
                    returns_dist_df,
                    nbins=50,
                    opacity=0.7,
                    barmode="overlay",
                    marginal="box",
                    title=f"{dist_freq} Return Distribution"
                )
                
                # Update layout
                fig.update_layout(
                    height=400, 
                    xaxis_title=f"{dist_freq} Return", 
                    yaxis_title="Frequency"
                )
                fig.update_xaxes(tickformat='.0%')
                
                # Display
                st.plotly_chart(fig, use_container_width=True)
                
                # Rolling performance metrics - already full width
                st.subheader("📈 Rolling Performance")
                metric_type = st.selectbox(
                    "Performance metric", 
                    ["Rolling Sharpe Ratio", "Rolling Volatility", "Rolling Return"],
                    index=0,
                    key="rolling_performance_metric"
                )
                
                window = st.slider(
                    "Rolling window (months)", 
                    min_value=3, 
                    max_value=36, 
                    value=12, 
                    step=3,
                    key="rolling_window_slider"
                )
                
                # Calculate rolling metrics
                rolling_data = {}
                for model in selected_models:
                    if model in portfolio_returns:
                        rets = portfolio_returns[model]
                        
                        if metric_type == "Rolling Sharpe Ratio":
                            rolling = rets.rolling(window=window*21).apply(
                                lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else 0
                            )
                            title = f"{window}-Month Rolling Sharpe Ratio"
                            y_title = "Sharpe Ratio"
                        
                        elif metric_type == "Rolling Volatility":
                            rolling = rets.rolling(window=window*21).std() * np.sqrt(252) * 100
                            title = f"{window}-Month Rolling Volatility"
                            y_title = "Annualized Volatility (%)"
                        
                        else:  # Rolling Return
                            rolling = rets.rolling(window=window*21).apply(
                                lambda x: (1 + x).prod() - 1
                            ) * 100
                            title = f"{window}-Month Rolling Return"
                            y_title = "Return (%)"
                        
                        rolling_data[model] = rolling
                
                # Create DataFrame
                rolling_df = pd.DataFrame(rolling_data)
                
                # Plot
                fig = px.line(
                    rolling_df,
                    labels={"value": y_title, "index": "Date", "variable": "Model"},
                    title=title
                )
                
                # Add split line
                fig.add_shape(
                    type="line",
                    x0=split_date,
                    y0=0,
                    x1=split_date,
                    y1=1,
                    yref="paper",
                    line=dict(color="magenta", width=2, dash="dash"),
                )
                
                # Update layout
                # fig.update_layout(height=500, legend=dict(orientation="h", y=1.1))
                fig.update_layout(
                    height=500,
                    # Improve the legend positioning and appearance
                    legend=dict(
                        orientation="v",            # Vertical orientation
                        yanchor="top",             # Anchor point at the top
                        y=1.0,                     # Position at the top
                        xanchor="left",            # Anchor to the left
                        x=1.05,                    # Position outside the plot area
                        borderwidth=1,              # Border width
                        itemsizing="constant",      # Keep item size consistent
                        itemclick="toggle",         # Allow clicking to toggle visibility
                        font=dict(size=12),         # Adjust font size
                        tracegroupgap=5             # Add gap between legend groups
                    ),
                    # Add margin to accommodate the legend
                    margin=dict(r=150)  # Increase right margin to make room for legend
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # For the Train Period Tab:
            with viz_tabs[1]:
                st.subheader("📘 Train Period Performance")
                
                # Create masks specifically for values data
                train_mask = values_df.index < split_date
                train_values = values_df[train_mask]


                # Percentage gain for train period
                train_gain = pd.DataFrame()
                for col in train_values.columns:
                    train_gain[col] = (train_values[col] / train_values[col].iloc[0] - 1) * 100
                
                # Plot
                fig = px.line(
                    train_gain,
                    labels={"value": "Return (%)", "index": "Date", "variable": "Model"},
                    title=f"Train Period: {train_values.index[0].date()} to {train_values.index[-1].date()}"
                )
                
                # Update layout
                fig.update_layout(height=500, yaxis_tickformat='.1f')
                st.plotly_chart(fig, use_container_width=True)
                
                # Train period statistics
                train_stats = {}
                for model in selected_models:
                    if model in portfolio_returns:
                        # Create mask specific to this model's returns
                        returns_train_mask = portfolio_returns[model].index < split_date
                        model_rets = portfolio_returns[model][returns_train_mask]
                        
                        # The rest of the code remains the same
                        total_return = (train_values[model].iloc[-1] / train_values[model].iloc[0] - 1) * 100
                        
                        # Calculate annualized metrics
                        years = (train_values.index[-1] - train_values.index[0]).days / 365.25
                        annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100
                        annualized_vol = model_rets.std() * np.sqrt(252) * 100
                        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
                        
                        # Calculate drawdown
                        roll_max = train_values[model].cummax()
                        drawdown = (train_values[model] / roll_max - 1) * 100
                        max_drawdown = drawdown.min()
                        
                        train_stats[model] = {
                            "Total Return (%)": f"{total_return:.2f}",
                            "Annualized Return (%)": f"{annualized_return:.2f}",
                            "Annualized Volatility (%)": f"{annualized_vol:.2f}",
                            "Sharpe Ratio": f"{sharpe:.2f}",
                            "Maximum Drawdown (%)": f"{max_drawdown:.2f}"
                        }
                
                # Create DataFrame
                train_stats_df = pd.DataFrame(train_stats).T
                
                # Display
                st.dataframe(train_stats_df, use_container_width=True)

            # Similarly for the Test Period Tab:
            with viz_tabs[2]:
                st.subheader("📙 Test Period Performance")
                
                # Create masks specifically for values data
                test_mask = values_df.index >= split_date
                test_values = values_df[test_mask]
                
                # Percentage gain for test period
                test_gain = pd.DataFrame()
                for col in test_values.columns:
                    test_gain[col] = (test_values[col] / test_values[col].iloc[0] - 1) * 100
                
                # Plot
                fig = px.line(
                    test_gain,
                    labels={"value": "Return (%)", "index": "Date", "variable": "Model"},
                    title=f"Test Period: {test_values.index[0].date()} to {test_values.index[-1].date()}"
                )
                
                # Update layout
                fig.update_layout(height=500, yaxis_tickformat='.1f')
                st.plotly_chart(fig, use_container_width=True)
                
                # Test period statistics
                test_stats = {}
                for model in selected_models:
                    if model in portfolio_returns:
                        # Create mask specific to this model's returns
                        returns_test_mask = portfolio_returns[model].index >= split_date
                        model_rets = portfolio_returns[model][returns_test_mask]
                        
                        # Remaining calculations are the same
                        total_return = (test_values[model].iloc[-1] / test_values[model].iloc[0] - 1) * 100
                        
                        # Calculate annualized metrics
                        years = (test_values.index[-1] - test_values.index[0]).days / 365.25
                        annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100
                        annualized_vol = model_rets.std() * np.sqrt(252) * 100
                        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
                        
                        # Calculate drawdown
                        roll_max = test_values[model].cummax()
                        drawdown = (test_values[model] / roll_max - 1) * 100
                        max_drawdown = drawdown.min()
                        
                        test_stats[model] = {
                            "Total Return (%)": f"{total_return:.2f}",
                            "Annualized Return (%)": f"{annualized_return:.2f}",
                            "Annualized Volatility (%)": f"{annualized_vol:.2f}",
                            "Sharpe Ratio": f"{sharpe:.2f}",
                            "Maximum Drawdown (%)": f"{max_drawdown:.2f}"
                        }
                
                # Create DataFrame
                test_stats_df = pd.DataFrame(test_stats).T
                
                # Display
                st.dataframe(test_stats_df, use_container_width=True)
            
            # Tab 4: Weight Evolution
            with viz_tabs[3]:
                st.subheader("⚖️ Portfolio Weight Evolution")
                
                # Select a model for weight analysis
                weight_model = st.selectbox(
                    "Select model for weight evolution", 
                    selected_models
                )
                
                if weight_model in portfolio_weights_over_time:
                    weights_over_time = portfolio_weights_over_time[weight_model]
                    
                    # Resample to monthly for cleaner visualization
                    weights_monthly = weights_over_time.resample('M').last()
                    
                    # Plot
                    fig = px.area(
                        weights_monthly,
                        labels={"value": "Weight", "index": "Date", "variable": "Asset"},
                        title=f"Weight Evolution: {weight_model}"
                    )
                    
                    # Add split line
                    fig.add_vline(
                        x=split_date, 
                        line_dash="dash", 
                        line_color="magenta"
                    )
                    
                    # Update layout
                    fig.update_layout(height=500, yaxis_tickformat='.0%')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Weight statistics
                    st.subheader("Weight Statistics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Average weights
                        avg_weights = weights_over_time.mean()
                        avg_df = pd.DataFrame({
                            "Asset": avg_weights.index,
                            "Average Weight": avg_weights.values
                        }).sort_values("Average Weight", ascending=False)
                        
                        st.dataframe(
                            avg_df.style.format({"Average Weight": "{:.2%}"}),
                            use_container_width=True
                        )
                    
                    with col2:
                        # Weight volatility
                        weight_vol = weights_over_time.std()
                        vol_df = pd.DataFrame({
                            "Asset": weight_vol.index,
                            "Weight Volatility": weight_vol.values
                        }).sort_values("Weight Volatility", ascending=False)
                        
                        st.dataframe(
                            vol_df.style.format({"Weight Volatility": "{:.2%}"}),
                            use_container_width=True
                        )
                    
                    # Correlation of weights
                    st.subheader("Weight Correlation Matrix")
                    
                    weight_corr = weights_over_time.corr()
                    
                    fig = px.imshow(
                        weight_corr,
                        text_auto='.2f',
                        color_continuous_scale="RdBu_r",
                        labels=dict(x="Asset", y="Asset", color="Correlation"),
                        title="Correlation between Asset Weights"
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

            # Tab 5: Drawdown Analysis
            with viz_tabs[4]:
                st.subheader("📉 Drawdown Analysis")
                
                # Calculate drawdowns for each model
                drawdowns = {}
                for model in selected_models:
                    if model in portfolio_values:
                        values = portfolio_values[model]
                        roll_max = values.cummax()
                        dd = (values / roll_max - 1) * 100
                        drawdowns[model] = dd
                
                # Convert to DataFrame
                drawdown_df = pd.DataFrame(drawdowns)
                
                # Plot
                fig = px.line(
                    drawdown_df,
                    labels={"value": "Drawdown (%)", "index": "Date", "variable": "Model"},
                    title="Portfolio Drawdowns"
                )
                
                # Add split line
                fig.add_vline(
                    x=split_date, 
                    line_dash="dash", 
                    line_color="magenta"
                )
                
                # Update layout
                fig.update_layout(
                    height=500, 
                    yaxis_tickformat='.1f', 
                    yaxis_autorange="reversed"  # Invert y-axis for better visualization
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Top 5 drawdown periods for selected model
                dd_model = st.selectbox(
                    "Select model for detailed drawdown analysis", 
                    selected_models
                )
                
                if dd_model in drawdowns:
                    dd_series = drawdowns[dd_model]
                    
                    # Find drawdown periods
                    is_drawdown = dd_series < 0
                    
                    # Create groups of consecutive drawdown days
                    dd_groups = (is_drawdown.astype(int).diff() != 0).cumsum()
                    
                    # Get only the groups where there is a drawdown
                    dd_periods = dd_groups[is_drawdown]
                    
                    # Calculate statistics for each drawdown period
                    dd_stats = []
                    
                    for group in dd_periods.unique():
                        period = dd_series[dd_groups == group]
                        
                        if len(period) > 0:
                            start_date = period.index[0]
                            end_date = period.index[-1]
                            max_dd = period.min()
                            recovery_days = len(period)
                            
                            dd_stats.append({
                                "Start Date": start_date.date(),
                                "End Date": end_date.date(),
                                "Max Drawdown (%)": max_dd,
                                "Duration (Days)": recovery_days
                            })
                    
                    # Convert to DataFrame and sort by max drawdown
                    if dd_stats:
                        dd_stats_df = pd.DataFrame(dd_stats)
                        dd_stats_df = dd_stats_df.sort_values("Max Drawdown (%)")
                        
                        # Display top 5 worst drawdowns
                        st.subheader(f"Top 5 Worst Drawdowns: {dd_model}")
                        st.dataframe(
                            dd_stats_df.head(5).style.format({
                                "Max Drawdown (%)": "{:.2f}"
                            }),
                            use_container_width=True
                        )
            
            # ---------------- Performance Summary Table ----------------
            st.subheader("📊 Portfolio Performance Summary")
            
            # Calculate comprehensive performance metrics
            performance_stats = {}
            
            for model in selected_models:
                if model in portfolio_returns:
                    rets = portfolio_returns[model]
                    vals = portfolio_values[model]
                    
                    # Basic return metrics
                    total_return = (vals.iloc[-1] / vals.iloc[0] - 1) * 100
                    
                    # Annualized metrics
                    years = (vals.index[-1] - vals.index[0]).days / 365.25
                    ann_return = ((1 + total_return/100) ** (1/years) - 1) * 100
                    ann_vol = rets.std() * np.sqrt(252) * 100
                    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                    
                    # Calculate downside metrics
                    roll_max = vals.cummax()
                    drawdown = (vals / roll_max - 1) * 100
                    max_drawdown = drawdown.min()
                    
                    # Sortino ratio (downside deviation)
                    downside_returns = rets[rets < 0]
                    downside_deviation = downside_returns.std() * np.sqrt(252) * 100
                    sortino = ann_return / downside_deviation if downside_deviation > 0 else 0
                    
                    # Calmar ratio
                    calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
                    
                    # Value at Risk (95%)
                    var = rets.quantile(0.05) * 100
                    
                    # Calculate percentage of positive months
                    monthly_rets = rets.resample('ME').apply(lambda x: (1 + x).prod() - 1)
                    pos_months_pct = (monthly_rets > 0).mean() * 100
                    
                    # Store metrics
                    performance_stats[model] = {
                        "Total Return (%)": total_return,
                        "Annualized Return (%)": ann_return,
                        "Annualized Volatility (%)": ann_vol,
                        "Sharpe Ratio": sharpe,
                        "Sortino Ratio": sortino,
                        "Maximum Drawdown (%)": max_drawdown,
                        "Calmar Ratio": calmar,
                        "Daily VaR 95% (%)": var,
                        "Positive Months (%)": pos_months_pct,
                        "Final Value ($)": vals.iloc[-1]
                    }
            
            # Convert to DataFrame
            summary_df = pd.DataFrame(performance_stats).T
            
            # Sort by total return
            summary_df = summary_df.sort_values("Total Return (%)", ascending=False)
            
            # Format and display
            formatted_df = summary_df.copy()
            for col in formatted_df.columns:
                if col in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]:
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}")
                elif col == "Final Value ($)":
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:,.2f}")
                else:
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}%")
            
            # Display table
            st.dataframe(formatted_df, use_container_width=True)
            
            # Option to download results
            csv = summary_df.to_csv()
            st.download_button(
                "Download Performance Summary",
                csv,
                "portfolio_performance.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.info("Select at least one model to compare.")

# -------------------- 3. Mean Risk --------------------
# In the Mean Risk tab
with tabs[2]:
    if not tickers:
        st.warning("Please enter at least one ticker in the sidebar.")
        st.stop()

    allow_short = st.checkbox("Allow Short Selling", value=False, key="mean_risk_short_selling")
    num_portfolios = st.slider("Number of Portfolios", 100, 3000, 1000)
    alpha = st.slider("CVaR/VaR Confidence", 0.90, 0.99, 0.95)

    price_data = get_price_data(tickers, start, end)
    returns = compute_returns(price_data)
    if returns.empty:
        st.error("No returns could be calculated from the available price data.")
        st.stop()

    # Show a spinner while calculating portfolios
    with st.spinner("Generating efficient frontier..."):
        mean_r = returns.mean()
        cov = returns.cov()
        result = []

        for _ in range(num_portfolios):
            weights = np.random.uniform(-1, 1, len(tickers)) if allow_short else np.random.dirichlet(np.ones(len(tickers)))
            weights /= np.sum(np.abs(weights)) if allow_short else 1
            port_daily = returns.dot(weights)
            if port_daily.empty:
                continue
            port_r = np.dot(weights, mean_r)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            sr = port_r / vol
            losses = -port_daily
            var = np.percentile(losses, (1 - alpha) * 100)
            cvar = losses[losses >= var].mean()
            
            # Store results with weights dictionary for better reference
            weights_dict = {ticker: weight for ticker, weight in zip(tickers, weights)}
            result.append([port_r, vol, sr, cvar, weights_dict])

        df = pd.DataFrame(result, columns=["Return", "Volatility", "Sharpe", "CVaR", "Weights"])
        if df.empty:
            st.warning("No valid portfolios generated. Check your tickers and data availability.")
            st.stop()

        # Find special portfolios
        max_sharpe_idx = df["Sharpe"].idxmax()
        min_vol_idx = df["Volatility"].idxmin()
        max_return_idx = df["Return"].idxmax()
        
        # Create equal weights portfolio
        equal_weights = {ticker: 1/len(tickers) for ticker in tickers}
        equal_port_r = np.dot(list(equal_weights.values()), mean_r)
        equal_port_vol = np.sqrt(np.dot(list(equal_weights.values()), np.dot(cov, list(equal_weights.values()))))
        equal_port_sr = equal_port_r / equal_port_vol

        # Efficient Frontier visualization with Plotly
        st.subheader("📊 Efficient Frontier")
        
        # Create a custom dataframe for the special portfolios
        special_portfolios = pd.DataFrame([
            {
                'Portfolio': 'Max Sharpe',
                'Return': df.loc[max_sharpe_idx, 'Return'],
                'Volatility': df.loc[max_sharpe_idx, 'Volatility'],
                'Sharpe': df.loc[max_sharpe_idx, 'Sharpe'],
                'CVaR': df.loc[max_sharpe_idx, 'CVaR'],
                'color': 'red',
                'size': 15
            },
            {
                'Portfolio': 'Min Volatility',
                'Return': df.loc[min_vol_idx, 'Return'],
                'Volatility': df.loc[min_vol_idx, 'Volatility'],
                'Sharpe': df.loc[min_vol_idx, 'Sharpe'],
                'CVaR': df.loc[min_vol_idx, 'CVaR'],
                'color': 'green',
                'size': 15
            },
            {
                'Portfolio': 'Max Return',
                'Return': df.loc[max_return_idx, 'Return'],
                'Volatility': df.loc[max_return_idx, 'Volatility'],
                'Sharpe': df.loc[max_return_idx, 'Sharpe'],
                'CVaR': df.loc[max_return_idx, 'CVaR'],
                'color': 'blue',
                'size': 15
            },
            {
                'Portfolio': 'Equal Weight',
                'Return': equal_port_r,
                'Volatility': equal_port_vol,
                'Sharpe': equal_port_sr,
                'CVaR': 0,  # This would need to be calculated properly
                'color': 'purple',
                'size': 15
            }
        ])
        
        # Create the scatter plot with Plotly
        fig = px.scatter(
            df, 
            x="Volatility", 
            y="Return", 
            color="Sharpe",
            color_continuous_scale="viridis",
            hover_data=["Sharpe", "CVaR"],
            title="Portfolio Efficient Frontier",
            labels={"Volatility": "Volatility (σ)", "Return": "Expected Return (μ)"}
        )
        
        # Add the special portfolios
        for i, row in special_portfolios.iterrows():
            fig.add_scatter(
                x=[row['Volatility']], 
                y=[row['Return']], 
                mode='markers',
                marker=dict(color=row['color'], size=row['size']),
                name=row['Portfolio'],
                hoverinfo='text',
                hovertext=f"{row['Portfolio']}<br>Return: {row['Return']:.4f}<br>Volatility: {row['Volatility']:.4f}<br>Sharpe: {row['Sharpe']:.4f}"
            )
        
        # Update layout for better appearance
        fig.update_layout(
            height=600,
            xaxis_title="Volatility (σ)",
            yaxis_title="Expected Return (μ)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the optimal portfolio information
        st.subheader("🎯 Optimal Portfolio Details")
        
        # Create tabs for different portfolio types
        portfolio_tabs = st.tabs(["Max Sharpe", "Min Volatility", "Max Return", "Equal Weight"])
        
        # Max Sharpe Portfolio
        with portfolio_tabs[0]:
            best = df.iloc[max_sharpe_idx]
            weights = best["Weights"]
            
            # Display weights visualization
            st.write("#### Portfolio Weights")
            weights_df = pd.DataFrame({
                'Asset': list(weights.keys()),
                'Weight': list(weights.values())
            }).sort_values('Weight', ascending=False)
            
            # Create a bar chart of weights
            fig = px.bar(
                weights_df,
                x='Asset',
                y='Weight',
                title="Asset Allocation - Maximum Sharpe Ratio Portfolio",
                color='Weight',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display metrics
            opt_returns = returns.dot(list(weights.values()))
            downside = np.std(opt_returns[opt_returns < 0])
            var = np.percentile(-opt_returns, (1 - alpha) * 100)
            cvar = -opt_returns[opt_returns <= -var].mean()
            
            # Add more metrics
            try:
                spy = yf.download("^GSPC", start=start, end=end, progress=False)["Close"].pct_change(fill_method=None).dropna().loc[opt_returns.index]
                beta = np.cov(opt_returns, spy)[0, 1] / np.var(spy)
            except:
                beta = np.nan
                
            # Calculate additional metrics
            annualized_return = best["Return"] * 252
            annualized_vol = best["Volatility"] * np.sqrt(252)
            sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
            sortino = annualized_return / (downside * np.sqrt(252)) if downside > 0 else 0
            
            # Create metrics columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Expected Annual Return", f"{annualized_return:.2%}")
                st.metric("Annual Volatility", f"{annualized_vol:.2%}")
                st.metric("Sharpe Ratio", f"{sharpe:.4f}")
                st.metric("Beta", f"{beta:.4f}" if not np.isnan(beta) else "N/A")
                
            with col2:
                st.metric("Sortino Ratio", f"{sortino:.4f}")
                st.metric("Value at Risk (VaR)", f"{var:.2%}")
                st.metric("Conditional VaR", f"{cvar:.2%}")
                st.metric("Downside Deviation", f"{downside:.2%}")
                
            # Add an expander for the detailed asset weights
            with st.expander("Detailed Asset Weights"):
                st.dataframe(weights_df.style.format({'Weight': '{:.2%}'}))

        # Min Volatility Portfolio
        with portfolio_tabs[1]:
            min_vol = df.iloc[min_vol_idx]
            min_vol_weights = min_vol["Weights"]
            
            # Display weights visualization
            st.write("#### Portfolio Weights")
            min_vol_weights_df = pd.DataFrame({
                'Asset': list(min_vol_weights.keys()),
                'Weight': list(min_vol_weights.values())
            }).sort_values('Weight', ascending=False)
            
            # Create a bar chart of weights
            fig = px.bar(
                min_vol_weights_df,
                x='Asset',
                y='Weight',
                title="Asset Allocation - Minimum Volatility Portfolio",
                color='Weight',
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display metrics
            min_vol_returns = returns.dot(list(min_vol_weights.values()))
            min_vol_downside = np.std(min_vol_returns[min_vol_returns < 0])
            min_vol_var = np.percentile(-min_vol_returns, (1 - alpha) * 100)
            min_vol_cvar = -min_vol_returns[min_vol_returns <= -min_vol_var].mean()
            
            # Add more metrics
            try:
                spy = yf.download("^GSPC", start=start, end=end, progress=False)["Close"].pct_change(fill_method=None).dropna().loc[min_vol_returns.index]
                min_vol_beta = np.cov(min_vol_returns, spy)[0, 1] / np.var(spy)
            except:
                min_vol_beta = np.nan
                
            # Calculate additional metrics
            min_vol_annualized_return = min_vol["Return"] * 252
            min_vol_annualized_vol = min_vol["Volatility"] * np.sqrt(252)
            min_vol_sharpe = min_vol_annualized_return / min_vol_annualized_vol if min_vol_annualized_vol > 0 else 0
            min_vol_sortino = min_vol_annualized_return / (min_vol_downside * np.sqrt(252)) if min_vol_downside > 0 else 0
            
            # Create metrics columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Expected Annual Return", f"{min_vol_annualized_return:.2%}")
                st.metric("Annual Volatility", f"{min_vol_annualized_vol:.2%}")
                st.metric("Sharpe Ratio", f"{min_vol_sharpe:.4f}")
                st.metric("Beta", f"{min_vol_beta:.4f}" if not np.isnan(min_vol_beta) else "N/A")
                
            with col2:
                st.metric("Sortino Ratio", f"{min_vol_sortino:.4f}")
                st.metric("Value at Risk (VaR)", f"{min_vol_var:.2%}")
                st.metric("Conditional VaR", f"{min_vol_cvar:.2%}")
                st.metric("Downside Deviation", f"{min_vol_downside:.2%}")
                
            # Add an expander for the detailed asset weights
            with st.expander("Detailed Asset Weights"):
                st.dataframe(min_vol_weights_df.style.format({'Weight': '{:.2%}'}))
                
            # Add historical performance visualization
            st.subheader("Historical Performance")
            min_vol_cum_returns = (1 + min_vol_returns).cumprod()
            
            # Get benchmark for comparison (S&P 500)
            try:
                benchmark = yf.download("^GSPC", start=start, end=end, progress=False)["Close"]
                benchmark_returns = benchmark.pct_change().dropna()
                benchmark_cum_returns = (1 + benchmark_returns).cumprod()
                
                # Plot cumulative returns
                perf_df = pd.DataFrame({
                    'Min Volatility Portfolio': min_vol_cum_returns,
                    'S&P 500': benchmark_cum_returns
                })
                
                fig = px.line(
                    perf_df,
                    title="Cumulative Returns - Min Volatility Portfolio vs S&P 500",
                    labels={"value": "Growth of $1", "index": "Date", "variable": ""}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            except:
                # If S&P 500 data can't be fetched, just show portfolio returns
                fig = px.line(
                    min_vol_cum_returns,
                    title="Cumulative Returns - Min Volatility Portfolio",
                    labels={"value": "Growth of $1", "index": "Date"}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            # Add risk contribution analysis
            st.subheader("Risk Contribution Analysis")
            
            # Calculate risk contribution for each asset
            cov_matrix = returns.cov().values
            port_vol = min_vol["Volatility"]
            weights_array = np.array(list(min_vol_weights.values()))
            
            # Marginal contribution to risk
            marginal_contrib = np.dot(cov_matrix, weights_array) / port_vol
            
            # Risk contribution
            risk_contrib = weights_array * marginal_contrib
            
            # Normalize to sum to 100%
            risk_contrib_pct = risk_contrib / np.sum(risk_contrib)
            
            # Create DataFrame for visualization
            risk_contrib_df = pd.DataFrame({
                'Asset': list(min_vol_weights.keys()),
                'Weight': weights_array,
                'Risk Contribution': risk_contrib_pct
            })
            
            # Sort by risk contribution
            risk_contrib_df = risk_contrib_df.sort_values('Risk Contribution', ascending=False)
            
            # Plot risk contribution vs weight
            fig = px.bar(
                risk_contrib_df,
                x='Asset',
                y=['Weight', 'Risk Contribution'],
                title="Portfolio Weights vs Risk Contribution",
                barmode='group',
                labels={"value": "Percentage", "variable": ""}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        # Max Return Portfolio
        with portfolio_tabs[2]:
            max_ret = df.iloc[max_return_idx]
            max_ret_weights = max_ret["Weights"]
            
            # Display weights visualization
            st.write("#### Portfolio Weights")
            max_ret_weights_df = pd.DataFrame({
                'Asset': list(max_ret_weights.keys()),
                'Weight': list(max_ret_weights.values())
            }).sort_values('Weight', ascending=False)
            
            # Create a bar chart of weights
            fig = px.bar(
                max_ret_weights_df,
                x='Asset',
                y='Weight',
                title="Asset Allocation - Maximum Return Portfolio",
                color='Weight',
                color_continuous_scale='Blues_r'  # Using a reversed blue scale
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display metrics
            max_ret_returns = returns.dot(list(max_ret_weights.values()))
            max_ret_downside = np.std(max_ret_returns[max_ret_returns < 0])
            max_ret_var = np.percentile(-max_ret_returns, (1 - alpha) * 100)
            max_ret_cvar = -max_ret_returns[max_ret_returns <= -max_ret_var].mean()
            
            # Add more metrics
            try:
                spy = yf.download("^GSPC", start=start, end=end, progress=False)["Close"].pct_change(fill_method=None).dropna().loc[max_ret_returns.index]
                max_ret_beta = np.cov(max_ret_returns, spy)[0, 1] / np.var(spy)
            except:
                max_ret_beta = np.nan
                
            # Calculate additional metrics
            max_ret_annualized_return = max_ret["Return"] * 252
            max_ret_annualized_vol = max_ret["Volatility"] * np.sqrt(252)
            max_ret_sharpe = max_ret_annualized_return / max_ret_annualized_vol if max_ret_annualized_vol > 0 else 0
            max_ret_sortino = max_ret_annualized_return / (max_ret_downside * np.sqrt(252)) if max_ret_downside > 0 else 0
            
            # Create metrics columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Expected Annual Return", f"{max_ret_annualized_return:.2%}")
                st.metric("Annual Volatility", f"{max_ret_annualized_vol:.2%}")
                st.metric("Sharpe Ratio", f"{max_ret_sharpe:.4f}")
                st.metric("Beta", f"{max_ret_beta:.4f}" if not np.isnan(max_ret_beta) else "N/A")
                
            with col2:
                st.metric("Sortino Ratio", f"{max_ret_sortino:.4f}")
                st.metric("Value at Risk (VaR)", f"{max_ret_var:.2%}")
                st.metric("Conditional VaR", f"{max_ret_cvar:.2%}")
                st.metric("Downside Deviation", f"{max_ret_downside:.2%}")
                
            # Add an expander for the detailed asset weights
            with st.expander("Detailed Asset Weights"):
                st.dataframe(max_ret_weights_df.style.format({'Weight': '{:.2%}'}))
                
            # Add historical performance visualization
            st.subheader("Historical Performance")
            max_ret_cum_returns = (1 + max_ret_returns).cumprod()
            
            # Get benchmark for comparison (S&P 500)
            try:
                benchmark = yf.download("^GSPC", start=start, end=end, progress=False)["Close"]
                benchmark_returns = benchmark.pct_change().dropna()
                benchmark_cum_returns = (1 + benchmark_returns).cumprod()
                
                # Plot cumulative returns
                perf_df = pd.DataFrame({
                    'Max Return Portfolio': max_ret_cum_returns,
                    'S&P 500': benchmark_cum_returns
                })
                
                fig = px.line(
                    perf_df,
                    title="Cumulative Returns - Max Return Portfolio vs S&P 500",
                    labels={"value": "Growth of $1", "index": "Date", "variable": ""}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            except:
                # If S&P 500 data can't be fetched, just show portfolio returns
                fig = px.line(
                    max_ret_cum_returns,
                    title="Cumulative Returns - Max Return Portfolio",
                    labels={"value": "Growth of $1", "index": "Date"}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Add concentration analysis
            st.subheader("Concentration Analysis")
            
            # Calculate portfolio concentration (Herfindahl-Hirschman Index)
            weights_array = np.array(list(max_ret_weights.values()))
            hhi = np.sum(weights_array**2)
            
            # Create a gauge chart for diversification
            diversification = 1 - (hhi - 1/len(weights_array)) / (1 - 1/len(weights_array))
            
            # Add metrics
            st.write(f"**Portfolio Concentration (HHI):** {hhi:.4f}")
            st.write(f"**Diversification Score:** {diversification:.2%}")
            st.write("*Note: HHI ranges from 1/N (perfectly diversified) to 1 (completely concentrated). The Diversification Score ranges from 0% (concentrated in one asset) to 100% (equally weighted).*")
            
            # Highlight the most significant asset(s)
            top_asset = max_ret_weights_df.iloc[0]
            st.info(f"The portfolio is heavily weighted towards **{top_asset['Asset']}** ({top_asset['Weight']:.2%}), which is typical of Maximum Return portfolios that concentrate on the assets with the highest historical returns.")

        # Equal Weight Portfolio
        with portfolio_tabs[3]:
            # Create equal weights dictionary
            equal_weights_dict = {ticker: 1/len(tickers) for ticker in tickers}
            
            # Display weights visualization
            st.write("#### Portfolio Weights")
            equal_weights_df = pd.DataFrame({
                'Asset': list(equal_weights_dict.keys()),
                'Weight': list(equal_weights_dict.values())
            }).sort_values('Asset')
            
            # Create a bar chart of weights
            fig = px.bar(
                equal_weights_df,
                x='Asset',
                y='Weight',
                title="Asset Allocation - Equal Weight Portfolio",
                color='Asset',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(height=400, showlegend=False)  # Hide legend as it's redundant
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display metrics
            equal_weights_array = np.array(list(equal_weights_dict.values()))
            equal_returns = returns.dot(equal_weights_array)
            equal_downside = np.std(equal_returns[equal_returns < 0])
            equal_var = np.percentile(-equal_returns, (1 - alpha) * 100)
            equal_cvar = -equal_returns[equal_returns <= -equal_var].mean()
            
            # Add more metrics
            try:
                spy = yf.download("^GSPC", start=start, end=end, progress=False)["Close"].pct_change(fill_method=None).dropna().loc[equal_returns.index]
                equal_beta = np.cov(equal_returns, spy)[0, 1] / np.var(spy)
            except:
                equal_beta = np.nan
                
            # Calculate additional metrics
            equal_mean_return = np.dot(equal_weights_array, mean_r)
            equal_volatility = np.sqrt(np.dot(equal_weights_array.T, np.dot(cov, equal_weights_array)))
            equal_annualized_return = equal_mean_return * 252
            equal_annualized_vol = equal_volatility * np.sqrt(252)
            equal_sharpe = equal_annualized_return / equal_annualized_vol if equal_annualized_vol > 0 else 0
            equal_sortino = equal_annualized_return / (equal_downside * np.sqrt(252)) if equal_downside > 0 else 0
            
            # Create metrics columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Expected Annual Return", f"{equal_annualized_return:.2%}")
                st.metric("Annual Volatility", f"{equal_annualized_vol:.2%}")
                st.metric("Sharpe Ratio", f"{equal_sharpe:.4f}")
                st.metric("Beta", f"{equal_beta:.4f}" if not np.isnan(equal_beta) else "N/A")
                
            with col2:
                st.metric("Sortino Ratio", f"{equal_sortino:.4f}")
                st.metric("Value at Risk (VaR)", f"{equal_var:.2%}")
                st.metric("Conditional VaR", f"{equal_cvar:.2%}")
                st.metric("Downside Deviation", f"{equal_downside:.2%}")
                
            # Add historical performance comparison
            st.subheader("Performance Comparison")
            equal_cum_returns = (1 + equal_returns).cumprod()
            
            # Compare to other portfolio types
            comparison_df = pd.DataFrame({
                'Equal Weight': equal_cum_returns
            })
            
            # Add Max Sharpe, Min Vol, and Max Return to comparison if available
            max_sharpe_returns = returns.dot(list(df.iloc[max_sharpe_idx]["Weights"].values()))
            min_vol_returns = returns.dot(list(df.iloc[min_vol_idx]["Weights"].values()))
            max_ret_returns = returns.dot(list(df.iloc[max_return_idx]["Weights"].values()))
            
            comparison_df['Max Sharpe'] = (1 + max_sharpe_returns).cumprod()
            comparison_df['Min Volatility'] = (1 + min_vol_returns).cumprod()
            comparison_df['Max Return'] = (1 + max_ret_returns).cumprod()
            
            # Plot comparison
            fig = px.line(
                comparison_df,
                title="Portfolio Comparison - Growth of $1",
                labels={"value": "Growth of $1", "index": "Date", "variable": "Portfolio Type"}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add information about equal weight portfolios
            st.subheader("About Equal Weight Portfolios")
            st.markdown("""
            An equal-weight portfolio allocates the same percentage to each asset regardless of market capitalization, price, or other factors. This approach:
            
            * **Eliminates selection bias** and the need to forecast returns or estimate covariances
            * Provides **automatic rebalancing** away from assets that have become overvalued
            * Often outperforms cap-weighted indices due to **size premium** and **mean reversion**
            * Offers **simplicity** and transparency in portfolio construction
            
            However, equal weighting may not be optimal from a risk-return perspective and can lead to higher turnover costs from rebalancing.
            """)
            
            # Add risk contribution analysis
            st.subheader("Risk Contribution Analysis")
            
            # Calculate risk contribution for each asset
            cov_matrix = returns.cov().values
            equal_port_vol = equal_volatility
            
            # Marginal contribution to risk
            equal_marginal_contrib = np.dot(cov_matrix, equal_weights_array) / equal_port_vol
            
            # Risk contribution
            equal_risk_contrib = equal_weights_array * equal_marginal_contrib
            
            # Normalize to sum to 100%
            equal_risk_contrib_pct = equal_risk_contrib / np.sum(equal_risk_contrib)
            
            # Create DataFrame for visualization
            equal_risk_contrib_df = pd.DataFrame({
                'Asset': list(equal_weights_dict.keys()),
                'Weight': equal_weights_array,
                'Risk Contribution': equal_risk_contrib_pct
            })
            
            # Sort by risk contribution
            equal_risk_contrib_df = equal_risk_contrib_df.sort_values('Risk Contribution', ascending=False)
            
            # Plot risk contribution vs weight
            fig = px.bar(
                equal_risk_contrib_df,
                x='Asset',
                y=['Weight', 'Risk Contribution'],
                title="Equal Weights vs Risk Contribution",
                barmode='group',
                labels={"value": "Percentage", "variable": ""}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("While an equal-weight portfolio assigns the same weight to each asset, the risk contribution is typically unequal. Assets with higher volatility or stronger correlations with other portfolio components contribute disproportionately to the overall portfolio risk.")

     
# -------------------- 4. Risk Builder --------------------
# -------------------- 4. Risk Builder --------------------
with tabs[3]:
    st.header("🏗️ Portfolio Builder & Risk Manager")
    
    # Create columns for layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("⚖️ Set Portfolio Weights")
        
        if "custom_weights" not in st.session_state:
            # Initialize with equal weights
            st.session_state["custom_weights"] = {ticker: 1/len(tickers) for ticker in tickers}
        
        # Add options for weight initialization
        init_options = ["Equal Weight", "Market Cap Weight", "Minimum Volatility", "Maximum Sharpe", "Start Fresh"]
        init_strategy = st.selectbox("Initialize weights from:", init_options)
        
        if st.button("Apply Initialization"):
            if init_strategy == "Equal Weight":
                st.session_state["custom_weights"] = {ticker: 1/len(tickers) for ticker in tickers}
            elif init_strategy == "Market Cap Weight":
                # Get market caps from yfinance
                market_caps = {}
                total_cap = 0
                for ticker in tickers:
                    try:
                        info = yf.Ticker(ticker).info
                        market_cap = info.get('marketCap', 0)
                        if market_cap:
                            market_caps[ticker] = market_cap
                            total_cap += market_cap
                        else:
                            market_caps[ticker] = 0
                    except:
                        market_caps[ticker] = 0
                
                # Normalize to sum to 1
                if total_cap > 0:
                    st.session_state["custom_weights"] = {ticker: cap/total_cap for ticker, cap in market_caps.items()}
                else:
                    st.error("Could not retrieve market cap data. Using equal weights instead.")
                    st.session_state["custom_weights"] = {ticker: 1/len(tickers) for ticker in tickers}
            elif init_strategy == "Minimum Volatility":
                # Use the minimum volatility portfolio from the Mean Risk tab
                if 'min_vol_idx' in locals():
                    min_vol_weights = df.iloc[min_vol_idx]["Weights"]
                    st.session_state["custom_weights"] = {ticker: min_vol_weights.get(ticker, 0) for ticker in tickers}
                else:
                    st.error("Minimum volatility weights not available. Using equal weights instead.")
                    st.session_state["custom_weights"] = {ticker: 1/len(tickers) for ticker in tickers}
            elif init_strategy == "Maximum Sharpe":
                # Use the maximum Sharpe portfolio from the Mean Risk tab
                if 'max_sharpe_idx' in locals():
                    max_sharpe_weights = df.iloc[max_sharpe_idx]["Weights"]
                    st.session_state["custom_weights"] = {ticker: max_sharpe_weights.get(ticker, 0) for ticker in tickers}
                else:
                    st.error("Maximum Sharpe weights not available. Using equal weights instead.")
                    st.session_state["custom_weights"] = {ticker: 1/len(tickers) for ticker in tickers}
            elif init_strategy == "Start Fresh":
                st.session_state["custom_weights"] = {ticker: 0 for ticker in tickers}
                # Set the first asset to 100%
                if tickers:
                    st.session_state["custom_weights"][tickers[0]] = 1.0
        
        # Create a form for weight inputs
        with st.form("weight_form"):
            # Create sliders for each ticker
            total_weight = 0
            for ticker in tickers:
                weight = st.slider(
                    f"{ticker}", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=st.session_state["custom_weights"].get(ticker, 1/len(tickers)),
                    step=0.01,
                    format="%0.0f%%",
                    key=f"weight_{ticker}"
                )
                st.session_state["custom_weights"][ticker] = weight
                total_weight += weight
            
            # Show current total weight
            if not np.isclose(total_weight, 1.0, atol=0.01):
                st.warning(f"Total weight: {total_weight:.0%} (should be 100%)")
            else:
                st.success(f"Total weight: {total_weight:.0%}")
            
            # Add a normalize button inside the form
            normalize = st.form_submit_button("Normalize Weights to 100%")
            if normalize and total_weight > 0:
                for ticker in tickers:
                    st.session_state["custom_weights"][ticker] /= total_weight
        
        # Convert session state weights to list for calculations
        custom_weights_list = [st.session_state["custom_weights"].get(ticker, 0) for ticker in tickers]
        
        # Normalize weights if they don't sum to 1
        if not np.isclose(sum(custom_weights_list), 1.0, atol=0.01) and sum(custom_weights_list) > 0:
            custom_weights_list = [w/sum(custom_weights_list) for w in custom_weights_list]
            for i, ticker in enumerate(tickers):
                st.session_state["custom_weights"][ticker] = custom_weights_list[i]
    
    with col2:
        st.subheader("📊 Weight Allocation")
        
        # Create a pie chart of current weights
        weights_df = pd.DataFrame({
            'Asset': tickers,
            'Weight': custom_weights_list
        })
        
        # Only include assets with non-zero weights
        weights_df = weights_df[weights_df['Weight'] > 0.001]
        
        fig = px.pie(
            weights_df,
            values='Weight',
            names='Asset',
            title="Current Portfolio Allocation",
            hole=0.4
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add sector allocation if available
        try:
            sector_weights = {}
            for i, ticker in enumerate(tickers):
                if custom_weights_list[i] > 0:
                    try:
                        sector = yf.Ticker(ticker).info.get("sector", "Unknown")
                        sector_weights[sector] = sector_weights.get(sector, 0) + custom_weights_list[i]
                    except:
                        sector_weights["Unknown"] = sector_weights.get("Unknown", 0) + custom_weights_list[i]
            
            # Create sector allocation chart
            if sector_weights:
                sector_df = pd.DataFrame({
                    'Sector': list(sector_weights.keys()),
                    'Weight': list(sector_weights.values())
                }).sort_values('Weight', ascending=False)
                
                fig = px.bar(
                    sector_df,
                    x='Sector',
                    y='Weight',
                    title="Sector Allocation",
                    color='Sector',
                    text_auto='.0%'
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Could not retrieve sector data.")
    
    # Calculate portfolio metrics based on custom weights
    st.subheader("📈 Portfolio Performance Analysis")
    
    if not np.isclose(sum(custom_weights_list), 0.0):
        # Calculate returns
        returns = compute_returns(prices)
        portfolio_returns = returns.dot(custom_weights_list)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        # Calculate and display metrics
        mean_return = portfolio_returns.mean() * 252  # Annualized
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Calculate downside deviation and Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate max drawdown
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns/peak - 1)
        max_drawdown = drawdown.min()
        
        with col1:
            st.metric("Expected Annual Return", f"{mean_return:.2%}")
            st.metric("Max Drawdown", f"{max_drawdown:.2%}")
        
        with col2:
            st.metric("Annual Volatility", f"{volatility:.2%}")
            st.metric("Downside Deviation", f"{downside_deviation:.2%}")
        
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
        
        # Display performance chart
        st.subheader("📈 Cumulative Returns")
        
        # Get S&P 500 for comparison
        try:
            spy = yf.download("^GSPC", start=start, end=end, progress=False)["Close"]
            spy_returns = spy.pct_change().dropna()
            spy_cum_returns = (1 + spy_returns).cumprod()
            
            # Create DataFrame with both portfolio and S&P 500
            comparison_df = pd.DataFrame({
                'Portfolio': cumulative_returns,
                'S&P 500': spy_cum_returns.loc[cumulative_returns.index]
            })
            
            fig = px.line(
                comparison_df,
                title="Cumulative Returns Comparison",
                labels={"value": "Growth of $1", "index": "Date", "variable": ""}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except:
            # If S&P 500 data can't be fetched, just show portfolio returns
            fig = px.line(
                cumulative_returns,
                title="Portfolio Cumulative Returns",
                labels={"value": "Growth of $1", "index": "Date"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk analysis section
        st.subheader("🔍 Portfolio Risk Analysis")
        
        # Create tabs for different risk analyses
        risk_tabs = st.tabs(["Return Distribution", "Drawdown Analysis", "Risk Contribution", "Correlation Analysis"])
        
        # Tab 1: Return Distribution
        with risk_tabs[0]:
            # Monthly returns for better visualization
            monthly_returns = portfolio_returns.resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create histogram of returns
                fig = px.histogram(
                    monthly_returns,
                    title="Monthly Return Distribution",
                    labels={"value": "Monthly Return", "count": "Frequency"},
                    color_discrete_sequence=['blue'],
                    nbins=20
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Create box plot of returns
                fig = px.box(
                    monthly_returns,
                    title="Monthly Return Statistics",
                    labels={"value": "Monthly Return"},
                    color_discrete_sequence=['green']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Calculate return statistics
            mean_monthly = monthly_returns.mean()
            median_monthly = monthly_returns.median()
            min_monthly = monthly_returns.min()
            max_monthly = monthly_returns.max()
            
            # Calculate percentiles
            percentile_5 = monthly_returns.quantile(0.05)
            percentile_95 = monthly_returns.quantile(0.95)
            
            # Display statistics
            st.write("#### Return Statistics (Monthly)")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average", f"{mean_monthly:.2%}")
            with col2:
                st.metric("Median", f"{median_monthly:.2%}")
            with col3:
                st.metric("Min", f"{min_monthly:.2%}")
            with col4:
                st.metric("Max", f"{max_monthly:.2%}")
            
            st.write("#### Return Range")
            st.info(f"90% of monthly returns fall between {percentile_5:.2%} and {percentile_95:.2%}")
            
            # Calculate positive months percentage
            positive_months = (monthly_returns > 0).mean()
            st.metric("Positive Months", f"{positive_months:.1%}")
        
        # Tab 2: Drawdown Analysis
        with risk_tabs[1]:
            # Calculate drawdowns
            drawdown_series = drawdown * 100  # Convert to percentage
            
            # Plot drawdowns
            fig = px.area(
                drawdown_series,
                title="Portfolio Drawdowns",
                labels={"value": "Drawdown (%)", "index": "Date"},
                color_discrete_sequence=['red']
            )
            fig.update_layout(height=400, yaxis_tickformat='.1f')
            fig.update_yaxes(autorange="reversed")  # Invert y-axis for better visualization
            st.plotly_chart(fig, use_container_width=True)
            
            # Find top 5 drawdown periods
            is_drawdown = drawdown_series < 0
            
            # Create groups of consecutive drawdown days
            drawdown_groups = (is_drawdown.astype(int).diff() != 0).cumsum()
            
            # Get only the groups where there is a drawdown
            drawdown_periods = drawdown_groups[is_drawdown]
            
            # Calculate statistics for each drawdown period
            drawdown_stats = []
            
            for group in drawdown_periods.unique():
                period = drawdown_series[drawdown_groups == group]
                
                if len(period) > 5:  # Filter out very short drawdowns
                    start_date = period.index[0]
                    end_date = period.index[-1]
                    max_dd = period.min()
                    recovery_days = len(period)
                    
                    drawdown_stats.append({
                        "Start Date": start_date.date(),
                        "End Date": end_date.date(),
                        "Max Drawdown (%)": max_dd,
                        "Duration (Days)": recovery_days
                    })
            
            # Sort by max drawdown
            drawdown_stats = sorted(drawdown_stats, key=lambda x: x["Max Drawdown (%)"])
            
            if drawdown_stats:
                st.write("#### Top 5 Worst Drawdowns")
                drawdown_df = pd.DataFrame(drawdown_stats[:5])
                
                # Format the table
                st.dataframe(
                    drawdown_df.style.format({
                        "Max Drawdown (%)": "{:.2f}%",
                        "Duration (Days)": "{:.0f}"
                    }),
                    use_container_width=True
                )
            else:
                st.info("No significant drawdown periods identified.")
            
            # Calculate underwater statistics
            underwater_days = (drawdown_series < 0).sum()
            total_days = len(drawdown_series)
            underwater_percentage = underwater_days / total_days if total_days > 0 else 0
            
            recovery_time = []
            current_peak = 1
            days_since_peak = 0
            
            for value in cumulative_returns:
                if value > current_peak:
                    if days_since_peak > 0:
                        recovery_time.append(days_since_peak)
                    current_peak = value
                    days_since_peak = 0
                else:
                    days_since_peak += 1
            
            avg_recovery = np.mean(recovery_time) if recovery_time else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Time Underwater", f"{underwater_percentage:.1%}")
                st.caption("Percentage of time the portfolio is below previous peak")
            
            with col2:
                st.metric("Avg. Recovery Time", f"{avg_recovery:.0f} days")
                st.caption("Average time to recover from drawdowns")
        
        # Tab 3: Risk Contribution
        with risk_tabs[2]:
            # Calculate risk contributions
            cov_matrix = returns.cov().values
            weights_array = np.array(custom_weights_list)
            
            # Calculate portfolio volatility
            port_vol = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
            
            # Calculate marginal contribution to risk
            if port_vol > 0:
                marginal_contrib = np.dot(cov_matrix, weights_array) / port_vol
                
                # Calculate risk contribution
                risk_contrib = weights_array * marginal_contrib
                
                # Normalize to percentage
                risk_contrib_pct = risk_contrib / np.sum(risk_contrib)
                
                # Create DataFrame for visualization
                risk_contrib_df = pd.DataFrame({
                    'Asset': tickers,
                    'Weight': weights_array,
                    'Marginal Contribution': marginal_contrib,
                    'Risk Contribution': risk_contrib,
                    'Risk Contribution (%)': risk_contrib_pct
                })
                
                # Calculate risk/weight ratio
                risk_contrib_df['Risk/Weight Ratio'] = np.where(
                    risk_contrib_df['Weight'] > 0.001,
                    risk_contrib_df['Risk Contribution (%)'] / risk_contrib_df['Weight'],
                    0
                )
                
                # Show only assets with weight > 0
                risk_contrib_df = risk_contrib_df[risk_contrib_df['Weight'] > 0.001]
                
                # Sort by risk contribution
                risk_contrib_df = risk_contrib_df.sort_values('Risk Contribution (%)', ascending=False)
                
                # Create visualizations
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Bar chart comparing weights to risk contributions
                    fig = px.bar(
                        risk_contrib_df,
                        x='Asset',
                        y=['Weight', 'Risk Contribution (%)'],
                        title="Weight vs Risk Contribution",
                        barmode='group',
                        labels={"value": "Percentage", "variable": ""}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Treemap of risk contributions
                    fig = px.treemap(
                        risk_contrib_df,
                        path=['Asset'],
                        values='Risk Contribution (%)',
                        color='Risk/Weight Ratio',
                        color_continuous_scale='RdYlGn_r',  # Red for high values (more risky than weight)
                        title="Risk Contribution Breakdown"
                    )
                    fig.update_traces(textinfo="label+percent entry")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk concentration metrics
                st.write("#### Risk Concentration")
                
                # Calculate Herfindahl-Hirschman Index for risk
                hhi_risk = np.sum(risk_contrib_pct**2)
                hhi_weights = np.sum(weights_array**2)
                
                # Effective number of assets (by risk)
                effective_n_risk = 1 / hhi_risk
                effective_n_weights = 1 / hhi_weights
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk HHI", f"{hhi_risk:.4f}")
                    st.caption("Lower is better (more diversified)")
                
                with col2:
                    st.metric("Effective N (Risk)", f"{effective_n_risk:.1f}")
                    st.caption("Higher is better (more diversified)")
                
                with col3:
                    st.metric("Top Asset Risk", f"{risk_contrib_df.iloc[0]['Risk Contribution (%)']:.1%}")
                    st.caption("Risk from highest contributor")
                
                # Risk budget efficiency
                st.write("#### Risk Allocation Efficiency")
                
                # Color-code assets based on risk/weight ratio
                risk_weight_df = risk_contrib_df.sort_values('Risk/Weight Ratio', ascending=False)
                
                fig = px.bar(
                    risk_weight_df,
                    x='Asset',
                    y='Risk/Weight Ratio',
                    title="Risk/Weight Ratio by Asset",
                    color='Risk/Weight Ratio',
                    color_continuous_scale='RdYlGn_r',  # Red for high values
                    text_auto='.2f'
                )
                fig.update_layout(height=300)
                fig.add_hline(y=1, line_dash="dash", line_color="black", annotation_text="Equal Risk/Weight")
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations based on risk allocation
                st.subheader("Risk Allocation Recommendations")
                
                high_risk_assets = risk_weight_df[risk_weight_df['Risk/Weight Ratio'] > 1.3]
                low_risk_assets = risk_weight_df[risk_weight_df['Risk/Weight Ratio'] < 0.7]
                
                if not high_risk_assets.empty:
                    st.info(f"""
                    **Consider reducing** allocation to these high-risk assets:
                    {', '.join(high_risk_assets['Asset'].values)}
                    
                    These assets contribute disproportionately more to risk than their weight allocation.
                    """)
                
                if not low_risk_assets.empty:
                    st.success(f"""
                    **Consider increasing** allocation to these low-risk assets:
                    {', '.join(low_risk_assets['Asset'].values)}
                    
                    These assets contribute disproportionately less to risk than their weight allocation.
                    """)
                
                if effective_n_risk < len(tickers) / 3:
                    st.warning(f"""
                    Your portfolio has concentrated risk in a few assets. The effective number of assets 
                    by risk contribution is only {effective_n_risk:.1f} out of {len(tickers)} total assets.
                    Consider diversifying further to reduce concentration risk.
                    """)
            else:
                st.warning("Cannot calculate risk contributions for a zero-volatility portfolio.")
        
        # Tab 4: Correlation Analysis
        with risk_tabs[3]:
            st.write("#### Asset Correlation Matrix")
            
            # Calculate correlation matrix
            corr_matrix = returns.corr()
            
            # Create correlation heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                labels=dict(color='Correlation'),
                title="Asset Correlation Matrix"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate average correlation
            corr_values = corr_matrix.values
            avg_corr = (np.sum(corr_values) - len(corr_values)) / (len(corr_values)**2 - len(corr_values))
            
            st.metric("Average Correlation", f"{avg_corr:.2f}")
            st.caption("Lower values indicate better diversification potential")
            
            # Identify highly correlated pairs
            st.write("#### Highly Correlated Pairs")
            
            # Create a list of highly correlated pairs
            high_corr_pairs = []
            
            for i in range(len(tickers)):
                for j in range(i+1, len(tickers)):
                    if corr_matrix.iloc[i, j] > 0.7:  # Threshold for high correlation
                        high_corr_pairs.append({
                            "Asset 1": tickers[i],
                            "Asset 2": tickers[j],
                            "Correlation": corr_matrix.iloc[i, j]
                        })
            
            if high_corr_pairs:
                # Sort by correlation (highest first)
                high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x["Correlation"], reverse=True)
                
                # Create DataFrame
                high_corr_df = pd.DataFrame(high_corr_pairs)
                
                # Display table
                st.dataframe(
                    high_corr_df.style.format({"Correlation": "{:.2f}"}),
                    use_container_width=True
                )
                
                # Warning if many highly correlated pairs
                if len(high_corr_pairs) > len(tickers) / 2:
                    st.warning("""
                    Your portfolio contains many highly correlated assets. This could reduce the 
                    diversification benefit. Consider replacing some of these assets with others 
                    that have lower correlation.
                    """)
            else:
                st.success("No highly correlated asset pairs found (correlation > 0.7).")
            
            # Add correlation network graph
            st.write("#### Correlation Network")
            
            # Convert correlations to a suitable format for visualization
            # (Simplified implementation - a more detailed network graph would use networkx)
            # This is a placeholder for a network visualization
            
            st.info("""
            A correlation network visualization would typically be displayed here, showing how 
            assets are interconnected based on their correlations. Assets with stronger correlations 
            would be connected with thicker lines and positioned closer together.
            
            For a complete implementation, you could use libraries like NetworkX and Pyvis, 
            which would require some additional setup.
            """)
    
    # Portfolio saving and comparison
    st.subheader("💾 Save & Compare Portfolios")
    
    # Initialize session state for saved portfolios if not exists
    if "saved_portfolios" not in st.session_state:
        st.session_state["saved_portfolios"] = {}
    
    # Create columns for the save interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        portfolio_name = st.text_input("Portfolio Name", "My Portfolio")
    
    with col2:
        if st.button("Save Current Portfolio"):
            if portfolio_name:
                # Calculate key metrics
                returns = compute_returns(prices)
                portfolio_returns = returns.dot(custom_weights_list)
                mean_return = portfolio_returns.mean() * 252
                volatility = portfolio_returns.std() * np.sqrt(252)
                sharpe = mean_return / volatility if volatility > 0 else 0
                
                # Calculate downside deviation and Sortino
                downside_returns = portfolio_returns[portfolio_returns < 0]
                downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
                sortino = mean_return / downside_deviation if downside_deviation > 0 else 0
                
                # Calculate max drawdown
                cumulative_returns = (1 + portfolio_returns).cumprod()
                peak = cumulative_returns.cummax()
                drawdown = (cumulative_returns/peak - 1)
                max_drawdown = drawdown.min()
                
                # Save portfolio
                st.session_state["saved_portfolios"][portfolio_name] = {
                    "weights": {ticker: weight for ticker, weight in zip(tickers, custom_weights_list)},
                    "metrics": {
                        "Expected Return": mean_return,
                        "Volatility": volatility,
                        "Sharpe Ratio": sharpe,
                        "Sortino Ratio": sortino,
                        "Max Drawdown": max_drawdown
                    },
                    "date_saved": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                }
                st.success(f"Portfolio '{portfolio_name}' saved successfully!")
    
    # Compare saved portfolios
    if st.session_state["saved_portfolios"]:
        st.write("#### Compare Saved Portfolios")
        
        # Select portfolios to compare
        portfolio_names = list(st.session_state["saved_portfolios"].keys())
        selected_portfolios = st.multiselect(
            "Select portfolios to compare",
            portfolio_names,
            default=portfolio_names[:min(3, len(portfolio_names))]
        )
        
        if selected_portfolios:
            # Create comparison table
            comparison_data = []
            
            for name in selected_portfolios:
                portfolio = st.session_state["saved_portfolios"][name]
                
                comparison_data.append({
                    "Portfolio": name,
                    "Date Saved": portfolio.get("date_saved", "N/A"),
                    "Expected Return": portfolio["metrics"]["Expected Return"],
                    "Volatility": portfolio["metrics"]["Volatility"],
                    "Sharpe Ratio": portfolio["metrics"]["Sharpe Ratio"],
                    "Sortino Ratio": portfolio["metrics"].get("Sortino Ratio", 0),
                    "Max Drawdown": portfolio["metrics"].get("Max Drawdown", 0)
                })
            
            # Create DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display comparison table
            st.dataframe(
                comparison_df.set_index("Portfolio").style.format({
                    "Expected Return": "{:.2%}",
                    "Volatility": "{:.2%}",
                    "Sharpe Ratio": "{:.2f}",
                    "Sortino Ratio": "{:.2f}",
                    "Max Drawdown": "{:.2%}"
                }),
                use_container_width=True
            )
            
            # Create radar chart comparison
            metrics = ["Expected Return", "Sharpe Ratio", "Sortino Ratio"]
            
            # Prepare data for radar chart
            radar_data = []
            
            for name in selected_portfolios:
                portfolio = st.session_state["saved_portfolios"][name]
                for metric in metrics:
                    radar_data.append({
                        "Portfolio": name,
                        "Metric": metric,
                        "Value": portfolio["metrics"].get(metric, 0)
                    })
            
            # Convert to DataFrame
            radar_df = pd.DataFrame(radar_data)
            
            # Normalize values for radar chart
            for metric in metrics:
                max_val = radar_df[radar_df["Metric"] == metric]["Value"].max()
                if max_val > 0:
                    radar_df.loc[radar_df["Metric"] == metric, "Normalized Value"] = radar_df[radar_df["Metric"] == metric]["Value"] / max_val
                else:
                    radar_df.loc[radar_df["Metric"] == metric, "Normalized Value"] = 0
            
            # Create the radar chart
            fig = px.line_polar(
                radar_df, 
                r='Normalized Value', 
                theta='Metric', 
                color='Portfolio', 
                line_close=True,
                range_r=[0, 1],
                title="Portfolio Comparison"
            )
            fig.update_traces(fill='toself')
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            # Create weight comparison
            st.write("#### Asset Allocation Comparison")
            
            # Get all unique assets across portfolios
            all_assets = set()
            for name in selected_portfolios:
                all_assets.update(st.session_state["saved_portfolios"][name]["weights"].keys())
            
            # Create weight comparison data
            weight_data = []
            
            for name in selected_portfolios:
                portfolio_weights = st.session_state["saved_portfolios"][name]["weights"]
                for asset in all_assets:
                    weight_data.append({
                        "Portfolio": name,
                        "Asset": asset,
                        "Weight": portfolio_weights.get(asset, 0)
                    })
            
            # Convert to DataFrame
            weight_df = pd.DataFrame(weight_data)
            
            # Create grouped bar chart
            fig = px.bar(
                weight_df,
                x='Asset',
                y='Weight',
                color='Portfolio',
                barmode='group',
                title="Asset Allocation Comparison",
                labels={"Weight": "Weight", "Asset": "Asset"}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Option to delete saved portfolios
        if st.button("Delete All Saved Portfolios"):
            st.session_state["saved_portfolios"] = {}
            st.success("All saved portfolios deleted!")
    
    # Portfolio optimization suggestions
    st.subheader("💡 Portfolio Optimization Suggestions")
    
    # Calculate current metrics
    if not np.isclose(sum(custom_weights_list), 0):
        returns = compute_returns(prices)
        portfolio_returns = returns.dot(custom_weights_list)
        mean_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = mean_return / volatility if volatility > 0 else 0
        
        # Add optimization hints based on portfolio characteristics
        st.write("#### Current Portfolio Characteristics")
        
        # Create optimization target options
        optimization_target = st.selectbox(
            "Optimization Target",
            ["Maximize Sharpe Ratio", "Minimize Volatility", "Target Return", "Risk Parity"],
            index=0
        )
        
        if optimization_target == "Target Return":
            target_return = st.slider(
                "Target Annual Return (%)",
                min_value=int(min(mean_return * 0.5, 0.05) * 100),
                max_value=int(max(mean_return * 1.5, 0.2) * 100),
                value=int(mean_return * 100),
                step=1
            ) / 100
        else:
            target_return = mean_return
        
        if st.button("Generate Optimization Suggestion"):
            with st.spinner("Optimizing portfolio..."):
                # Create optimization function based on target
                cov_matrix = returns.cov().values
                mean_returns = returns.mean().values
                
                if optimization_target == "Maximize Sharpe Ratio":
                    # Define objective function for maximum Sharpe ratio
                    def portfolio_sharpe(weights):
                        weights = np.array(weights)
                        returns_mean = np.sum(mean_returns * weights) * 252
                        returns_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                        return -returns_mean / returns_std  # Negative for minimization
                    
                    objective_function = portfolio_sharpe
                    
                elif optimization_target == "Minimize Volatility":
                    # Define objective function for minimum volatility
                    def portfolio_variance(weights):
                        weights = np.array(weights)
                        return np.dot(weights.T, np.dot(cov_matrix, weights))
                    
                    objective_function = portfolio_variance
                    
                elif optimization_target == "Target Return":
                    # Define objective function for target return
                    def portfolio_target_return(weights):
                        weights = np.array(weights)
                        returns_mean = np.sum(mean_returns * weights) * 252
                        returns_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                        return returns_std  # Minimize risk for target return
                    
                    objective_function = portfolio_target_return
                    
                else:  # Risk Parity
                    # Define objective function for risk parity
                    def risk_parity_objective(weights):
                        weights = np.array(weights)
                        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        risk_contribution = weights * (np.dot(cov_matrix, weights)) / portfolio_vol
                        target_risk = portfolio_vol / len(weights)
                        return np.sum((risk_contribution - target_risk) ** 2)
                    
                    objective_function = risk_parity_objective
                
                # Define constraints
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                
                # Add target return constraint if needed
                if optimization_target == "Target Return":
                    constraints.append({
                        'type': 'eq', 
                        'fun': lambda x: np.sum(mean_returns * x) * 252 - target_return
                    })
                
                # Define bounds (non-negative weights)
                bounds = tuple((0, 1) for _ in range(len(tickers)))
                
                # Initial guess - current weights
                initial_weights = custom_weights_list
                
                # Run optimization
                result = minimize(
                    objective_function,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000}
                )
                
                if result.success:
                    optimized_weights = result.x
                    
                    # Calculate metrics with optimized weights
                    opt_portfolio_returns = returns.dot(optimized_weights)
                    opt_mean_return = opt_portfolio_returns.mean() * 252
                    opt_volatility = opt_portfolio_returns.std() * np.sqrt(252)
                    opt_sharpe = opt_mean_return / opt_volatility if opt_volatility > 0 else 0
                    
                    # Calculate improvement
                    return_change = opt_mean_return - mean_return
                    volatility_change = opt_volatility - volatility
                    sharpe_change = opt_sharpe - sharpe
                    
                    # Display results
                    st.write("#### Optimization Results")
                    
                    # Create DataFrame for comparison
                    comparison = pd.DataFrame({
                        'Portfolio': ['Current', 'Optimized'],
                        'Expected Return': [mean_return, opt_mean_return],
                        'Volatility': [volatility, opt_volatility],
                        'Sharpe Ratio': [sharpe, opt_sharpe]
                    })
                    
                    # Display comparison table
                    st.dataframe(
                        comparison.set_index('Portfolio').style.format({
                            'Expected Return': '{:.2%}',
                            'Volatility': '{:.2%}',
                            'Sharpe Ratio': '{:.2f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Create metrics for key changes
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Return",
                            f"{opt_mean_return:.2%}",
                            f"{return_change:+.2%}"
                        )
                    
                    with col2:
                        st.metric(
                            "Volatility",
                            f"{opt_volatility:.2%}",
                            f"{volatility_change:+.2%}",
                            delta_color="inverse"  # Lower volatility is better
                        )
                    
                    with col3:
                        st.metric(
                            "Sharpe Ratio",
                            f"{opt_sharpe:.2f}",
                            f"{sharpe_change:+.2f}"
                        )
                    
                    # Show suggested portfolio weights
                    st.write("#### Suggested Portfolio Weights")
                    
                    # Create DataFrame for weight comparison
                    weight_changes = []
                    
                    for i, ticker in enumerate(tickers):
                        current_weight = custom_weights_list[i]
                        suggested_weight = optimized_weights[i]
                        weight_change = suggested_weight - current_weight
                        
                        if abs(weight_change) > 0.005:  # Only show meaningful changes
                            weight_changes.append({
                                'Asset': ticker,
                                'Current Weight': current_weight,
                                'Suggested Weight': suggested_weight,
                                'Change': weight_change
                            })
                    
                    # Sort by absolute change
                    weight_changes = sorted(weight_changes, key=lambda x: abs(x['Change']), reverse=True)
                    
                    if weight_changes:
                        # Create DataFrame
                        changes_df = pd.DataFrame(weight_changes)
                        
                        # Display table
                        st.dataframe(
                            changes_df.style.format({
                                'Current Weight': '{:.2%}',
                                'Suggested Weight': '{:.2%}',
                                'Change': '{:+.2%}'
                            }).background_gradient(
                                cmap='RdYlGn', subset=['Change'], vmin=-0.1, vmax=0.1
                            ),
                            use_container_width=True
                        )
                        
                        # Add action button to apply suggested weights
                        if st.button("Apply Suggested Weights"):
                            for i, ticker in enumerate(tickers):
                                st.session_state["custom_weights"][ticker] = optimized_weights[i]
                            st.success("Suggested weights applied! Refresh the page to see the updated portfolio.")
                            st.experimental_rerun()
                    else:
                        st.info("Your current portfolio is already well-optimized for your selected target.")
                else:
                    st.error("Optimization failed. Try a different target or constraint.")
    
        # Portfolio export options
        st.subheader("📤 Export Portfolio")
        
        export_format = st.selectbox(
            "Select export format",
            ["CSV", "Excel", "JSON", "Text Report"],
            index=0
        )
        
        if not np.isclose(sum(custom_weights_list), 0):
            # Calculate returns for export
            returns = compute_returns(prices)
            portfolio_returns = returns.dot(custom_weights_list)
            mean_return = portfolio_returns.mean() * 252
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe = mean_return / volatility if volatility > 0 else 0
            
            # Create a dictionary with portfolio details
            portfolio_details = {
                "Portfolio Name": portfolio_name,
                "Date Created": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "Assets": tickers,
                "Weights": custom_weights_list,
                "Expected Annual Return": mean_return,
                "Annual Volatility": volatility,
                "Sharpe Ratio": sharpe
            }
            
            # Create export data based on selected format
            if export_format == "CSV":
                # Create DataFrame for weights
                export_df = pd.DataFrame({
                    'Asset': tickers,
                    'Weight': custom_weights_list
                })
                
                # Convert to CSV
                csv = export_df.to_csv(index=False)
                
                # Create download button
                st.download_button(
                    "Download Portfolio as CSV",
                    csv,
                    f"{portfolio_name.replace(' ', '_')}_weights.csv",
                    "text/csv",
                    key="download_csv"
                )
                
            elif export_format == "Excel":
                # Create DataFrame for weights
                export_df = pd.DataFrame({
                    'Asset': tickers,
                    'Weight': custom_weights_list
                })
                
                # Create a BytesIO object to store the Excel file
                buffer = io.BytesIO()
                
                # Write to Excel
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, sheet_name="Weights", index=False)
                    
                    # Create a sheet for metrics
                    metrics_df = pd.DataFrame({
                        'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio'],
                        'Value': [mean_return, volatility, sharpe]
                    })
                    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
                    
                    # Get the xlsxwriter workbook and worksheet objects
                    workbook = writer.book
                    
                    # Add formatting
                    percent_format = workbook.add_format({'num_format': '0.00%'})
                    worksheet = writer.sheets["Weights"]
                    worksheet.set_column('B:B', 12, percent_format)
                    
                    worksheet = writer.sheets["Metrics"]
                    worksheet.set_column('B:B', 12, percent_format)
                
                # Set the buffer position to the beginning
                buffer.seek(0)
                
                # Create download button
                st.download_button(
                    "Download Portfolio as Excel",
                    buffer,
                    f"{portfolio_name.replace(' ', '_')}_portfolio.xlsx",
                    "application/vnd.ms-excel",
                    key="download_excel"
                )
                
            elif export_format == "JSON":
                # Convert portfolio details to JSON
                json_data = json.dumps(portfolio_details, default=str, indent=4)
                
                # Create download button
                st.download_button(
                    "Download Portfolio as JSON",
                    json_data,
                    f"{portfolio_name.replace(' ', '_')}_portfolio.json",
                    "application/json",
                    key="download_json"
                )
                
            else:  # Text Report
                # Create a detailed text report
                report = f"""
                    PORTFOLIO ANALYSIS REPORT
                    ========================
                    Portfolio Name: {portfolio_name}
                    Date: {pd.Timestamp.now().strftime("%Y-%m-%d")}
                    Analysis Period: {start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}

                    PERFORMANCE METRICS
                    ------------------
                    Expected Annual Return: {mean_return:.2%}
                    Annual Volatility: {volatility:.2%}
                    Sharpe Ratio: {sharpe:.2f}

                    ASSET ALLOCATION
                    ---------------
                    """
                
                # Add asset allocation
                weight_data = sorted(zip(tickers, custom_weights_list), key=lambda x: x[1], reverse=True)
                for ticker, weight in weight_data:
                    report += f"{ticker}: {weight:.2%}\n"
                
                report += """
                    RISK ANALYSIS
                    ------------
                    """
                                
                # Add risk metrics if available
                if 'risk_contrib_df' in locals():
                    top_risk_contributors = risk_contrib_df.head(3)
                    report += "Top 3 Risk Contributors:\n"
                    for _, row in top_risk_contributors.iterrows():
                        report += f"{row['Asset']}: {row['Risk Contribution (%)']:.2%} of total risk\n"
                    
                    report += f"\nRisk Concentration (HHI): {hhi_risk:.4f}\n"
                    report += f"Effective Number of Assets (by risk): {effective_n_risk:.1f}\n"
                
                report += """
                    CORRELATION ANALYSIS
                    -------------------
                    """
                
                # Add correlation info
                corr_matrix = returns.corr()
                report += f"Average Asset Correlation: {avg_corr:.2f}\n\n"
                
                if 'high_corr_pairs' in locals() and high_corr_pairs:
                    report += "Highly Correlated Pairs (>0.7):\n"
                    for pair in high_corr_pairs[:5]:  # Top 5 most correlated pairs
                        report += f"{pair['Asset 1']} & {pair['Asset 2']}: {pair['Correlation']:.2f}\n"
                
                report += """
                    RECOMMENDATIONS
                    --------------
                    """
                
                # Add some generic recommendations
                if 'high_risk_assets' in locals() and not high_risk_assets.empty:
                    report += f"Consider reducing allocation to high-risk assets: {', '.join(high_risk_assets['Asset'].values)}\n"
                
                if 'low_risk_assets' in locals() and not low_risk_assets.empty:
                    report += f"Consider increasing allocation to low-risk assets: {', '.join(low_risk_assets['Asset'].values)}\n"
                
                if 'high_corr_pairs' in locals() and len(high_corr_pairs) > len(tickers) / 2:
                    report += "Consider diversifying your portfolio to reduce correlation between assets.\n"
                
                # Create download button
                st.download_button(
                    "Download Portfolio Analysis Report",
                    report,
                    f"{portfolio_name.replace(' ', '_')}_report.txt",
                    "text/plain",
                    key="download_report"
                )