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
with tabs[2]:
    # show_model_inputs = False
    # st.header("📉 Mean-Variance Optimization")
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
        result.append([port_r, vol, sr, cvar, weights])

    df = pd.DataFrame(result, columns=["Return", "Volatility", "Sharpe", "CVaR", "Weights"])
    if df.empty:
        st.warning("No valid portfolios generated. Check your tickers and data availability.")
        st.stop()

    best = df.iloc[df["Sharpe"].idxmax()]
    st.subheader("Efficient Frontier")
    fig, ax = plt.subplots()
    ax.scatter(df["Volatility"], df["Return"], c=df["Sharpe"], cmap="viridis")
    ax.scatter(best["Volatility"], best["Return"], color="red", s=100, label="Max Sharpe")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Optimal Portfolio Stats")
    weights = best["Weights"]
    opt_returns = returns.dot(weights)
    downside = np.std(opt_returns[opt_returns < 0])
    var = np.percentile(-opt_returns, (1 - alpha) * 100)
    cvar = -opt_returns[opt_returns <= -var].mean()

    try:
        spy = yf.download("^GSPC", start=start, end=end, progress=False)["Close"].pct_change(fill_method=None).dropna().loc[opt_returns.index]
        beta = np.cov(opt_returns, spy)[0, 1] / np.var(spy)
    except:
        beta = np.nan

    st.write({
        "Expected Return": round(best["Return"], 4),
        "Volatility": round(best["Volatility"], 4),
        "Sharpe": round(best["Sharpe"], 4),
        "CVaR": round(cvar, 4),
        "Downside Deviation": round(downside, 4),
        "Beta": round(beta, 4) if not np.isnan(beta) else "N/A"
    })

# -------------------- 4. Risk Builder --------------------
with tabs[3]:
    # show_model_inputs = False
    # st.header("🏗️ Risk Builder")
    if "saved_portfolios" not in st.session_state:
        st.session_state["saved_portfolios"] = {}

    if not tickers:
        st.warning("Please enter at least one ticker in the sidebar.")
        st.stop()

    alpha = st.slider("CVaR/VaR Confidence Level", 0.90, 0.99, 0.95)
    prices = get_price_data(tickers, start, end)
    if prices.empty:
        st.warning("No price data returned for selected tickers.")
        st.stop()

    valid_tickers = list(prices.columns)
    if set(tickers) != set(valid_tickers):
        st.warning(f"Some tickers dropped due to missing data: {set(tickers) - set(valid_tickers)}")
    tickers = valid_tickers

    weights = []
    st.subheader("⚖️ Set Portfolio Weights")
    for t in tickers:
        w = st.number_input(f"Weight for {t}", 0.0, 1.0, round(1 / len(tickers), 2))
        weights.append(w)

    if not np.isclose(sum(weights), 1.0):
        st.warning("Weights must sum to 1. Adjust them accordingly.")
        st.stop()

    returns = compute_returns(prices)
    daily = returns.dot(weights)
    cumulative = (1 + daily).cumprod()
    st.subheader("📈 Cumulative Performance")
    st.line_chart(cumulative.rename("Portfolio"))

    st.subheader("📉 Return Distribution")
    fig1, ax1 = plt.subplots()
    ax1.hist(daily, bins=40, alpha=0.7)
    ax1.set_title("Daily Return Histogram")
    st.pyplot(fig1)

    downside = np.std(daily[daily < 0])
    var = np.percentile(-daily, (1 - alpha) * 100)
    cvar = -daily[daily <= -var].mean()
    mean = daily.mean()
    vol = daily.std()
    sharpe = mean / vol

    try:
        spy = yf.download("^GSPC", start=start, end=end, progress=False)["Close"].pct_change(fill_method=None).dropna().loc[daily.index]
        beta = np.cov(daily, spy)[0, 1] / np.var(spy)
    except:
        beta = np.nan

    st.subheader("📊 Risk Metrics")
    st.write({
        "Expected Return": round(mean, 4),
        "Volatility": round(vol, 4),
        "Sharpe Ratio": round(sharpe, 4),
        "CVaR": round(cvar, 4),
        "VaR": round(var, 4),
        "Downside Deviation": round(downside, 4),
        "Beta": round(beta, 4) if not np.isnan(beta) else "N/A"
    })

    st.subheader("💾 Save Portfolio")
    name = st.text_input("Name this portfolio", "My Portfolio")
    if st.button("Save Portfolio"):
        st.session_state.saved_portfolios[name] = {
            "weights": weights,
            "metrics": {
                "Expected Return": mean,
                "Volatility": vol,
                "Sharpe Ratio": sharpe,
                "CVaR": cvar,
                "VaR": var,
                "Downside Deviation": downside,
                "Beta": beta
            }
        }
        st.success(f"Saved portfolio '{name}'!")

    if st.session_state.saved_portfolios:
        st.subheader("📎 Compare Saved Portfolios")
        selected = st.multiselect("Choose portfolios to compare", list(st.session_state.saved_portfolios.keys()))
        if selected:
            metrics = list(st.session_state.saved_portfolios[selected[0]]['metrics'].keys())
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            for name in selected:
                values = list(st.session_state.saved_portfolios[name]['metrics'].values())
                norm = (np.array(values) - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
                norm = norm.tolist() + norm[:1]
                ax.plot(angles, norm, label=name)
                ax.fill(angles, norm, alpha=0.2)
            ax.set_thetagrids(np.degrees(angles), metrics)
            ax.set_title("📊 Risk Radar Comparison")
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            st.pyplot(fig)

    st.subheader("⬇️ Export Portfolio Metrics")
    if name in st.session_state.saved_portfolios:
        df = pd.DataFrame.from_dict(
            st.session_state.saved_portfolios[name]['metrics'],
            orient="index", columns=["Value"]
        )
        st.dataframe(df.style.format("{:.4f}"))
        st.download_button("Download CSV", df.to_csv().encode("utf-8"), "portfolio_metrics.csv", "text/csv")
