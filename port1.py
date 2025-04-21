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
st.title("📊 Portfolio Analysis App")

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
    st.header("📈 Asset Analysis")
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
    st.header("🔄 Portfolio Comparison")
    
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

    vol = returns.std()

    # st.subheader("Select Models to Compare")
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
    selected_models = st.multiselect("📍 Select Models to Compare", model_options, default=["Equal Weighted", "Minimum CVaR"])

    # -------------------- Dynamic Inputs for Models --------------------
    if "Target Return Portfolio" in selected_models:
        st.subheader("🎯 Set Target Return")
        target_return = st.number_input(
            "Minimum Expected Return (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            help="Set the minimum annual return you'd like this portfolio to target."
        ) / 100

        if target_return < 0.05:
            st.caption("🧮 Very conservative target")
        elif target_return <= 0.15:
            st.caption("⚖️ Balanced growth objective")
        else:
            st.caption("🚀 Aggressive return target")
    else:
        target_return = 0.05


    if "Bayesian Mean-Variance" in selected_models:
        shrinkage_level = st.slider(
            "Shrinkage Level (τ): Prior vs. Sample Balance",
            min_value=0.01,
            max_value=1.0,
            value=0.05,
            step=0.01,
            help="Lower = rely more on historical data. Higher = rely more on prior belief."
        )

        if shrinkage_level <= 0.05:
            st.caption("🧮 Mostly trusting historical data (Sample-driven)")
        elif shrinkage_level <= 0.25:
            st.caption("⚖️ Balanced trust between prior and data")
        else:
            st.caption("📐 Strong trust in prior (Shrinkage-dominant)")
    else:
        shrinkage_level = 0.05


    if "Minimum CVaR" in selected_models or "Risk Budgeting (CVaR)" in selected_models:
        alpha = st.slider(
            "CVaR Confidence Level (α)",
            min_value=0.85,
            max_value=0.99,
            step=0.01,
            value=0.95,
            help="Tail threshold for CVaR calculations."
        )

        if alpha <= 0.90:
            st.caption("⚠️ Looser tail: more risk assumed")
        elif alpha <= 0.97:
            st.caption("✅ Balanced tail confidence")
        else:
            st.caption("🛡️ Very conservative tail control")
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

        if shrinkage_lambda <= 0.1:
            st.caption("📊 Mostly sample covariance")
        elif shrinkage_lambda <= 0.4:
            st.caption("⚖️ Blended covariance estimate")
        else:
            st.caption("📐 Mostly shrinkage target matrix")
    else:
        shrinkage_lambda = 0.1

    weight_dict = {}

    if "Equal Weighted" in selected_models:
        w_eq = np.ones(len(tickers)) / len(tickers)
        weight_dict["Equal Weighted"] = w_eq

    if "Inverse Volatility" in selected_models:
        inv_vol = 1 / vol
        inv_vol = inv_vol / inv_vol.sum()
        weight_dict["Inverse Volatility"] = inv_vol.values

    if "Random" in selected_models:
        w = np.random.random(len(tickers))
        w /= w.sum()
        weight_dict["Random"] = w


    if "Minimum Variance" in selected_models:
        # Get the covariance matrix
        cov = returns.cov().values
        
        # Add a small regularization term to ensure positive definiteness
        # This helps with numerical stability
        n_assets = cov.shape[0]
        cov_reg = cov + np.eye(n_assets) * 1e-8
        
        # Define the objective function for portfolio variance
        def portfolio_variance(w):
            return np.dot(w.T, np.dot(cov_reg, w))
        
        # Define constraints: weights sum to 1
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Define bounds: no short selling (all weights between 0 and 1)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
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
                # Log the error but don't interrupt the loop
                st.debug(f"Optimization attempt {attempt+1} failed: {str(e)}")
        
        # Use the best result if available
        if best_result is not None and best_result.success:
            # Ensure weights sum to 1 (fix any small numerical issues)
            weights = best_result.x
            weights = weights / np.sum(weights)
            
            # Store in the weight dictionary
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
        bounds = tuple((0, 1) for _ in range(len(tickers)))
        init_guess = np.ones(len(tickers)) / len(tickers)

        result = minimize(portfolio_var, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        if result.success:
            weight_dict["Target Return Portfolio"] = result.x

    if "Maximum Diversification" in selected_models:
        cov = returns.cov()
        vol_vec = vol.values
        def diversification_ratio(w):
            return - (np.dot(w, vol_vec) / np.sqrt(np.dot(w.T, np.dot(cov.values, w))))
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(len(tickers)))
        init_guess = np.ones(len(tickers)) / len(tickers)
        result = minimize(diversification_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        if result.success:
            weight_dict["Maximum Diversification"] = result.x

    if "Maximum Sharpe Ratio" in selected_models:
        mean_ret = returns.mean()
        cov = returns.cov()
        def negative_sharpe(w):
            port_ret = np.dot(w, mean_ret)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov.values, w)))
            return -port_ret / port_vol
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(len(tickers)))
        init_guess = np.ones(len(tickers)) / len(tickers)
        result = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        if result.success:
            weight_dict["Maximum Sharpe Ratio"] = result.x

    if "Minimum CVaR" in selected_models:
        # alpha = 0.95
        returns_array = returns.values
        def portfolio_cvar(weights):
            port_ret = np.dot(returns_array, weights)
            var_thresh = np.percentile(port_ret, (1 - alpha) * 100)
            cvar = -port_ret[port_ret <= var_thresh].mean()
            return cvar
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(len(tickers)))
        init_guess = np.ones(len(tickers)) / len(tickers)
        result = minimize(portfolio_cvar, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        if result.success:
            weight_dict["Minimum CVaR"] = result.x


    if "Risk Parity (Variance)" in selected_models:
        cov = returns.cov().values
        
        def risk_parity_objective(weights):
            # Avoid division by zero by setting a minimum weight
            weights = np.maximum(weights, 1e-8)
            weights = weights / np.sum(weights)  # Normalize weights
            
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
        bounds = tuple((0.001, 1.0) for _ in range(len(tickers)))
        
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
            best_weights = best_weights / np.sum(best_weights)
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
                w_up = w_up / np.sum(w_up)
                
                # Calculate marginal change in CVaR
                cvar_up = compute_cvar(w_up)
                mcvar[i] = (cvar_up - base_cvar) / epsilon
            
            return mcvar
        
        # Define risk budget objective function
        def cvar_risk_budget_obj(weights):
            # Ensure weights are positive and sum to 1
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            
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
        
        # Bound weights to be between 0.01 and 1
        bounds = tuple((0.01, 1) for _ in range(len(tickers)))
        
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
            result_weights = result_weights / np.sum(result_weights)
            weight_dict["Risk Budgeting (CVaR)"] = result_weights
    

    if "Risk Parity (Covariance Shrinkage)" in selected_models:
        shrinkage_lambda = shrinkage_lambda
        S = returns.cov().values
        avg_var = np.mean(np.diag(S))
        T = np.eye(len(tickers)) * avg_var
        shrunk_cov = shrinkage_lambda * T + (1 - shrinkage_lambda) * S

        def risk_parity_shrinkage(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(shrunk_cov, weights)))
            mrc = np.dot(shrunk_cov, weights) / port_vol
            rc = weights * mrc
            rc = rc / port_vol
            target = np.ones(len(weights)) * (port_vol / len(weights))
            return np.sum((rc - target) ** 2)

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(len(tickers)))
        init_guess = np.ones(len(tickers)) / len(tickers)
        result = minimize(risk_parity_shrinkage, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        if result.success:
            weight_dict["Risk Parity (Covariance Shrinkage)"] = result.x


    if "Hierarchical Risk Parity (HRP)" in selected_models:
        cov = returns.cov().values
        corr = returns.corr().values

        # Step 1: Compute distance matrix from correlation
        dist = np.sqrt(0.5 * (1 - corr))
        dist = np.clip(dist, 0, 1)  # just in case

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

                    cov_sub = pd.DataFrame(cov).iloc[cluster, cluster].values
                    var_left = np.dot(np.ones(len(left)), np.dot(cov[np.ix_(left, left)], np.ones(len(left))))
                    var_right = np.dot(np.ones(len(right)), np.dot(cov[np.ix_(right, right)], np.ones(len(right))))
                    alpha = 1 - var_left / (var_left + var_right)

                    weights[left] *= alpha
                    weights[right] *= (1 - alpha)

                    new_clusters += [left, right]
                cluster_items = new_clusters
            return weights / weights.sum()

        hrp_weights = hrp_allocation(cov, sorted_idx)
        # Reorder to match the original ticker order
        w_final = np.zeros(len(tickers))
        for i, idx in enumerate(hrp_weights.index):
            w_final[idx] = hrp_weights.iloc[i]
        weight_dict["Hierarchical Risk Parity (HRP)"] = w_final


    if "Bayesian Mean-Variance" in selected_models:
        sample_mean = returns.mean()
        sample_cov = returns.cov()

        mu_0 = pd.Series(sample_mean.mean(), index=sample_mean.index)
        tau = shrinkage_level


        # prior_cov = np.eye(len(tickers)) * sample_cov.values.mean()
        prior_cov = tau * np.eye(len(tickers)) * sample_cov.values.mean()


        post_cov = np.linalg.inv(
            np.linalg.inv(sample_cov.values) + np.linalg.inv(prior_cov)
        )
        post_mean = post_cov @ (
            np.linalg.inv(sample_cov.values) @ sample_mean.values +
            np.linalg.inv(prior_cov) @ mu_0.values
        )

        def neg_sharpe(w):
            port_ret = np.dot(w, post_mean)
            port_vol = np.sqrt(np.dot(w.T, np.dot(sample_cov.values, w)))
            return -port_ret / port_vol

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(len(tickers)))
        init_guess = np.ones(len(tickers)) / len(tickers)

        result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        if result.success:
            weight_dict["Bayesian Mean-Variance"] = result.x


    if weight_dict:
        weight_df = pd.DataFrame(weight_dict, index=tickers)
        weight_df = weight_df.T  # portfolios as rows

        # st.subheader("📊 Portfolio Composition")
        fig = px.bar(
            weight_df,
            barmode="stack",
            orientation="v",
            title="Portfolio Allocation by Model",
            labels={"value": "Weight", "index": "Portfolios", "variable": "Assets"},
        )
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Portfolios", yaxis_title="Weight")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one model to compare.")


# -------------------- 3. Mean Risk --------------------
with tabs[2]:
    # show_model_inputs = False
    st.header("📉 Mean-Variance Optimization")
    if not tickers:
        st.warning("Please enter at least one ticker in the sidebar.")
        st.stop()

    allow_short = st.checkbox("Allow Short Selling", value=False)
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
    st.header("🏗️ Risk Builder")
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
