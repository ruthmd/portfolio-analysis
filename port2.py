import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# Set page config
st.set_page_config(
    page_title="Portfolio Analysis App",
    page_icon="📊",
    layout="wide"
)

# Main title
st.title("Financial Portfolio Analysis")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Asset Analysis", 
    "Portfolio Comparison", 
    "Mean Risk", 
    "Risk Building"
])

# Asset Analysis Tab
with tab1:
    st.header("Asset Analysis")
    
    # Input for stock symbols
    with st.expander("Select Assets", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            stock_input = st.text_input(
                "Enter stock symbols (comma-separated):",
                value="AAPL,MSFT,AMZN,GOOGL",
                help="Example: AAPL,MSFT,AMZN,GOOGL"
            )
            
        with col2:
            today = datetime.now()
            start_date = st.date_input(
                "Start Date",
                value=today - timedelta(days=365*3),  # 3 years default
                max_value=today
            )
            end_date = st.date_input(
                "End Date",
                value=today,
                max_value=today
            )
    
    # Process the input
    if stock_input:
        stocks = [s.strip() for s in stock_input.split(',')]
        
        # Download data
        @st.cache_data(ttl=3600)  # Cache data for 1 hour
        def load_stock_data(tickers, start, end):
            data = yf.download(tickers, start=start, end=end)['Close']
            # If only one stock, convert to DataFrame
            if isinstance(data, pd.Series):
                data = data.to_frame(name=tickers[0])
            return data
        
        try:
            # Load data
            df_stocks = load_stock_data(stocks, start_date, end_date)
            
            # Show the data
            st.subheader("Stock Prices")
            st.dataframe(df_stocks.head())
            
            # Normalize prices (for comparison)
            norm_data = df_stocks.div(df_stocks.iloc[0]).mul(100)
            
            # Price chart
            st.subheader("Stock Price Chart")
            price_chart = px.line(
                df_stocks, 
                title="Historical Adjusted Close Prices"
            )
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Normalized price chart
            st.subheader("Normalized Price Chart (Base=100)")
            norm_chart = px.line(
                norm_data, 
                title="Normalized Prices (Base=100)"
            )
            st.plotly_chart(norm_chart, use_container_width=True)
            
            # Calculate returns
            returns = df_stocks.pct_change().dropna()
            
            # Display statistics
            st.subheader("Asset Statistics")
            
            # Calculate statistics
            stats = pd.DataFrame({
                'Mean Daily Return (%)': returns.mean() * 100,
                'Annual Return (%)': returns.mean() * 252 * 100,  # Approx 252 trading days
                'Daily Volatility (%)': returns.std() * 100,
                'Annual Volatility (%)': returns.std() * np.sqrt(252) * 100,
                'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                'Max Drawdown (%)': ((df_stocks / df_stocks.cummax() - 1) * 100).min()
            }).T
            
            st.dataframe(stats)
            
            # Returns distribution
            st.subheader("Returns Distribution")
            
            # Create subplot
            fig = plt.figure(figsize=(12, 4 * len(stocks)))
            for i, stock in enumerate(stocks):
                if stock in returns.columns:  # Check if the stock exists in the dataframe
                    ax = fig.add_subplot(len(stocks), 1, i+1)
                    sns.histplot(returns[stock], kde=True, ax=ax)
                    ax.set_title(f"Daily Returns Distribution for {stock}")
                    ax.axvline(x=0, color='r', linestyle='--')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Correlation matrix
            st.subheader("Correlation Matrix")
            
            # Calculate correlation
            corr = returns.corr()
            
            # Plot correlation heatmap
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title="Asset Correlation Matrix"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Please check your stock symbols and try again. Make sure all symbols are valid.")

# Portfolio Comparison Tab
with tab2:
    st.header("Portfolio Comparison")
    
    # Input for stock symbols
    with st.expander("Select Assets", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            stock_input = st.text_input(
                "Enter stock symbols (comma-separated):",
                value="AAPL,MSFT,AMZN,GOOGL,BRK-B,JNJ,V,PG",
                help="Example: AAPL,MSFT,AMZN,GOOGL,BRK-B,JNJ,V,PG",
                key="stocks_tab2"
            )
            
        with col2:
            today = datetime.now()
            start_date = st.date_input(
                "Start Date",
                value=today - timedelta(days=365*3),  # 3 years default
                max_value=today,
                key="start_date_tab2"
            )
            end_date = st.date_input(
                "End Date",
                value=today,
                max_value=today,
                key="end_date_tab2"
            )
    
    # Process the input
    if stock_input:
        stocks = [s.strip() for s in stock_input.split(',')]
        
        # Download data
        @st.cache_data(ttl=3600)  # Cache data for 1 hour
        def load_stock_data(tickers, start, end):
            data = yf.download(tickers, start=start, end=end)['Close']
            # If only one stock, convert to DataFrame
            if isinstance(data, pd.Series):
                data = data.to_frame(name=tickers[0])
            return data
        
        try:
            # Load data
            df_stocks = load_stock_data(stocks, start_date, end_date)
            
            # Calculate daily returns
            returns = df_stocks.pct_change().dropna()
            
            # Portfolio construction
            st.subheader("Create Portfolios")
            
            # Create an empty list to store portfolios
            portfolios = []
            
            # Predefined portfolio weights
            st.write("#### Pre-defined Portfolios")
            col1, col2 = st.columns(2)
            
            with col1:
                # Equal Weight Portfolio
                if st.button("Add Equal Weight Portfolio"):
                    weight = 1 / len(stocks)
                    weights = [weight] * len(stocks)
                    portfolios.append({
                        "name": "Equal Weight",
                        "weights": weights
                    })
                    st.success("Equal Weight Portfolio added!")
            
            with col2:
                # 60/40 Portfolio (60% in first half, 40% in second half)
                if st.button("Add 60/40 Allocation"):
                    half = len(stocks) // 2
                    weights = [0.6 / half] * half + [0.4 / (len(stocks) - half)] * (len(stocks) - half)
                    portfolios.append({
                        "name": "60/40 Allocation",
                        "weights": weights
                    })
                    st.success("60/40 Portfolio added!")
            
            # Custom portfolios
            st.write("#### Custom Portfolio")
            
            # Initialize weights for the custom portfolio
            custom_weights = []
            
            # Create sliders for each stock
            for stock in stocks:
                weight = st.slider(f"Weight for {stock} (%)", 0, 100, 100 // len(stocks))
                custom_weights.append(weight)
            
            # Normalize weights to sum to 100%
            total_weight = sum(custom_weights)
            
            # Show total weight
            st.write(f"Total weight: {total_weight}%")
            
            if total_weight != 100:
                st.warning("Total weight should be 100%. Weights will be normalized.")
            
            # Add custom portfolio button
            custom_name = st.text_input("Custom Portfolio Name", "My Portfolio")
            
            if st.button("Add Custom Portfolio"):
                # Normalize weights
                norm_weights = [w / total_weight for w in custom_weights]
                portfolios.append({
                    "name": custom_name,
                    "weights": norm_weights
                })
                st.success(f"Portfolio '{custom_name}' added!")
            
            # Compare portfolios
            if portfolios:
                st.subheader("Portfolio Comparison")
                
                # Calculate performance for each portfolio
                performance_data = []
                
                for portfolio in portfolios:
                    weights = portfolio["weights"]
                    portfolio_return = (returns @ weights) * 100  # Daily returns in percentage
                    
                    # Calculate cumulative returns
                    cumulative_return = (1 + portfolio_return / 100).cumprod() - 1
                    
                    # Portfolio statistics
                    annual_return = portfolio_return.mean() * 252  # Annualized return
                    annual_volatility = portfolio_return.std() * np.sqrt(252)  # Annualized volatility
                    sharpe_ratio = annual_return / annual_volatility  # Sharpe ratio (assuming 0% risk-free rate)
                    
                    # Maximum drawdown
                    cumulative_returns = (1 + portfolio_return / 100).cumprod()
                    max_drawdown = ((cumulative_returns / cumulative_returns.cummax()) - 1).min() * 100
                    
                    # Add to performance data
                    performance_data.append({
                        "Portfolio": portfolio["name"],
                        "Annual Return (%)": annual_return,
                        "Annual Volatility (%)": annual_volatility,
                        "Sharpe Ratio": sharpe_ratio,
                        "Max Drawdown (%)": max_drawdown,
                        "Cumulative Return": cumulative_return
                    })
                
                # Display portfolio composition
                st.write("#### Portfolio Composition")
                
                # Create a DataFrame to display portfolio weights
                weights_df = pd.DataFrame({
                    p["name"]: p["weights"] for p in portfolios
                }, index=stocks)
                
                # Show weights
                st.dataframe(weights_df.style.format("{:.2%}"))
                
                # Create a bar chart to visualize portfolio weights
                weights_fig = go.Figure()
                
                for portfolio in portfolios:
                    weights_fig.add_trace(go.Bar(
                        x=stocks,
                        y=portfolio["weights"],
                        name=portfolio["name"]
                    ))
                
                weights_fig.update_layout(
                    title="Portfolio Allocations",
                    xaxis_title="Assets",
                    yaxis_title="Weight",
                    barmode='group'
                )
                
                st.plotly_chart(weights_fig, use_container_width=True)
                
                # Display portfolio statistics
                st.write("#### Portfolio Statistics")
                
                # Create a DataFrame for statistics
                stats_df = pd.DataFrame([
                    {
                        "Portfolio": p["Portfolio"],
                        "Annual Return (%)": p["Annual Return (%)"],
                        "Annual Volatility (%)": p["Annual Volatility (%)"],
                        "Sharpe Ratio": p["Sharpe Ratio"],
                        "Max Drawdown (%)": p["Max Drawdown (%)"]
                    } for p in performance_data
                ])
                
                # Display statistics
                st.dataframe(stats_df.set_index("Portfolio").style.format({
                    "Annual Return (%)": "{:.2f}",
                    "Annual Volatility (%)": "{:.2f}",
                    "Sharpe Ratio": "{:.2f}",
                    "Max Drawdown (%)": "{:.2f}"
                }))
                
                # Plot cumulative returns
                st.write("#### Cumulative Returns")
                
                # Create a DataFrame for cumulative returns
                cum_returns_df = pd.DataFrame({
                    p["Portfolio"]: p["Cumulative Return"] for p in performance_data
                }, index=returns.index)
                
                # Plot cumulative returns
                cum_fig = px.line(
                    cum_returns_df,
                    title="Cumulative Portfolio Returns"
                )
                
                cum_fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    yaxis_tickformat=".2%"
                )
                
                st.plotly_chart(cum_fig, use_container_width=True)
                
                # Create a risk vs return scatter plot
                st.write("#### Risk vs Return")
                
                # Create a DataFrame for risk vs return
                risk_return_df = pd.DataFrame([
                    {
                        "Portfolio": p["Portfolio"],
                        "Annual Return (%)": p["Annual Return (%)"],
                        "Annual Volatility (%)": p["Annual Volatility (%)"],
                        "Sharpe Ratio": p["Sharpe Ratio"]
                    } for p in performance_data
                ])
                
                # Plot risk vs return
                risk_fig = px.scatter(
                    risk_return_df,
                    x="Annual Volatility (%)",
                    y="Annual Return (%)",
                    text="Portfolio",
                    size="Sharpe Ratio",
                    size_max=20,
                    title="Risk vs Return"
                )
                
                risk_fig.update_traces(textposition='top center')
                
                st.plotly_chart(risk_fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Please check your stock symbols and try again. Make sure all symbols are valid.")

# Mean Risk Tab
with tab3:
    st.header("Mean Risk")
    st.write("This tab visualizes the efficient frontier and optimal portfolios based on Modern Portfolio Theory.")
    
    # Input for stock symbols
    with st.expander("Select Assets", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            stock_input = st.text_input(
                "Enter stock symbols (comma-separated):",
                value="AAPL,MSFT,AMZN,GOOGL,BRK-B,JNJ",
                help="Example: AAPL,MSFT,AMZN,GOOGL,BRK-B,JNJ",
                key="stocks_tab3"
            )
            
        with col2:
            today = datetime.now()
            start_date = st.date_input(
                "Start Date",
                value=today - timedelta(days=365*3),  # 3 years default
                max_value=today,
                key="start_date_tab3"
            )
            end_date = st.date_input(
                "End Date",
                value=today,
                max_value=today,
                key="end_date_tab3"
            )
    
    # Process the input
    if stock_input:
        stocks = [s.strip() for s in stock_input.split(',')]
        
        # Download data
        @st.cache_data(ttl=3600)  # Cache data for 1 hour
        def load_stock_data(tickers, start, end):
            data = yf.download(tickers, start=start, end=end)['Close']
            # If only one stock, convert to DataFrame
            if isinstance(data, pd.Series):
                data = data.to_frame(name=tickers[0])
            return data
        
        try:
            # Load data
            df_stocks = load_stock_data(stocks, start_date, end_date)
            
            # Calculate daily returns
            daily_returns = df_stocks.pct_change().dropna()
            
            # Calculate mean returns and covariance matrix
            mean_returns = daily_returns.mean() * 252  # Annualized returns
            cov_matrix = daily_returns.cov() * 252  # Annualized covariance
            
            # Display mean returns and covariance matrix
            st.subheader("Asset Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Mean Annual Returns")
                st.dataframe(mean_returns.to_frame("Annual Return (%)").style.format("{:.2%}"))
            
            with col2:
                st.write("Correlation Matrix")
                corr_matrix = daily_returns.corr()
                fig_corr = px.imshow(
                    corr_matrix, 
                    text_auto=True, 
                    color_continuous_scale='RdBu_r',
                    title="Asset Correlation Matrix"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Risk-free rate
            risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0) / 100
            
            # Functions for portfolio optimization
            def portfolio_perf(weights, mean_returns, cov_matrix, risk_free_rate):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
                return portfolio_return, portfolio_stddev, sharpe_ratio
            
            def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
                # Negative Sharpe ratio for minimization
                portfolio_return, portfolio_stddev, sharpe_ratio = portfolio_perf(weights, mean_returns, cov_matrix, risk_free_rate)
                return -sharpe_ratio
            
            def min_variance(weights, mean_returns, cov_matrix):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            def max_return(weights, mean_returns):
                return -np.sum(mean_returns * weights)
            
            def portfolio_return_at_risk(weights, mean_returns, cov_matrix, target_volatility):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                penalty = 100 * abs(portfolio_stddev - target_volatility)
                return -portfolio_return + penalty
            
            # Optimize for different objectives
            num_assets = len(stocks)
            args = (mean_returns, cov_matrix, risk_free_rate)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for asset in range(num_assets))
            
            initial_guess = num_assets * [1. / num_assets]
            
            # Optimize for maximum Sharpe ratio
            optimal_sharpe = minimize(
                neg_sharpe_ratio, 
                initial_guess, 
                args=args, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
            max_sharpe_weights = optimal_sharpe['x']
            max_sharpe_return, max_sharpe_volatility, max_sharpe_ratio = portfolio_perf(
                max_sharpe_weights, 
                mean_returns, 
                cov_matrix, 
                risk_free_rate
            )
            
            # Optimize for minimum volatility
            optimal_variance = minimize(
                min_variance, 
                initial_guess, 
                args=(mean_returns, cov_matrix), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
            min_vol_weights = optimal_variance['x']
            min_vol_return, min_vol_volatility, min_vol_ratio = portfolio_perf(
                min_vol_weights, 
                mean_returns, 
                cov_matrix, 
                risk_free_rate
            )
            
            # Optimize for maximum return
            optimal_return = minimize(
                max_return, 
                initial_guess, 
                args=(mean_returns,), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
            max_return_weights = optimal_return['x']
            max_ret_return, max_ret_volatility, max_ret_ratio = portfolio_perf(
                max_return_weights, 
                mean_returns, 
                cov_matrix, 
                risk_free_rate
            )
            
            # Generate efficient frontier
            st.subheader("Efficient Frontier")
            
            # Number of portfolios for the frontier
            num_portfolios = 1000
            
            # Arrays to store risk and returns
            ef_returns = []
            ef_volatilities = []
            
            # Target volatilities for the efficient frontier
            target_volatilities = np.linspace(min_vol_volatility, max_ret_volatility, num_portfolios)
            
            with st.spinner("Generating efficient frontier..."):
                # Generate efficient frontier
                for target in target_volatilities:
                    args = (mean_returns, cov_matrix, target)
                    ef_opt = minimize(
                        portfolio_return_at_risk, 
                        initial_guess, 
                        args=args, 
                        method='SLSQP', 
                        bounds=bounds, 
                        constraints=constraints
                    )
                    ef_weights = ef_opt['x']
                    ef_ret, ef_vol, _ = portfolio_perf(ef_weights, mean_returns, cov_matrix, risk_free_rate)
                    ef_returns.append(ef_ret)
                    ef_volatilities.append(ef_vol)
            
            # Create data for the scatter plot
            portfolio_data = {
                "Annual Return (%)": [max_sharpe_return, min_vol_return, max_ret_return],
                "Annual Volatility (%)": [max_sharpe_volatility, min_vol_volatility, max_ret_volatility],
                "Sharpe Ratio": [max_sharpe_ratio, min_vol_ratio, max_ret_ratio],
                "Portfolio": ["Maximum Sharpe", "Minimum Volatility", "Maximum Return"]
            }
            
            portfolio_df = pd.DataFrame(portfolio_data)
            
            # Create the efficient frontier plot
            fig = go.Figure()
            
            # Add efficient frontier
            fig.add_trace(
                go.Scatter(
                    x=ef_volatilities,
                    y=ef_returns,
                    mode='lines',
                    line=dict(color='blue', width=1),
                    name='Efficient Frontier'
                )
            )
            
            # Add optimal portfolios
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df["Annual Volatility (%)"],
                    y=portfolio_df["Annual Return (%)"],
                    mode='markers+text',
                    marker=dict(
                        size=10,
                        color=['red', 'green', 'blue']
                    ),
                    text=portfolio_df["Portfolio"],
                    textposition="top center",
                    name='Optimal Portfolios'
                )
            )
            
            # Add individual assets
            for i, stock in enumerate(stocks):
                fig.add_trace(
                    go.Scatter(
                        x=[np.sqrt(cov_matrix.iloc[i, i])],
                        y=[mean_returns[i]],
                        mode='markers+text',
                        marker=dict(size=8, color='purple'),
                        text=[stock],
                        textposition="top center",
                        name=stock
                    )
                )
            
            # Add capital allocation line
            if risk_free_rate > 0:
                max_sharpe_slope = (max_sharpe_return - risk_free_rate) / max_sharpe_volatility
                cal_x = np.linspace(0, max(ef_volatilities) * 1.2, 100)
                cal_y = risk_free_rate + max_sharpe_slope * cal_x
                
                fig.add_trace(
                    go.Scatter(
                        x=cal_x,
                        y=cal_y,
                        mode='lines',
                        line=dict(color='black', width=1, dash='dash'),
                        name='Capital Allocation Line'
                    )
                )
            
            # Update layout
            fig.update_layout(
                title='Efficient Frontier and Optimal Portfolios',
                xaxis_title='Annual Volatility (%)',
                yaxis_title='Annual Return (%)',
                xaxis=dict(tickformat='.0%'),
                yaxis=dict(tickformat='.0%'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Show the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display optimal portfolio weights
            st.subheader("Optimal Portfolio Weights")
            
            # Create a DataFrame for weights
            weights_data = {
                "Maximum Sharpe": max_sharpe_weights,
                "Minimum Volatility": min_vol_weights,
                "Maximum Return": max_return_weights
            }
            
            weights_df = pd.DataFrame(weights_data, index=stocks)
            
            # Show weights
            st.dataframe(weights_df.style.format("{:.2%}").background_gradient(cmap='Blues'))
            
            # Create a bar chart to visualize portfolio weights
            weights_fig = go.Figure()
            
            for portfolio, weights in weights_data.items():
                weights_fig.add_trace(go.Bar(
                    x=stocks,
                    y=weights,
                    name=portfolio
                ))
            
            weights_fig.update_layout(
                title="Optimal Portfolio Allocations",
                xaxis_title="Assets",
                yaxis_title="Weight",
                barmode='group',
                yaxis=dict(tickformat='.0%'),
            )
            
            st.plotly_chart(weights_fig, use_container_width=True)
            
            # Display portfolio statistics
            st.subheader("Optimal Portfolio Statistics")
            
            # Display statistics
            st.dataframe(portfolio_df.set_index("Portfolio").style.format({
                "Annual Return (%)": "{:.2%}",
                "Annual Volatility (%)": "{:.2%}",
                "Sharpe Ratio": "{:.2f}"
            }))
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Please check your stock symbols and try again. Make sure all symbols are valid.")

# Risk Building Tab
with tab4:
    st.header("Risk Building")
    st.write("This tab helps you build a portfolio based on your risk preference and target return.")
    
    # Input for stock symbols
    with st.expander("Select Assets", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            stock_input = st.text_input(
                "Enter stock symbols (comma-separated):",
                value="AAPL,MSFT,AMZN,GOOGL,BRK-B,JNJ,V,PG,JPM,NVDA",
                help="Example: AAPL,MSFT,AMZN,GOOGL,BRK-B,JNJ,V,PG,JPM,NVDA",
                key="stocks_tab4"
            )
            
        with col2:
            today = datetime.now()
            start_date = st.date_input(
                "Start Date",
                value=today - timedelta(days=365*3),  # 3 years default
                max_value=today,
                key="start_date_tab4"
            )
            end_date = st.date_input(
                "End Date",
                value=today,
                max_value=today,
                key="end_date_tab4"
            )
    
    # Process the input
    if stock_input:
        stocks = [s.strip() for s in stock_input.split(',')]
        
        # Download data
        @st.cache_data(ttl=3600)  # Cache data for 1 hour
        def load_stock_data(tickers, start, end):
            data = yf.download(tickers, start=start, end=end)['Close']
            # If only one stock, convert to DataFrame
            if isinstance(data, pd.Series):
                data = data.to_frame(name=tickers[0])
            return data
        
        try:
            # Load data
            df_stocks = load_stock_data(stocks, start_date, end_date)
            
            # Calculate daily returns
            daily_returns = df_stocks.pct_change().dropna()
            
            # Calculate mean returns and covariance matrix
            mean_returns = daily_returns.mean() * 252  # Annualized returns
            cov_matrix = daily_returns.cov() * 252  # Annualized covariance
            
            # Display asset info
            st.subheader("Asset Information")
            
            # Create a DataFrame for asset information
            asset_info = pd.DataFrame({
                "Annual Return (%)": mean_returns * 100,
                "Annual Volatility (%)": np.sqrt(np.diag(cov_matrix)) * 100
            })
            
            # Display asset information
            st.dataframe(asset_info)
            
            # Plot asset info as a scatter plot
            fig = px.scatter(
                asset_info, 
                x="Annual Volatility (%)", 
                y="Annual Return (%)",
                text=asset_info.index,
                title="Risk-Return Profile of Individual Assets"
            )
            
            fig.update_traces(textposition='top center', marker=dict(size=10))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk preferences
            st.subheader("Build Your Portfolio")
            
            # Risk-free rate
            risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, key="rf_tab4") / 100
            
            # Create tabs for different portfolio building approaches
            build_tab1, build_tab2, build_tab3 = st.tabs([
                "Based on Risk Tolerance", 
                "Based on Target Return", 
                "Custom Constraints"
            ])
            
            # Portfolio optimization functions
            def portfolio_perf(weights, mean_returns, cov_matrix, risk_free_rate):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
                return portfolio_return, portfolio_stddev, sharpe_ratio
            
            def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
                portfolio_return, portfolio_stddev, sharpe_ratio = portfolio_perf(weights, mean_returns, cov_matrix, risk_free_rate)
                return -sharpe_ratio
            
            def min_variance(weights, mean_returns, cov_matrix):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            def portfolio_return(weights, mean_returns):
                return np.sum(mean_returns * weights)
            
            def portfolio_return_at_risk(weights, mean_returns, cov_matrix, target_volatility):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                penalty = 100 * abs(portfolio_stddev - target_volatility)
                return -portfolio_return + penalty
            
            def portfolio_volatility_at_return(weights, mean_returns, cov_matrix, target_return):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                penalty = 100 * abs(portfolio_return - target_return)
                return portfolio_stddev + penalty
            
            # Based on Risk Tolerance
            with build_tab1:
                st.write("Build a portfolio based on your risk tolerance (volatility).")
                
                # Risk tolerance slider
                min_vol = min(np.sqrt(np.diag(cov_matrix)))
                max_vol = max(np.sqrt(np.diag(cov_matrix)))
                
                risk_tolerance = st.slider(
                    "Select your annual risk tolerance (volatility %)",
                    float(min_vol * 100),
                    float(max_vol * 100),
                    float((min_vol + max_vol) / 2 * 100),
                    step=0.5
                ) / 100
                
                # Calculate optimal portfolio for selected risk tolerance
                num_assets = len(stocks)
                args = (mean_returns, cov_matrix, risk_tolerance)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for asset in range(num_assets))
                
                initial_guess = num_assets * [1. / num_assets]
                
                # Optimize for maximum return at given risk
                optimal_portfolio = minimize(
                    portfolio_return_at_risk,
                    initial_guess,
                    args=args,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                risk_based_weights = optimal_portfolio['x']
                
                # Calculate performance metrics
                risk_based_return, risk_based_volatility, risk_based_sharpe = portfolio_perf(
                    risk_based_weights,
                    mean_returns,
                    cov_matrix,
                    risk_free_rate
                )
                
                # Display portfolio
                st.write("### Optimized Portfolio for Selected Risk Level")
                
                # Create a DataFrame for weights
                weights_df = pd.DataFrame({
                    "Asset": stocks,
                    "Weight (%)": risk_based_weights * 100
                })
                
                # Sort by weight
                weights_df = weights_df.sort_values("Weight (%)", ascending=False)
                
                # Display weights as a bar chart
                fig = px.bar(
                    weights_df,
                    x="Asset",
                    y="Weight (%)",
                    title="Portfolio Allocation",
                    color="Weight (%)",
                    color_continuous_scale="Viridis"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display performance metrics
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Expected Annual Return", f"{risk_based_return*100:.2f}%")
                col2.metric("Expected Annual Volatility", f"{risk_based_volatility*100:.2f}%")
                col3.metric("Sharpe Ratio", f"{risk_based_sharpe:.2f}")
                
                # Display weights
                st.dataframe(weights_df)
            
            # Based on Target Return
            with build_tab2:
                st.write("Build a portfolio based on your target return.")
                
                # Target return slider
                min_ret = min(mean_returns)
                max_ret = max(mean_returns)
                
                target_return = st.slider(
                    "Select your target annual return (%)",
                    float(min_ret * 100),
                    float(max_ret * 100),
                    float((min_ret + max_ret) / 2 * 100),
                    step=0.5
                ) / 100
                
                # Calculate optimal portfolio for selected target return
                num_assets = len(stocks)
                args = (mean_returns, cov_matrix, target_return)
                constraints = (
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns) - target_return}
                )
                bounds = tuple((0, 1) for asset in range(num_assets))
                
                initial_guess = num_assets * [1. / num_assets]
                
                # Optimize for minimum volatility at given return
                optimal_portfolio = minimize(
                    min_variance,
                    initial_guess,
                    args=(mean_returns, cov_matrix),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                return_based_weights = optimal_portfolio['x']
                
                # Calculate performance metrics
                return_based_return, return_based_volatility, return_based_sharpe = portfolio_perf(
                    return_based_weights,
                    mean_returns,
                    cov_matrix,
                    risk_free_rate
                )
                
                # Display portfolio
                st.write("### Optimized Portfolio for Target Return")
                
                # Create a DataFrame for weights
                weights_df = pd.DataFrame({
                    "Asset": stocks,
                    "Weight (%)": return_based_weights * 100
                })
                
                # Sort by weight
                weights_df = weights_df.sort_values("Weight (%)", ascending=False)
                
                # Display weights as a bar chart
                fig = px.bar(
                    weights_df,
                    x="Asset",
                    y="Weight (%)",
                    title="Portfolio Allocation",
                    color="Weight (%)",
                    color_continuous_scale="Viridis"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display performance metrics
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Expected Annual Return", f"{return_based_return*100:.2f}%")
                col2.metric("Expected Annual Volatility", f"{return_based_volatility*100:.2f}%")
                col3.metric("Sharpe Ratio", f"{return_based_sharpe:.2f}")
                
                # Display weights
                st.dataframe(weights_df)
            
            # Custom Constraints
            with build_tab3:
                st.write("Build a portfolio with custom constraints.")
                
                # Maximum weight per asset
                max_weight = st.slider(
                    "Maximum weight per asset (%)",
                    0,
                    100,
                    20,
                    step=5
                ) / 100
                
                # Minimum weight per asset
                min_weight = st.slider(
                    "Minimum weight per asset (%)",
                    0,
                    50,
                    0,
                    step=5
                ) / 100
                
                # Optimization objective
                objective = st.selectbox(
                    "Optimization Objective",
                    ["Maximize Sharpe Ratio", "Minimize Volatility", "Maximize Return"]
                )
                
                # Calculate optimal portfolio based on selected constraints and objective
                num_assets = len(stocks)
                
                # Bounds based on selected constraints
                bounds = tuple((min_weight, max_weight) for asset in range(num_assets))
                
                # Constraints
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                
                # Initial guess
                initial_guess = num_assets * [1. / num_assets]
                
                # Optimize based on selected objective
                if objective == "Maximize Sharpe Ratio":
                    args = (mean_returns, cov_matrix, risk_free_rate)
                    optimal_portfolio = minimize(
                        neg_sharpe_ratio,
                        initial_guess,
                        args=args,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints
                    )
                elif objective == "Minimize Volatility":
                    args = (mean_returns, cov_matrix)
                    optimal_portfolio = minimize(
                        min_variance,
                        initial_guess,
                        args=args,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints
                    )
                else:  # Maximize Return
                    args = (mean_returns,)
                    optimal_portfolio = minimize(
                        lambda x, mean_returns: -portfolio_return(x, mean_returns),
                        initial_guess,
                        args=args,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints
                    )
                
                custom_weights = optimal_portfolio['x']
                
                # Calculate performance metrics
                custom_return, custom_volatility, custom_sharpe = portfolio_perf(
                    custom_weights,
                    mean_returns,
                    cov_matrix,
                    risk_free_rate
                )
                
                # Display portfolio
                st.write("### Custom Optimized Portfolio")
                
                # Create a DataFrame for weights
                weights_df = pd.DataFrame({
                    "Asset": stocks,
                    "Weight (%)": custom_weights * 100
                })
                
                # Sort by weight
                weights_df = weights_df.sort_values("Weight (%)", ascending=False)
                
                # Display weights as a bar chart
                fig = px.bar(
                    weights_df,
                    x="Asset",
                    y="Weight (%)",
                    title="Portfolio Allocation",
                    color="Weight (%)",
                    color_continuous_scale="Viridis"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display performance metrics
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Expected Annual Return", f"{custom_return*100:.2f}%")
                col2.metric("Expected Annual Volatility", f"{custom_volatility*100:.2f}%")
                col3.metric("Sharpe Ratio", f"{custom_sharpe:.2f}")
                
                # Display weights
                st.dataframe(weights_df)
            
            # Historical performance simulation
            st.subheader("Simulated Historical Performance")
            
            # Select portfolio to simulate
            portfolio_option = st.selectbox(
                "Select Portfolio to Simulate",
                ["Risk-Based Portfolio", "Return-Based Portfolio", "Custom Portfolio"]
            )
            
            # Get selected weights
            if portfolio_option == "Risk-Based Portfolio":
                selected_weights = risk_based_weights
                portfolio_name = "Risk-Based Portfolio"
            elif portfolio_option == "Return-Based Portfolio":
                selected_weights = return_based_weights
                portfolio_name = "Return-Based Portfolio"
            else:
                selected_weights = custom_weights
                portfolio_name = "Custom Portfolio"
            
            # Calculate historical performance
            historical_returns = daily_returns @ selected_weights
            cumulative_returns = (1 + historical_returns).cumprod()
            
            # Calculate drawdown
            previous_peaks = cumulative_returns.cummax()
            drawdown = (cumulative_returns / previous_peaks) - 1
            
            # Calculate performance metrics
            annual_return = historical_returns.mean() * 252
            annual_volatility = historical_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility
            max_drawdown = drawdown.min()
            
            # Display performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Historical Annual Return", f"{annual_return*100:.2f}%")
            col2.metric("Historical Annual Volatility", f"{annual_volatility*100:.2f}%")
            col3.metric("Historical Sharpe Ratio", f"{sharpe_ratio:.2f}")
            col4.metric("Maximum Drawdown", f"{max_drawdown*100:.2f}%")
            
            # Plot cumulative returns
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns,
                    mode='lines',
                    name=f"{portfolio_name} Cumulative Returns"
                )
            )
            
            fig.update_layout(
                title="Historical Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                yaxis=dict(tickformat='.2%')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot drawdown
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    mode='lines',
                    name=f"{portfolio_name} Drawdown",
                    fill='tozeroy',
                    line=dict(color='red')
                )
            )
            
            fig.update_layout(
                title="Historical Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown",
                yaxis=dict(tickformat='.0%')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly returns heatmap
            monthly_returns = historical_returns.resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            
            # Create a DataFrame with month and year for heatmap
            monthly_returns_df = pd.DataFrame(monthly_returns)
            monthly_returns_df.index = pd.to_datetime(monthly_returns_df.index)
            monthly_returns_df['Year'] = monthly_returns_df.index.year
            monthly_returns_df['Month'] = monthly_returns_df.index.strftime('%b')
            
            # Pivot for heatmap
            heatmap_data = monthly_returns_df.pivot_table(
                index='Year',
                columns='Month',
                values=0,
                aggfunc='sum'
            )
            
            # Month order
            months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Reorder columns
            heatmap_data = heatmap_data[
                [month for month in months_order if month in heatmap_data.columns]
            ]
            
            # Create heatmap
            fig = px.imshow(
                heatmap_data,
                text_auto='.1%',
                aspect="auto",
                color_continuous_scale='RdYlGn',
                title=f"Monthly Returns Heatmap for {portfolio_name}"
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Year",
                coloraxis_colorbar=dict(title="Return")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Please check your stock symbols and try again. Make sure all symbols are valid.")