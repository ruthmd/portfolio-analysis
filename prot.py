import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime, timedelta
import io
import base64

# Set page config
st.set_page_config(
    page_title="Portfolio Optimization App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<div class='main-header'>Portfolio Optimization App</div>", unsafe_allow_html=True)
st.markdown("""
This app helps you optimize your investment portfolio based on Modern Portfolio Theory (MPT).
You can select stocks, analyze historical performance, and find optimal asset allocations based on your risk preferences.
""")

# Sidebar for inputs
st.sidebar.markdown("## Configuration")

# Date selection
st.sidebar.markdown("### Historical Data Period")
start_date = st.sidebar.date_input(
    "Start Date",
    datetime.now() - timedelta(days=365*5)
)
end_date = st.sidebar.date_input(
    "End Date",
    datetime.now()
)

# Asset selection method
asset_selection_method = st.sidebar.radio(
    "Asset Selection Method",
    ["Manual Entry", "Upload CSV", "Sample Portfolios"]
)

# Function to download Yahoo Finance data
@st.cache_data(ttl=3600)  # Cache the data for 1 hour
def download_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        # If only one ticker is provided, the output is a Series, not a DataFrame
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data, columns=[tickers])
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    # Expected portfolio return
    portfolio_return = np.sum(mean_returns * weights) * 252
    
    # Expected portfolio volatility (risk)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return {
        'return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    }

# Function to optimize for maximum Sharpe ratio
def optimize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0.02):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    initial_guess = np.array(num_assets * [1. / num_assets])
    
    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        portfolio_metrics = calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate)
        return -portfolio_metrics['sharpe_ratio']
    
    result = minimize(neg_sharpe_ratio, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result['x']

# Function to optimize for minimum volatility
def optimize_min_volatility(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    initial_guess = np.array(num_assets * [1. / num_assets])
    
    def portfolio_volatility(weights, mean_returns, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    result = minimize(portfolio_volatility, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result['x']

# Function to generate efficient frontier
def generate_efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficient_portfolios = []
    num_assets = len(mean_returns)
    
    for ret in returns_range:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) * 252 - ret}
        )
        bounds = tuple((0, 1) for asset in range(num_assets))
        initial_guess = np.array(num_assets * [1. / num_assets])
        
        def portfolio_volatility(weights, mean_returns, cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        result = minimize(
            portfolio_volatility, 
            initial_guess,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result['success']:
            efficient_portfolios.append({
                'return': ret,
                'volatility': portfolio_volatility(result['x'], mean_returns, cov_matrix),
                'weights': result['x']
            })
    
    return efficient_portfolios

# Function to create downloadable links
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href

# Sample portfolios
sample_portfolios = {
    "S&P 500 Top 10": "AAPL,MSFT,AMZN,NVDA,GOOGL,META,BRK-B,TSLA,UNH,JPM",
    "Tech Giants": "AAPL,MSFT,AMZN,NVDA,GOOGL,META,TSLA,AMD,INTC,CSCO",
    "Dividend Champions": "JNJ,PG,KO,PEP,VZ,MMM,T,IBM,MRK,XOM",
    "Global ETFs": "SPY,QQQ,VEA,VWO,AGG,BND,GLD,VNQ,TLT,IEF"
}

# Main app logic
# Asset selection
if asset_selection_method == "Manual Entry":
    ticker_input = st.text_input(
        "Enter ticker symbols (comma separated, e.g., AAPL,MSFT,GOOG)",
        "AAPL,MSFT,AMZN,GOOGL,JNJ,JPM,V,PG,UNH,HD"
    )
    tickers = [ticker.strip() for ticker in ticker_input.split(',')]
    
elif asset_selection_method == "Upload CSV":
    st.markdown("Upload a CSV file with ticker symbols (one per row, with a header 'Ticker')")
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    
    if uploaded_file is not None:
        try:
            ticker_df = pd.read_csv(uploaded_file)
            if 'Ticker' in ticker_df.columns:
                tickers = ticker_df['Ticker'].tolist()
                st.success(f"Loaded {len(tickers)} tickers from CSV")
            else:
                st.error("CSV must have a 'Ticker' column")
                tickers = []
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            tickers = []
    else:
        tickers = []
        
elif asset_selection_method == "Sample Portfolios":
    selected_portfolio = st.sidebar.selectbox(
        "Choose a sample portfolio",
        list(sample_portfolios.keys())
    )
    tickers = [ticker.strip() for ticker in sample_portfolios[selected_portfolio].split(',')]
    st.markdown(f"**Selected Sample Portfolio:** {selected_portfolio}")
    st.markdown(f"**Tickers:** {', '.join(tickers)}")

# Only proceed if we have tickers
if tickers and len(tickers) > 0:
    # Download data
    with st.spinner('Downloading stock data...'):
        stock_data = download_stock_data(tickers, start_date, end_date)
    
    if stock_data is not None and not stock_data.empty:
        # Show raw data if requested
        with st.expander("View Raw Price Data"):
            st.dataframe(stock_data)
            st.markdown(get_table_download_link(stock_data, "price_data", "Download Price Data"), unsafe_allow_html=True)
        
        # Calculate returns
        returns = stock_data.pct_change().dropna()
        
        # Display correlation matrix
        with st.expander("Asset Correlation Matrix"):
            corr_matrix = returns.corr()
            
            # Heatmap using plotly
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title="Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display returns and volatility
        with st.expander("Asset Returns & Volatility"):
            annual_returns = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)
            
            metrics_df = pd.DataFrame({
                'Annual Return': annual_returns,
                'Annual Volatility': annual_volatility,
                'Return/Risk Ratio': annual_returns / annual_volatility
            })
            
            st.dataframe(metrics_df)
            
            # Scatter plot of risk vs return
            fig = px.scatter(
                x=annual_volatility,
                y=annual_returns,
                text=metrics_df.index,
                color=annual_returns / annual_volatility,
                color_continuous_scale='viridis',
                title="Risk vs Return for Individual Assets",
                labels={'x': 'Annual Volatility (Risk)', 'y': 'Annual Return', 'color': 'Return/Risk Ratio'}
            )
            fig.update_traces(textposition='top center', marker=dict(size=15))
            st.plotly_chart(fig, use_container_width=True)
        
        # Historical performance visualization
        with st.expander("Historical Performance"):
            # Normalize price data to 100
            normalized_data = stock_data / stock_data.iloc[0] * 100
            
            # Line chart using plotly
            fig = px.line(
                normalized_data,
                title="Historical Performance (Normalized to 100)",
                labels={'value': 'Normalized Price', 'variable': 'Asset'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod() - 1
            
            # Line chart for cumulative returns
            fig = px.line(
                cumulative_returns,
                title="Cumulative Returns",
                labels={'value': 'Cumulative Return', 'variable': 'Asset'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio Optimization
        st.markdown("<div class='sub-header'>Portfolio Optimization</div>", unsafe_allow_html=True)
        
        # Risk-free rate input
        risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Optimization methods
        optimization_methods = st.multiselect(
            "Select Optimization Methods",
            ["Maximum Sharpe Ratio", "Minimum Volatility", "Efficient Frontier", "Equal Weights"],
            default=["Maximum Sharpe Ratio", "Minimum Volatility", "Equal Weights"]
        )
        
        # Calculate and display optimized portfolios
        portfolios = {}
        colors = {
            "Maximum Sharpe Ratio": "green",
            "Minimum Volatility": "blue",
            "Equal Weights": "grey"
        }
        
        # For efficient frontier
        efficient_frontier_portfolios = None
        
        if "Maximum Sharpe Ratio" in optimization_methods:
            try:
                max_sharpe_weights = optimize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
                max_sharpe_metrics = calculate_portfolio_metrics(max_sharpe_weights, mean_returns, cov_matrix, risk_free_rate)
                
                portfolios["Maximum Sharpe Ratio"] = {
                    "weights": max_sharpe_weights,
                    "metrics": max_sharpe_metrics
                }
            except Exception as e:
                st.error(f"Error optimizing for Maximum Sharpe Ratio: {e}")
        
        if "Minimum Volatility" in optimization_methods:
            try:
                min_vol_weights = optimize_min_volatility(mean_returns, cov_matrix)
                min_vol_metrics = calculate_portfolio_metrics(min_vol_weights, mean_returns, cov_matrix, risk_free_rate)
                
                portfolios["Minimum Volatility"] = {
                    "weights": min_vol_weights,
                    "metrics": min_vol_metrics
                }
            except Exception as e:
                st.error(f"Error optimizing for Minimum Volatility: {e}")
        
        if "Equal Weights" in optimization_methods:
            equal_weights = np.array([1/len(tickers)] * len(tickers))
            equal_weights_metrics = calculate_portfolio_metrics(equal_weights, mean_returns, cov_matrix, risk_free_rate)
            
            portfolios["Equal Weights"] = {
                "weights": equal_weights,
                "metrics": equal_weights_metrics
            }
        
        if "Efficient Frontier" in optimization_methods:
            try:
                # Get the range of returns
                min_return = min(annual_returns)
                max_return = max(annual_returns)
                
                returns_range = np.linspace(min_return, max_return, 50)
                efficient_frontier_portfolios = generate_efficient_frontier(mean_returns, cov_matrix, returns_range)
            except Exception as e:
                st.error(f"Error generating Efficient Frontier: {e}")
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["Portfolio Allocation", "Risk-Return Profile", "Efficient Frontier"])
        
        with tab1:
            # Display portfolio weights for each optimization method
            for method, portfolio in portfolios.items():
                weights = portfolio["weights"]
                metrics = portfolio["metrics"]
                
                st.markdown(f"### {method}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Annual Return", f"{metrics['return']*100:.2f}%")
                with col2:
                    st.metric("Expected Annual Volatility", f"{metrics['volatility']*100:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                
                # Create a DataFrame for the weights
                weights_df = pd.DataFrame({
                    'Asset': tickers,
                    'Weight': weights * 100  # Convert to percentage
                })
                
                # Sort by weight descending
                weights_df = weights_df.sort_values('Weight', ascending=False)
                
                # Display as a table
                st.dataframe(weights_df)
                
                # Pie chart for weights
                fig = px.pie(
                    weights_df,
                    values='Weight',
                    names='Asset',
                    title=f"{method} - Portfolio Allocation",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
                # Download link for weights
                st.markdown(get_table_download_link(weights_df, f"{method.replace(' ', '_')}_weights", f"Download {method} Weights"), unsafe_allow_html=True)
                
                st.markdown("---")
        
        with tab2:
            # Create a plot showing the risk-return profile of all portfolios
            fig = go.Figure()
            
            # Add a scatter point for each asset
            for i, ticker in enumerate(tickers):
                fig.add_trace(go.Scatter(
                    x=[annual_volatility[i] * 100],
                    y=[annual_returns[i] * 100],
                    mode='markers+text',
                    name=ticker,
                    text=[ticker],
                    textposition="top center",
                    marker=dict(size=10)
                ))
            
            # Add a point for each optimized portfolio
            for method, portfolio in portfolios.items():
                metrics = portfolio["metrics"]
                fig.add_trace(go.Scatter(
                    x=[metrics['volatility'] * 100],
                    y=[metrics['return'] * 100],
                    mode='markers+text',
                    name=method,
                    text=[method],
                    textposition="top center",
                    marker=dict(
                        size=15,
                        color=colors.get(method, "purple"),
                        symbol='star'
                    )
                ))
            
            # Update layout
            fig.update_layout(
                title="Risk-Return Profile",
                xaxis_title="Expected Volatility (%)",
                yaxis_title="Expected Return (%)",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if "Efficient Frontier" in optimization_methods and efficient_frontier_portfolios:
                # Extract data for plotting
                ef_returns = [p['return'] * 100 for p in efficient_frontier_portfolios]
                ef_volatility = [p['volatility'] * 100 for p in efficient_frontier_portfolios]
                
                # Create the efficient frontier plot
                fig = go.Figure()
                
                # Add the efficient frontier line
                fig.add_trace(go.Scatter(
                    x=ef_volatility,
                    y=ef_returns,
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='purple', width=3)
                ))
                
                # Add individual assets
                for i, ticker in enumerate(tickers):
                    fig.add_trace(go.Scatter(
                        x=[annual_volatility[i] * 100],
                        y=[annual_returns[i] * 100],
                        mode='markers+text',
                        name=ticker,
                        text=[ticker],
                        textposition="top center",
                        marker=dict(size=10)
                    ))
                
                # Add optimized portfolios
                for method, portfolio in portfolios.items():
                    metrics = portfolio["metrics"]
                    fig.add_trace(go.Scatter(
                        x=[metrics['volatility'] * 100],
                        y=[metrics['return'] * 100],
                        mode='markers+text',
                        name=method,
                        text=[method],
                        textposition="top center",
                        marker=dict(
                            size=15,
                            color=colors.get(method, "purple"),
                            symbol='star'
                        )
                    ))
                
                # Update layout
                fig.update_layout(
                    title="Efficient Frontier",
                    xaxis_title="Expected Volatility (%)",
                    yaxis_title="Expected Return (%)",
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Allow exploration of portfolios on the efficient frontier
                st.markdown("### Explore Efficient Frontier Portfolios")
                
                # Slider to select a point on the efficient frontier
                selected_return = st.slider(
                    "Select Target Return (%)",
                    min_value=min(ef_returns),
                    max_value=max(ef_returns),
                    value=(min(ef_returns) + max(ef_returns)) / 2,
                    step=0.1
                )
                
                # Find the closest portfolio
                closest_idx = min(range(len(ef_returns)), key=lambda i: abs(ef_returns[i] - selected_return))
                selected_portfolio = efficient_frontier_portfolios[closest_idx]
                
                # Display the selected portfolio
                st.markdown("#### Selected Portfolio")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Expected Annual Return", f"{selected_portfolio['return']*100:.2f}%")
                with col2:
                    st.metric("Expected Annual Volatility", f"{selected_portfolio['volatility']*100:.2f}%")
                
                # Create a DataFrame for the weights
                weights_df = pd.DataFrame({
                    'Asset': tickers,
                    'Weight': selected_portfolio['weights'] * 100  # Convert to percentage
                })
                
                # Sort by weight descending
                weights_df = weights_df.sort_values('Weight', ascending=False)
                
                # Display as a table
                st.dataframe(weights_df)
                
                # Pie chart for weights
                fig = px.pie(
                    weights_df,
                    values='Weight',
                    names='Asset',
                    title="Portfolio Allocation",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
                # Download link for weights
                st.markdown(get_table_download_link(weights_df, "efficient_frontier_weights", "Download Portfolio Weights"), unsafe_allow_html=True)
            else:
                st.info("Select 'Efficient Frontier' as one of the optimization methods to view this tab.")
        
        # Backtesting section
        st.markdown("<div class='sub-header'>Portfolio Backtesting</div>", unsafe_allow_html=True)
        
        # Select a portfolio for backtesting
        if portfolios:
            backtest_portfolio = st.selectbox(
                "Select Portfolio for Backtesting",
                list(portfolios.keys())
            )
            
            selected_weights = portfolios[backtest_portfolio]["weights"]
            
            # Calculate historical portfolio performance
            weighted_returns = returns.dot(selected_weights)
            
            # Cumulative returns
            cumulative_portfolio_return = (1 + weighted_returns).cumprod() - 1
            
            # Benchmark comparison (using equal weights as a simple benchmark)
            equal_weights = np.array([1/len(tickers)] * len(tickers))
            benchmark_returns = returns.dot(equal_weights)
            cumulative_benchmark_return = (1 + benchmark_returns).cumprod() - 1
            
            # Plot the results
            performance_df = pd.DataFrame({
                'Optimized Portfolio': cumulative_portfolio_return,
                'Equal-Weight Benchmark': cumulative_benchmark_return
            })
            
            fig = px.line(
                performance_df,
                title="Portfolio Backtesting",
                labels={'value': 'Cumulative Return', 'variable': 'Portfolio'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate performance metrics
            annual_return = weighted_returns.mean() * 252
            annual_volatility = weighted_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            
            # Calculate drawdowns
            rolling_max = (1 + weighted_returns).cumprod().cummax()
            drawdown = ((1 + weighted_returns).cumprod() / rolling_max) - 1
            max_drawdown = drawdown.min()
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cumulative Return", f"{cumulative_portfolio_return.iloc[-1]*100:.2f}%")
            with col2:
                st.metric("Annual Return", f"{annual_return*100:.2f}%")
            with col3:
                st.metric("Annual Volatility", f"{annual_volatility*100:.2f}%")
            with col4:
                st.metric("Maximum Drawdown", f"{max_drawdown*100:.2f}%")
            
            # Display drawdown chart
            fig = px.area(
                drawdown,
                title="Portfolio Drawdown",
                labels={'value': 'Drawdown', 'index': 'Date'}
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly returns heatmap
            monthly_returns = weighted_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns_matrix = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
            
            # Convert month numbers to names
            month_names = {
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            }
            monthly_returns_matrix.columns = [month_names[col] for col in monthly_returns_matrix.columns]
            
            # Create heatmap
            fig = px.imshow(
                monthly_returns_matrix * 100,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title="Monthly Returns (%)",
                labels={"x": "Month", "y": "Year", "color": "Return (%)"}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to download stock data. Please check the ticker symbols and try again.")

# Footer with disclaimer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='disclaimer'>Disclaimer: This app is for educational purposes only. Past performance is not indicative of future results. Always conduct your own research before making investment decisions.</p>", unsafe_allow_html=True)