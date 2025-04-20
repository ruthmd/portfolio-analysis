import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime, timedelta
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
def optimize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0.02, method="SLSQP"):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    initial_guess = np.array(num_assets * [1. / num_assets])
    
    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        portfolio_metrics = calculate_portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate)
        return -portfolio_metrics['sharpe_ratio']
    
    result = minimize(neg_sharpe_ratio, initial_guess, args=args, method=method, bounds=bounds, constraints=constraints)
    
    return result['x']

# Function to optimize for minimum volatility
def optimize_min_volatility(mean_returns, cov_matrix, method="SLSQP"):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    initial_guess = np.array(num_assets * [1. / num_assets])
    
    def portfolio_volatility(weights, mean_returns, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    result = minimize(portfolio_volatility, initial_guess, args=args, method=method, bounds=bounds, constraints=constraints)
    
    return result['x']

# Function to optimize for target return (used in efficient frontier)
def optimize_for_target_return(mean_returns, cov_matrix, target_return, method="SLSQP"):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) * 252 - target_return}
    )
    
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    initial_guess = np.array(num_assets * [1. / num_assets])
    
    def portfolio_volatility(weights, mean_returns, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    result = minimize(
        portfolio_volatility, 
        initial_guess,
        args=args,
        method=method,
        bounds=bounds,
        constraints=constraints
    )
    
    return result['x'] if result['success'] else None

# Function to generate efficient frontier
def generate_efficient_frontier(mean_returns, cov_matrix, returns_range, method="SLSQP"):
    efficient_portfolios = []
    
    for ret in returns_range:
        weights = optimize_for_target_return(mean_returns, cov_matrix, ret, method)
        
        if weights is not None:
            portfolio_return = np.sum(mean_returns * weights) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            
            efficient_portfolios.append({
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'weights': weights
            })
    
    return efficient_portfolios

# Function to create downloadable links
def get_table_download_link(df, filename, text, file_format="CSV"):
    if file_format == "CSV":
        csv = df.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    else:  # Excel
        buffer = io.BytesIO()
        df.to_excel(buffer, index=True)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">{text}</a>'
    
    return href

# Educational resources
def show_educational_resources():
    st.markdown("<div class='main-header'>Educational Resources</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ## Modern Portfolio Theory (MPT)
    
    Modern Portfolio Theory is a mathematical framework for constructing a portfolio of assets to maximize expected return for a given level of risk.
    
    ### Key Concepts:
    
    **Expected Return**: The anticipated return of an investment over a period of time.
    
    **Risk (Volatility)**: Measured by standard deviation, it represents the variation of returns around the expected return.
    
    **Diversification**: The practice of spreading investments across various financial instruments to reduce risk.
    
    **Correlation**: The degree to which two securities move in relation to each other.
    
    **Efficient Frontier**: The set of optimal portfolios that offer the highest expected return for a defined level of risk.
    
    **Sharpe Ratio**: A measure of risk-adjusted return, calculated as (portfolio return - risk-free rate) / portfolio standard deviation.
    """)
    
    # Add an interactive demonstration of diversification benefits
    st.markdown("### Interactive Demonstration: Benefits of Diversification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        corr_value = st.slider("Correlation between assets", min_value=-1.0, max_value=1.0, value=0.0, step=0.1, key="demo_correlation")
    
    with col2:
        weight_a = st.slider("Weight of Asset A (%)", min_value=0, max_value=100, value=50, step=5, key="demo_weight") / 100
        weight_b = 1 - weight_a
    
    # Generate some returns data
    np.random.seed(42)
    returns_a = np.random.normal(0.10, 0.20, 1000)  # Mean=10%, SD=20%
    
    # Generate correlated returns for asset B
    z = np.random.normal(0.08, 0.15, 1000)  # Mean=8%, SD=15%
    returns_b = corr_value * returns_a + np.sqrt(1 - corr_value**2) * z
    
    # Calculate portfolio returns and statistics
    portfolio_returns = weight_a * returns_a + weight_b * returns_b
    portfolio_mean = portfolio_returns.mean()
    portfolio_std = portfolio_returns.std()
    
    # Scatter plot of returns
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=returns_a,
        y=returns_b,
        mode='markers',
        marker=dict(
            size=5,
            color='rgba(0, 0, 255, 0.5)',
        ),
        name=f'Correlation: {corr_value:.2f}'
    ))
    
    fig.update_layout(
        title=f"Returns Correlation: {corr_value:.2f}",
        xaxis_title="Asset A Returns",
        yaxis_title="Asset B Returns",
        showlegend=True,
        template=theme
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display portfolio statistics
    st.markdown("### Portfolio Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Return", f"{portfolio_mean*100:.2f}%")
    with col2:
        st.metric("Risk (Standard Deviation)", f"{portfolio_std*100:.2f}%")
    with col3:
        st.metric("Risk-Adjusted Return", f"{portfolio_mean/portfolio_std:.2f}")
    
    # More educational content
    st.markdown("""
    ## Key Portfolio Optimization Techniques
    
    ### Maximum Sharpe Ratio Portfolio
    
    This portfolio maximizes the Sharpe ratio, which represents the excess return (above the risk-free rate) per unit of risk.
    It essentially finds the portfolio with the best risk-adjusted return.
    
    ### Minimum Volatility Portfolio
    
    Also known as the minimum variance portfolio, this approach seeks to create a portfolio with the lowest possible risk, regardless of the expected return.
    It's suitable for risk-averse investors who prioritize capital preservation.
    
    ### Efficient Frontier
    
    The efficient frontier represents a set of optimal portfolios that provide the maximum expected return for a given level of risk, or minimum risk for a given level of expected return.
    Portfolios that lie below the efficient frontier are considered sub-optimal.
    
    ## Risk Measures
    
    ### Value at Risk (VaR)
    
    VaR estimates the maximum loss expected over a specific time period at a given confidence level under normal market conditions.
    For example, a 1-day 95% VaR of $100,000 means there's a 5% chance of losing more than $100,000 in a single day.
    
    ### Conditional Value at Risk (CVaR)
    
    Also known as Expected Shortfall, CVaR measures the expected loss in the worst-case scenarios beyond the VaR threshold.
    It provides a more comprehensive view of tail risk than VaR.
    
    ### Maximum Drawdown
    
    The maximum observed loss from a peak to a trough of a portfolio before a new peak is achieved.
    It's an indicator of downside risk over a specified time period.
    """)
    
    # Add a glossary
    with st.expander("Financial Terms Glossary"):
        st.markdown("""
        **Alpha**: Excess return of an investment relative to the return of a benchmark index.
        
        **Beta**: A measure of a stock's volatility in relation to the overall market.
        
        **Capital Asset Pricing Model (CAPM)**: A model that describes the relationship between systematic risk and expected return for assets.
        
        **Diversification**: Spreading investments across various financial instruments to reduce risk.
        
        **Efficient Market Hypothesis (EMH)**: The theory that stock prices reflect all available information.
        
        **Fundamental Analysis**: Evaluating a security's intrinsic value by examining related economic, financial, and other qualitative and quantitative factors.
        
        **Liquidity**: The degree to which an asset can be quickly bought or sold without affecting its price.
        
        **Market Capitalization**: The total dollar market value of a company's outstanding shares.
        
        **Rebalancing**: The process of realigning the weightings of a portfolio of assets by periodically buying or selling assets to maintain the original or desired level of asset allocation.
        
        **Technical Analysis**: A method of evaluating securities by analyzing statistics generated by market activity.
        
        **Yield**: The income return on an investment, such as the interest or dividends received from holding a particular security.
        """)

# Risk assessment function
def show_risk_assessment():
    st.markdown("<div class='main-header'>Risk Tolerance Assessment</div>", unsafe_allow_html=True)
    
    st.markdown("""
    This section helps you evaluate your risk tolerance and understand what investment strategy might be suitable for you.
    Answer the following questions to get a personalized risk profile.
    """)
    
    # Risk assessment questions
    q1 = st.selectbox(
        "1. What is your primary investment objective?",
        [
            "Preserving capital (lowest risk)",
            "Generating income",
            "Income with some growth",
            "Growth with some income",
            "Maximizing long-term growth (highest risk)"
        ],
        index=2
    )
    
    q2 = st.selectbox(
        "2. How long do you plan to invest before you need the money?",
        [
            "Less than 1 year",
            "1-3 years",
            "3-5 years",
            "5-10 years",
            "More than 10 years"
        ],
        index=2
    )
    
    q3 = st.selectbox(
        "3. How would you react if your portfolio suddenly decreased in value by 20%?",
        [
            "Sell all investments and move to cash",
            "Sell some investments to reduce risk",
            "Do nothing and wait for recovery",
            "Keep the investments and add a little more",
            "Significantly increase investments to buy at lower prices"
        ],
        index=2
    )
    
    q4 = st.selectbox(
        "4. Which statement best describes your investment experience?",
        [
            "I have no investment experience",
            "I have some experience with conservative investments",
            "I have moderate experience with various investments",
            "I have significant experience across different asset classes",
            "I am a very experienced investor with advanced knowledge"
        ],
        index=2
    )
    
    q5 = st.selectbox(
        "5. How stable is your current and future income from sources like employment?",
        [
            "Very unstable",
            "Somewhat unstable",
            "Moderately stable",
            "Stable",
            "Very stable"
        ],
        index=2
    )
    
    if st.button("Calculate Risk Profile"):
        # Map answers to scores
        scores = {
            q1: {"Preserving capital (lowest risk)": 1, "Generating income": 2, "Income with some growth": 3, 
                 "Growth with some income": 4, "Maximizing long-term growth (highest risk)": 5},
            q2: {"Less than 1 year": 1, "1-3 years": 2, "3-5 years": 3, "5-10 years": 4, "More than 10 years": 5},
            q3: {"Sell all investments and move to cash": 1, "Sell some investments to reduce risk": 2, 
                 "Do nothing and wait for recovery": 3, "Keep the investments and add a little more": 4, 
                 "Significantly increase investments to buy at lower prices": 5},
            q4: {"I have no investment experience": 1, "I have some experience with conservative investments": 2, 
                 "I have moderate experience with various investments": 3, "I have significant experience across different asset classes": 4, 
                 "I am a very experienced investor with advanced knowledge": 5},
            q5: {"Very unstable": 1, "Somewhat unstable": 2, "Moderately stable": 3, "Stable": 4, "Very stable": 5}
        }
        
        # Calculate total score
        total_score = scores[q1][q1] + scores[q2][q2] + scores[q3][q3] + scores[q4][q4] + scores[q5][q5]
        
        # Determine risk profile
        if total_score <= 7:
            risk_profile = "Conservative"
            description = "You prioritize protecting your capital over growth. Consider a portfolio with a large allocation to bonds and fixed income, and a small allocation to blue-chip stocks."
            allocation = {"Stocks": 20, "Bonds": 60, "Cash": 15, "Alternative Investments": 5}
        elif total_score <= 13:
            risk_profile = "Moderately Conservative"
            description = "You prefer stability but are willing to accept some risk for growth. Consider a balanced portfolio with more bonds than stocks."
            allocation = {"Stocks": 40, "Bonds": 45, "Cash": 10, "Alternative Investments": 5}
        elif total_score <= 19:
            risk_profile = "Moderate"
            description = "You seek a balance between risk and return. Consider a portfolio with roughly equal allocations to stocks and bonds."
            allocation = {"Stocks": 50, "Bonds": 35, "Cash": 5, "Alternative Investments": 10}
        elif total_score <= 23:
            risk_profile = "Moderately Aggressive"
            description = "You're comfortable with significant risk for potentially higher returns. Consider a portfolio with more stocks than bonds."
            allocation = {"Stocks": 70, "Bonds": 20, "Cash": 0, "Alternative Investments": 10}
        else:
            risk_profile = "Aggressive"
            description = "You prioritize maximum growth and can tolerate high volatility. Consider a portfolio heavily weighted toward stocks, including international and small-cap stocks."
            allocation = {"Stocks": 85, "Bonds": 5, "Cash": 0, "Alternative Investments": 10}
        
        # Display results
        st.markdown(f"### Your Risk Profile: {risk_profile}")
        st.markdown(f"**Score:** {total_score}/25")
        st.markdown(f"**Description:** {description}")
        
        # Create allocation pie chart
        fig = px.pie(
            values=list(allocation.values()),
            names=list(allocation.keys()),
            title="Suggested Asset Allocation",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template=theme)
        st.plotly_chart(fig, use_container_width=True)
        
        # Suggested ETFs based on risk profile
        st.markdown("### Suggested ETFs for Your Risk Profile")
        
        if risk_profile == "Conservative":
            etfs = {
                "Bonds": ["BND (Vanguard Total Bond Market ETF)", "AGG (iShares Core U.S. Aggregate Bond ETF)", "VCSH (Vanguard Short-Term Corporate Bond ETF)"],
                "Stocks": ["VIG (Vanguard Dividend Appreciation ETF)", "HDV (iShares Core High Dividend ETF)"],
                "Cash/Short-term": ["SHV (iShares Short Treasury Bond ETF)", "BIL (SPDR Bloomberg 1-3 Month T-Bill ETF)"],
                "Alternative": ["VCIT (Vanguard Intermediate-Term Corporate Bond ETF)"]
            }
        elif risk_profile == "Moderately Conservative":
            etfs = {
                "Bonds": ["BND (Vanguard Total Bond Market ETF)", "MUB (iShares National Muni Bond ETF)", "VCIT (Vanguard Intermediate-Term Corporate Bond ETF)"],
                "Stocks": ["VIG (Vanguard Dividend Appreciation ETF)", "VYM (Vanguard High Dividend Yield ETF)", "SPY (SPDR S&P 500 ETF)"],
                "Cash/Short-term": ["SHY (iShares 1-3 Year Treasury Bond ETF)"],
                "Alternative": ["REET (iShares Global REIT ETF)"]
            }
        elif risk_profile == "Moderate":
            etfs = {
                "Bonds": ["BND (Vanguard Total Bond Market ETF)", "VCIT (Vanguard Intermediate-Term Corporate Bond ETF)"],
                "Stocks": ["VTI (Vanguard Total Stock Market ETF)", "VOO (Vanguard S&P 500 ETF)", "VEA (Vanguard FTSE Developed Markets ETF)"],
                "Cash/Short-term": ["SHY (iShares 1-3 Year Treasury Bond ETF)"],
                "Alternative": ["VNQ (Vanguard Real Estate ETF)", "GLD (SPDR Gold Shares)"]
            }
        elif risk_profile == "Moderately Aggressive":
            etfs = {
                "Bonds": ["LQD (iShares iBoxx $ Investment Grade Corporate Bond ETF)", "VCIT (Vanguard Intermediate-Term Corporate Bond ETF)"],
                "Stocks": ["VTI (Vanguard Total Stock Market ETF)", "VO (Vanguard Mid-Cap ETF)", "VB (Vanguard Small-Cap ETF)", "VWO (Vanguard FTSE Emerging Markets ETF)", "VGT (Vanguard Information Technology ETF)"],
                "Alternative": ["VNQ (Vanguard Real Estate ETF)", "GLD (SPDR Gold Shares)", "PDBC (Invesco Optimum Yield Diversified Commodity Strategy)"]
            }
        else:  # Aggressive
            etfs = {
                "Bonds": ["HYG (iShares iBoxx $ High Yield Corporate Bond ETF)"],
                "Stocks": ["VTI (Vanguard Total Stock Market ETF)", "VB (Vanguard Small-Cap ETF)", "VWO (Vanguard FTSE Emerging Markets ETF)", "VGT (Vanguard Information Technology ETF)", "XBI (SPDR S&P Biotech ETF)"],
                "Alternative": ["VNQ (Vanguard Real Estate ETF)", "GDX (VanEck Gold Miners ETF)", "ARKK (ARK Innovation ETF)"]
            }
        
        # Display ETF recommendations
        for category, category_etfs in etfs.items():
            st.markdown(f"**{category}:**")
            for etf in category_etfs:
                st.markdown(f"- {etf}")
        
        # Additional guidance
        st.markdown("""
        ### Important Considerations
        
        - **Diversification:** Spread investments across different asset classes, sectors, and geographies to reduce risk.
        - **Rebalancing:** Periodically adjust your portfolio back to your target allocation.
        - **Cost:** Pay attention to expense ratios and fees when selecting investments.
        - **Tax Efficiency:** Consider tax implications of investment selections and account types.
        - **Time Horizon:** Your optimal asset allocation may change as you approach your financial goals.
        
        *This assessment provides general guidance. For personalized advice, consult with a financial professional.*
        """)

# Sample portfolios
sample_portfolios = {
    "S&P 500 Top 10": "AAPL,MSFT,AMZN,NVDA,GOOGL,META,BRK-B,TSLA,UNH,JPM",
    "Tech Giants": "AAPL,MSFT,AMZN,NVDA,GOOGL,META,TSLA,AMD,INTC,CSCO",
    "Dividend Champions": "JNJ,PG,KO,PEP,VZ,MMM,T,IBM,MRK,XOM",
    "Global ETFs": "SPY,QQQ,VEA,VWO,AGG,BND,GLD,VNQ,TLT,IEF"
}

# Sidebar for inputs
st.sidebar.markdown("## Configuration")

# Add app mode selection
app_mode = st.sidebar.selectbox(
    "App Mode",
    ["Portfolio Optimization", "Risk Assessment", "Educational Resources"]
)

# Date selection
st.sidebar.markdown("### Historical Data Period")
start_date = st.sidebar.date_input(
    "Start Date",
    datetime.now() - timedelta(days=365*5)  # Default to 5 years ago
)
end_date = st.sidebar.date_input(
    "End Date",
    datetime.now()  # Default to today
)

# Asset selection method
asset_selection_method = st.sidebar.radio(
    "Asset Selection Method",
    ["Manual Entry", "Upload CSV", "Sample Portfolios"]
)

# Add risk-free rate to sidebar
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1, key="sidebar_rfr") / 100

# Add theme selection
theme = st.sidebar.selectbox(
    "Chart Theme",
    ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"]
)

# Add advanced settings
with st.sidebar.expander("Advanced Settings"):
    # Optimization settings
    st.markdown("### Optimization Settings")
    optimization_method = st.selectbox(
        "Optimization Solver",
        ["SLSQP", "trust-constr", "COBYLA"],
        index=0
    )
    
    # Number of points on efficient frontier
    ef_points = st.slider(
        "Efficient Frontier Points",
        min_value=20,
        max_value=100,
        value=50,
        step=5,
        key="ef_points_slider"
    )
    
    # Monte Carlo simulation settings
    st.markdown("### Monte Carlo Settings")
    mc_samples = st.slider(
        "Number of Simulations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        key="mc_samples_slider"
    )
    
    # Visualization settings
    st.markdown("### Visualization Settings")
    show_annotations = st.checkbox("Show Chart Annotations", value=True)
    
    # Download settings
    st.markdown("### Download Settings")
    download_format = st.selectbox(
        "Download Format",
        ["CSV", "Excel"],
        index=0
    )

# Title and introduction
if app_mode == "Portfolio Optimization":
    st.markdown("<div class='main-header'>Portfolio Optimization App</div>", unsafe_allow_html=True)
    st.markdown("""
    This app helps you optimize your investment portfolio based on Modern Portfolio Theory (MPT).
    You can select stocks, analyze historical performance, and find optimal asset allocations based on your risk preferences.
    """)
elif app_mode == "Risk Assessment":
    show_risk_assessment()
elif app_mode == "Educational Resources":
    show_educational_resources()

# Main app logic based on selected mode
if app_mode == "Portfolio Optimization":
    # Initialize tickers as empty list
    tickers = []
    
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
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
            
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
                st.markdown(get_table_download_link(stock_data, "price_data", "Download Price Data", download_format), unsafe_allow_html=True)
            
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
                fig.update_layout(template=theme)
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
                fig.update_layout(template=theme)
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
                fig.update_layout(template=theme)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate cumulative returns
                cumulative_returns = (1 + returns).cumprod() - 1
                
                # Line chart for cumulative returns
                fig = px.line(
                    cumulative_returns,
                    title="Cumulative Returns",
                    labels={'value': 'Cumulative Return', 'variable': 'Asset'}
                )
                fig.update_layout(template=theme)
                st.plotly_chart(fig, use_container_width=True)
            
            # Portfolio Optimization
            st.markdown("<div class='sub-header'>Portfolio Optimization</div>", unsafe_allow_html=True)
            
            # Calculate mean returns and covariance matrix
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Optimization methods
            optimization_methods = st.multiselect(
                "Select Optimization Methods",
                ["Maximum Sharpe Ratio", "Minimum Volatility", "Efficient Frontier", "Equal Weights"],
                default=["Maximum Sharpe Ratio", "Minimum Volatility", "Equal Weights"],
                key="optimization_methods"
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
                    max_sharpe_weights = optimize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, optimization_method)
                    max_sharpe_metrics = calculate_portfolio_metrics(max_sharpe_weights, mean_returns, cov_matrix, risk_free_rate)
                    
                    portfolios["Maximum Sharpe Ratio"] = {
                        "weights": max_sharpe_weights,
                        "metrics": max_sharpe_metrics
                    }
                except Exception as e:
                    st.error(f"Error optimizing for Maximum Sharpe Ratio: {e}")
            
            if "Minimum Volatility" in optimization_methods:
                try:
                    min_vol_weights = optimize_min_volatility(mean_returns, cov_matrix, optimization_method)
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
                    
                    returns_range = np.linspace(min_return, max_return, ef_points)
                    efficient_frontier_portfolios = generate_efficient_frontier(mean_returns, cov_matrix, returns_range, optimization_method)
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
                    fig.update_layout(template=theme)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download link for weights
                    st.markdown(get_table_download_link(weights_df, f"{method.replace(' ', '_')}_weights", f"Download {method} Weights", download_format), unsafe_allow_html=True)
                    
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
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    template=theme
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
                    
                    # Update layout with the selected theme
                    fig.update_layout(
                        title="Efficient Frontier",
                        xaxis_title="Expected Volatility (%)",
                        yaxis_title="Expected Return (%)",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                        template=theme
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
                        step=0.1,
                        key="ef_slider"
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
                    fig.update_layout(template=theme)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download link for weights
                    st.markdown(get_table_download_link(weights_df, "efficient_frontier_weights", "Download Portfolio Weights", download_format), unsafe_allow_html=True)
                else:
                    st.info("Select 'Efficient Frontier' as one of the optimization methods to view this tab.")
            
            # Add a custom portfolio section
            st.markdown("<div class='sub-header'>Custom Portfolio Comparison</div>", unsafe_allow_html=True)
            
            st.markdown("""
            In this section, you can create a custom portfolio by specifying your own asset weights. 
            This allows you to compare your allocation against the optimized portfolios.
            """)
            
            # Option to create custom portfolio
            create_custom = st.checkbox("Create Custom Portfolio")
            
            if create_custom:
                st.markdown("### Enter Custom Weights")
                st.markdown("The sum of weights must equal 100%")
                
                # Initialize custom weights
                custom_weights = []
                total_weight = 0
                
                # Create columns for weights input
                cols = st.columns(min(4, len(tickers)))
                for i, ticker in enumerate(tickers):
                    col_idx = i % len(cols)
                    with cols[col_idx]:
                        weight = st.number_input(
                            f"{ticker} (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=round(100.0 / len(tickers), 1),
                            step=0.1,
                            key=f"custom_weight_{i}"
                        )
                        custom_weights.append(weight / 100)
                        total_weight += weight
                
                # Check if weights sum to 100%
                if abs(total_weight - 100) > 0.1:
                    st.warning(f"Total weight is {total_weight}%. Please adjust weights to sum to 100%.")
                else:
                    # Convert to numpy array
                    custom_weights = np.array(custom_weights)
                    
                    # Calculate metrics for custom portfolio
                    custom_metrics = calculate_portfolio_metrics(custom_weights, mean_returns, cov_matrix, risk_free_rate)
                    
                    # Add to portfolios dictionary
                    portfolios["Custom Portfolio"] = {
                        "weights": custom_weights,
                        "metrics": custom_metrics
                    }
                    
                    colors["Custom Portfolio"] = "orange"
                    
                    # Display custom portfolio metrics
                    st.markdown("### Custom Portfolio Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Expected Annual Return", f"{custom_metrics['return']*100:.2f}%")
                    with col2:
                        st.metric("Expected Annual Volatility", f"{custom_metrics['volatility']*100:.2f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{custom_metrics['sharpe_ratio']:.2f}")
                    
                    # Create a DataFrame for the weights
                    weights_df = pd.DataFrame({
                        'Asset': tickers,
                        'Weight': custom_weights * 100  # Convert to percentage
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
                        title="Custom Portfolio Allocation",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(template=theme)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Compare with optimized portfolios
                    st.markdown("### Portfolio Comparison")
                    
                    # Prepare data for comparison
                    comparison_data = []
                    for method, portfolio in portfolios.items():
                        metrics = portfolio["metrics"]
                        comparison_data.append({
                            'Portfolio': method,
                            'Expected Return (%)': metrics['return'] * 100,
                            'Volatility (%)': metrics['volatility'] * 100,
                            'Sharpe Ratio': metrics['sharpe_ratio']
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Sort by Sharpe ratio
                    comparison_df = comparison_df.sort_values('Sharpe Ratio', ascending=False)
                    
                    # Display as a table
                    st.dataframe(comparison_df.set_index('Portfolio'))
                    
                    # Create a bar chart for comparison
                    fig = go.Figure()
                    
                    for metric in ['Expected Return (%)', 'Volatility (%)', 'Sharpe Ratio']:
                        fig.add_trace(go.Bar(
                            x=comparison_df['Portfolio'],
                            y=comparison_df[metric],
                            name=metric,
                            text=comparison_df[metric].round(2),
                            textposition='auto'
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Portfolio Metrics Comparison",
                        barmode='group',
                        template=theme
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Backtesting section
            st.markdown("<div class='sub-header'>Portfolio Backtesting</div>", unsafe_allow_html=True)
            
            # Select a portfolio for backtesting
            if portfolios:
                backtest_portfolio = st.selectbox(
                    "Select Portfolio for Backtesting",
                    list(portfolios.keys()),
                    key="backtest_portfolio"
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
                
                # Add benchmark comparison with S&P 500
                try:
                    spy_data = yf.download('SPY', start=start_date, end=end_date)['Close']
                    spy_returns = spy_data.pct_change().dropna()
                    cumulative_spy_return = (1 + spy_returns).cumprod() - 1
                    
                    # Plot the results
                    performance_df = pd.DataFrame({
                        'Optimized Portfolio': cumulative_portfolio_return,
                        'Equal-Weight Benchmark': cumulative_benchmark_return,
                        'S&P 500 (SPY)': cumulative_spy_return
                    })
                except:
                    # If SPY data fails, just use the portfolio and equal weight
                    performance_df = pd.DataFrame({
                        'Optimized Portfolio': cumulative_portfolio_return,
                        'Equal-Weight Benchmark': cumulative_benchmark_return
                    })
                
                fig = px.line(
                    performance_df,
                    title="Portfolio Backtesting",
                    labels={'value': 'Cumulative Return', 'variable': 'Portfolio'}
                )
                fig.update_layout(template=theme)
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
                fig.update_layout(yaxis_tickformat=".0%", template=theme)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate benchmark drawdowns
                benchmark_rolling_max = (1 + benchmark_returns).cumprod().cummax()
                benchmark_drawdown = ((1 + benchmark_returns).cumprod() / benchmark_rolling_max) - 1
                benchmark_max_drawdown = benchmark_drawdown.min()

                # Add right after calculating benchmark_max_drawdown
                benchmark_annual_return = benchmark_returns.mean() * 252
                benchmark_annual_volatility = benchmark_returns.std() * np.sqrt(252)
                benchmark_sharpe = (benchmark_annual_return - risk_free_rate) / benchmark_annual_volatility
                
                # SPY metrics if available
                if 'S&P 500 (SPY)' in performance_df.columns:
                    spy_annual_return = spy_returns.mean() * 252
                    spy_annual_volatility = spy_returns.std() * np.sqrt(252)
                    spy_sharpe = (spy_annual_return - risk_free_rate) / spy_annual_volatility
                    
                    spy_rolling_max = (1 + spy_returns).cumprod().cummax()
                    spy_drawdown = ((1 + spy_returns).cumprod() / spy_rolling_max) - 1
                    spy_max_drawdown = spy_drawdown.min()
                    
                    # Create comparison dataframe
                    comparison_df = pd.DataFrame({
                        'Metric': ['Cumulative Return (%)', 'Annual Return (%)', 'Annual Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)'],
                        'Optimized Portfolio': [
                            f"{cumulative_portfolio_return.iloc[-1]*100:.2f}%",
                            f"{annual_return*100:.2f}%",
                            f"{annual_volatility*100:.2f}%",
                            f"{sharpe_ratio:.2f}",
                            f"{max_drawdown*100:.2f}%"
                        ],
                        'Equal-Weight Portfolio': [
                            f"{cumulative_benchmark_return.iloc[-1]*100:.2f}%",
                            f"{benchmark_annual_return*100:.2f}%",
                            f"{benchmark_annual_volatility*100:.2f}%",
                            f"{benchmark_sharpe:.2f}",
                            f"{benchmark_max_drawdown*100:.2f}%"
                        ],
                        'S&P 500': [
                            f"{cumulative_spy_return.iloc[-1]*100:.2f}%",
                            f"{spy_annual_return*100:.2f}%",
                            f"{spy_annual_volatility*100:.2f}%",
                            f"{spy_sharpe:.2f}",
                            f"{spy_max_drawdown*100:.2f}%"
                        ]
                    })
                else:
                    # Create comparison dataframe without SPY
                    comparison_df = pd.DataFrame({
                        'Metric': ['Cumulative Return (%)', 'Annual Return (%)', 'Annual Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)'],
                        'Optimized Portfolio': [
                            f"{cumulative_portfolio_return.iloc[-1]*100:.2f}%",
                            f"{annual_return*100:.2f}%",
                            f"{annual_volatility*100:.2f}%",
                            f"{sharpe_ratio:.2f}",
                            f"{max_drawdown*100:.2f}%"
                        ],
                        'Equal-Weight Portfolio': [
                            f"{cumulative_benchmark_return.iloc[-1]*100:.2f}%",
                            f"{benchmark_annual_return*100:.2f}%",
                            f"{benchmark_annual_volatility*100:.2f}%",
                            f"{benchmark_sharpe:.2f}",
                            f"{benchmark_max_drawdown*100:.2f}%"
                        ]
                    })
                
                st.dataframe(comparison_df.set_index('Metric'))
                
                # Add rolling metrics
                st.markdown("### Rolling Performance Metrics")
                
                rolling_window = st.slider("Rolling Window (Days)", 
                                          min_value=30, 
                                          max_value=252, 
                                          value=90, 
                                          step=30,
                                          key="rolling_window")
                
                # Calculate rolling returns
                rolling_returns = weighted_returns.rolling(window=rolling_window).mean() * 252
                rolling_volatility = weighted_returns.rolling(window=rolling_window).std() * np.sqrt(252)
                rolling_sharpe = rolling_returns / rolling_volatility
                
                # Calculate benchmark rolling metrics
                benchmark_rolling_returns = benchmark_returns.rolling(window=rolling_window).mean() * 252
                benchmark_rolling_volatility = benchmark_returns.rolling(window=rolling_window).std() * np.sqrt(252)
                benchmark_rolling_sharpe = benchmark_rolling_returns / benchmark_rolling_volatility
                
                # Create dataframes for rolling metrics
                rolling_metrics_df = pd.DataFrame({
                    'Portfolio Return': rolling_returns,
                    'Benchmark Return': benchmark_rolling_returns,
                    'Portfolio Volatility': rolling_volatility,
                    'Benchmark Volatility': benchmark_rolling_volatility,
                    'Portfolio Sharpe': rolling_sharpe,
                    'Benchmark Sharpe': benchmark_rolling_sharpe
                })
                
                # Create tabs for different rolling metrics
                rolling_tab1, rolling_tab2, rolling_tab3 = st.tabs(["Rolling Returns", "Rolling Volatility", "Rolling Sharpe"])
                
                with rolling_tab1:
                    fig = px.line(
                        rolling_metrics_df[['Portfolio Return', 'Benchmark Return']],
                        title=f"{rolling_window}-Day Rolling Annual Returns",
                        labels={'value': 'Annual Return', 'variable': 'Portfolio'}
                    )
                    fig.update_layout(template=theme)
                    st.plotly_chart(fig, use_container_width=True)
                    
                with rolling_tab2:
                    fig = px.line(
                        rolling_metrics_df[['Portfolio Volatility', 'Benchmark Volatility']],
                        title=f"{rolling_window}-Day Rolling Annual Volatility",
                        labels={'value': 'Annual Volatility', 'variable': 'Portfolio'}
                    )
                    fig.update_layout(template=theme)
                    st.plotly_chart(fig, use_container_width=True)
                    
                with rolling_tab3:
                    fig = px.line(
                        rolling_metrics_df[['Portfolio Sharpe', 'Benchmark Sharpe']],
                        title=f"{rolling_window}-Day Rolling Sharpe Ratio",
                        labels={'value': 'Sharpe Ratio', 'variable': 'Portfolio'}
                    )
                    fig.update_layout(template=theme)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add section for risk analysis
                st.markdown("<div class='sub-header'>Risk Analysis</div>", unsafe_allow_html=True)
                
                # Calculate value at risk (VaR)
                confidence_level = st.slider("Confidence Level (%)", 
                                           min_value=90, 
                                           max_value=99, 
                                           value=95, 
                                           step=1,
                                           key="confidence_level")
                
                # Calculate historical VaR
                var_percentile = 100 - confidence_level
                historical_var = np.percentile(weighted_returns, var_percentile) * 100
                
                # Calculate conditional VaR (Expected Shortfall)
                cvar = weighted_returns[weighted_returns <= np.percentile(weighted_returns, var_percentile)].mean() * 100
                
                # Display VaR metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"Value at Risk (VaR) - {confidence_level}%", f"{historical_var:.2f}%")
                    st.markdown(f"There is a {100-confidence_level}% chance that the portfolio will lose more than {abs(historical_var):.2f}% in a single day.")
                with col2:
                    st.metric(f"Conditional VaR (Expected Shortfall) - {confidence_level}%", f"{cvar:.2f}%")
                    st.markdown(f"If the {100-confidence_level}% worst case scenario occurs, the expected loss is {abs(cvar):.2f}% in a single day.")
                
                # Display return distribution
                fig = px.histogram(
                    weighted_returns * 100,
                    nbins=50,
                    title="Daily Returns Distribution",
                    labels={'value': 'Daily Return (%)', 'count': 'Frequency'},
                    marginal="box"
                )
                
                # Add VaR line
                fig.add_vline(x=historical_var, line_dash="dash", line_color="red",
                              annotation_text=f"{confidence_level}% VaR: {historical_var:.2f}%",
                              annotation_position="top right")
                
                # Add CVaR line
                fig.add_vline(x=cvar, line_dash="dash", line_color="darkred",
                              annotation_text=f"{confidence_level}% CVaR: {cvar:.2f}%",
                              annotation_position="bottom right")
                
                fig.update_layout(template=theme)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to download stock data. Please check the ticker symbols and try again.")

    # Add Monte Carlo simulation
    if 'stock_data' in locals() and stock_data is not None and not stock_data.empty and 'portfolios' in locals() and portfolios:
        st.markdown("<div class='sub-header'>Monte Carlo Simulation</div>", unsafe_allow_html=True)
        
        st.markdown("""
        This section simulates possible future portfolio values using Monte Carlo methods.
        It generates random scenarios based on historical returns and volatility.
        """)
        
        # Portfolio selection for simulation
        sim_portfolio = st.selectbox(
            "Select Portfolio for Simulation",
            list(portfolios.keys()),
            key="simulation_portfolio"
        )
        
        # Simulation parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            sim_years = st.number_input("Simulation Years", min_value=1, max_value=30, value=5, step=1)
        with col2:
            initial_investment = st.number_input("Initial Investment ($)", min_value=1000, max_value=10000000, value=10000, step=1000)
        with col3:
            num_simulations = st.number_input("Number of Simulations", min_value=100, max_value=10000, value=mc_samples, step=100)
        
        if st.button("Run Monte Carlo Simulation"):
            with st.spinner("Running simulation..."):
                # Get the selected portfolio weights
                selected_weights = portfolios[sim_portfolio]["weights"]
                
                # Calculate portfolio expected return and volatility
                portfolio_return = np.sum(mean_returns * selected_weights) * 252
                portfolio_volatility = np.sqrt(np.dot(selected_weights.T, np.dot(cov_matrix, selected_weights))) * np.sqrt(252)
                
                # Simulation parameters
                trading_days = 252 * sim_years
                
                # Initialize array for simulation results
                simulation_results = np.zeros((trading_days, num_simulations))
                
                # Initial investment
                simulation_results[0] = initial_investment
                
                # Generate random daily returns
                np.random.seed(42)  # For reproducibility
                
                # Run simulations
                for i in range(num_simulations):
                    daily_returns = np.random.normal(
                        portfolio_return / 252,
                        portfolio_volatility / np.sqrt(252),
                        trading_days
                    )
                    
                    # Calculate cumulative returns
                    for t in range(1, trading_days):
                        simulation_results[t, i] = simulation_results[t-1, i] * (1 + daily_returns[t])
                
                # Convert to DataFrame for easier analysis
                sim_df = pd.DataFrame(simulation_results)
                
                # Plot results
                fig = go.Figure()
                
                # Add a sample of simulations
                sample_size = min(100, num_simulations)
                for i in range(sample_size):
                    fig.add_trace(go.Scatter(
                        y=sim_df.iloc[:, i],
                        mode='lines',
                        opacity=0.3,
                        line=dict(width=1),
                        showlegend=False
                    ))
                
                # Add mean, min, max
                fig.add_trace(go.Scatter(
                    y=sim_df.mean(axis=1),
                    mode='lines',
                    name='Mean',
                    line=dict(color='blue', width=2)
                ))
                
                # Add percentiles
                fig.add_trace(go.Scatter(
                    y=np.percentile(sim_df, 5, axis=1),
                    mode='lines',
                    name='5th Percentile',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    y=np.percentile(sim_df, 95, axis=1),
                    mode='lines',
                    name='95th Percentile',
                    line=dict(color='green', width=2, dash='dash')
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"Monte Carlo Simulation ({num_simulations} runs, {sim_years} years)",
                    xaxis_title="Trading Days",
                    yaxis_title="Portfolio Value ($)",
                    showlegend=True,
                    template=theme
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Final value statistics
                final_values = sim_df.iloc[-1]
                
                st.markdown("### Simulation Results")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Final Value", f"${final_values.mean():.2f}")
                with col2:
                    st.metric("Median Final Value", f"${final_values.median():.2f}")
                with col3:
                    st.metric("5th Percentile", f"${np.percentile(final_values, 5):.2f}")
                with col4:
                    st.metric("95th Percentile", f"${np.percentile(final_values, 95):.2f}")
                
                # Distribution of final values
                fig = px.histogram(
                    final_values,
                    nbins=50,
                    title="Distribution of Final Portfolio Values",
                    labels={'value': 'Portfolio Value ($)', 'count': 'Frequency'}
                )
                
                # Add percentile lines
                for percentile, color, label in [(5, 'red', '5th'), (50, 'green', 'Median'), (95, 'blue', '95th')]:
                    value = np.percentile(final_values, percentile)
                    fig.add_vline(x=value, line_dash="dash", line_color=color,
                                  annotation_text=f"{label}: ${value:.2f}",
                                  annotation_position="top right")
                
                fig.update_layout(template=theme)
                st.plotly_chart(fig, use_container_width=True)

# Footer with disclaimer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='disclaimer'>Disclaimer: This app is for educational purposes only. Past performance is not indicative of future results. Always conduct your own research before making investment decisions.</p>", unsafe_allow_html=True)