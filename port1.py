# portfolio_app.py (final version with Sector Allocation and all fixes)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(layout="wide")
st.title("📊 Portfolio Analysis")

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
    st.header("📈 Asset Analysis")

    # st.subheader("📍 Benchmark Comparison")

    # benchmark_options = {
    #     "^GSPC": "S&P 500",
    #     "^DJI": "Dow Jones",
    #     "^IXIC": "NASDAQ",
    #     "^RUT": "Russell 2000",
    #     "^FTSE": "FTSE 100",
    #     "^N225": "Nikkei 225"
    # }

    # selected_benchmark = st.selectbox("Choose a Benchmark", options=list(benchmark_options.keys()), format_func=lambda x: benchmark_options[x])


    tickers = [t.strip().upper() for t in st.text_input("Enter ticker symbols", "AAPL, MSFT, GOOGL").split(",")]
    start = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end = st.date_input("End Date", pd.to_datetime("2024-12-31"))
    prices = get_price_data(tickers, start, end)

    if not prices.empty:

        st.subheader("ℹ️ Asset Information")

        asset_info = []

        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                asset_info.append({
                    "Ticker": ticker,
                    "Sector": info.get("sector", "N/A"),
                    "Industry": info.get("industry", "N/A"),
                    "Market Cap": f"${info.get('marketCap', 0) / 1e9:.2f} B" if info.get('marketCap') else "N/A",
                    "Country": info.get("country", "N/A")
                })
            except:
                asset_info.append({
                    "Ticker": ticker,
                    "Sector": "N/A",
                    "Industry": "N/A",
                    "Market Cap": "N/A",
                    "Country": "N/A"
                })

        df_info = pd.DataFrame(asset_info)
        df_info.set_index("Ticker", inplace=True)
        st.dataframe(df_info)


        st.subheader("Price Chart")
        st.line_chart(prices)


        # benchmark_data = yf.download(selected_benchmark, start=start, end=end, progress=False)["Close"]
        # benchmark_data.name = selected_benchmark

        # combined_prices = prices.copy()
        # combined_prices[selected_benchmark] = benchmark_data

        # st.subheader("Price Chart (with Benchmark)")
        # st.line_chart(combined_prices)


        returns = compute_returns(prices)
        st.subheader("Returns")
        st.line_chart(returns)

        st.subheader("Basic Statistics")
        stats = pd.DataFrame({
            "Mean Return": returns.mean(),
            "Volatility": returns.std(),
            "Sharpe Ratio": returns.mean() / returns.std()
        })
        # st.dataframe(stats.style.format("{:.4f}"))

        # Reset the index and prepare the data
        stats.index.name = 'Ticker'  # Set the index name before reset
        stats_reset = stats.reset_index()

        # Melt the DataFrame into long format for plotting
        stats_melt = stats_reset.melt(id_vars="Ticker", var_name="Metric", value_name="Value")

        # Plotly bar chart
        fig = px.bar(
            stats_melt,
            x="Ticker",
            y="Value",
            color="Metric",
            barmode="group",
            title="Mean Return, Volatility, and Sharpe Ratio by Asset"
        )
        fig.update_layout(xaxis_title="Ticker", yaxis_title="Value", legend_title="Metric")

        st.plotly_chart(fig, use_container_width=True)

        # # Optional raw table toggle
        # if st.toggle("📋 Show raw statistics table"):
        #     st.dataframe(stats.style.format("{:.4f}"))

        

        st.subheader("🏢 Sector Allocation (Equal Weighted)")

        equal_weights = [1 / len(tickers)] * len(tickers)
        sectors = {}

        for i, ticker in enumerate(tickers):
            try:
                sector = yf.Ticker(ticker).info.get("sector", "Unknown")
                sectors[sector] = sectors.get(sector, 0) + equal_weights[i]
            except:
                sectors["Unknown"] = sectors.get("Unknown", 0) + equal_weights[i]

        # Normalize
        total = sum(sectors.values())
        sector_weights = {k: v / total for k, v in sectors.items()}
        sector_df = pd.DataFrame(list(sector_weights.items()), columns=["Sector", "Weight"])

        fig = px.pie(sector_df, values="Weight", names="Sector", title="Sector Allocation", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)


        st.subheader("📊 Asset Correlation")
        corr_matrix = returns.corr().round(2)

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu",
            title="Correlation Matrix of Returns"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No data available for selection.")

# -------------------- 2. Portfolio Comparison --------------------
with tabs[1]:
    st.header("🔄 Portfolio Comparison")
    tickers = [t.strip().upper() for t in st.text_input("Tickers (comma-separated)", "AAPL, MSFT, GOOGL").split(",")]
    start = st.date_input("Start Date", pd.to_datetime("2022-01-01"), key="pc_start")
    end = st.date_input("End Date", pd.to_datetime("2024-12-31"), key="pc_end")
    default_weights = [round(1 / len(tickers), 2)] * len(tickers)
    weights = list(map(float, st.text_input("Weights", ",".join(map(str, default_weights))).split(",")))

    if len(weights) != len(tickers) or not np.isclose(sum(weights), 1.0):
        st.warning("Ensure weights match number of tickers and sum to 1.")
    else:
        data = get_price_data(tickers, start, end)
        if not data.empty:
            returns, cumulative = compute_portfolio_returns(data, weights)

            st.subheader("Portfolio vs Individual Assets")
            all_cum = pd.DataFrame({"Portfolio": cumulative})
            for ticker in data.columns:
                all_cum[ticker] = (1 + data[ticker].pct_change(fill_method=None).dropna()).cumprod()
            st.line_chart(all_cum)

            st.subheader("Final Cumulative Returns")
            final_returns = all_cum.iloc[-1].to_frame(name="Final Return")
            st.bar_chart(final_returns)

            st.subheader("Stats")
            mean, vol, sharpe = portfolio_stats(returns)
            st.write(f"Return: {mean:.4f}, \nVolatility: {vol:.4f}, \nSharpe Ratio: {sharpe:.4f}")
        else:
            st.warning("No data returned for these tickers.")

# -------------------- 3. Mean Risk --------------------
with tabs[2]:
    st.header("📉 Mean-Variance Optimization")
    tickers = [t.strip().upper() for t in st.text_input("Enter ticker symbols", "AAPL, MSFT, GOOGL", key="mv_t").split(",")]
    start = st.date_input("Start Date", pd.to_datetime("2022-01-01"), key="mv_s")
    end = st.date_input("End Date", pd.to_datetime("2024-12-31"), key="mv_e")
    allow_short = st.checkbox("Allow Short Selling", value=False)
    num_portfolios = st.slider("Number of Portfolios", 100, 3000, 1000)
    alpha = st.slider("CVaR/VaR Confidence", 0.90, 0.99, 0.95)

    price_data = get_price_data(tickers, start, end)
    returns = compute_returns(price_data)
    if returns.empty:
        st.error("No returns could be calculated from the available price data.")
        st.stop()

    valid_tickers = list(returns.columns)
    if set(tickers) != set(valid_tickers):
        st.warning(f"Dropped due to missing data: {set(tickers) - set(valid_tickers)}")
    tickers = valid_tickers

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
    sc = ax.scatter(df["Volatility"], df["Return"], c=df["Sharpe"], cmap="viridis")
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
    st.header("🏗️ Risk Builder")

    if "saved_portfolios" not in st.session_state:
        st.session_state["saved_portfolios"] = {}

    tickers = [t.strip().upper() for t in st.text_input("Enter ticker symbols", "AAPL, MSFT, GOOGL", key="rb_t").split(",")]
    start = st.date_input("Start Date", pd.to_datetime("2022-01-01"), key="rb_s")
    end = st.date_input("End Date", pd.to_datetime("2024-12-31"), key="rb_e")
    alpha = st.slider("CVaR/VaR Confidence Level", 0.90, 0.99, 0.95, key="rb_alpha")

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

    st.subheader("🏢 Sector Allocation")
    sectors = {}
    for i, ticker in enumerate(tickers):
        try:
            sector = yf.Ticker(ticker).info.get("sector", "Unknown")
            sectors[sector] = sectors.get(sector, 0) + weights[i]
        except:
            sectors["Unknown"] = sectors.get("Unknown", 0) + weights[i]

    total_weight = sum(sectors.values())
    sector_weights = {k: v / total_weight for k, v in sectors.items()}
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sector_weights.values(), labels=sector_weights.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

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
