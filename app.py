"""Interactive educational portfolio risk dashboard."""
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from analytics import daily_returns, portfolio_series, metrics, growth_index

st.set_page_config(page_title='Portfolio Risk Analyzer', page_icon='📊', layout='wide')
st.title('📊 Portfolio Risk Analyzer')
st.caption('Explore historical performance, diversification, risk and drawdowns. Educational use only.')

with st.sidebar:
    st.header('Portfolio settings')
    tickers_raw = st.text_input('Stock tickers (comma-separated)', 'AAPL, MSFT, GOOGL')
    weights_raw = st.text_input('Weights (comma-separated)', '40, 35, 25')
    benchmark = st.text_input('Benchmark ticker', 'SPY').strip().upper()
    start = st.date_input('Start date', date.today() - timedelta(days=365 * 3))
    end = st.date_input('End date (exclusive)', date.today())
    rf = st.number_input('Annual risk-free rate (%)', -20.0, 30.0, 4.0, 0.5)
    rebalance = st.selectbox('Portfolio assumption', ['daily', 'buy-and-hold'], index=1)
    st.caption('Example Indian tickers: RELIANCE.NS, TCS.NS, INFY.NS; benchmark ^NSEI.')

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(symbols, start_date, end_date):
    data = yf.download(list(symbols), start=str(start_date), end=str(end_date),
                       auto_adjust=True, progress=False, threads=False)
    if data.empty:
        raise ValueError('No market data returned. Check tickers, dates and internet access.')
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' not in data.columns.get_level_values(0):
            raise ValueError('Close prices are unavailable.')
        close = data['Close']
    else:
        close = data[['Close']].rename(columns={'Close': symbols[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbols[0])
    return close.reindex(columns=list(symbols)).sort_index()

try:
    symbols = [s.strip().upper() for s in tickers_raw.split(',') if s.strip()]
    weights = np.asarray([float(s.strip()) for s in weights_raw.split(',')], dtype=float)
    if not symbols or len(symbols) != len(set(symbols)) or len(symbols) > 10:
        raise ValueError('Enter 1–10 unique stock tickers.')
    if len(weights) != len(symbols) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Enter one nonnegative weight per ticker, with a positive total.')
    if start >= end:
        raise ValueError('Start date must be earlier than end date.')
    if benchmark in symbols:
        raise ValueError('Choose a benchmark ticker different from the portfolio holdings.')
    with st.spinner('Fetching market prices...'):
        all_prices = load_prices(tuple(symbols + [benchmark]), start, end)
    missing = [s for s in symbols + [benchmark] if all_prices[s].notna().sum() < 3]
    if missing:
        raise ValueError('Insufficient historical data for: ' + ', '.join(missing))
    # Shared dates prevent mismatched observation windows between portfolio and benchmark.
    prices = all_prices.dropna(how='any')
    returns = daily_returns(prices)
    portfolio = portfolio_series(returns[symbols], weights, rebalance)
    benchmark_returns = returns[benchmark].rename('Benchmark')
    stats = metrics(portfolio, rf)
    bench_stats = metrics(benchmark_returns, rf)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.caption(f'{len(returns)} aligned trading-day returns | Weights normalized to 100% | Adjusted closing prices')
cols = st.columns(5)
for col, label, value in zip(cols, ['Annualized return', 'Annualized volatility', 'Sharpe ratio', 'Maximum drawdown', 'Total return'],
                              [f"{stats['annualized_return']:.1%}", f"{stats['annualized_volatility']:.1%}",
                               f"{stats['sharpe_ratio']:.2f}" if np.isfinite(stats['sharpe_ratio']) else 'N/A',
                               f"{stats['max_drawdown']:.1%}", f"{stats['total_return']:.1%}"]):
    col.metric(label, value)

series = pd.DataFrame({'Portfolio': growth_index(portfolio), 'Benchmark': growth_index(benchmark_returns)})
fig = px.line(series, title='Growth of a hypothetical ₹10,000 / $10,000 (same currency as prices)',
              labels={'value': 'Indexed value', 'index': 'Date', 'variable': 'Series'})
st.plotly_chart(fig, use_container_width=True)
left, right = st.columns(2)
with left:
    corr = returns[symbols].corr()
    heat = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale='RdBu_r',
                     title='Daily return correlation matrix', text_auto='.2f')
    st.plotly_chart(heat, use_container_width=True)
with right:
    wealth = (1 + portfolio).cumprod()
    running_peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    drawdown = wealth.to_numpy() / running_peak - 1
    fig2 = go.Figure(go.Scatter(x=portfolio.index, y=drawdown, fill='tozeroy', name='Drawdown'))
    fig2.update_layout(title='Portfolio drawdown', yaxis_tickformat='.0%', xaxis_title='Date')
    st.plotly_chart(fig2, use_container_width=True)

st.subheader('Portfolio weights')
st.dataframe(pd.DataFrame({'Ticker': symbols, 'Weight (%)': (weights / weights.sum() * 100).round(2)}), hide_index=True)
st.subheader('Performance comparison')
comparison = pd.DataFrame([stats, bench_stats], index=['Portfolio', benchmark])
st.dataframe(comparison.style.format({'total_return': '{:.2%}', 'annualized_return': '{:.2%}',
                                     'annualized_volatility': '{:.2%}', 'sharpe_ratio': '{:.2f}',
                                     'max_drawdown': '{:.2%}'}))
output = pd.DataFrame({'Portfolio daily return': portfolio, 'Benchmark daily return': benchmark_returns,
                       'Portfolio indexed value': series['Portfolio'], 'Benchmark indexed value': series['Benchmark']})
st.download_button('Download daily analysis (CSV)', output.to_csv().encode(), 'portfolio_analysis.csv', 'text/csv')
st.warning('Historical returns do not predict future returns. Prices are adjusted for corporate actions where available. '
           'The model ignores fees, taxes, slippage, and cash flows; daily rebalancing assumes costless trades. '
           'Annualized returns are geometric (252 trading days/year); Sharpe uses average daily excess returns. '
           'Compare assets and benchmark denominated in the same currency.')
