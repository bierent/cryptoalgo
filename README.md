Code is still being developed, this is verison 0.1. This is NO trading bot. I build this code to help me see indicators fast via Telegram to help me make better decisions.
Also to give me alerts when certain levels are being crossed. For the future I want to build in more alerts and use some machine learning to make it better.

The main.py can be used and has to put in the same folder as the config.ini, you do have to add the api key from cryptocompare(free) and telegram(just create free bot). After that insert your user id and you can run it.


A Python-based market analysis tool with automated Telegram bot integration. This project provides advanced technical analysis and charting capabilities for tickers, including:

- Real-time price and volume tracking using APIs like CryptoCompare and Yahoo Finance.
- Technical indicators such as RSI, MACD, Bollinger Bands, Stochastic Oscillator, and more.
- Custom indicators like Pi Cycle, Hash Ribbons, Fibonacci Bollinger Bands, and Supertrend.
- Automated chart generation for candlestick charts, moving averages, and other indicators.
- Telegram bot integration for real-time alerts, reports, and user interaction.
- Periodic updates with configurable alerts for price thresholds, divergences, and indicator signals.
Key features:

- Built with Python libraries: pandas, matplotlib, mplfinance, talib, and yfinance.
- Asynchronous Telegram bot using telebot for real-time notifications.
- Modular design with classes for market data and bot functionality.
- Configurable via config.ini for API keys, user IDs, and symbols.
