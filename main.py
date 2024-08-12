##webinterface diango

import types
import requests
import pandas as pd
#import telebot
import pandas_ta as ta
import pandas_ta as tk
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mplfinance as mpf
import io
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf
from mplfinance.original_flavor import candlestick_ohlc
from pandas.tseries.offsets import DateOffset
from scipy.signal import argrelextrema
import talib
import asyncio
from telebot.async_telebot import AsyncTeleBot

import os
import configparser
#import logging
import ast
from types import SimpleNamespace
import ta


class MarketData:
    
    def __init__(self, weektype='day', symbol='BTC', limit=100, period='1y', aggregate=1, config_file = 'config.ini'):
        self.weektype = weektype
        self.symbol = symbol
        self.limit = limit
        self.aggregate = aggregate
        self.period = period
        self.api_key, self.token, self.ids, self.symbols = self.load_config(config_file)
        master_user = self.ids[0]

    def load_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), config_file))
        api_key = config.get('API', 'key')
        telegram_token = config.get('Credentials', 'telegram_token')
        identities = config.get('USER_ID', 'user_id')
        list_id = identities.strip("[]").replace("'", "").split(", ")
        symbols = config.get('Tokens', 'symbols').strip("[]").replace("'", "").split(", ")
        return api_key, telegram_token, list_id, symbols
    
    def get_current_price(self, symbol='BTC'):
        try:
            url = f'https://min-api.cryptocompare.com/data/price?fsym={symbol}&tsyms=USD'
            headers = {'authorization': f'Apikey {self.api_key}'}
            response = requests.get(url, headers=headers)
            data = response.json()
            return data.get('USD')
        except Exception as e:
            print("Error fetching data:", e)
            return None

    def get_current_volume(self, symbol='BTC'):
        url = f'https://min-api.cryptocompare.com/data/top/exchanges/full?fsym={symbol}&tsym=USD'
        headers = {'authorization': f'Apikey {self.api_key}'}
        response = requests.get(url, headers=headers)
        data = response.json()
    
        try:
            if data.get('Response') == 'Success':
                volume = data['Data']['AggregatedData']['TOTALVOLUME24H']
                return volume
            else:
                return None
        except KeyError:
            return "No volume data found"

    def get_historical_data(self, weektype=None, aggregate=None):
        if weektype is None:
            weektype = self.weektype
        if aggregate is None:
            aggregate = self.aggregate
        url = f'https://min-api.cryptocompare.com/data/v2/histo{weektype}?fsym={self.symbol}&tsym=USD&limit={self.limit}&aggregate={aggregate}'
        headers = {'authorization': f'Apikey {self.api_key}'}
        response = requests.get(url, headers=headers)
        data = response.json()
        if data['Response'] == 'Success':
            df = pd.DataFrame(data['Data']['Data'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.drop(columns=['conversionType', 'conversionSymbol'], inplace=True)
            return df
        else:
            print("Error fetching data:", data['Message'])
            return None

    def generate_candlestick_chart(self, weektype='hour', symbol='BTC'):
        self.weektype = weektype
        self.symbol = symbol

        df = self.get_historical_data(weektype=weektype, aggregate=1)
        if df is not None:
            df = df[['open', 'high', 'low', 'close', 'volumefrom']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            file_path = f'{self.symbol}_{weektype}_candlestick_chart.png'
            mpf.plot(df, type='candle', style='charles', volume=True, title=f'{self.symbol} {weektype} Candlestick Chart', ylabel='Price (USD)', ylabel_lower='Volume', savefig=file_path)
            return file_path
        return None
    
    def calculate_bollinger_bands(self, df, window=34, multiplier=2.0):
        df['basis'] = df['close'].rolling(window=window).mean()
        df['std_dev'] = df['close'].rolling(window=window).std()
        df['upper_band1'] = df['basis'] + df['std_dev']
        df['lower_band1'] = df['basis'] - df['std_dev']
        df['upper_band2'] = df['basis'] + multiplier * df['std_dev']
        df['lower_band2'] = df['basis'] - multiplier * df['std_dev']
        return df

    def generate_bollinger_bands_chart(self, weektype='hour', symbol='BTC'):
        self.weektype = weektype
        self.symbol = symbol
        self.aggregate = 4 if weektype == 'hour' else 1

        df = self.get_historical_data(weektype=weektype, aggregate=self.aggregate)
        if df is not None:
            df = self.calculate_bollinger_bands(df)

            file_path = f'{self.symbol}_{weektype}_bollinger_bands_chart.png'
            title = f'{symbol} Bollinger Bands ({weektype.capitalize()})'

            fig, ax = mpf.plot(df, type='candle', style='charles', title=title, ylabel='Price',
                               mav=(34),
                               addplot=[mpf.make_addplot(df[['upper_band1', 'lower_band1']], color='blue'),
                                        mpf.make_addplot(df[['upper_band2', 'lower_band2']], color='orange')],
                               figratio=(12, 8), volume=False, returnfig=True)

            buffer = io.BytesIO()  
            fig.savefig(buffer, format='png')  
            buffer.seek(0)  

            plt.close(fig)  
            return buffer
        return None
    
    def daily_report(self, symbol: str):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=120)  # 4 months

        stock_data = yf.download(symbol, start=start_date, end=end_date, interval='1d')

        stock_data.ta.macd(append=True)
        stock_data.ta.rsi(append=True)
        stock_data.ta.bbands(append=True)
        stock_data.ta.obv(append=True)
        stock_data.ta.sma(length=20, append=True)
        stock_data.ta.ema(length=50, append=True)
        stock_data.ta.stoch(append=True)
        stock_data.ta.adx(append=True)
        stock_data.ta.willr(append=True)
        stock_data.ta.cmf(append=True)
        stock_data.ta.psar(append=True)

        stock_data['OBV_in_million'] = stock_data['OBV'] / 1e7
        stock_data['MACD_histogram_12_26_9'] = stock_data['MACDh_12_26_9']

        last_day_summary = stock_data.iloc[-1][[
            'Adj Close', 'MACD_12_26_9', 'MACD_histogram_12_26_9', 'RSI_14', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0',
            'SMA_20', 'EMA_50', 'OBV_in_million', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ADX_14', 'WILLR_14', 'CMF_20',
            'PSARl_0.02_0.2', 'PSARs_0.02_0.2'
        ]]

        plt.figure(figsize=(14, 8))

    # Price trends
        plt.subplot(3, 3, 1)
        plt.plot(stock_data.index, stock_data['Adj Close'], label='Adj Close', color='blue')
        plt.plot(stock_data.index, stock_data['EMA_50'], label='EMA_50', color='green')
        plt.plot(stock_data['SMA_20'], label='SMA_20', color='orange')
        plt.title('Price Trend')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45, fontsize=8)
        plt.legend()

    # On-Balance-Volume
        plt.subplot(3, 3, 2)
        plt.plot(stock_data['OBV'], label='On-Balance Volume')
        plt.title('On-Balance Volume (OBV) Indicator')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45, fontsize=8)
        plt.legend()

    # MACD plot
        plt.subplot(3, 3, 3)
        plt.plot(stock_data['MACD_12_26_9'], label='MACD')
        plt.plot(stock_data['MACDh_12_26_9'], label='MACD Histogram')
        plt.title('MACD Indicator')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45, fontsize=8)
        plt.legend()

    # RSI Plot
        plt.subplot(3, 3, 4)
        plt.plot(stock_data['RSI_14'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
        plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45, fontsize=8)
        plt.title('RSI Indicator')

    # Bollinger Bands Plot
        plt.subplot(3, 3, 5)
        plt.plot(stock_data.index, stock_data['BBU_5_2.0'], label='Upper BB')
        plt.plot(stock_data.index, stock_data['BBM_5_2.0'], label='Middle BB')
        plt.plot(stock_data.index, stock_data['BBL_5_2.0'], label='Lower BB')
        plt.plot(stock_data.index, stock_data['Adj Close'], label='Adj Close', color='brown')
        plt.title("Bollinger Bands")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45, fontsize=8)
        plt.legend()

    # Stochastic Oscillator Plot
        plt.subplot(3, 3, 6)
        plt.plot(stock_data.index, stock_data['STOCHk_14_3_3'], label='Stoch %K')
        plt.plot(stock_data.index, stock_data['STOCHd_14_3_3'], label='Stoch %D')
        plt.title("Stochastic Oscillator")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45, fontsize=8)
        plt.legend()

    # Williams %R Plot
        plt.subplot(3, 3, 7)
        plt.plot(stock_data.index, stock_data['WILLR_14'])
        plt.title("Williams %R")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45, fontsize=8)

    # ADX Plot
        plt.subplot(3, 3, 8)
        plt.plot(stock_data.index, stock_data['ADX_14'])
        plt.title("Average Directional Index (ADX)")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45, fontsize=8)

    # CMF Plot
        plt.subplot(3, 3, 9)
        plt.plot(stock_data.index, stock_data['CMF_20'])
        plt.title("Chaikin Money Flow (CMF)")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45, fontsize=8)
        plt.tight_layout()
        #plt.show()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()

        photo_data = buffer.getvalue()
        last_day_summary = last_day_summary.round()
        summary_json = last_day_summary.to_json(orient='index', indent=4)

        return photo_data, summary_json

    def pi_cycle_plot(self, symbol:str):
        #symbol = self.symbol
        limit = 2000
        aggregate = 24
        api_key = self.api_key
        
        url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={limit}'
        headers = {'authorization': f'Apikey {api_key}'}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        except requests.RequestException as e:
            print(f"Error fetching data: {e}")
            return
        
        data = response.json()
        if data['Response'] != 'Success':
            print(f"Error processing data: {data['Message']}")
            return
        
        df = pd.DataFrame(data['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['close'], label=f'{symbol} Price')

        df['ma_111'] = df['close'].rolling(window=111).mean()
        df['ma_350x2'] = df['close'].rolling(window=350).mean() * 2

        plt.plot(df['ma_111'], label='MA 111')
        plt.plot(df['ma_350x2'], label='MA 350x2')

        crossover_points = df[(df['ma_111'] > df['ma_350x2']) & (df['ma_111'].shift(1) < df['ma_350x2'].shift(1))].index
        for point in crossover_points:
            plt.vlines(point, ymin=df['close'].min(), ymax=df['close'].max(), linestyles='dashed', colors='r', label='Crossover')
            plt.text(point, df['close'][point], f"${df['close'][point]:.0f}", ha='center', va='bottom')

        plt.legend(loc='upper left')
        symbol = symbol.upper()
        plt.title(f'{symbol} Pi Cycle')
        #plt.show()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        photo_data = buffer.getvalue()

        ma_111_value = df['ma_111'].iloc[-1]
        ma_350x2_value = df['ma_350x2'].iloc[-1]
        abs_difference = abs(ma_111_value - ma_350x2_value)
        
        #print(f"Absolute difference between 350x2ma and 111ma at the latest data point: {abs_difference:.2f}")
        return photo_data, abs_difference
    
    def two_yma(self, symbol:str):
        #symbol = self.symbol
        limit = 2000
        api_key = self.api_key

        url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={limit}'
        headers = {'authorization': f'Apikey {api_key}'}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        except requests.RequestException as e:
            print(f"Error fetching data: {e}")
            return None, None

        data = response.json()
        if data['Response'] != 'Success':
            print(f"Error processing data: {data['Message']}")
            return None, None

        df = pd.DataFrame(data['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        window_size = 2 * 365
        df['2YMA'] = df['close'].rolling(window=window_size, min_periods=1).mean()
        df['5x2YMA'] = df['2YMA'] * 5

        latest_date = df.index[-1]
        latest_price = df['close'].iloc[-1]
        latest_5x2YMA = df['5x2YMA'].iloc[-1]

        difference = latest_5x2YMA - latest_price
        symbol = symbol.upper()

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Close Price', color='blue')
        plt.plot(df.index, df['2YMA'], label='2YMA', color='green')
        plt.plot(df.index, df['5x2YMA'], label='5x2YMA', color='red')
        plt.title(f'{symbol.upper()} Closing Price and 2-Year Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        #plt.show()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        photo_data = buffer.getvalue()

        return photo_data, difference

    def twohundredweek_ma(self, symbol:str, weektype: str = 'week'):
        #symbol = self.symbol
        limit = 2000
        api_key = self.api_key
        weektype = 'day'

        url = f'https://min-api.cryptocompare.com/data/v2/histo{weektype}?fsym={symbol}&tsym=USD&limit={limit}'
        headers = {'authorization': f'Apikey {api_key}'}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        except requests.RequestException as e:
            print(f"Error fetching data: {e}")
            return None

        data = response.json()
        if data['Response'] != 'Success':
            print(f"Error processing data: {data['Message']}")
            return None

        nested_df = pd.DataFrame(data['Data'])
        df = nested_df['Data'].apply(pd.Series)
        if 'time' not in df.columns:
            print("The 'time' column is missing from the data.")
            return None

        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        df['200wma'] = df['close'].rolling(window=1400, min_periods=1).mean()  # 200 weeks * 7 days = 1400 days
        df_weekly = df['close'].resample('W').mean()
        start_date = df.index[0] + pd.DateOffset(years=1)
        df_filtered = df[df.index >= start_date]
        df_weekly_filtered = df_weekly[df_weekly.index >= start_date]
        symbol=symbol.upper()

        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax1.plot(df_filtered.index, df_filtered['close'], label=f'{symbol} Price', color='blue')
        ax1.plot(df_filtered.index, df_filtered['200wma'], label='200WMA', color='red')
        ax1.set_title(f'{symbol.upper()} Price, 200-Week Moving Average, and Weekly Price Heatmap')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD)')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        scatter = ax1.scatter(df_weekly_filtered.index, df_weekly_filtered, c=df_weekly_filtered, cmap='plasma', edgecolor='none', alpha=0.6)
        plt.colorbar(scatter, ax=ax1, label='Weekly Close Price')
        #plt.show()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        photo_data = buffer.getvalue()

        return photo_data

    def twohundredday_ma(self, symbol: str):
        limit = 2000
        api_key = self.api_key
        weektype = 'day'

        url = f'https://min-api.cryptocompare.com/data/v2/histo{weektype}?fsym={symbol}&tsym=USD&limit={limit}'
        headers = {'authorization': f'Apikey {api_key}'}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        except requests.RequestException as e:
            print(f"Error fetching data: {e}")
            return None

        data = response.json()
        if data['Response'] != 'Success':
            print(f"Error processing data: {data['Message']}")
            return None

        nested_df = pd.DataFrame(data['Data'])
        df = nested_df['Data'].apply(pd.Series)
        if 'time' not in df.columns:
            print("The 'time' column is missing from the data.")
            return None

        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        df['200dma'] = df['close'].rolling(window=200, min_periods=1).mean()
        start_date = df.index[0] + pd.DateOffset(years=1)
        df_filtered = df[df.index >= start_date]
        symbol = symbol.upper()

        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax1.plot(df_filtered.index, df_filtered['close'], label=f'{symbol} Price', color='blue')
        ax1.plot(df_filtered.index, df_filtered['200dma'], label='200DMA', color='red')
        ax1.set_title(f'{symbol} Price and 200-Day Moving Average')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD)')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        scatter = ax1.scatter(df_filtered.index, df_filtered['close'], c=df_filtered['close'], cmap='plasma', edgecolor='none', alpha=0.6)
        plt.colorbar(scatter, ax=ax1, label='Daily Close Price')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        photo_data = buffer.getvalue()

        return photo_data
    
    def fetch_madr(self):
        symbol = 'BTC-USD'
        ma_period = 21
        no_std_dev = 2
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='7y', interval='1wk')
        df.index = pd.to_datetime(df.index)

        df['SMA'] = df['Close'].rolling(window=ma_period).mean()
        df['rate'] = (df['Close'] / df['SMA']) * 100 - 100

        extended_period = ma_period * no_std_dev
        df['stdCenter'] = df['rate'].rolling(window=extended_period).mean()
        df['std'] = df['rate'].rolling(window=extended_period).std()

        df['plusDev'] = df['stdCenter'] + df['std'] * no_std_dev
        df['minusDev'] = df['stdCenter'] - df['std'] * no_std_dev

        top_2017 = df.loc['2017-12'].rate.idxmax()
        top_2021 = df.loc['2021-01':'2021-12'].rate.idxmax()

        top_2017_value = df.loc[top_2017].rate
        top_2021_value = df.loc[top_2021].rate

        slope = (top_2021_value - top_2017_value) / ((top_2021 - top_2017).days)
        intercept = top_2017_value - slope * (top_2017 - df.index[0]).days

        collinear_line = {date: slope * (date - df.index[0]).days + intercept for date in df.index}

        latest_date = df.index[-1]
        latest_deviation_rate = df.loc[latest_date, 'rate']
        collinear_value = collinear_line[latest_date]
        difference = abs(latest_deviation_rate - collinear_value)

        print(f"Difference between deviation rate and collinear line at the latest date: {difference:.1f}")

        plt.figure(figsize=(14, 10))
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['Close'], label='Close', color='blue')
        plt.plot(df.index, df['SMA'], label='21-SMA', color='orange')
        plt.title('Bitcoin Weekly Close Price and SMA')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['rate'], label='Deviation Rate', color='purple')
        plt.plot(df.index, df['plusDev'], label='Plus Deviation', color='green')
        plt.plot(df.index, df['minusDev'], label='Minus Deviation', color='red')
        plt.plot(df.index, list(collinear_line.values()), label='Collinear Line', color='cyan')
        plt.title('Deviation Rate with Plus and Minus Deviation Lines')
        plt.legend()

        plt.tight_layout()
    
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        photo_data = buffer.getvalue()

        return photo_data, difference
    
    def smi_indicator(self):
    
        symbol = 'BTC-USD'
        start_date = None
        end_date = None
        fast_k = 5
        slow_k = 13
        signal_period = 3

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=12 * 365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        data = yf.download(symbol, start=start_date, end=end_date, interval='1mo')

        dates = data.index.to_pydatetime()
        close_prices = data['Close'].values
        high_prices = data['High'].values
        low_prices = data['Low'].values

        high_max = np.array([np.max(high_prices[i - fast_k:i]) for i in range(fast_k, len(high_prices) + 1)])
        low_min = np.array([np.min(low_prices[i - fast_k:i]) for i in range(fast_k, len(low_prices) + 1)])
        mid_point = (high_max + low_min) / 2
        close_trimmed = close_prices[fast_k - 1:]

        smi = (close_trimmed - mid_point) / (high_max - low_min) * 100

        smi_ema = np.convolve(smi, np.ones(slow_k), 'valid') / slow_k
        smi_signal = np.convolve(smi_ema, np.ones(signal_period), 'valid') / signal_period

        smi_ema = smi_ema[-len(smi_signal):]

        def normalize(data):
            max_value = np.max(data)
            return (data / max_value) * 100

        smi_values_normalized = normalize(smi_ema)
        signal_values_normalized = normalize(smi_signal)

        adjusted_dates = dates[-len(smi_values_normalized):]

        plt.figure(figsize=(12, 6))
        plt.plot(adjusted_dates, smi_values_normalized, label='SMI')
        plt.plot(adjusted_dates, signal_values_normalized, label='Signal Line', linestyle='--')
        plt.axhline(y=100, color='r', linestyle='-')
        plt.axhline(y=90, color='y', linestyle='-')
        plt.xlabel('Date')
        plt.ylabel('Normalized SMI Value')
        plt.title(f'Normalized SMI Ergodic Indicator for {symbol} Over the Last 12 Years (Monthly)')
        plt.legend()
        plt.grid(True)
        #plt.show()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        photo_data = buffer.getvalue()

        return photo_data
               
    def generate_bull_market_support_band_chart(self, symbol:str, interval='1wk'):
        
        if interval not in ['1d', '1wk']:
            raise ValueError("Interval must be '1d' or '1wk'")
    
        data = yf.download(symbol, period='5y', interval=interval)
   
        data_weekly = yf.download(symbol, period='5y', interval='1wk')
        data_weekly['20w_SMA'] = data_weekly['Close'].rolling(window=20).mean()
        data_weekly['21w_EMA'] = data_weekly['Close'].ewm(span=21, adjust=False).mean()
    
        data['20w_SMA'] = data_weekly['20w_SMA'].reindex(data.index, method='nearest').interpolate(method='linear')
        data['21w_EMA'] = data_weekly['21w_EMA'].reindex(data.index, method='nearest').interpolate(method='linear')
    
        plt.figure(figsize=(14, 8))
        plt.plot(data.index, data['20w_SMA'], label='20w SMA', color='red')
        plt.plot(data.index, data['21w_EMA'], label='21w EMA', color='green')

        for idx, row in data.iterrows():
            if row['Close'] > row['Open']:
                color = 'green'
                lower = row['Open']
                height = row['Close'] - row['Open']
            else:
                color = 'red'
                lower = row['Close']
                height = row['Open'] - row['Close']
            plt.bar(idx, height, bottom=lower, color=color, width=0.8 if interval == '1d' else 5, edgecolor='black')
            plt.vlines(x=idx, ymin=row['Low'], ymax=row['High'], color='black', linewidth=0.5 if interval == '1d' else 1)

        plt.fill_between(data.index, data['20w_SMA'], data['21w_EMA'], where=(data['20w_SMA'] > data['21w_EMA']), facecolor='orange', alpha=0.5, interpolate=True)
        plt.fill_between(data.index, data['20w_SMA'], data['21w_EMA'], where=(data['20w_SMA'] <= data['21w_EMA']), facecolor='orange', alpha=0.5, interpolate=True)

        plt.legend()
        plt.title(f'{symbol} {interval.upper()} Candlesticks with 20w SMA and 21w EMA')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        photo_data = buffer.getvalue()

        return photo_data
            
    def tenkanline_plot(self, symbol:str, start_date='2024-01-01', end_date=datetime.now().strftime('%Y-%m-%d'), interval='1d'):
    
        def calculate_midpoint(high, low, period):
        
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            midpoint = (highest_high + lowest_low) / 2
            return midpoint

   
        btc_data = yf.download(symbol, start=start_date, end=end_date, interval=interval)

        btc_weekly = btc_data.resample('W').agg({
            'High': 'max',
            'Low': 'min',
            'Open': 'first',
            'Close': 'last',
            'Volume': 'sum'
        })

        btc_weekly['Tenkansen'] = calculate_midpoint(btc_weekly['High'], btc_weekly['Low'], 9)

        btc_data['Tenkansen'] = btc_weekly['Tenkansen'].reindex(btc_data.index, method='ffill')

        btc_data['Kijunsen'] = calculate_midpoint(btc_data['High'], btc_data['Low'], 26)

        add_plots = [
            mpf.make_addplot(btc_data['Tenkansen'], color='blue', width=2),
            mpf.make_addplot(btc_data['Kijunsen'], color='red', width=2)
        ]

        buffer = io.BytesIO()

        fig, axlist = mpf.plot(btc_data, type='candle', style='charles', addplot=add_plots, title=f'{symbol.split('-')[0].upper()} Candlestick Chart with Tenkansen and Kijunsen Lines', ylabel='Price (USD)', volume=True, returnfig=True)

        fig.savefig(buffer, format='png')
        plt.close(fig)

        buffer.seek(0)

        return buffer
        
    def plot_superguppy(self, symbol:str):

        data = yf.download(symbol, start='2023-12-01', end=pd.Timestamp.today())

        fast_ema_lengths = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        slow_ema_lengths = [25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70]
        ema_200_length = 200

        fast_emas = [data['Close'].ewm(span=length, adjust=False).mean() for length in fast_ema_lengths]
        slow_emas = [data['Close'].ewm(span=length, adjust=False).mean() for length in slow_ema_lengths]
        ema_200 = data['Close'].ewm(span=ema_200_length, adjust=False).mean()

        fast_avg = sum(fast_emas) / len(fast_emas)
        slow_avg = sum(slow_emas) / len(slow_emas)

        fig, ax = plt.subplots(figsize=(12, 6))

        data['Date'] = mdates.date2num(data.index.to_pydatetime())
        candlestick_data = data[['Date', 'Open', 'High', 'Low', 'Close']].values.tolist()
        candlestick_ohlc(ax, candlestick_data, width=0.6, colorup='g', colordown='r')

        for i, length in enumerate(fast_ema_lengths):
            ax.plot(data.index, fast_emas[i], label=f'EMA {length}', color='blue', alpha=0.5)

        ax.plot(data.index, fast_avg, label='Fast Avg', color='gold', linewidth=2)

        for i, length in enumerate(slow_ema_lengths):
         ax.plot(data.index, slow_emas[i], label=f'EMA {length}', color='red', alpha=0.5)

        ax.plot(data.index, slow_avg, label='Slow Avg', color='fuchsia', linewidth=2)

        ax.plot(data.index, ema_200, label='EMA 200', color='black', linewidth=2)

        ax.set_title(f'{symbol.split('-')[0].upper()} Price with Multiple EMAs')
        ax.set_ylabel('Price')
        plt.legend(loc='center left',  fontsize=8)
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
    
        return buffer
    
    def plot_fibonacci_bollinger_bands(self, ticker = 'BTC-USD', interval='1d', length=200, mult=3.0):
   
        if not ticker.endswith('-USD'):
            ticker = f"{ticker}-USD"
        end_date = datetime.now()
  
        if interval == '4h':
            start_date = end_date - timedelta(days=120)
        elif interval == '1d':
            start_date = end_date - timedelta(days=500)
        else:
            print(f"Interval '{interval}' not supported.")
            return
    
        if interval == '4h':
       
            data = yf.download(ticker, start=start_date, end=end_date, interval='1h')
       
            data = data.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna().ffill()  # Fill missing values forward
        else:
   
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
        if data.empty:
            print(f"No valid data available for {ticker} with interval {interval} from {start_date} to {end_date}.")
            return
    
        #print(f"Data fetched for {ticker} with interval {interval} from {start_date} to {end_date}:\n{data.head()}")

        last_open = data['Open'].iloc[-1]

        hlc3 = (data['High'] + data['Low'] + data['Close']) / 3

        vwma = (hlc3 * data['Volume']).rolling(window=length).sum() / data['Volume'].rolling(window=length).sum()

        stdev = hlc3.rolling(window=length).std()

        basis = vwma
        dev = mult * stdev

        upper_1 = basis + (0.236 * dev)
        upper_2 = basis + (0.382 * dev)
        upper_3 = basis + (0.5 * dev)
        upper_4 = basis + (0.618 * dev)
        upper_5 = basis + (0.764 * dev)
        upper_6 = basis + (1 * dev)
        lower_1 = basis - (0.236 * dev)
        lower_2 = basis - (0.382 * dev)
        lower_3 = basis - (0.5 * dev)
        lower_4 = basis - (0.618 * dev)
        lower_5 = basis - (0.764 * dev)
        lower_6 = basis - (1 * dev)

        if basis.isnull().all() or upper_1.isnull().all() or lower_1.isnull().all():
            print(f"Insufficient valid data after calculations for {ticker} with interval {interval}.")
            return

        data['Basis'] = basis
        data['Upper 0.236'] = upper_1
        data['Upper 0.382'] = upper_2
        data['Upper 0.5'] = upper_3
        data['Upper 0.618'] = upper_4
        data['Upper 0.764'] = upper_5
        data['Upper 1.0'] = upper_6
        data['Lower 0.236'] = lower_1
        data['Lower 0.382'] = lower_2
        data['Lower 0.5'] = lower_3
        data['Lower 0.618'] = lower_4
        data['Lower 0.764'] = lower_5
        data['Lower 1.0'] = lower_6

        #print(f"Fibonacci Bollinger Bands calculated for {ticker}:\n{data[['Basis', 'Upper 0.236', 'Lower 0.236']].head()}")

        fig, ax = mpf.plot(
            data,
            type='candle',
            style='charles',
            title=f'{ticker} Fibonacci Bollinger Bands ({interval} interval)',
            ylabel='Price (USD)',
            addplot=[
                mpf.make_addplot(data['Basis'], color='fuchsia', linestyle='-', width=2),
                mpf.make_addplot(data['Upper 0.236'], color='gray', linestyle='-', width=1),
                mpf.make_addplot(data['Upper 0.382'], color='gray', linestyle='-', width=1),
                mpf.make_addplot(data['Upper 0.5'], color='gray', linestyle='-', width=1),
                mpf.make_addplot(data['Upper 0.618'], color='orange', linestyle='-', width=1),
                mpf.make_addplot(data['Upper 0.764'], color='gray', linestyle='-', width=1),
                mpf.make_addplot(data['Upper 1.0'], color='red', linestyle='-', width=2),
                mpf.make_addplot(data['Lower 0.236'], color='gray', linestyle='-', width=1),
                mpf.make_addplot(data['Lower 0.382'], color='gray', linestyle='-', width=1),
                mpf.make_addplot(data['Lower 0.5'], color='gray', linestyle='-', width=1),
                mpf.make_addplot(data['Lower 0.618'], color='orange', linestyle='-', width=1),
                mpf.make_addplot(data['Lower 0.764'], color='gray', linestyle='-', width=1),
                mpf.make_addplot(data['Lower 1.0'], color='green', linestyle='-', width=2),
            ],
            volume=False,
            figsize=(14, 7),
            returnfig=True
        )
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        plt.close(fig)

        buffer.seek(0)
        
        current_price = data['Close'].iloc[-1]
        basis_price = data['Basis'].iloc[-1]
        price_below_basis = current_price < basis_price
      

        return buffer, price_below_basis
    
    def calculate_obv(slef, symbol: str, timeframe='1d'):
        end_date = datetime.today()

        if timeframe == '4h':
            start_date = end_date - timedelta(days=120)  
            interval = '1h' 
        elif timeframe == '1w':
            start_date = end_date - timedelta(days=1500)  
            interval = '1d'  
        else:  
            start_date = end_date - timedelta(days=365)  
            interval = timeframe

        btc_data = yf.download(symbol, start=start_date, end=end_date, interval=interval)

        if btc_data.empty:
            print(f"No data found for ticker {symbol.split('-')[0].upper()} within the specified date range.")
            return

        if timeframe == '4h':
            btc_data = btc_data.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Adj Close': 'last',
                'Volume': 'sum'
            }).dropna().ffill()
        elif timeframe == '1w':
            btc_data = btc_data.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Adj Close': 'last',
                'Volume': 'sum'
            }).dropna().ffill()

        btc_data['Price Change'] = btc_data['Adj Close'].diff()
        btc_data['OBV'] = 0
        btc_data.loc[btc_data['Price Change'] > 0, 'OBV'] = btc_data['Volume']
        btc_data.loc[btc_data['Price Change'] < 0, 'OBV'] = -btc_data['Volume']
        btc_data['OBV'] = btc_data['OBV'].cumsum()

        plt.figure(figsize=(14, 7))
        plt.plot(btc_data['OBV'], label='OBV', color='purple')
        plt.title(f'On-Balance Volume (OBV) of {symbol}')
        plt.xlabel('Date')
        plt.ylabel('OBV')
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf

    def plot_btc_rsi_collinear(self):
        btc = yf.download("BTC-USD", interval="1mo")

        close_prices = np.array(btc['Close']) 
        rsi_period = 14  
        rsi_values = talib.RSI(close_prices, timeperiod=rsi_period) 

        btc['RSI'] = rsi_values


        peaks = argrelextrema(rsi_values, np.greater_equal, order=10)[0]  

        peaks_2017 = [peak for peak in peaks if btc.index[peak].year == 2017]
        peaks_2021 = [peak for peak in peaks if btc.index[peak].year == 2021]

        if peaks_2017:
            peak_2017_idx = max(peaks_2017)
            peak_2017_date = btc.index[peak_2017_idx]
            peak_2017_value = rsi_values[peak_2017_idx]
        else:
            print("No peaks found in 2017")
            return

        if peaks_2021:
            peak_2021_idx = max(peaks_2021)
            peak_2021_date = btc.index[peak_2021_idx]
            peak_2021_value = rsi_values[peak_2021_idx]
        else:
            print("No peaks found in 2021")
            return

        months_between_peaks = peak_2021_idx - peak_2017_idx
        slope = (peak_2021_value - peak_2017_value) / months_between_peaks
        intercept = peak_2017_value - slope * peak_2017_idx

        extend_months = 40  
        collinear_start = peak_2021_date  
        collinear_end = peak_2021_date + DateOffset(months=extend_months)  

        months_end = (collinear_end.year - peak_2017_date.year) * 12 + collinear_end.month - peak_2017_date.month
        collinear_end_rsi = peak_2017_value + slope * months_end

        latest_rsi_value = rsi_values[-1]

        plt.figure(figsize=(14, 10))

        plt.subplot(2, 1, 1)
        plt.plot(btc.index, btc['Close'], label='Monthly Close', color='blue')
        plt.title('Bitcoin Monthly Closing Prices')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(btc.index, btc['RSI'], label='RSI (14)', color='red')
        plt.title('RSI (14)')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.axhline(y=70, color='orange', linestyle='--', label='Overbought (RSI = 70)')
        plt.axhline(y=30, color='green', linestyle='--', label='Oversold (RSI = 30)')

        plt.plot([peak_2017_date, peak_2021_date], [peak_2017_value, peak_2021_value], linestyle='-', color='black', linewidth=2)
        plt.plot([collinear_start, collinear_end], [peak_2021_value, collinear_end_rsi], linestyle='--', color='gray', linewidth=2, label='Extended Collinear Line')
        plt.legend()
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)  

        #plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf, latest_rsi_value
    
    def plot_ticker_with_sar(self, symbol:str, period='1mo', interval='1d'):
 
        data = yf.download(symbol, period=period, interval=interval)
    
        if data is None or data.empty:
            print(f"No data available for {symbol.split('-')[0].upper()} for the given period.")
            return None
    
        data['SAR'] = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)

        ap0 = [mpf.make_addplot(data['SAR'], type='scatter', markersize=5, marker='o', color='blue')]
        fig, ax = mpf.plot(data, type='candle', style='charles', title=f'{symbol.split('-')[0].upper()} Price with SAR', ylabel='Price (USD)', addplot=ap0, returnfig=True)
    
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf

    def plot_hash_ribbons(self):
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=365&interval=daily"
        response = requests.get(url)

        try:
            data = response.json()
        except ValueError:
            print("Error: Unable to parse JSON response. Response content:")
            print(response.text)
            return

        if 'prices' not in data:
            print("Error: Unexpected response structure. Response content:")
            print(data)
            return

        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        len_s = 30
        len_l = 60
        df['HR_short'] = df['price'].rolling(window=len_s).mean()
        df['HR_long'] = df['price'].rolling(window=len_l).mean()

        df['capitulation'] = df['HR_short'] < df['HR_long']
        df['recovering'] = (df['HR_short'] > df['HR_short'].shift(1)) & (df['HR_short'] > df['HR_short'].shift(2)) & (df['HR_short'] > df['HR_short'].shift(3)) & df['capitulation']
        df['recovered'] = df['HR_short'] > df['HR_long']
        df['buy'] = ((df['HR_short'].shift(10) < df['HR_long'].shift(10)) & (df['HR_short'] > df['HR_long']))

        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['HR_short'], label='Short SMA (30 days)', color='lime')
        plt.plot(df.index, df['HR_long'], label='Long SMA (60 days)', color='gray')
        plt.fill_between(df.index, df['HR_short'], df['HR_long'], where=(df['HR_short'] < df['HR_long']), color='red', alpha=0.3)
        plt.scatter(df[df['capitulation']].index, df[df['capitulation']]['HR_short'], label='Capitulation', color='gray', marker='o')
        plt.scatter(df[df['recovering']].index, df[df['recovering']]['HR_short'], label='Recovering', color='green', marker='o')
        plt.scatter(df[df['recovered']].index, df[df['recovered']]['HR_short'], label='Recovered', color='lime', marker='o')
        plt.scatter(df[df['buy']].index, df[df['buy']]['HR_short'], label='Buy', color='blue', marker='o')

        plt.xlabel('Date')
        plt.ylabel('Hash Rate')
        plt.title('Hash Ribbons')
        plt.legend()
        plt.grid(True)
        #plt.show()

        buf = io.BytesIO() 
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf

    def plot_cmf(self, symbol, start_date='2020-01-01', end_date=None, interval='1wk', cmf_period=20):

        btc_data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        btc_data['Money Flow Multiplier'] = (2 * btc_data['Close'] - btc_data['Low'] - btc_data['High']) / (btc_data['High'] - btc_data['Low'])
        btc_data['Money Flow Volume'] = btc_data['Money Flow Multiplier'] * btc_data['Volume']
        btc_data['CMF'] = btc_data['Money Flow Volume'].rolling(window=cmf_period).sum() / btc_data['Volume'].rolling(window=cmf_period).sum()

        if symbol == 'BTC-USD':
            top_cmf_2020 = btc_data.loc['2020-01-01':'2020-12-31']['CMF'].idxmax()
            top_cmf_2023 = btc_data.loc['2023-01-01':'2023-12-31']['CMF'].idxmax()

            cmf_2020 = btc_data.loc[top_cmf_2020]['CMF']
            cmf_2023 = btc_data.loc[top_cmf_2023]['CMF']

            m = (cmf_2023 - cmf_2020) / ((top_cmf_2023 - top_cmf_2020).days)
            b = cmf_2020 - m * (top_cmf_2020 - btc_data.index[0]).days

            btc_data['Collinear Line CMF'] = m * (btc_data.index - btc_data.index[0]).days + b

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        ax1.set_title(f'{symbol.split('-')[0].upper()} Price')
        ax1.set_ylabel('Price (USD)')
        ax1.plot(btc_data.index, btc_data['Close'], label='BTC Price', color='tab:blue')
        ax1.legend(loc='upper left')

        ax2.set_title('Chaikin Money Flow (CMF)')
        ax2.set_ylabel('CMF')
        ax2.plot(btc_data.index, btc_data['CMF'], label='CMF', color='tab:green')
        if symbol == 'BTC-USD':
            ax2.plot([top_cmf_2020, top_cmf_2023], [cmf_2020, cmf_2023], 'ro') 
            ax2.plot(btc_data.index, btc_data['Collinear Line CMF'], label='Collinear Line', color='tab:red', linestyle='--')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax2.legend(loc='upper left')
        ax2.set_xlabel('Date')

        fig.tight_layout(pad=3.0)

        #plt.show()
        buf = io.BytesIO() 
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf
    

    def plot_sma_crossovers(self, symbol: str, start_date='2020-07-01', end_date=None):

        if end_date is None:
            data = yf.download(symbol, start=start_date)
        else:
            data = yf.download(symbol, start=start_date, end=end_date)
    
        data['SMA10'] = data['Close'].rolling(window=10).mean()
        data['SMA35'] = data['Close'].rolling(window=35).mean()

        data['Signal'] = np.where(data['SMA10'] > data['SMA35'], 1, 0)

        data.loc[data.index[10:], 'Signal'] = np.where(data['SMA10'][10:] > data['SMA35'][10:], 1, 0)
        data['Position'] = data['Signal'].diff()

        plt.figure(figsize=(14, 7))
        plt.plot(data['Close'], label=f'{symbol.split("-")[0].upper()} Close Price', alpha=0.5, color='blue')
        plt.plot(data['SMA10'], label='10-Day SMA', alpha=0.75, color='orange')
        plt.plot(data['SMA35'], label='35-Day SMA', alpha=0.75, color='green')

        buy_signals = data[data['Position'] == 1]
        sell_signals = data[data['Position'] == -1]

        plt.plot(buy_signals.index, data['SMA10'][buy_signals.index], '^', markersize=10, color='g', lw=0, label='Buy Signal')
        plt.plot(sell_signals.index, data['SMA10'][sell_signals.index], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

        if not buy_signals.empty:
            last_buy_date = buy_signals.index[-1]
        else:
            last_buy_date = None

        if not sell_signals.empty:
            last_sell_date = sell_signals.index[-1]
        else:
            last_sell_date = None

        if last_buy_date is not None and (last_sell_date is None or last_buy_date > last_sell_date):
            last_signal = 'Buy'
            last_signal_date = last_buy_date
        elif last_sell_date is not None and (last_buy_date is None or last_sell_date > last_buy_date):
            last_signal = 'Sell'
            last_signal_date = last_sell_date
        else:
            last_signal = 'No Signal'
            last_signal_date = None

        plt.title(f'{symbol.split("-")[0].upper()} Price with SMA Crossovers')
        plt.legend()
        #plt.show()

        buf = io.BytesIO() 
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        print(last_signal)
        print(last_signal_date)

        return buf, last_signal, last_signal_date

   
    def calculate_stoch(self, ticker, interval):
        today = datetime.today()
    
        if interval == '1d':
            requested_days = 300  # Approx. 1 year back
        elif interval == '1wk':
            requested_days = 365 * 1.5  # Approx. 1.5 years back
        elif interval == '4h':
            requested_days = 30  # Approx. 1 month back
        else:
            raise ValueError('Unsupported interval. Choose from "1d", "1wk", "4h".')
        
        def fetch_data(start_date):
            try:
                if interval == '4h':
                    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=None, interval='1h')
                    data = data.resample('4h').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }).dropna()
                else:
                    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=None, interval=interval)
                return data
            except Exception as e:
                print(f"Error fetching data: {e}")
                return pd.DataFrame()  


        start_date = today - timedelta(days=requested_days)
        data = fetch_data(start_date)
 
        if data.empty:
            for reduction_factor in [0.5, 0.25, 0.1]:
                try:
                    adjusted_days = int(requested_days * reduction_factor)
                    print(f"Data not available for requested period. Trying with a shorter period ({adjusted_days} days).")
                    start_date = today - timedelta(days=adjusted_days)
                    data = fetch_data(start_date)
                    if not data.empty:
                        break
                except Exception as e:
                    print(f"Error during fallback data fetch: {e}")
                    continue

        if data.empty:
            raise ValueError(f'No price data found for {ticker}. Symbol may be delisted or not available for the specified interval.')

        stoch = ta.momentum.StochasticOscillator(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            window=14,
            smooth_window=3
        )
        data['%K'] = stoch.stoch()
        data['%D'] = data['%K'].rolling(window=3).mean()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        ax1.plot(data.index, data['Close'], label=f'{ticker} Price')
        ax1.set_title(f'{ticker} Price Chart ({interval})')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(data.index, data['%K'], label='%K')
        ax2.plot(data.index, data['%D'], label='%D')
        ax2.axhline(y=20, color='r', linestyle='--', label='Oversold (20)')
        ax2.axhline(y=80, color='g', linestyle='--', label='Overbought (80)')
        ax2.set_title('Stochastic Oscillator')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        buf = io.BytesIO() 
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()


        latest_k_value = data['%K'].iloc[-1] if not data['%K'].empty else None
        if latest_k_value is None:
            raise ValueError(f'Unable to retrieve the latest %K value for {ticker}.')

        return buf, latest_k_value
           
    
    def supertrend(self, ticker, interval, periods, atr_multipliers):

        today = datetime.today()
        if interval == '1d':
            start_date = today - timedelta(days=365)
        elif interval == '1wk':
            start_date = today - timedelta(days=365 * 1.5)
        elif interval == '4h':
            start_date = today - timedelta(days=30)
        else:
            raise ValueError('Unsupported interval. Choose from "1d", "1wk", "4h".')

        if interval == '4h':
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=None, interval='1h')
            data = data.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        else:
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=None, interval=interval)

        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax1.plot(data.index, data['Close'], label=f'{ticker} Price')
        
        colors = ['green', 'blue', 'orange', 'purple', 'brown']
        for i, (period, atr_multiplier) in enumerate(zip(periods, atr_multipliers)):
            source = (data["High"] + data["Low"]) / 2

            atr = ta.volatility.AverageTrueRange(
                high=data['High'], low=data['Low'], close=data['Close'], window=period
            ).average_true_range()

            up = source - (atr_multiplier * atr)
            dn = source + (atr_multiplier * atr)

            up_list = [up.iloc[0]]
            dn_list = [dn.iloc[0]]
            trend = 1
            trend_list = [trend]

            for j in range(1, len(data)):
                if trend == 1:
                    if data['Close'].iloc[j] > up_list[-1]:
                        up_list.append(max(up_list[-1], up.iloc[j]))
                        dn_list.append(np.nan)
                    else:
                        trend = -1
                        dn_list.append(dn.iloc[j])
                        up_list.append(np.nan)
                else:
                    if data['Close'].iloc[j] < dn_list[-1]:
                        dn_list.append(min(dn_list[-1], dn.iloc[j]))
                        up_list.append(np.nan)
                    else:
                        trend = 1
                        up_list.append(up.iloc[j])
                        dn_list.append(np.nan)
                trend_list.append(trend)

            supertrend_df = pd.DataFrame({
                'uptrend': up_list,
                'downtrend': dn_list,
                'trend': trend_list
            }, index=data.index)

            ax1.plot(supertrend_df.index, supertrend_df['uptrend'], label=f'Uptrend P{period} M{atr_multiplier}', color=colors[i % len(colors)], linestyle='--')
            ax1.plot(supertrend_df.index, supertrend_df['downtrend'], label=f'Downtrend P{period} M{atr_multiplier}', color='red')

        ax1.set_title(f'{ticker} Price Chart ({interval}) with Multiple SuperTrend Indicators')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True)

        plt.tight_layout()
        #plt.show()
   
        buf = io.BytesIO() 
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf






    def calculate_emas(self, ticker, interval):
        ema_periods = [7, 30, 100, 200]

        today = datetime.today()
        if interval == '1d':
            start_date = today - timedelta(days=365)
        elif interval == '1wk':
            start_date = today - timedelta(days=365 * 1.5)
        elif interval == '4h':
            start_date = today - timedelta(days=30)
        else:
            raise ValueError('Unsupported interval. Choose from "1d", "1wk", "4h".')

        if interval == '4h':
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=None, interval='1h')
            data = data.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        else:
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=None, interval=interval)

        for period in ema_periods:
            data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()

        upward_cross = False
        downward_cross = False

        # Check for crossover in the last 10 data points
        if len(data) >= 2:
            for i in range(-2, 0):
                if data['EMA_7'].iloc[i] > data['EMA_30'].iloc[i] and data['EMA_7'].iloc[i-1] <= data['EMA_30'].iloc[i-1]:
                    upward_cross = True
                    break
                elif data['EMA_7'].iloc[i] < data['EMA_30'].iloc[i] and data['EMA_7'].iloc[i-1] >= data['EMA_30'].iloc[i-1]:
                    downward_cross = True
                    break

        fig, ax1 = plt.subplots(figsize=(14, 8))

        ax1.plot(data.index, data['Close'], label=f'{ticker} Price', color='black')
        ax1.plot(data.index, data['EMA_7'], label='EMA 7', color='green', linewidth=1)
        ax1.plot(data.index, data['EMA_30'], label='EMA 30', color='blue', linewidth=2)
        ax1.plot(data.index, data['EMA_100'], label='EMA 100', color='orange', linewidth=3)
        ax1.plot(data.index, data['EMA_200'], label='EMA 200', color='red', linewidth=4)

        ax1.set_title(f'{ticker} Price Chart ({interval}) with EMAs')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        print(upward_cross)
        print(downward_cross)

        return buf, upward_cross, downward_cross
    

    def detect_bearish_engulfing(self, ticker='BTC-USD', interval='1d', detection: str = None):

        today = datetime.today()

        if interval == '1d':
            start_date = today - timedelta(days=365)
        elif interval == '1wk':
            start_date = today - timedelta(days=365 * 1.5)
        elif interval == '4h':
            start_date = today - timedelta(days=30)
        else:
            raise ValueError('Unsupported interval. Choose from "1d", "1wk", "4h".')

        if interval == '4h':
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=None, interval='1h')
            data = data.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        else:
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=None, interval=interval)

        if detection == "SMA50":
            sma50 = pd.Series(data["Close"]).rolling(window=50).mean().values
            up_trend = np.where(data["Close"].values > sma50, True, False)
        elif detection == "SMA50/200":
            sma50 = pd.Series(data["Close"]).rolling(window=50).mean().values
            sma200 = pd.Series(data["Close"]).rolling(window=200).mean().values
            up_trend = np.where(
                (data["Close"].values > sma50) & (data["Close"].values > sma200),
                True,
                False,
            )
        else:
            up_trend = np.full(len(data), True)

        body_len = 14  

        body_high = np.maximum(data["Close"].values, data["Open"].values)
        body_low = np.minimum(data["Close"].values, data["Open"].values)
        body = body_high - body_low

        body_avg = pd.Series(body).ewm(span=body_len, adjust=False).mean().values
        short_body = body < body_avg
        long_body = body > body_avg

        white_body = data["Open"].values < data["Close"].values
        black_body = data["Open"].values > data["Close"].values

        engulfing_bearish = [False]
        for i in range(1, len(data)):
            condition = (
                up_trend[i]
                & black_body[i]
                & long_body[i]
                & white_body[i - 1]
                & short_body[i - 1]
                & (data["Close"].values[i] <= data["Open"].values[i - 1])
                & (data["Open"].values[i] >= data["Close"].values[i - 1])
                & (
                    (data["Close"].values[i] < data["Open"].values[i - 1])
                    | (data["Open"].values[i] > data["Close"].values[i - 1])
                )
            )
            engulfing_bearish.append(condition)

        data['Bearish_Engulfing'] = engulfing_bearish

        now_or_one_ago = np.any(data['Bearish_Engulfing'][-2:])
        engulfing_dates = data[data['Bearish_Engulfing']].index
        #print(f"Bearish Engulfing patterns detected for {ticker} on {interval} timeframe:\n", engulfing_dates)
        #print(f"Bearish Engulfing signal is {'current or from the previous timeframe' if now_or_one_ago else 'not recent'}.")

        fig, ax = plt.subplots(figsize=(14, 8))

        ax.plot(data.index, data['Close'], label=f'{ticker} Close Price', color='black')

        ax.plot(data.index[data['Bearish_Engulfing']], data['Close'][data['Bearish_Engulfing']], 
                marker='o', linestyle='None', color='red', label='Bearish Engulfing')

        ax.set_title(f'{ticker} Price with Bearish Engulfing Patterns ({interval} Timeframe)')
        ax.set_ylabel('Price (USD)')
        ax.set_xlabel('Date')
        ax.legend()

        plt.grid(True)
        plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return buf, now_or_one_ago

    def detect_bullish_engulfing(self, ticker='BTC-USD', interval='1d', detection: str = None):
    
        today = datetime.today()

        if interval == '1d':
            start_date = today - timedelta(days=365)
        elif interval == '1wk':
            start_date = today - timedelta(days=365 * 1.5)
        elif interval == '4h':
            start_date = today - timedelta(days=30)
        else:
            raise ValueError('Unsupported interval. Choose from "1d", "1wk", "4h".')

        if interval == '4h':
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=None, interval='1h')
            data = data.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        else:
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=None, interval=interval)

        if detection == "SMA50":
            sma50 = pd.Series(data["Close"]).rolling(window=50).mean().values
            down_trend = np.where(data["Close"].values < sma50, True, False)
        elif detection == "SMA50/200":
            sma50 = pd.Series(data["Close"]).rolling(window=50).mean().values
            sma200 = pd.Series(data["Close"]).rolling(window=200).mean().values
            down_trend = np.where(
                (data["Close"].values < sma50) & (data["Close"].values < sma200),
                True,
                False,
            )
        else:
            down_trend = np.full(len(data), True)

        body_len = 14
        body_high = np.maximum(data["Close"].values, data["Open"].values)
        body_low = np.minimum(data["Close"].values, data["Open"].values)
        body = body_high - body_low
        body_avg = pd.Series(body).ewm(span=body_len, adjust=False).mean().values
        short_body = body < body_avg
        long_body = body > body_avg
        white_body = data["Open"].values < data["Close"].values
        black_body = data["Open"].values > data["Close"].values

        engulfing_bullish = [False]
        for i in range(1, len(data)):
            condition = (
                down_trend[i]
                & white_body[i]
                & long_body[i]
                & black_body[i - 1]
                & short_body[i - 1]
                & (data["Close"].values[i] >= data["Open"].values[i - 1])
                & (data["Open"].values[i] <= data["Close"].values[i - 1])
                & (
                    (data["Close"].values[i] > data["Open"].values[i - 1])
                    | (data["Open"].values[i] < data["Close"].values[i - 1])
                )
            )
            engulfing_bullish.append(condition)

        data['Bullish_Engulfing'] = engulfing_bullish

        now_or_one_ago = np.any(data['Bullish_Engulfing'][-2:])

        engulfing_dates = data[data['Bullish_Engulfing']].index
        #print(f"Bullish Engulfing patterns detected for {ticker} on {interval} timeframe:\n", engulfing_dates)
        #print(f"Bullish Engulfing signal is {'current or from the previous timeframe' if now_or_one_ago else 'not recent'}.")

        fig, ax = plt.subplots(figsize=(14, 8))

        ax.plot(data.index, data['Close'], label=f'{ticker} Close Price', color='black')

        ax.plot(engulfing_dates, data['Close'][data['Bullish_Engulfing']], 
                marker='o', linestyle='None', color='green', label='Bullish Engulfing')

        ax.set_title(f'{ticker} Price with Bullish Engulfing Patterns ({interval} Timeframe)')
        ax.set_ylabel('Price (USD)')
        ax.set_xlabel('Date')
        ax.legend()

        plt.grid(True)
        plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return buf, now_or_one_ago
    
    def plot_rsi_fib_cross(self, btc_symbol='BTC-USD', start_date='2016-01-01'):
       
        btc_data = yf.download(btc_symbol, start=start_date, end=None, interval='1d')

        # Resampling to 3-week periods
        btc_3w = btc_data.resample('3W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })

        # Calculating RSI on the resampled data using pandas_ta
        rsi_3w = tk.rsi(btc_3w['Close'], length=14)

        # Plotting the RSI with Reversed Fibonacci Channel
        plt.figure(figsize=(12, 6))
        plt.plot(rsi_3w, label='RSI', color='blue')

        # Adding Reversed Fibonacci retracement levels to the RSI
        low_rsi = rsi_3w.min()
        high_rsi = rsi_3w.max()
        diff_rsi = high_rsi - low_rsi

        # Reversed Fibonacci levels (0% at the top, 100% at the bottom)
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 1]

        # Initialize variables to store the last cross
        last_cross_type = None
        last_cross_level = None
        last_cross_value = None
        last_cross_date = None

        for i in range(1, len(rsi_3w)):
            for level in fib_levels:
                # Reverse the level calculation
                level_value = high_rsi - level * diff_rsi
                plt.axhline(y=level_value, color='red', linestyle='--')
                
                # Adjusting the label position even further to the right
                plt.text(rsi_3w.index[-1] + pd.Timedelta(weeks=6), level_value, f'{level*100:.1f}%', color='red', fontsize=12,
                        verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
                
                # Check for crossing up using iloc for positional indexing
                if rsi_3w.iloc[i-1] < level_value and rsi_3w.iloc[i] >= level_value:
                    last_cross_type = 'Up'
                    last_cross_level = level
                    last_cross_value = rsi_3w.iloc[i]
                    last_cross_date = rsi_3w.index[i]
                
                # Check for crossing down using iloc for positional indexing
                if rsi_3w.iloc[i-1] > level_value and rsi_3w.iloc[i] <= level_value:
                    last_cross_type = 'Down'
                    last_cross_level = level
                    last_cross_value = rsi_3w.iloc[i]
                    last_cross_date = rsi_3w.index[i]

        # Check if the last cross happened within the last 3 weeks
        recent_cross = False
        if last_cross_date and (rsi_3w.index[-1] - last_cross_date).days <= 21:
            recent_cross = True
            cross_info = {
                'type': last_cross_type,
                'level': last_cross_level,
                'value': last_cross_value,
                'date': last_cross_date
            }
        else:
            cross_info = None

        plt.title('RSI with Reversed Fibonacci Channel (3-Week Period)')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid()

        # Save the plot to a buffer
        buf = io.BytesIO() 
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return buf, cross_info


class Bot:
    
    def __init__(self, user_id, token, weektype='day', symbol='BTC', limit=100, aggregate=1):
        self.token = token
        self.bot = AsyncTeleBot(token=self.token)
        self.user_id = user_id
        self.market_data = MarketData(weektype=weektype, symbol=symbol, limit=limit, aggregate=aggregate)
        fetch_symbols = False
        
         
        if not fetch_symbols:
            self.tickers = [symbol + '-USD' for symbol in self.market_data.symbols]
            fetch_symbols = True
        self.last_signal = {ticker: {'signal': None, 'date': None} for ticker in self.tickers}
        
        

        print(self.tickers)
        

        self.fetched_above_threshold = {ticker: False for ticker in self.tickers}
        self.fetched_above_threshold['PiCycle'] = False 
        self.fetched_above_threshold['Fibbol'] = False 
        self.fetched_above_threshold['RSI_HIGH'] = False
        self.fetched_above_threshold['RSI_LOW'] = False
        self.fetched_above_threshold['Stoch'] = False
        self.ema_buy_action = False
        self.ema_sell_action = False
        self.check = True
        self.bearish_engulf_action = False
        self.bullish_engulf_action = False
        self.fibrsi = False
        
 
        @self.bot.message_handler(commands=['start'])
        async def handle_start(message):
            await self.process_start(message)

        @self.bot.message_handler(commands=['price'])
        async def handle_price(message):
            await self.process_price(message)

        @self.bot.message_handler(commands=['volume'])
        async def handle_volume(message):
            await self.process_volume(message)

        @self.bot.message_handler(commands=['price_chart'])
        async def handle_price_chart(message):
            await self.process_price_chart(message)

        @self.bot.message_handler(commands=['bollinger_bands'])
        async def handle_bollinger_bands(message):
            await self.process_bollinger_bands(message)

        @self.bot.message_handler(commands=['daily_report'])
        async def handle_daily_report(message):
            await self.process_daily_report(message)

        @self.bot.message_handler(commands=['pi_cycle'])
        async def handle_pi_cycle(message):
            await self.process_pi_cycle(message)

        @self.bot.message_handler(commands=['2yma'])
        async def handle_two_yma(message):
            await self.process_two_yma(message)

        @self.bot.message_handler(commands=['200wma'])
        async def handle_twohundredweek_ma(message):
            await self.process_twohundredweek_ma(message)

        @self.bot.message_handler(commands=['200dma'])
        async def handle_twohundredday_ma(message):
            await self.process_twohundredday_ma(message)

        @self.bot.message_handler(commands=['madr'])
        async def handle_moving_average_deviation_rate(message):
            await self.process_madr(message)

        @self.bot.message_handler(commands=['smi'])
        async def handle_smi(message):
            await self.process_smi(message)

        @self.bot.message_handler(commands=['superguppy'])
        async def handle_superguppy(message):
            await self.process_superguppy(message)

        @self.bot.message_handler(commands=['support_band'])
        async def handle_bull_market_support_band(message):
            await self.process_bull_market_support_band(message)    
            
        @self.bot.message_handler(commands=['tenkanline'])
        async def handle_tenkanline(message):
            await self.process_tenkanline(message) 
            
        @self.bot.message_handler(commands=['fibbol'])
        async def handle_fibonacci_bollinger(message):
            await self.process_fibonacci_bollinger_bands(message)
            
        @self.bot.message_handler(commands=['obv'])
        async def handle_obv(message):
            await self.process_obv(message)
            
        @self.bot.message_handler(commands=['rsi_month'])
        async def handle_rsi_monthly(message):
            await self.process_rsi_monthly(message)
            
        @self.bot.message_handler(commands=['sar'])
        async def handle_sar(message):
            await self.process_sar(message)
            
        @self.bot.message_handler(commands=['hash_ribbon'])
        async def handle_hash_ribbon(message):
            await self.process_hash_ribbon(message)
            
        @self.bot.message_handler(commands=['everything'])
        async def handle_everything(message):
            await self.process_everything(message)

        @self.bot.message_handler(commands=['cmf'])
        async def handle_cmf(message):
            await self.process_cmf(message)

        @self.bot.message_handler(commands=['sma'])
        async def handle_sma_crossover(message):
            await self.process_sma_crossover(message)

        @self.bot.message_handler(commands=['stoch'])
        async def handle_stoch(message):
            await self.process_stoch(message)

        @self.bot.message_handler(commands=['supertrend'])
        async def handle_supertrend(message):
            await self.process_supertrend(message)

        @self.bot.message_handler(commands=['emas'])
        async def handle_emas(message):
            await self.process_emas(message)

        @self.bot.message_handler(commands=['bearish_engulf'])
        async def handle_bearish_engulf(message):
            await self.process_bearish_engulf(message)

        @self.bot.message_handler(commands=['bullish_engulf'])
        async def handle_bullish_engulf(message):
            await self.process_bullish_engulf(message)

        @self.bot.message_handler(commands=['3w_rsi'])
        async def handle_3wrsi(message):
            await self.process_3wrsi(message)

        @self.bot.message_handler(func=lambda message: True)
        async def handle_unknown(message):
            await self.bot.send_message(chat_id=message.chat.id, text="Sorry, I don't understand that command.")

      
        
    async def process_start(self, message):
        chat_id = message.chat.id
        await self.bot.send_message(chat_id=chat_id, text="It works!\nThere are many posibilities with this code.\nPrice, volume and price_chart are able to work with all tickers and price_chart can work on hourly next to day" )

    async def process_price(self, message):
        chat_id = message.chat.id
        text = message.text.split()
        symbol = text[1] if len(text) > 1 else 'BTC'
        price = self.market_data.get_current_price(symbol)
        if price:
            if price >= 1:
                await self.bot.send_message(chat_id=chat_id, text=f"The current price of {symbol} is ${price:.2f}")
            else:
                await self.bot.send_message(chat_id=chat_id, text=f"The current price of {symbol} is ${price:.7f}")
        else:
            await self.bot.send_message(chat_id=chat_id, text=f"Sorry, I couldn't fetch the current price for {symbol}.")

    async def process_volume(self, message):
        chat_id = message.chat.id
        text = message.text.split()
        symbol = text[1] if len(text) > 1 else 'BTC'
        volume = self.market_data.get_current_volume(symbol)
        if isinstance(volume, float):
            await self.bot.send_message(chat_id=chat_id, text=f"The current 24-hour volume of {symbol} is ${volume:.2f} USD")
        elif volume == "No volume data found":
            await self.bot.send_message(chat_id=chat_id, text=f"Sorry, no volume data found for {symbol}.")
        else:
            await self.bot.send_message(chat_id=chat_id, text=f"Sorry, I couldn't fetch the current volume for {symbol}.")

    async def process_price_chart(self, message):
        chat_id = message.chat.id
        text = message.text.split()
        symbol = text[1] if len(text) > 1 else 'BTC'
        weektype = text[2] if len(text) > 2 else 'day'
        
        file_path = self.market_data.generate_candlestick_chart(weektype=weektype, symbol=symbol)
        
        if file_path:
            with open(file_path, 'rb') as file:
                await self.bot.send_photo(chat_id=chat_id, photo=file)
        else:
            await self.bot.send_message(chat_id=chat_id, text=f"Sorry, I couldn't generate the candlestick chart for {symbol}.")

    async def process_bollinger_bands(self, message):
        chat_id = message.chat.id
        text = message.text.split()
        symbol = text[1] if len(text) > 1 else 'BTC'
        weektype = text[2] if len(text) > 2 else 'hour'
        
        buffer = self.market_data.generate_bollinger_bands_chart(weektype=weektype, symbol=symbol)
        
        if buffer:
            await self.bot.send_photo(chat_id=chat_id, photo=buffer)
        else:
            await self.bot.send_message(chat_id=chat_id, text=f"Sorry, I couldn't generate the Bollinger Bands chart for {symbol}.")

        
    async def process_daily_report(self, message):
        chat_id = message.chat.id
        command_parts = message.text.split()
        symbol = command_parts[1].upper() + '-USD' if len(command_parts) > 1 else 'BTC-USD'
        photo_data, summary_json = self.market_data.daily_report(symbol)
        await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        await self.bot.send_message(chat_id=chat_id, text=f"Daily Report Summary for {symbol}:\n{summary_json}")

    async def process_pi_cycle(self, message):
        chat_id = message.chat.id
        text = message.text.split()
        symbol = text[1] if len(text) > 1 else 'BTC'
        photo_data, abs_difference = self.market_data.pi_cycle_plot(symbol)
        if photo_data:
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
            await self.bot.send_message(chat_id=chat_id, text=f"The abs difference between lines is: ${abs_difference:.0f}")
            if abs_difference < 5000:
                await self.bot.send_message(chat_id=chat_id, text=f"The pi cycle seems to be getting topped, WATCH IT! DANGER DANGER DANGER")
            else:
                await self.bot.send_message(chat_id=chat_id, text=f"Pi cycle seems quite low still, from this indicator no danger ;p")
        else:
            await self.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't generate the Pi Cycle plot.")

    async def process_two_yma(self, message):
        chat_id = message.chat.id
        text = message.text.split()
        symbol = text[1] if len(text) > 1 else 'BTC'
        photo_data, difference = self.market_data.two_yma(symbol)
        if photo_data:
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
            await self.bot.send_message(chat_id=chat_id, text=f"The difference between 5x2YMA and the latest price is: ${difference:.2f}")
        else:
            await self.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't generate the 2-Year Moving Average plot.")


    async def process_twohundredweek_ma(self, message):
        chat_id = message.chat.id
        text = message.text.split()
        symbol = text[1] if len(text) > 1 else 'BTC'
        photo_data = self.market_data.twohundredweek_ma(symbol)
        if photo_data:
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        else:
            await self.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't generate the 200-Week Moving Average plot.")

    async def process_twohundredday_ma(self, message):
        chat_id = message.chat.id
        text = message.text.split()
        symbol = text[1] if len(text) > 1 else 'BTC'
        photo_data = self.market_data.twohundredday_ma(symbol)
        if photo_data:
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        else:
            await self.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't generate the 200-Week Moving Average plot.")



    async def process_madr(self, message):
        chat_id = message.chat.id
        photo_data, difference = self.market_data.fetch_madr()
        if photo_data:
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
            await self.bot.send_message(chat_id=chat_id, text=f"The difference between madr and the latest price is: {difference:.1f}")
        else:
            await self.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't generate the Moving Average Deviation Rate plot.")

    async def process_smi(self, message):
        chat_id = message.chat.id
        photo_data = self.market_data.smi_indicator()
        if photo_data:
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        else:
            await self.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't generate the SMI indicator plot.")

    async def process_superguppy(self, message):
        chat_id = message.chat.id
        text = message.text.split()
        symbol = text[1].upper() + '-USD' if len(text) > 1 else 'BTC-USD'
        if not symbol.upper().endswith('-USD'):
            symbol = f"{symbol.upper()}-USD"
        photo_data = self.market_data.plot_superguppy(symbol)
        if photo_data:
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        else:
            await self.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't generate the super guppy chart.")
  
    async def process_bull_market_support_band(self, message):
        chat_id = message.chat.id
        text = message.text.split()
        symbol = text[1].upper() + '-USD' if len(text) > 1 else 'BTC-USD'
        if not symbol.upper().endswith('-USD'):
            symbol = f"{symbol.upper()}-USD"
        photo_data = self.market_data.generate_bull_market_support_band_chart(symbol)
        if photo_data:
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        else:
            await self.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't generate the Bull Market Support Band chart.")
            
            
    async def process_tenkanline(self, message):
        chat_id = message.chat.id
        text = message.text.split()
        symbol = text[1].upper() + '-USD' if len(text) > 1 else 'BTC-USD'
        if not symbol.upper().endswith('-USD'):
            symbol = f"{symbol.upper()}-USD"
        photo_data = self.market_data.tenkanline_plot(symbol)
        if photo_data:
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        else:
            await self.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't generate the tenkan line chart.")
            
            
    async def process_fibonacci_bollinger_bands(self, message):
        chat_id=message.chat.id
        command_parts = message.text.split()
        ticker = 'BTC'
        interval = '1d'
        
        if len(command_parts) > 1:
            ticker = command_parts[1].upper()
        if len(command_parts) > 2:
            interval = command_parts[2]

        photo_data, equivalent = self.market_data.plot_fibonacci_bollinger_bands(ticker, interval)
        if photo_data:
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        else:
            await self.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't generate the obv lines.")
            
            
    async def process_obv(self, message):
        interval = '1d'
        chat_id=message.chat.id
        text = message.text.split()
        symbol = text[1].upper() + '-USD' if len(text) > 1 else 'BTC-USD'
        if not symbol.upper().endswith('-USD'):
            symbol = f"{symbol.upper()}-USD"

        if len(text) > 2:
            interval = text[2].lower()
        if interval not in ['1d', '4h', '1w']:
            await self.bot.send_message(chat_id=chat_id, text="Invalid interval. Please use '1d', '4h', or '1w'.")

        photo_data = self.market_data.calculate_obv(symbol, interval)
        if photo_data:
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        else:
            await self.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't generate the fibonacci bollinger lines.")
    
    async def process_rsi_monthly(self, message):
        chat_id = message.chat.id
        photo_data, latest = self.market_data.plot_btc_rsi_collinear()
        if photo_data:
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        else:
            await self.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't generate the rsi monthly lines.")
           
           
    async def process_sar(self, message):
        chat_id = message.chat.id
        interval = '1d'
        text = message.text.split()
        symbol = text[1].upper() + '-USD' if len(text) > 1 else 'BTC-USD'
        if not symbol.upper().endswith('-USD'):
            symbol = f"{symbol.upper()}-USD"
        
        if len(text) > 2:
            interval = text[2].lower()
        if interval not in ['1d', '4h', '1w']:
            await self.bot.send_message(chat_id=chat_id, text="Invalid interval. Please use '1d', '4h', or '1w'.")

        try:
            photo_data = self.market_data.plot_ticker_with_sar(symbol, period='1mo', interval=interval)
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        except ValueError as e:
            await self.bot.send_message(chat_id=chat_id, text=f"Sorry, I couldn't generate the SAR")
            
            
    async def process_hash_ribbon(self, message):
        chat_id = message.chat.id
        try:
            photo_data = self.market_data.plot_hash_ribbons()
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        except ValueError as e:
            await self.bot.send_message(chat_id=chat_id, text=f"Sorry, I couldn't generate the hash ribbon")


    async def process_cmf(self, message):
        chat_id = message.chat.id
        text = message.text.split()
        symbol = text[1].upper() + '-USD' if len(text) > 1 else 'BTC-USD'
        if not symbol.upper().endswith('-USD'):
            symbol = f"{symbol.upper()}-USD"
        try:
            photo_data = self.market_data.plot_cmf(symbol)
        
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
            
        except ValueError as e:
            await self.bot.send_message(chat_id=chat_id, text=f"Sorry couldn't generate the CMF")
    
    async def process_sma_crossover(self, message=None, chat_id=None):
        if message and message.text:
            if chat_id is None:
                chat_id = message.chat.id
            if message.text.endswith('-USD'):    
                symbol=message.text
            else:
                text = message.text.split()
                symbol = text[1].upper() + '-USD' if len(text) > 1 else 'BTC-USD'
                if not symbol.upper().endswith('-USD'):
                        symbol = f"{symbol.upper()}-USD"
        try:
            photo_data, buy, date = self.market_data.plot_sma_crossovers(symbol)
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        except ValueError as e:
            await self.bot.send_message(chat_id=chat_id, text=f"Sorry, I couldn't generate the sma crossover")


    async def process_stoch(self, message=None, chat_id=None):
        interval = '1d'
        if message and message.text:
            if chat_id is None:
                chat_id = message.chat.id
            if message.text.endswith('-USD'):    
                    symbol=message.text
            else:
                text = message.text.split()
                symbol = text[1].upper() + '-USD' if len(text) > 1 else 'BTC-USD'
                if not symbol.upper().endswith('-USD'):
                    symbol = f"{symbol.upper()}-USD"
        
                if len(text) > 2:
                    interval = text[2].lower()
                if interval not in ['1d', '4h', '1wk']:
                    await self.bot.send_message(chat_id=chat_id, text="Invalid interval. Please use '1d', '4h', or '1wk'.")


    async def process_supertrend(self, message=None):
        chat_id = message.chat.id
        interval = '1d'
        periods = [10, 12] 
        atr_multipliers = [1.0, 3.0]
        text = message.text.split()
        symbol = text[1].upper() + '-USD' if len(text) > 1 else 'BTC-USD'
        if not symbol.upper().endswith('-USD'):
            symbol = f"{symbol.upper()}-USD"
        
        if len(text) > 2:
            interval = text[2].lower()
        if interval not in ['1d', '4h', '1wk']:
            await self.bot.send_message(chat_id=chat_id, text="Invalid interval. Please use '1d', '4h', or '1w'.")

        try:
            photo_data = self.market_data.supertrend(symbol, interval, periods, atr_multipliers)
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        except ValueError as e:
            await self.bot.send_message(chat_id=chat_id, text=f"Sorry, I couldn't generate the supertrend")


    async def process_emas(self, message=None):
        chat_id = message.chat.id
        interval = '4h'
        text = message.text.split()
        symbol = text[1].upper() + '-USD' if len(text) > 1 else 'BTC-USD'
        if not symbol.upper().endswith('-USD'):
            symbol = f"{symbol.upper()}-USD"
        
        if len(text) > 2:
            interval = text[2].lower()
        if interval not in ['1d', '4h', '1wk']:
            await self.bot.send_message(chat_id=chat_id, text="Invalid interval. Please use '1d', '4h', or '1w'.")

        try:
            photo_data, up, down = self.market_data.calculate_emas(symbol, interval)
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        except ValueError as e:
            await self.bot.send_message(chat_id=chat_id, text=f"Sorry, I couldn't generate the emas")

    async def process_bearish_engulf(self, message=None):
        chat_id = message.chat.id
        interval = '1d'
        detection="SMA50/200"
        text = message.text.split()
        symbol = text[1].upper() + '-USD' if len(text) > 1 else 'BTC-USD'
        if not symbol.upper().endswith('-USD'):
            symbol = f"{symbol.upper()}-USD"
        
        if len(text) > 2:
            interval = text[2].lower()
        if interval not in ['1d', '4h', '1wk']:
            await self.bot.send_message(chat_id=chat_id, text="Invalid interval. Please use '1d', '4h', or '1w'.")

        try:
            photo_data, signal_bearish = self.market_data.detect_bearish_engulfing(symbol, interval, detection)
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        except ValueError as e:
            await self.bot.send_message(chat_id=chat_id, text=f"Sorry, I couldn't generate the bearish_engulfing")
            

    async def process_bullish_engulf(self, message=None):
        chat_id = message.chat.id
        interval = '1d'
        detection="SMA50/200"
        text = message.text.split()
        symbol = text[1].upper() + '-USD' if len(text) > 1 else 'BTC-USD'
        if not symbol.upper().endswith('-USD'):
            symbol = f"{symbol.upper()}-USD"
        
        if len(text) > 2:
            interval = text[2].lower()
        if interval not in ['1d', '4h', '1wk']:
            await self.bot.send_message(chat_id=chat_id, text="Invalid interval. Please use '1d', '4h', or '1w'.")

        try:
            photo_data, signal_bearish = self.market_data.detect_bullish_engulfing(symbol, interval, detection)
            await self.bot.send_photo(chat_id=chat_id, photo=photo_data)
        except ValueError as e:
            await self.bot.send_message(chat_id=chat_id, text=f"Sorry, I couldn't generate the bullish_engulfing")


    async def process_3wrsi(self, message):
        chat_id = message.chat.id
        try:
            buf, cross_info = self.market_data.plot_rsi_fib_cross()
            await self.bot.send_photo(chat_id=chat_id, photo=buf)
        except ValueError as e:
            await self.bot.send_message(chat_id=chat_id, text=f"Sorry, I couldn't generate the 3w rsi") 

    async def process_everything(self, message):
        chat_id = message.chat.id
        text = message.text.split()
        symbol = text[1].upper() + '-USD' if len(text) > 1 else 'BTC-USD'

        mock_message = types.SimpleNamespace(chat=types.SimpleNamespace(id=chat_id), text=symbol)
        print(mock_message)
        await self.process_price_chart(mock_message)
        await self.process_bollinger_bands(mock_message)
        await self.process_daily_report(mock_message)
        await self.process_pi_cycle(mock_message)
        await self.process_two_yma(mock_message)
        await self.process_twohundredweek_ma(mock_message)
        await self.process_twohundredday_ma(mock_message)
        if symbol == 'BTC' or 'BTC-USD':
            await self.process_madr(mock_message)
            await self.process_smi(mock_message)
        await self.process_superguppy(mock_message)
        await self.process_bull_market_support_band(mock_message)
        await self.process_tenkanline(mock_message)
        await self.process_fibonacci_bollinger_bands(mock_message)
        await self.process_obv(mock_message)
        await self.process_rsi_monthly(mock_message)
        await self.process_sar(mock_message)
        await self.process_hash_ribbon(mock_message)   
        await self.process_cmf(mock_message)
        await self.process_sma_crossover(mock_message)
        await self.process_stoch(mock_message)
        await self.process_supertrend(mock_message)  
        await self.process_emas(mock_message)  
        await self.process_3wrsi(mock_message)
            
    ######################################################################################################################################
    async def periodic_task(self):
        await self.some_function()
        while True:
            await asyncio.sleep(1800)  # Sleep for 30 minutes
            await self.some_function()
            
    async def some_function(self):
        print("Executing periodic task...")
        market_data = MarketData()
        chat_ids = market_data.ids
        master_id = market_data.ids[0]
        second_id = market_data.ids[1]
        now = datetime.now()
        try:
            
            if now.weekday()==6 or now.weekday == 2:
                if now.hour==20 and self.check == True:
                    for chat_id in chat_ids:
                        await self.bot.send_message(chat_id=chat_id, text=f'Time for ya Wednesday or Sunday evening update!')
                        mock_message = types.SimpleNamespace(chat=types.SimpleNamespace(id=chat_id), text='BTC')
                        await self.process_price_chart(mock_message)
                        await self.process_bollinger_bands(mock_message)
                        await self.process_daily_report(mock_message)
                        await self.process_pi_cycle(mock_message)
                        await self.process_two_yma(mock_message)
                        await self.process_twohundredweek_ma(mock_message)
                        await self.process_twohundredday_ma(mock_message)
                        await self.process_madr(mock_message)
                        await self.process_smi(mock_message)
                        await self.process_superguppy(mock_message)
                        await self.process_bull_market_support_band(mock_message)
                        await self.process_tenkanline(mock_message)
                        await self.process_fibonacci_bollinger_bands(mock_message)
                        await self.process_obv(mock_message)
                        await self.process_rsi_monthly(mock_message)
                        await self.process_sar(mock_message)
                        await self.process_hash_ribbon(mock_message)  
                        await self.process_cmf(mock_message)
                        await self.process_sma_crossover(mock_message)
                        await self.process_stoch(mock_message)  
                        await self.process_supertrend(mock_message)   
                        await self.process_emas(mock_message)   
                        await self.process_3wrsi(mock_message)  
                        print('weekly shit')       
                    self.check=False       
            else:
                self.check=True
        except ValueError as e:
                    for chat_id in chat_ids:
                        await self.bot.send_message(chat_id=chat_id, text=f'Some error in the weekly update')
        
        for ticker in self.tickers:
            for chat_id in chat_ids:
                pass
            symbol = ticker.replace('-USD','')
            #print(symbol)
            price = self.market_data.get_current_price(symbol)
            #print(price)
            #if price < 0.1:  # Assuming tickers with prices less than .1 should have more decimals
               # formatted_price = f"{price:.6f}"
            #else:
                #formatted_price = f"{price:.2f}"
            ###Price alert################################################
            price_alert = 60000
            try:
                if ticker == 'BTC' and price > price_alert and not self.fetched_above_threshold['BTC']:
                    
                    await self.bot.send_message(chat_id=master_id, text=f"{ticker} alert going of: ${price}")
                    self.fetched_above_threshold['BTC'] = True
                
                #else:
                    #await self.bot.send_message(chat_id=chat_id, text=f"The current price of {ticker} is ${formatted_price}")
                
            # Reset fetched_above_threshold if price goes below 60000 after being above
                if ticker == 'BTC' and price <= price_alert:
                    self.fetched_above_threshold['BTC'] = False
                
            except ValueError as e:
                for chat_id in chat_ids:
                    await self.bot.send_message(chat_id=chat_id, text=f'Sorry, could not retrieve price for {ticker}')
                
                
            ####pi cycle######
        symbol='BTC'
        pi_im, pi_dif = self.market_data.pi_cycle_plot(symbol)
        pi_alert = 5000
            
        try:
            if ticker == 'BTC' and pi_dif is not None and pi_dif < pi_alert and not self.fetched_above_threshold['PiCycle']:
                for chat_id in chat_ids:
                    await self.bot.send_message(chat_id=chat_id, text=f'BTC Pi cycle going off')
                self.fetched_above_threshold['PiCycle'] = True

            elif ticker == 'BTC' and pi_dif is not None and pi_dif >= pi_alert:
                self.fetched_above_threshold['PiCycle'] = False

                

        except ValueError as e:
            await self.bot.send_message(chat_id=chat_id, text=f'Failed pi cycle')
            
            
            #####fibbol, test if basisline is tested######
        fibbol_im, fibbol_dif = self.market_data.plot_fibonacci_bollinger_bands()
        try:
            if fibbol_dif and not self.fetched_above_threshold['Fibbol']:
                for chat_id in chat_ids:
                    await self.bot.send_message(chat_id=chat_id, text=f'BTC Fibbol going off')
                    mock_message = types.SimpleNamespace(chat=types.SimpleNamespace(id=chat_id), text='BTC')
                    await self.process_fibonacci_bollinger_bands(mock_message)
                self.fetched_above_threshold['Fibbol'] = True
            elif fibbol_dif is False:
                self.fetched_above_threshold['Fibbol'] = False
        except ValueError as e:
            for chat_id in chat_ids:
                await self.bot.send_message(chat_id=chat_id, text=f'Failed bol basis band')
                
            ###########ris month##############
        rsi_im, rsi_value = self.market_data.plot_btc_rsi_collinear()

            # Define the thresholds
        high_rsi_threshold = 80
        low_rsi_threshold = 50

        try:
                # Check if RSI is above the high threshold and not already fetched
            if rsi_value >= high_rsi_threshold and not self.fetched_above_threshold['RSI_HIGH']:
                for chat_id in chat_ids:
                    await self.bot.send_message(chat_id=chat_id, text='RSI is quite high')
                    mock_message = types.SimpleNamespace(chat=types.SimpleNamespace(id=chat_id), text='BTC')
                    await self.process_rsi_monthly(mock_message)
                self.fetched_above_threshold['RSI_HIGH'] = True
                self.fetched_above_threshold['RSI_LOW'] = False  # Ensure RSI low flag is reset

                # Check if RSI is below the low threshold and not already fetched
            elif rsi_value < low_rsi_threshold and not self.fetched_above_threshold['RSI_LOW']:
                for chat_id in chat_ids:
                    await self.bot.send_message(chat_id=chat_id, text='RSI is quite low')
                    mock_message = types.SimpleNamespace(chat=types.SimpleNamespace(id=chat_id), text='BTC')
                    await self.process_rsi_monthly(mock_message)
                self.fetched_above_threshold['RSI_LOW'] = True
                self.fetched_above_threshold['RSI_HIGH'] = False  # Ensure RSI high flag is reset

                # Reset both flags if RSI is within thresholds
            elif low_rsi_threshold <= rsi_value <= high_rsi_threshold:
                self.fetched_above_threshold['RSI_HIGH'] = False
                self.fetched_above_threshold['RSI_LOW'] = False

        except ValueError as e:
            for chat_id in chat_ids:
                    await self.bot.send_message(chat_id=chat_id, text='Failed to process RSI data')

           ############stoch buy oppurtunity###############
        #for ticker in self.tickers[0]:
                   # try:
                     #   interval = '1wk'
                       # photo, kdata = self.market_data.calculate_stoch(ticker, interval)
                       # if kdata <20 and not self.fetched_above_threshold['Stoch']:
                           # for chat_id in chat_ids:
                               # await self.bot.send_message(chat_id=chat_id, text=f'Weekly Stoch of {ticker} value <20')
                                #mock_message = SimpleNamespace(chat=SimpleNamespace(id=chat_id), text=ticker)
                                #await self.process_stoch(mock_message, chat_id)
                                #self.fetched_above_threshold['Stoch'] = True
                          #  else:
                                #self.fetched_above_threshold['Stoch'] = False
                        
                   # except Exception as e:
                       # for chat_id in chat_ids:
                         #   print(f"Error processing stoch {ticker}: {e}")

 ##########sma crossover buy and sell#############
        for ticker in self.tickers:
            try:
                photo, signal, signal_date = self.market_data.plot_sma_crossovers(ticker)
                if signal in ['Buy', 'Sell']:
                    if signal_date is not None:
                        today = datetime.now().date()
                        
                        if (today - signal_date.date()).days <= 5:
                            last_signal_info = self.last_signal.get(ticker, {'signal': None, 'date': None})
                            if signal != last_signal_info['signal'] or last_signal_info['date'] != signal_date.date():
                                self.last_signal[ticker] = {'signal': signal, 'date': signal_date.date()}
                                
                                if signal == 'Buy':
                                    self.take_buy_action = True
                                    
                                    await self.bot.send_message(chat_id=master_id, text=f'Buy signal {ticker} given on SMA crossover')
                                    mock_message = SimpleNamespace(chat=SimpleNamespace(id=master_id), text=ticker)
                                    await self.process_sma_crossover(mock_message, master_id)
                                elif signal == 'Sell':
                                    self.take_sell_action = True
                                    
                                    await self.bot.send_message(chat_id=master_id, text=f'Sell signal {ticker} given on SMA crossover')
                                    mock_message = SimpleNamespace(chat=SimpleNamespace(id=master_id), text=ticker)
                                    await self.process_sma_crossover(mock_message, chat_id)
            except Exception as e:
                for chat_id in chat_ids:
                    print(f"Error processing sma crossover{ticker}: {e}")

        for ticker in self.tickers[0:1]:
            try:
                photo, signal, signal_date = self.market_data.plot_sma_crossovers(ticker)
                if signal in ['Buy', 'Sell']:
                    if signal_date is not None:
                        today = datetime.now().date()
                        
                        if (today - signal_date.date()).days <= 5:
                            last_signal_info = self.last_signal.get(ticker, {'signal': None, 'date': None})
                            if signal != last_signal_info['signal'] or last_signal_info['date'] != signal_date.date():
                                self.last_signal[ticker] = {'signal': signal, 'date': signal_date.date()}
                                
                                if signal == 'Buy':
                                    self.take_buy_action = True
                                    
                                    await self.bot.send_message(chat_id=second_id, text=f'Buy signal {ticker} given on SMA crossover')
                                    mock_message = SimpleNamespace(chat=SimpleNamespace(id=second_id), text=ticker)
                                    await self.process_sma_crossover(mock_message, chat_id)
                                elif signal == 'Sell':
                                    self.take_sell_action = True
                                    
                                    await self.bot.send_message(chat_id=second_id, text=f'Sell signal {ticker} given on SMA crossover')
                                    mock_message = SimpleNamespace(chat=SimpleNamespace(id=second_id), text=ticker)
                                    await self.process_sma_crossover(mock_message, chat_id)
            except Exception as e:
                for chat_id in chat_ids:
                    print(f"Error processing sma crossover{ticker}: {e}")   

        for ticker in self.tickers[0:9]:
            try:
                interval = '4h'
                photo1, upwards, downwards  = self.market_data.calculate_emas(ticker, interval)

                if upwards and not self.ema_buy_action:
                    await self.bot.send_message(chat_id=master_id, text=f'Upwards trend detected on {ticker}')
                    mock_message = SimpleNamespace(chat=SimpleNamespace(id=master_id), text=ticker)
                    #await self.process_emas(mock_message, chat_id)
                    self.ema_buy_action = True
                    self.ema_sell_action = False

                elif downwards and not self.ema_sell_action:
                    await self.bot.send_message(chat_id=master_id, text=f'Downwards trend detected on {ticker}')
                    mock_message = SimpleNamespace(chat=SimpleNamespace(id=master_id), text=ticker)
                    #await self.process_emas(mock_message, chat_id)
                    self.ema_sell_action = True
                    self.ema_buy_action = False
                
            except Exception as e:
                    for chat_id in chat_ids:
                        print(f"Error processing ema crossover{ticker}: {e}")     


        for ticker in self.tickers[0:9]:
            try:
                interval = '1d'
                detection="SMA50/200"
                photodata, signal_bearish = self.market_data.detect_bearish_engulfing(ticker, interval, detection)

                if signal_bearish and not self.bearish_engulf_action:
                    
                    await self.bot.send_message(chat_id=master_id, text=f'Bearish engulfing Sma Ema detected on {ticker}')
                    mock_message = SimpleNamespace(chat=SimpleNamespace(id=master_id), text=ticker)
                    self.bearish_engulf_action = True
                elif not signal_bearish:
                    self.bearish_engulf_action = False
            except Exception as e:
                    for chat_id in chat_ids:
                        print(f"Error processing bearish engulfing {ticker}: {e}")   

        try:
            interval = '1d'
            detection="SMA50/200"
            ticker = 'BTC-USD'
            photodata, signal_bullish = self.market_data.detect_bullish_engulfing(ticker, interval, detection)

            if signal_bullish and not self.bullish_engulf_action:
                await self.bot.send_message(chat_id=master_id, text=f'Bullish engulfing Sma Ema detected on {ticker}')
                mock_message = SimpleNamespace(chat=SimpleNamespace(id=master_id), text=ticker)
                self.bullish_engulf_action = True
            elif not signal_bearish:
                self.bullish_engulf_action = False
        except Exception as e:
                for chat_id in chat_ids:
                    print(f"Error processing bearish engulfing {ticker}: {e}")   


        try:
            buf, cross_info = self.market_data.plot_rsi_fib_cross()
            if cross_info and not self.fibrsi:
                await self.bot.send_message(chat_id=master_id,text=f"Recent RSI cross: {cross_info['type']} at {cross_info['level']*100:.1f}% on {cross_info['date']} with RSI {cross_info['value']:.2f}")
                mock_message = SimpleNamespace(chat=SimpleNamespace(id=master_id), text=ticker)
                await self.process_3wrsi(mock_message)
                self.fibrsi = True
            elif not cross_info:
                self.fibrsi = False
            else:
                print("No recent RSI cross detected.")
        except Exception as e:
                for chat_id in chat_ids:
                    print(f"Error processing 3w rsi {ticker}: {e}")   

    async def start_polling(self):
        try:
            await self.bot.polling(none_stop=True, interval=0, timeout=20)
        except Exception as e:
            print(f"Error starting polling: {e}")
        #finally:
            #await self.bot.close_session()

async def main(): 
    market_data = MarketData()
    user_id = market_data.ids
    bot = Bot(user_id, token = market_data.token)
    asyncio.create_task(bot.periodic_task())
    #print(f"Telegram bot token: {market_data.token}")
    #print("Starting bot polling...")
    await bot.start_polling()    

if __name__ == "__main__":
    asyncio.run(main())
