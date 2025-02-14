# Alpha Vantage

>[Alpha Vantage](https://www.alphavantage.co) Alpha Vantage provides realtime and historical financial market data through a set of powerful and developer-friendly data APIs and spreadsheets. 

Use the ``AlphaVantageAPIWrapper`` to get currency exchange rates.


```python
import getpass
import os

os.environ["ALPHAVANTAGE_API_KEY"] = getpass.getpass()
```


```python
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
```


```python
alpha_vantage = AlphaVantageAPIWrapper()
alpha_vantage._get_exchange_rate("USD", "JPY")
```




    {'Realtime Currency Exchange Rate': {'1. From_Currency Code': 'USD',
      '2. From_Currency Name': 'United States Dollar',
      '3. To_Currency Code': 'JPY',
      '4. To_Currency Name': 'Japanese Yen',
      '5. Exchange Rate': '148.19900000',
      '6. Last Refreshed': '2023-11-30 21:43:02',
      '7. Time Zone': 'UTC',
      '8. Bid Price': '148.19590000',
      '9. Ask Price': '148.20420000'}}



The `_get_time_series_daily` method returns the date, daily open, daily high, daily low, daily close, and daily volume of the global equity specified, covering the 100 latest data points.


```python
alpha_vantage._get_time_series_daily("IBM")
```

The `_get_time_series_weekly` method returns the last trading day of the week, weekly open, weekly high, weekly low, weekly close, and weekly volume of the global equity specified, covering 20+ years of historical data.


```python
alpha_vantage._get_time_series_weekly("IBM")
```

The `_get_quote_endpoint` method is a lightweight alternative to the time series APIs and returns the latest price and volume info for the specified symbol.


```python
alpha_vantage._get_quote_endpoint("IBM")
```




    {'Global Quote': {'01. symbol': 'IBM',
      '02. open': '156.9000',
      '03. high': '158.6000',
      '04. low': '156.8900',
      '05. price': '158.5400',
      '06. volume': '6640217',
      '07. latest trading day': '2023-11-30',
      '08. previous close': '156.4100',
      '09. change': '2.1300',
      '10. change percent': '1.3618%'}}



The `search_symbol` method returns a list of symbols and the matching company information based on the text entered.


```python
alpha_vantage.search_symbols("IB")
```

The `_get_market_news_sentiment` method returns live and historical market news sentiment for a given asset.


```python
alpha_vantage._get_market_news_sentiment("IBM")
```

The `_get_top_gainers_losers` method returns the top 20 gainers, losers and most active stocks in the US market.


```python
alpha_vantage._get_top_gainers_losers()
```

The `run` method of the wrapper takes the following parameters: from_currency, to_currency. 

It Gets the currency exchange rates for the given currency pair.


```python
alpha_vantage.run("USD", "JPY")
```




    {'1. From_Currency Code': 'USD',
     '2. From_Currency Name': 'United States Dollar',
     '3. To_Currency Code': 'JPY',
     '4. To_Currency Name': 'Japanese Yen',
     '5. Exchange Rate': '148.19900000',
     '6. Last Refreshed': '2023-11-30 21:43:02',
     '7. Time Zone': 'UTC',
     '8. Bid Price': '148.19590000',
     '9. Ask Price': '148.20420000'}


